# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class PrototypeManager:
    """全局类别原型管理器，用于跨模态原型对比学习。

    维护每个模态每个类别的原型向量，通过 EMA 在服务端聚合更新。
    替代逐样本 MSE 回归，用原型级对比损失实现更鲁棒的跨模态对齐。
    """

    def __init__(self, num_classes, feature_dim, modality_names, device='cpu',
                 ema_momentum=0.9, proto_temperature=0.1):
        """
        Args:
            num_classes: 类别数
            feature_dim: 每个模态的特征维度 (int)
            modality_names: 模态特征名列表，如 ["acc_feat", "gyro_feat"]
            device: 存储设备
            ema_momentum: 原型 EMA 更新动量
            proto_temperature: 对比损失温度
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.modality_names = list(modality_names)
        self.device = device
        self.ema_momentum = ema_momentum
        self.proto_temperature = proto_temperature

        # 全局原型: {mod_name: Tensor[num_classes, feature_dim]}
        self.prototypes = {}
        for mod_name in self.modality_names:
            self.prototypes[mod_name] = torch.zeros(num_classes, feature_dim, device=device)

        # 每类更新计数: {mod_name: Tensor[num_classes]}
        self.prototype_counts = {}
        for mod_name in self.modality_names:
            self.prototype_counts[mod_name] = torch.zeros(num_classes, device=device)

    def update(self, local_prototypes, local_counts):
        """用客户端上传的局部原型做 EMA 更新。

        Args:
            local_prototypes: {mod_name: Tensor[num_classes, dim]}
            local_counts: {mod_name: Tensor[num_classes]}
        """
        for mod_name in self.modality_names:
            if mod_name not in local_prototypes:
                continue
            if mod_name not in local_counts:
                continue
            lp = local_prototypes[mod_name].to(self.device)
            lc = local_counts[mod_name].to(self.device)
            for c in range(self.num_classes):
                n = lc[c].item()
                if n <= 0:
                    continue
                if self.prototype_counts[mod_name][c] == 0:
                    self.prototypes[mod_name][c] = lp[c].detach()
                else:
                    self.prototypes[mod_name][c] = (
                        self.ema_momentum * self.prototypes[mod_name][c]
                        + (1.0 - self.ema_momentum) * lp[c].detach()
                    )
                self.prototype_counts[mod_name][c] += n

    def get_prototypes(self, mod_name):
        """获取某模态的全局原型矩阵 [num_classes, dim]"""
        return self.prototypes[mod_name]

    def has_prototypes(self, mod_name, class_idx=None):
        """检查原型是否已初始化（至少有一个类别有非零原型）"""
        if class_idx is not None:
            return self.prototype_counts[mod_name][class_idx] > 0
        return (self.prototype_counts[mod_name] > 0).any()

    def num_valid_classes(self, mod_name):
        """有效类别原型数（count>0 的类别数）。"""
        return int((self.prototype_counts[mod_name] > 0).sum().item())

    def get_stats(self):
        """诊断用：返回各模态原型统计信息。"""
        stats = {}
        for mod_name in self.modality_names:
            counts = self.prototype_counts[mod_name]
            protos = self.prototypes[mod_name]
            active_mask = counts > 0
            n_active = active_mask.sum().item()
            total = self.num_classes
            if n_active > 0:
                active_protos = protos[active_mask]  # [n_active, dim]
                mean_norm = active_protos.norm(p=2, dim=1).mean().item()
                # 原型间余弦方差 (衡量多样性)
                active_norm = F.normalize(active_protos, p=2, dim=1)
                cos_sim = torch.matmul(active_norm, active_norm.T)  # [n_active, n_active]
                # 排除对角线
                mask = ~torch.eye(n_active, dtype=torch.bool, device=cos_sim.device)
                cosine_variance = cos_sim[mask].var().item() if mask.sum() > 0 else 0.0
            else:
                mean_norm = 0.0
                cosine_variance = 0.0
            stats[mod_name] = {
                'active': n_active,
                'total': total,
                'mean_norm': mean_norm,
                'cosine_variance': cosine_variance
            }
        return stats

    def compute_contrastive_loss(self, projected_features, labels, mod_name,
                                  multilabel=False, support_reference=None, return_info=False):
        """计算原型对比损失。

        Args:
            projected_features: [B, dim] 跨模态投影后的特征
            labels: [B] (单标签) 或 [B, num_classes] (多标签 one-hot)
            mod_name: 目标模态名
            multilabel: 是否多标签
            support_reference: None -> 旧行为（无可靠性门控，FedPCD-Old 用）。
                若为数值 S0（FedPCD-New），启用类别-模态级可靠性门控:
                  * 正类原型不存在(count==0)的样本被跳过;
                  * 对比分母只用有效类别(count>0)的原型;
                  * 每样本按 q = min(1, count[y]/S0) 加权;
                  * 有效类别<2 时对比损失无意义，返回 0（info 标记）。
            return_info: True 时额外返回 info dict（用于跳过原因统计，FedPCD-New 用）。

        Returns:
            loss (或 (loss, info) 当 return_info=True)
        """
        device = projected_features.device
        info = {'n_valid_classes': self.num_valid_classes(mod_name),
                'n_pos_valid': 0, 'n_total': projected_features.size(0),
                'skipped_reason': None}

        if not self.has_prototypes(mod_name):
            info['skipped_reason'] = 'no_prototypes'
            return (torch.tensor(0.0, device=device), info) if return_info else torch.tensor(0.0, device=device)

        # Server prototypes are FIXED targets: never backprop into the prototype bank.
        prototypes = self.prototypes[mod_name].detach()

        # L2 归一化
        proj_norm = F.normalize(projected_features, p=2, dim=1)  # [B, dim]
        proto_norm = F.normalize(prototypes, p=2, dim=1)         # [C, dim]

        # 相似度矩阵: [B, C]
        sim = torch.matmul(proj_norm, proto_norm.T) / self.proto_temperature

        if not multilabel:
            if support_reference is None:
                # 旧行为（FedPCD-Old）
                loss = F.cross_entropy(sim, labels, reduction='mean')
                return (loss, info) if return_info else loss
            else:
                # FedPCD-New 可靠性门控
                counts = self.prototype_counts[mod_name]            # [C]
                valid_class = counts > 0                             # [C]
                # 有效类别 < 2: 对比损失退化(分母只剩正类)，无有效梯度 -> 跳过
                if int(valid_class.sum()) < 2:
                    info['skipped_reason'] = 'insufficient_classes'
                    loss = torch.tensor(0.0, device=device)
                    return (loss, info) if return_info else loss
                # 分母只用有效类别原型（无效类别置 -inf → softmax 权重 0）
                sim_masked = sim.masked_fill(~valid_class.unsqueeze(0), float('-inf'))
                pos_valid = valid_class[labels]                     # [B] 正类原型是否存在
                info['n_pos_valid'] = int(pos_valid.sum().item())
                if pos_valid.sum() == 0:
                    info['skipped_reason'] = 'no_positive_proto'
                    loss = torch.tensor(0.0, device=device)
                    return (loss, info) if return_info else loss
                per = F.cross_entropy(sim_masked[pos_valid], labels[pos_valid], reduction='none')
                rel = (counts[labels[pos_valid]] / float(support_reference)).clamp(max=1.0)
                loss = (rel * per).sum() / pos_valid.sum().clamp(min=1).float()
                info['mean_reliability'] = float(rel.mean().item())
                return (loss, info) if return_info else loss
        else:
            # 多标签: 多正类对比损失
            # labels: [B, C], 每行可能有多个 1
            # 对每个正类计算正样本 log-sum-exp
            pos_mask = labels.bool()   # [B, C]
            neg_mask = ~pos_mask       # [B, C]

            # 负样本 log-sum-exp: [B]
            neg_sim = sim.masked_fill(pos_mask, float('-inf'))
            lse_neg = torch.logsumexp(neg_sim, dim=1)  # [B]

            # 正样本平均: [B]
            pos_sim = sim.masked_fill(neg_mask, float('-inf'))
            # 对每个样本，计算所有正类的 log-sum-exp，然后减去 log(N_pos)
            lse_pos = torch.logsumexp(pos_sim, dim=1)  # [B]
            n_pos = pos_mask.sum(dim=1).clamp(min=1)    # [B]

            # 损失: -log( Σ_{p∈P} exp(sim_p/τ) / ( Σ_{p∈P} exp(sim_p/τ) + Σ_{n∈N} exp(sim_n/τ) ) )
            # = -log( exp(lse_pos) / (exp(lse_pos) + exp(lse_neg)) )
            # 数值稳定版本:
            # = -lse_pos + log(exp(lse_pos) + exp(lse_neg))
            max_lse = torch.max(lse_pos, lse_neg)
            loss = -lse_pos + max_lse + torch.log(
                torch.exp(lse_pos - max_lse) + torch.exp(lse_neg - max_lse) + 1e-8
            )
            # 按正类数量归一化，避免多正类样本主导
            loss = (loss / n_pos).mean()

        return (loss, info) if return_info else loss

    def state_dict(self):
        return {
            'prototypes': {k: v.cpu().clone() for k, v in self.prototypes.items()},
            'prototype_counts': {k: v.cpu().clone() for k, v in self.prototype_counts.items()},
        }

    def load_state_dict(self, state_dict):
        for mod_name in self.modality_names:
            if mod_name in state_dict['prototypes']:
                self.prototypes[mod_name] = state_dict['prototypes'][mod_name].to(self.device)
            if mod_name in state_dict['prototype_counts']:
                self.prototype_counts[mod_name] = state_dict['prototype_counts'][mod_name].to(self.device)

    def to(self, device):
        self.device = device
        for mod_name in self.modality_names:
            self.prototypes[mod_name] = self.prototypes[mod_name].to(device)
            self.prototype_counts[mod_name] = self.prototype_counts[mod_name].to(device)
        return self


class ProtoGenManager:
    """Prototype-based Feature Generation (P3).

    Maintains per-class prototype pairs across modalities and learns a mapping
    between them. When one modality is missing, generates pseudo-features by
    mapping prototypes from the available modality.
    """

    def __init__(self, num_classes, feature_dim, modality_names, device='cpu',
                 ema_momentum=0.97):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.modality_names = list(modality_names)
        self.device = device
        self.ema_momentum = ema_momentum

        # Global prototypes: {mod_name: [num_classes, dim]}
        self.prototypes = {}
        self.proto_counts = {}
        for mod_name in self.modality_names:
            self.prototypes[mod_name] = torch.zeros(num_classes, feature_dim, device=device)
            self.proto_counts[mod_name] = torch.zeros(num_classes, device=device)

        # Prototype mapping networks: mod_a ↔ mod_b (for first two modalities)
        if len(modality_names) >= 2:
            self.map_a_to_b = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            ).to(device)
            self.map_b_to_a = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            ).to(device)

    def update(self, local_prototypes, local_counts):
        for mod_name in self.modality_names:
            if mod_name not in local_prototypes or mod_name not in local_counts:
                continue
            lp = local_prototypes[mod_name].to(self.device)
            lc = local_counts[mod_name].to(self.device)
            for c in range(self.num_classes):
                n = lc[c].item()
                if n <= 0:
                    continue
                if self.proto_counts[mod_name][c] == 0:
                    self.prototypes[mod_name][c] = lp[c].detach()
                else:
                    self.prototypes[mod_name][c] = (
                        self.ema_momentum * self.prototypes[mod_name][c]
                        + (1.0 - self.ema_momentum) * lp[c].detach()
                    )
                self.proto_counts[mod_name][c] += n

    def compute_gen_loss(self, labels):
        """Compute prototype-level cross-modal generation loss.

        Maps prototypes across modalities and penalizes inconsistency.
        Loss = MSE(mapped_proto_a, proto_b) + MSE(mapped_proto_b, proto_a)

        Args:
            labels: [B] class indices (only classes present in batch are used)

        Returns:
            scalar loss
        """
        if len(self.modality_names) < 2:
            return torch.tensor(0.0, device=self.device)

        mod_a = self.modality_names[0]
        mod_b = self.modality_names[1]

        # Get unique classes that have prototypes in both modalities
        mask_a = self.proto_counts[mod_a] > 0
        mask_b = self.proto_counts[mod_b] > 0
        valid = mask_a & mask_b
        if valid.sum() < 2:
            return torch.tensor(0.0, device=self.device)

        proto_a = self.prototypes[mod_a][valid]  # [K, dim]
        proto_b = self.prototypes[mod_b][valid]  # [K, dim]

        mapped_a_to_b = self.map_a_to_b(proto_a)
        mapped_b_to_a = self.map_b_to_a(proto_b)

        loss = F.mse_loss(mapped_a_to_b, proto_b) + F.mse_loss(mapped_b_to_a, proto_a)
        return loss

    def generate_missing_features(self, present_feat, present_mod_name, target_mod_name, labels=None):
        """Generate features for the missing modality using prototype mapping.

        Args:
            present_feat: [B, dim] features from the available modality
            present_mod_name: name of the available modality
            target_mod_name: name of the missing modality
            labels: optional [B] class labels for prototype lookup

        Returns:
            generated_feat: [B, dim] estimated features for the missing modality
        """
        B = present_feat.size(0)
        proto_present = self.prototypes[present_mod_name]  # [C, dim]
        proto_target = self.prototypes[target_mod_name]    # [C, dim]

        # Use class labels if available, otherwise nearest prototype
        if labels is not None:
            generated = torch.zeros(B, self.feature_dim, device=present_feat.device)
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() == 0:
                    continue
                if self.proto_counts[present_mod_name][c] > 0 and self.proto_counts[target_mod_name][c] > 0:
                    p_present = self.prototypes[present_mod_name][c:c+1]  # [1, dim]
                    p_target = self.prototypes[target_mod_name][c:c+1]    # [1, dim]
                    if present_mod_name == self.modality_names[0]:
                        mapped = self.map_a_to_b(p_present)
                        # Blend: mapped prototype + direct prototype
                        gen = 0.5 * mapped + 0.5 * p_target
                    else:
                        mapped = self.map_b_to_a(p_present)
                        gen = 0.5 * mapped + 0.5 * p_target
                    generated[mask] = gen.expand(mask.sum(), -1)
                else:
                    generated[mask] = torch.zeros(self.feature_dim, device=present_feat.device)
            return generated
        else:
            # No labels: use nearest prototype
            sim = F.cosine_similarity(
                F.normalize(present_feat, dim=1).unsqueeze(1),  # [B, 1, dim]
                F.normalize(proto_present, dim=1).unsqueeze(0),  # [1, C, dim]
                dim=2
            )  # [B, C]
            nearest = sim.argmax(dim=1)  # [B]
            generated = torch.zeros(B, self.feature_dim, device=present_feat.device)
            for i in range(B):
                c = nearest[i].item()
                if self.proto_counts[target_mod_name][c] > 0:
                    generated[i] = self.prototypes[target_mod_name][c]
            return generated

    def get_parameters(self):
        """Return mapping network parameters for federated aggregation."""
        params = list(self.map_a_to_b.parameters()) + list(self.map_b_to_a.parameters())
        return params

    def state_dict(self):
        return {
            'prototypes': {k: v.cpu().clone() for k, v in self.prototypes.items()},
            'proto_counts': {k: v.cpu().clone() for k, v in self.proto_counts.items()},
            'map_a_to_b': self.map_a_to_b.state_dict(),
            'map_b_to_a': self.map_b_to_a.state_dict(),
        }

    def load_state_dict(self, state_dict):
        for mod_name in self.modality_names:
            if mod_name in state_dict['prototypes']:
                self.prototypes[mod_name] = state_dict['prototypes'][mod_name].to(self.device)
            if mod_name in state_dict['proto_counts']:
                self.proto_counts[mod_name] = state_dict['proto_counts'][mod_name].to(self.device)
        if 'map_a_to_b' in state_dict:
            self.map_a_to_b.load_state_dict(state_dict['map_a_to_b'])
        if 'map_b_to_a' in state_dict:
            self.map_b_to_a.load_state_dict(state_dict['map_b_to_a'])

    def to(self, device):
        self.device = device
        for mod_name in self.modality_names:
            self.prototypes[mod_name] = self.prototypes[mod_name].to(device)
            self.proto_counts[mod_name] = self.proto_counts[mod_name].to(device)
        self.map_a_to_b = self.map_a_to_b.to(device)
        self.map_b_to_a = self.map_b_to_a.to(device)
        return self


def compute_local_prototypes(features, labels, num_classes, multilabel=False):
    """在客户端本地计算每个类别的平均特征（原型）。

    Args:
        features: [B, dim] 模态特征
        labels: [B] (单标签) 或 [B, C] (多标签 one-hot)
        num_classes: 类别数
        multilabel: 是否多标签

    Returns:
        local_protos: [num_classes, dim] 各类别特征均值
        local_counts: [num_classes] 各类别样本数
    """
    dim = features.size(1)
    device = features.device
    local_protos = torch.zeros(num_classes, dim, device=device)
    local_counts = torch.zeros(num_classes, device=device)

    if not multilabel:
        for c in range(num_classes):
            mask = (labels == c)
            n = mask.sum().item()
            if n > 0:
                local_protos[c] = features[mask].mean(dim=0)
                local_counts[c] = n
    else:
        # 多标签: 一个样本可能属于多个类别
        for c in range(num_classes):
            if labels.dim() == 1:
                # labels 是浮点 multi-hot (已在 dataloader 中转为 multi-hot)
                mask = (labels == c)
            else:
                mask = (labels[:, c] > 0.5)
            n = mask.sum().item()
            if n > 0:
                local_protos[c] = features[mask].mean(dim=0)
                local_counts[c] = n

    return local_protos, local_counts
