# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleDistillationLoss(nn.Module):
    """双蒸馏损失：logit层 + 特征层（共同模态对齐 + 跨模态蒸馏）。

    跨模态投影头已移入主模型（全局共享，参与联邦聚合），
    本模块只负责计算损失，不再持有投影头参数。

    支持两种跨模态损失模式：
    - 'mse': 逐样本 MSE 回归（原始方法，适合冗余模态）
    - 'proto': 原型对比损失（新增，适合互补模态）
    """

    def __init__(self, temperature=3.0, alpha_feat=0.5, lambda_cross=0.3,
                 modality_names=None, multilabel=False,
                 cross_mode='proto', proto_temperature=0.1,
                 en_logit=True, en_feat=True, en_cross=True,
                 ):
        super().__init__()
        self.temperature = temperature
        self.alpha_feat = alpha_feat
        self.lambda_cross = lambda_cross
        self.multilabel = multilabel
        self.cross_mode = cross_mode      # 'mse' or 'proto'
        self.proto_temperature = proto_temperature
        self.en_logit = en_logit          # enable logit-level KD
        self.en_feat = en_feat            # enable feature-level MSE
        self.en_cross = en_cross          # enable cross-modal MSE
        if modality_names is None:
            modality_names = ["acc_feat", "gyro_feat"]
        self.modality_names = modality_names

    def forward(self, student_logits, teacher_logits,
                student_feats, teacher_feats,
                cross_modal_projections=None,
                missing_modalities=None,
                cross_modalities=None,
                labels=None, criterion_task=None,
                prototype_manager=None):
        """
        Args:
            student_logits, teacher_logits: [B, num_classes]
            student_feats, teacher_feats: dict with modality feature keys
            cross_modal_projections: dict, 投影头输出 {mod_name_cross_proj: [B, d_hid]}
            missing_modalities: 学生缺失的模态名称列表（用于知识保持，仅对齐可用模态）
            cross_modalities: 需计算 Lcross 的目标模态列表（完整客户端传入所有模态，
                             缺失客户端传入空列表）
            labels: [B] (单标签) 或 [B, num_classes] (多标签)
            criterion_task: 任务损失函数 (可选)
            prototype_manager: PrototypeManager 实例 (cross_mode='proto' 时需要)

        Returns:
            loss_task, loss_logit, loss_feat_common, loss_feat_cross
        """
        # 1. 任务损失
        if labels is not None and criterion_task is not None:
            if not hasattr(criterion_task, 'multilabel') or not criterion_task.multilabel:
                soft_outputs = torch.log_softmax(student_logits, dim=1)
                loss_task = criterion_task(soft_outputs, labels)
            else:
                loss_task = criterion_task(student_logits, labels)
        else:
            loss_task = torch.tensor(0.0, device=student_logits.device)

        # 2. Logit 层蒸馏损失
        T = self.temperature
        if self.multilabel:
            # 余弦 KD：匹配预测方向（类别排序），而非精确概率值
            # 与 BCE 任务损失互补，对 FL 教师噪声更鲁棒
            # 用温度锐化教师 logits，放大教师确信的类别差异
            teacher_sharp = teacher_logits / T
            loss_logit = 1.0 - F.cosine_similarity(
                student_logits, teacher_sharp.detach(), dim=1).mean()
        else:
            p_s = F.log_softmax(student_logits / T, dim=1)
            p_t = F.softmax(teacher_logits / T, dim=1)
            loss_logit = F.kl_div(p_s, p_t, reduction='batchmean') * (T ** 2)

        # 3. 共同模态特征蒸馏损失（仅对齐学生存在的模态）
        loss_feat_common = torch.tensor(0.0, device=student_logits.device)
        valid_modalities = 0

        if missing_modalities is not None:
            present_modalities = set(self.modality_names) - set(missing_modalities)
        else:
            present_modalities = set(self.modality_names)

        for mod_name in present_modalities:
            if (student_feats.get(mod_name) is not None
                    and teacher_feats.get(mod_name) is not None):
                loss_feat_common = loss_feat_common + F.mse_loss(
                    student_feats[mod_name], teacher_feats[mod_name].detach())
                valid_modalities += 1

        if valid_modalities > 0:
            loss_feat_common = loss_feat_common / valid_modalities

        # 4. 跨模态特征蒸馏损失（仅在完整模态客户端上计算）
        loss_feat_cross = torch.tensor(0.0, device=student_logits.device)

        if (cross_modal_projections is not None
                and cross_modalities is not None and len(cross_modalities) > 0):
            n_cross = 0

            if self.cross_mode == 'mse':
                # 完整客户端：利用配对双模态特征学习跨模态映射
                for mod_name in cross_modalities:
                    proj_key = f"{mod_name}_cross_proj"
                    if (cross_modal_projections.get(proj_key) is not None
                            and teacher_feats.get(mod_name) is not None):
                        projected = cross_modal_projections[proj_key]
                        loss_feat_cross = loss_feat_cross + F.mse_loss(
                            projected, teacher_feats[mod_name].detach())
                        n_cross += 1
                if n_cross > 0:
                    loss_feat_cross = loss_feat_cross / n_cross

            elif self.cross_mode == 'proto':
                # 原型对比损失: 投影 → 拉近正类原型 / 推远负类原型
                if prototype_manager is not None and labels is not None:
                    for mod_name in cross_modalities:
                        proj_key = f"{mod_name}_cross_proj"
                        if cross_modal_projections.get(proj_key) is None:
                            continue
                        projected = cross_modal_projections[proj_key]  # [B, d_hid]

                        # 将 labels 转为用于 prototype_manager 的格式
                        if self.multilabel:
                            # PTB-XL: labels 是 multi-hot [B, C]
                            label_for_proto = labels
                        else:
                            # UCI-HAR: labels 是类别索引 [B]
                            label_for_proto = labels

                        proto_loss = prototype_manager.compute_contrastive_loss(
                            projected, label_for_proto, mod_name,
                            multilabel=self.multilabel
                        )
                        loss_feat_cross = loss_feat_cross + proto_loss
                        n_cross += 1
                    if n_cross > 0:
                        loss_feat_cross = loss_feat_cross / n_cross

        # Apply component gating for ablation studies
        if not self.en_logit:
            loss_logit = torch.tensor(0.0, device=loss_logit.device)
        if not self.en_feat:
            loss_feat_common = torch.tensor(0.0, device=loss_feat_common.device)
        if not self.en_cross:
            loss_feat_cross = torch.tensor(0.0, device=loss_feat_cross.device)

        return loss_task, loss_logit, loss_feat_common, loss_feat_cross
