# ---------------------------------------------------------------------------
# Derived from FedMultimodal (https://github.com/usc-sail/fed-multimodal),
# licensed under the Apache License, Version 2.0.
# Modified for FedPCD (2026). See NOTICE for a summary of changes.
# ---------------------------------------------------------------------------
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch
import torch.nn.functional as F
import random

from torch import nn
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score

# import optimizer
from .optimizer import FedProxOptimizer

from utils.double_distillation import DoubleDistillationLoss
from utils.prototype_manager import compute_local_prototypes

warnings.filterwarnings('ignore')
from .evaluation import EvalMetric


class ClientFedAvg(object):
    def __init__(
        self,
        args,
        device,
        criterion,
        dataloader,
        model,
        label_dict=None,
        num_class=None,
        distill_scheduler=None,
        current_epoch=0,
        missing_rate=0.0,
        prototype_manager=None
    ):
        self.args = args
        self.model = model

        # 蒸馏相关设置
        self.distill_scheduler = distill_scheduler
        self.current_epoch = current_epoch
        self.missing_rate = missing_rate
        self.prototype_manager = prototype_manager
        self.num_class = num_class

        # 扩展配置
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        self.en_double_distill = getattr(args, 'en_double_distill', False)
        self.en_proto_cross = getattr(args, 'en_proto_cross', False)
        # FedPCD-New leak-free path defaults (overridden below when DD is enabled)
        self.leak_free_cross = getattr(args, 'leak_free_cross', False)
        self.leakfree_loss = None
        # Checkpoint 4C task-coupled completion
        self.en_task_coupled = getattr(args, 'en_task_coupled', False)
        self.lambda_task_rec = getattr(args, 'lambda_task_rec', 0.5)
        self.task_rec_warmup = getattr(args, 'task_rec_warmup', 10)

        # 教师模型（用于蒸馏）- 只有当启用蒸馏时才创建
        if distill_scheduler and getattr(args, 'en_distill', False):
                self.teacher_model = copy.deepcopy(model)
                self.teacher_model.eval()
        else:
                self.teacher_model = None

        # 双蒸馏损失函数（不再持有投影头，投影头已在主模型中）
        if self.en_double_distill and self.teacher_model is not None:
            if args.dataset == 'ptb-xl':
                modality_names = ["i_to_avf_feat", "v1_to_v6_feat"]
            elif args.dataset in ["hateful_memes", "crisis-mmd"]:
                modality_names = ["img_feat", "text_feat"]
            elif args.dataset == "meld":
                modality_names = ["audio_feat", "text_feat"]
                if hasattr(args, 'video_feat') and args.video_feat is not None:
                    modality_names.append("video_feat")
            elif args.dataset == "mhealth" and hasattr(args, 'mag_feat'):
                modality_names = ["acc_feat", "gyro_feat", "mag_feat"]
            else:
                modality_names = ["acc_feat", "gyro_feat"]

            cross_mode = 'proto' if self.en_proto_cross else 'mse'
            # Parse DD component flags for ablation studies
            dd_comp_str = getattr(args, 'dd_components', 'logit+feat+cross')
            dd_components = set(dd_comp_str.lower().split('+'))

            # FedPCD-New leak-free path: knowledge preservation (logit+feat) is kept,
            # but the old mutually-exclusive cross slot is DISABLED here; the leak-free
            # rec/proto terms are computed separately by LeakFreeCrossLoss.
            self.leak_free_cross = getattr(args, 'leak_free_cross', False)
            en_cross_dd = ('cross' in dd_components) and (not self.leak_free_cross)

            self.double_distill_loss = DoubleDistillationLoss(
                temperature=1.0,  # 温度由调度器控制
                alpha_feat=getattr(args, 'alpha_feat', 0.5),
                lambda_cross=getattr(args, 'lambda_cross', 0.3),
                modality_names=modality_names,
                multilabel=self.multilabel,
                cross_mode=cross_mode,
                proto_temperature=getattr(args, 'proto_temperature', 0.1),
                en_logit=('logit' in dd_components),
                en_feat=('feat' in dd_components),
                en_cross=en_cross_dd,
            )

            # Leak-free cross-modal loss module (FedPCD-New)
            if self.leak_free_cross:
                from utils.leakfree_cross import LeakFreeCrossLoss
                lf_str = getattr(args, 'leakfree_components', 'rec+proto_paired+proto_missing')
                lf_comps = set(lf_str.lower().split('+'))
                self.lf_use_rec = ('rec' in lf_comps)
                self.lf_use_proto_paired = ('proto_paired' in lf_comps)
                self.lf_use_proto_missing = ('proto_missing' in lf_comps)
                self.lf_use_proto_same = ('proto_same' in lf_comps)  # C1: same-modality control
                self.lf_lambda_rec = getattr(args, 'lambda_rec', 1.0)
                self.lf_lambda_proto = getattr(args, 'lambda_proto', 0.05)
                self.lf_warmup_start = getattr(args, 'proto_warmup_start', 0)
                self.lf_warmup_len = getattr(args, 'proto_warmup_len', 0)
                # class-modality reliability gate: S0. Default 32 (Full model). S0=1 -> correctness
                # gating only (skip empty-positive-class samples; mask invalid classes; <2 valid
                # classes skipped). S0>1 additionally soft-weights by min(1,count/S0).
                self.lf_support_reference = getattr(args, 'lf_proto_support_ref', 32)
                self.leakfree_loss = LeakFreeCrossLoss(
                    modality_names=modality_names, multilabel=self.multilabel)
            else:
                self.leakfree_loss = None
        else:
            self.double_distill_loss = None

        # 局部原型累积器
        self.local_proto_sums = None    # {mod_name: Tensor[num_classes, dim]}
        self.local_proto_counts = None  # {mod_name: Tensor[num_classes]}

        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader


    def get_parameters(self):
        # Return model parameters
        sd = self.model.state_dict()
        # Ablation: exclude cross-modal projection heads from FedAvg
        if getattr(self.args, 'no_federated_proj', False):
            sd = {k: v for k, v in sd.items()
                  if 'cross_modal_heads' not in k and 'cross_modal_adapt' not in k}
        return sd

    def get_model_result(self):
        # Return model results
        return self.result

    def get_test_true(self):
        # Return test labels
        return self.test_true

    def get_test_pred(self):
        # Return test predictions
        return self.test_pred

    def get_train_groundtruth(self):
        # Return groundtruth used for training
        return self.train_groundtruth

    def get_local_prototypes(self):
        """返回本地累积的原型均值和计数。"""
        if self.local_proto_sums is None:
            return None, None
        protos, counts = dict(), dict()
        for mod_name in self.local_proto_sums:
            s = self.local_proto_sums[mod_name]
            c = self.local_proto_counts[mod_name]
            mask = (c > 0)
            avg = torch.zeros_like(s)
            avg[mask] = s[mask] / c[mask].unsqueeze(1)
            protos[mod_name] = avg
            counts[mod_name] = c.clone()
        return protos, counts

    def _init_proto_accumulators(self, modality_names, dim):
        """初始化原型累积器。"""
        self.local_proto_sums = {}
        self.local_proto_counts = {}
        for mod_name in modality_names:
            self.local_proto_sums[mod_name] = torch.zeros(self.num_class, dim, device=self.device)
            self.local_proto_counts[mod_name] = torch.zeros(self.num_class, device=self.device)

    def _accumulate_prototypes(self, feats, labels, modality_names, teacher_logits=None):
        """累积 teacher 特征用于计算局部原型。

        多标签场景下，使用 teacher logit argmax 做硬分配，
        避免同一特征分散到多个类原型导致坍塌。
        """
        if self.local_proto_sums is None:
            dim = list(feats.values())[0].size(1)
            self._init_proto_accumulators(modality_names, dim)

        # 确保所有 modality_names 都有累加器（首个 batch 可能缺失某模态，后续 batch 又出现）
        for mod_name in modality_names:
            if mod_name not in self.local_proto_sums:
                dim = list(feats.values())[0].size(1)
                self.local_proto_sums[mod_name] = torch.zeros(self.num_class, dim, device=self.device)
                self.local_proto_counts[mod_name] = torch.zeros(self.num_class, device=self.device)

        # 多标签 → 单标签硬分配：每个样本只分配给 teacher 预测最强的那个类
        if self.multilabel and teacher_logits is not None:
            labels = teacher_logits.argmax(dim=1)  # [B]
            multilabel = False
        else:
            multilabel = self.multilabel

        for mod_name in modality_names:
            if mod_name not in feats or feats[mod_name] is None:
                continue
            f = feats[mod_name].detach()
            lp, lc = compute_local_prototypes(f, labels, self.num_class, multilabel)
            self.local_proto_sums[mod_name] += lp * lc.unsqueeze(1)
            self.local_proto_counts[mod_name] += lc

    def update_weights(self):
        # Set mode to train model
        self.model.train()
        # 获取动态蒸馏参数
        if self.distill_scheduler and getattr(self.args, 'en_distill', False):
            alpha, T = self.distill_scheduler.get_params(self.current_epoch, self.missing_rate)
        else:
            alpha, T = 0.0, 1.0

        # 同步温度到 double_distill_loss（修复 T 未从调度器传递的 bug）
        if self.double_distill_loss is not None:
            self.double_distill_loss.temperature = T

        # initialize eval
        self.eval = EvalMetric(self.multilabel)

        # optimizer（投影头参数已在模型中，无需额外添加）
        if self.args.fed_alg in ['fed_avg', 'fed_opt']:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5
            )
        else:
            optimizer = FedProxOptimizer(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5,
                mu=self.args.mu
            )

        # AMP (mixed precision) support. Enabled only with --amp on CUDA.
        # autocast is entered once around the whole local-epoch loop to avoid
        # re-indenting the batch body; GradScaler handles gradient scaling.
        self._use_amp = bool(getattr(self.args, 'amp', False)) and self.device.type == 'cuda'
        self._scaler = None
        self._amp_ctx = None
        if self._use_amp:
            self._scaler = torch.cuda.amp.GradScaler(enabled=True)
            self._amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
            self._amp_ctx.__enter__()

        # last global model
        last_global_model = copy.deepcopy(self.model)

        # 跨模态蒸馏权重参数
        lambda_cross_base = getattr(self.args, 'lambda_cross', 0.3)
        cross_eta = getattr(self.args, 'cross_eta', 1.0)

        if self.args.dataset == 'ptb-xl':
            mod_names = ["i_to_avf_feat", "v1_to_v6_feat"]
        elif self.args.dataset in ["hateful_memes", "crisis-mmd"]:
            mod_names = ["img_feat", "text_feat"]
        elif self.args.dataset == "meld":
            mod_names = ["audio_feat", "text_feat"]
            if hasattr(self.args, 'video_feat') and self.args.video_feat is not None:
                mod_names.append("video_feat")
        elif self.args.dataset == "mhealth" and hasattr(self.args, 'mag_feat'):
            mod_names = ["acc_feat", "gyro_feat", "mag_feat"]
        else:
            mod_names = ["acc_feat", "gyro_feat"]

        loss_components = {'task': 0.0, 'logit': 0.0, 'feat_common': 0.0, 'feat_cross': 0.0,
                           'rec': 0.0, 'proto_paired': 0.0, 'proto_missing': 0.0, 'proto_same': 0.0,
                           'task_rec': 0.0}
        n_batches = 0

        # [ActCheck] env-gated diagnostic instrumentation (FEDPCD_ACTCHECK=1).
        # Records cross-branch activation; does NOT touch loss/optim logic.
        _actcheck = os.environ.get('FEDPCD_ACTCHECK') == '1'
        _ac_cross_nonzero = 0
        _ac_sum_cross_nz = 0.0
        _ac_complete_batches = 0
        _lf_pp_nz = 0; _lf_pm_nz = 0; _lf_rec_nz = 0   # leak-free activation counters
        _lf_skip_nosrc = 0; _lf_skip_nopos = 0; _lf_skip_insuf = 0  # leak-free skip counters
        cross_modalities = []

        # 诊断日志：每轮记录蒸馏参数
        if self.teacher_model is not None:
            import logging
            logging.info(f'[Diag] E{self.current_epoch} alpha={alpha:.4f} T={T:.2f} '
                         f'miss_rate={self.missing_rate:.2f} '
                         f'en_dd={self.en_double_distill} en_proto={self.en_proto_cross}')

        for local_epoch in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20: continue
                self.model.zero_grad()
                optimizer.zero_grad()

                is_3mod = (self.args.dataset == "mhealth" and hasattr(self.args, 'mag_feat')) or \
                          (self.args.dataset == "meld" and hasattr(self.args, 'video_feat') and self.args.video_feat is not None)
                if self.args.modality == "multimodal":
                    if is_3mod:
                        # 3-modality: acc+gyro+mag or audio+text+video
                        x_a, x_b, x_c, l_a, l_b, l_c, y = batch_data
                        x_a = x_a.to(self.device); x_b = x_b.to(self.device); x_c = x_c.to(self.device)
                        y = y.to(self.device)
                        l_a = l_a.to(self.device); l_b = l_b.to(self.device); l_c = l_c.to(self.device)

                        missing_modalities = []
                        mod_a_name, mod_b_name, mod_c_name = mod_names[0], mod_names[1], mod_names[2]
                        if (l_a == 0).all(): missing_modalities.append(mod_a_name)
                        if (l_b == 0).all(): missing_modalities.append(mod_b_name)
                        if (l_c == 0).all(): missing_modalities.append(mod_c_name)

                        cross_modalities = [] if len(missing_modalities) > 0 else mod_names[:3]

                        if self.args.dataset == "meld":
                            # SERClassifier3Mod: x_audio, x_text, len_a, len_t, x_video, len_v
                            outputs, x_mm, feats = self.model(
                                x_a.float(), x_b.float(), l_a, l_b, x_c.float(), l_c)
                        else:
                            outputs, x_mm, feats = self.model(
                                x_a.float(), x_b.float(), x_c.float(), l_a, l_b, l_c)

                        t_outputs = None; t_feats = None; t_x_mm = None
                        if self.teacher_model is not None and alpha > 0:
                            with torch.no_grad():
                                if self.args.dataset == "meld":
                                    t_outputs, t_x_mm, t_feats = self.teacher_model(
                                        x_a.float(), x_b.float(), l_a, l_b, x_c.float(), l_c)
                                else:
                                    t_outputs, t_x_mm, t_feats = self.teacher_model(
                                        x_a.float(), x_b.float(), x_c.float(), l_a, l_b, l_c)

                        if (self.en_proto_cross or self.leak_free_cross) and self.teacher_model is not None and t_feats is not None:
                            present_mods = [m for m in mod_names if m not in missing_modalities]
                            if present_mods:
                                self._accumulate_prototypes(t_feats, y, present_mods,
                                                           teacher_logits=t_outputs)
                    else:
                        x_a, x_b, l_a, l_b, y = batch_data
                        x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                        l_a, l_b = l_a.to(self.device), l_b.to(self.device)

                        # 判断缺失模态（batch中所有样本len=0 表示该模态整体缺失）
                        missing_modalities = []
                        mod_a_name, mod_b_name = mod_names[0], mod_names[1]
                        if (l_a == 0).all():
                            missing_modalities.append(mod_a_name)
                        if (l_b == 0).all():
                            missing_modalities.append(mod_b_name)

                        # 跨模态损失仅在完整模态客户端上计算（利用配对双模态特征学习映射）
                        # 缺失模态客户端跳过 Lcross，投影头通过 FedAvg 聚合获得映射能力
                        cross_modalities = [] if len(missing_modalities) > 0 else mod_names[:2]

                        # forward（feat_dict 中包含跨模态投影结果）
                        outputs, x_mm, feats = self.model(x_a.float(), x_b.float(), l_a, l_b)

                        # 教师前向传播（如果启用蒸馏）
                        t_outputs = None
                        t_feats = None
                        t_x_mm = None
                        if self.teacher_model is not None and alpha > 0:
                            with torch.no_grad():
                                t_outputs, t_x_mm, t_feats = self.teacher_model(
                                    x_a.float(), x_b.float(), l_a, l_b)

                        # 累积 teacher 特征用于原型计算
                        # 只累积当前客户端存在的模态（缺失模态的 teacher 特征是噪声）
                        # 多标签：用 teacher logit argmax 做硬分配，防止原型坍塌
                        if (self.en_proto_cross or self.leak_free_cross) and self.teacher_model is not None and t_feats is not None:
                            present_mods = [m for m in mod_names if m not in missing_modalities]
                            if present_mods:
                                self._accumulate_prototypes(t_feats, y, present_mods,
                                                           teacher_logits=t_outputs)

                else:
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)

                    # forward
                    outputs, _, feats = self.model(
                        x.float(), l
                    )
                    t_outputs = None
                    t_feats = None
                    t_x_mm = None
                    missing_modalities = []

                # 1. 基础任务损失
                if not self.multilabel:
                    soft_outputs = torch.log_softmax(outputs, dim=1)
                    loss_task = self.criterion(soft_outputs, y)
                else:
                    loss_task = self.criterion(outputs, y)

                # 诊断：记录 teacher vs student logit 统计（第一个 batch 每个 client）
                if t_outputs is not None and batch_idx == 0 and local_epoch == 0:
                    with torch.no_grad():
                        s_mean, s_std = outputs.mean().item(), outputs.std().item()
                        t_mean, t_std = t_outputs.mean().item(), t_outputs.std().item()
                        cos_sim = F.cosine_similarity(outputs, t_outputs, dim=1).mean().item()
                        import logging
                        logging.info(f'[Diag] E{self.current_epoch} batch0 '
                                     f'stu(logit mean={s_mean:.3f} std={s_std:.3f}) '
                                     f'tea(logit mean={t_mean:.3f} std={t_std:.3f}) '
                                     f'cos_sim={cos_sim:.4f} task_loss={loss_task.item():.4f} '
                                     f'missing={missing_modalities[:2]}')

                # 2. 蒸馏损失计算
                loss_distill = 0
                if t_outputs is not None and alpha > 0:
                    if self.en_double_distill and self.double_distill_loss is not None and t_feats is not None:
                        # 双蒸馏模式：logit层 + 特征层（共同模态） + 跨模态蒸馏
                        cross_projs = {k: v for k, v in feats.items() if k.endswith('_cross_proj')}

                        # 为原型对比损失准备 labels
                        if self.en_proto_cross:
                            loss_labels = y  # 直接传递原始 labels
                        else:
                            loss_labels = None

                        _, loss_logit, loss_feat_common, loss_feat_cross = self.double_distill_loss(
                            outputs, t_outputs, feats, t_feats,
                            cross_modal_projections=cross_projs,
                            missing_modalities=missing_modalities,
                            cross_modalities=cross_modalities,
                            labels=loss_labels,
                            prototype_manager=self.prototype_manager
                        )
                        alpha_feat = getattr(self.args, 'alpha_feat', 0.5)
                        # 跨模态损失权重与缺失率正相关
                        lambda_cross = lambda_cross_base * (1.0 + cross_eta * self.missing_rate)

                        # 总蒸馏损失（logit KD 使用 BCE-with-soft-targets，alpha 线性调度提供自然预热）
                        loss_logit_term = alpha * loss_logit
                        loss = ((1 - alpha) * loss_task
                                + loss_logit_term
                                + alpha_feat * alpha * loss_feat_common
                                + alpha * lambda_cross * loss_feat_cross)

                        # ===== FedPCD-New: leak-free cross-modal terms =====
                        # Knowledge preservation (logit+feat) above; the old cross slot
                        # (loss_feat_cross) is 0 here because en_cross was disabled.
                        # Add rec / proto_paired / proto_missing on LEAK-FREE predictions.
                        if (self.leak_free_cross and self.leakfree_loss is not None
                                and not is_3mod and hasattr(self.model, 'predict_target_features')):
                            present_mods = [m for m in mod_names if m not in missing_modalities]
                            target_preds = self.model.predict_target_features(
                                x_a.float(), x_b.float(), l_a, l_b)
                            loss_rec, loss_pp, loss_pm, lf_stats = self.leakfree_loss(
                                target_preds, t_feats, present_mods, y,
                                prototype_manager=self.prototype_manager,
                                use_instance_recovery=self.lf_use_rec,
                                use_prototype_alignment=self.lf_use_proto_paired,
                                use_missing_client_proto=self.lf_use_proto_missing,
                                support_reference=self.lf_support_reference)
                            # self.current_epoch is the GLOBAL communication round.
                            current_round = self.current_epoch
                            # warm-up weight for prototype terms (w_t ramps 0->1).
                            if self.lf_warmup_len > 0:
                                w_t = min(1.0, max(0.0,
                                    (current_round - self.lf_warmup_start) / float(self.lf_warmup_len)))
                            else:
                                w_t = 1.0
                            # Recovery is gated by alpha_t (teacher-maturity schedule, like logit/feat KD);
                            # Prototype is gated by w_t (server-prototype maturity) OR by alpha_t when
                            # --lf_proto_use_alpha is set (to match A-path weight schedule).
                            if getattr(self.args, 'lf_proto_use_alpha', False):
                                proto_weight = alpha * self.lf_lambda_proto
                            else:
                                proto_weight = self.lf_lambda_proto * w_t
                            loss = (loss
                                    + alpha * self.lf_lambda_rec * loss_rec
                                    + proto_weight * (loss_pp + loss_pm))
                            loss_components['rec'] += float(loss_rec)
                            loss_components['proto_paired'] += float(loss_pp)
                            loss_components['proto_missing'] += float(loss_pm)
                            if _actcheck:
                                _lf_pp_nz += 1 if float(loss_pp) > 1e-12 else 0
                                _lf_pm_nz += 1 if float(loss_pm) > 1e-12 else 0
                                _lf_rec_nz += 1 if float(loss_rec) > 1e-12 else 0
                                _lf_skip_nosrc += lf_stats.get('n_skipped_no_source', 0)
                                _lf_skip_nopos += lf_stats.get('n_skipped_no_positive_proto', 0)
                                _lf_skip_insuf += lf_stats.get('n_skipped_insufficient_classes', 0)

                            # C1: same-modality prototype alignment (control).
                            # f_m^observed → mu_y^m for each present m. No cross-modal projection.
                            # Uses the same prototype bank, warm-up, and reliability gate as the
                            # leak-free proto terms, so the only difference is cross-modal vs same-modal.
                            if (self.lf_use_proto_same and self.prototype_manager is not None
                                    and t_feats is not None and not is_3mod):
                                loss_pp_same = torch.tensor(0.0, device=self.device)
                                n_pp_same = 0
                                for m in present_mods:
                                    if (t_feats.get(m) is not None
                                            and self.prototype_manager.num_valid_classes(m) >= 2):
                                        l, info = self.prototype_manager.compute_contrastive_loss(
                                            t_feats[m], y, m, multilabel=self.multilabel,
                                            support_reference=self.lf_support_reference,
                                            return_info=True)
                                        if info['skipped_reason'] is None:
                                            loss_pp_same = loss_pp_same + l
                                            n_pp_same += 1
                                if n_pp_same > 0:
                                    loss_pp_same = loss_pp_same / n_pp_same
                                loss = loss + self.lf_lambda_proto * w_t * loss_pp_same
                                loss_components['proto_same'] += float(loss_pp_same)

                            # Checkpoint 4C: task-coupled completion.
                            # Uses extracted features (one-pass) to build completed
                            # representations, then classifies through the SHARED classifier.
                            if (self.en_task_coupled and hasattr(self.model, 'extract_completion_features')
                                    and not is_3mod):
                                cf = self.model.extract_completion_features(
                                    x_a.float(), x_b.float(), l_a, l_b)
                                loss_task_rec = torch.tensor(0.0, device=self.device)
                                n_task_rec = 0
                                present = set(present_mods)
                                # Direction 1: predict gyro from acc (acc present, gyro may be missing)
                                if 'acc_feat' in present:
                                    logits = self.model.classify_completed_features(
                                        cf['acc_real'], cf['gyro_pred'])
                                    loss_task_rec = loss_task_rec + F.cross_entropy(logits, y)
                                    n_task_rec += 1
                                # Direction 2: predict acc from gyro (gyro present, acc may be missing)
                                if 'gyro_feat' in present:
                                    logits = self.model.classify_completed_features(
                                        cf['acc_pred'], cf['gyro_real'])
                                    loss_task_rec = loss_task_rec + F.cross_entropy(logits, y)
                                    n_task_rec += 1
                                if n_task_rec > 0:
                                    loss_task_rec = loss_task_rec / n_task_rec
                                    tw = 1.0 if self.task_rec_warmup <= 0 else min(1.0, current_round / float(self.task_rec_warmup))
                                    loss = loss + self.lambda_task_rec * tw * loss_task_rec
                                    loss_components['task_rec'] += float(loss_task_rec)
                    else:
                        # 原始logit蒸馏
                        if self.multilabel:
                            teacher_sharp = t_outputs / T
                            loss_distill = 1.0 - F.cosine_similarity(
                                outputs, teacher_sharp.detach(), dim=1).mean()
                        else:
                            p_s = F.log_softmax(outputs / T, dim=1)
                            p_t = F.softmax(t_outputs / T, dim=1)
                            loss_distill = F.kl_div(p_s, p_t, reduction='batchmean') * (T**2)
                        loss = (1 - alpha) * loss_task + alpha * loss_distill
                else:
                    loss = loss_task

                # 累积损失分量用于分析
                n_batches += 1
                loss_components['task'] += loss_task.item()
                if t_outputs is not None and alpha > 0:
                    if self.en_double_distill:
                        loss_components['logit'] += loss_logit.item()
                        loss_components['feat_common'] += loss_feat_common.item()
                        loss_components['feat_cross'] += loss_feat_cross.item()
                        # [ActCheck] cross-branch activation accounting
                        if _actcheck:
                            if len(cross_modalities) > 0:
                                _ac_complete_batches += 1
                            _fc = loss_feat_cross.item()
                            if _fc > 1e-12:
                                _ac_cross_nonzero += 1
                                _ac_sum_cross_nz += _fc
                    else:
                        loss_components['logit'] += loss_distill.item()

                # 反向传播
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()

                # [ActCheck] gradient-norm diagnostic (first batch each round, env-gated)
                if _actcheck and self.leak_free_cross and batch_idx == 0 and local_epoch == 0:
                    import logging
                    def _gn(mods):
                        s = 0.0
                        for mm in mods:
                            for p in mm.parameters():
                                if p.grad is not None:
                                    s += p.grad.detach().pow(2).sum().item()
                        return s ** 0.5
                    g_src = _gn([self.model.acc_conv, self.model.acc_rnn,
                                 self.model.gyro_conv, self.model.gyro_rnn])
                    g_head = _gn([self.model.cross_modal_heads])
                    g_adapt = _gn([self.model.cross_modal_adapt])
                    logging.info(f'[ActCheck-Grad] E{self.current_epoch} '
                                 f'grad_src_encoder={g_src:.4e} grad_proj_heads={g_head:.4e} '
                                 f'grad_adapt={g_adapt:.4e}')

                if self._scaler is not None:
                    self._scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    optimizer.step()

                # save results
                if not self.multilabel:
                    self.eval.append_classification_results(
                        y,
                        torch.log_softmax(outputs, dim=1),
                        loss_task
                    )
                else:
                    self.eval.append_multilabel_results(y, outputs, loss_task)

        # epoch train results
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
        if n_batches > 0:
            for k in loss_components:
                self.result[f'loss_{k}'] = loss_components[k] / n_batches

        # [ActCheck] per-client cross-branch activation summary (env-gated, diagnostic only)
        if _actcheck and self.en_double_distill:
            import logging
            _cm = self.double_distill_loss.cross_mode if self.double_distill_loss is not None else 'none'
            _mean_nz = (_ac_sum_cross_nz / _ac_cross_nonzero) if _ac_cross_nonzero > 0 else 0.0
            logging.info(
                f'[ActCheck] E{self.current_epoch} cross_mode={_cm} en_proto={self.en_proto_cross} '
                f'complete_batches={_ac_complete_batches}/{n_batches} '
                f'cross_nonzero_batches={_ac_cross_nonzero} mean_cross_nz={_mean_nz:.6f} '
                f'mean_feat_common={self.result.get("loss_feat_common", 0.0):.6f} '
                f'mean_feat_cross={self.result.get("loss_feat_cross", 0.0):.6f}')
            if self.leak_free_cross:
                logging.info(
                    f'[ActCheck-LF] E{self.current_epoch} '
                    f'rec_nz={_lf_rec_nz}/{n_batches} proto_paired_nz={_lf_pp_nz} '
                    f'proto_missing_nz={_lf_pm_nz} '
                    f'skip_nosrc={_lf_skip_nosrc} skip_nopos={_lf_skip_nopos} skip_insuf={_lf_skip_insuf} '
                    f'valid_cls_acc={self.prototype_manager.num_valid_classes("acc_feat") if self.prototype_manager else 0} '
                    f'valid_cls_gyro={self.prototype_manager.num_valid_classes("gyro_feat") if self.prototype_manager else 0} '
                    f'mean_rec={self.result.get("loss_rec", 0.0):.6f} '
                    f'mean_proto_paired={self.result.get("loss_proto_paired", 0.0):.6f} '
                    f'mean_proto_missing={self.result.get("loss_proto_missing", 0.0):.6f}')

        # Diagnostic logging for FedPCD (every 10 epochs to avoid spam)
        if self.current_epoch % 10 == 0 and (self.en_double_distill or self.en_proto_cross):
            import logging
            diag_parts = [f'[Diag-Feat] E{self.current_epoch}']

            # Feature norms per modality
            if hasattr(self.model, 'cross_modal_heads'):
                diag_parts.append(f'cross_heads={list(self.model.cross_modal_heads.keys())}')

            # Prototype accumulator stats
            if self.local_proto_sums is not None:
                for mod_name in self.local_proto_sums:
                    s = self.local_proto_sums[mod_name]
                    c = self.local_proto_counts[mod_name]
                    n_active = (c > 0).sum().item()
                    proto_norm = s[c > 0].norm(dim=1).mean().item() if n_active > 0 else 0
                    diag_parts.append(f'{mod_name}:active={n_active}/{self.num_class},|p|={proto_norm:.2f}')

            logging.info(' '.join(diag_parts))

        if self._amp_ctx is not None:
            self._amp_ctx.__exit__(None, None, None)
