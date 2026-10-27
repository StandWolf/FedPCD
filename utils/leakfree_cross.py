# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""FedPCD-New: leak-free cross-modal losses.

Operates on LEAK-FREE target predictions produced by
`HARClassifier.predict_target_features`, where each prediction f_hat[m] is
computed from the OTHER modality only (target modality removed from the input).

Three independent loss terms (each toggled separately, unlike the old
mutually-exclusive mse/proto slot):

  loss_rec          instance-level MSE  || f_hat[m] - sg(teacher_real[m]) ||^2
                    computed ONLY when the SOURCE (other) modality AND the target m
                    are both present  ->  complete/paired clients.

  loss_proto_paired prototype contrastive on complete clients (source & target present):
                    pull f_hat[m] toward the class-y prototype of modality m.

  loss_proto_missing prototype contrastive on single-modality clients
                    (source present, target m MISSING): align f_hat[m] to the
                    m-prototype so the missing modality receives a class-level
                    target even without a real paired feature.

Guard: a prediction f_hat[m] is only USED when its SOURCE modality is actually
present; otherwise the prediction is driven by zero-padded input and is skipped.
The prototype bank is NEVER updated here (only real teacher features update it
elsewhere); this module only READS prototypes.
"""
import torch
import torch.nn.functional as F


class LeakFreeCrossLoss:
    def __init__(self, modality_names=None, multilabel=False):
        if modality_names is None:
            modality_names = ["acc_feat", "gyro_feat"]
        assert len(modality_names) == 2, \
            "LeakFreeCrossLoss currently supports exactly 2 modalities"
        self.modality_names = list(modality_names)
        self.multilabel = multilabel

    def _source_of(self, target):
        a, b = self.modality_names
        return b if target == a else a

    def __call__(self, target_preds, teacher_feats, present_mods, labels,
                 prototype_manager=None,
                 use_instance_recovery=True,
                 use_prototype_alignment=True,
                 use_missing_client_proto=True,
                 support_reference=None):
        # device / zero helper
        device = next(iter(target_preds.values())).device if target_preds else \
            (labels.device if labels is not None else torch.device('cpu'))
        zero = lambda: torch.tensor(0.0, device=device)

        loss_rec = zero(); n_rec = 0
        loss_proto_paired = zero(); n_pp = 0
        loss_proto_missing = zero(); n_pm = 0
        n_skipped_no_source = 0            # source modality absent
        n_skipped_no_positive_proto = 0   # batch has no positive-class prototype
        n_skipped_insufficient_classes = 0  # < 2 valid prototype classes

        present = set(present_mods)
        for m in self.modality_names:
            src = self._source_of(m)
            pred = target_preds.get(m)
            if pred is None:
                continue
            if src not in present:
                # HARD SKIP: source modality absent -> prediction is zero-driven; never enters loss.
                n_skipped_no_source += 1
                continue
            target_present = m in present
            if target_present:
                # paired direction: real target modality feature exists
                if use_instance_recovery and teacher_feats.get(m) is not None:
                    loss_rec = loss_rec + F.mse_loss(pred, teacher_feats[m].detach())
                    n_rec += 1
                if use_prototype_alignment and prototype_manager is not None and labels is not None:
                    if prototype_manager.num_valid_classes(m) < 2:
                        n_skipped_insufficient_classes += 1
                    else:
                        l, info = prototype_manager.compute_contrastive_loss(
                            pred, labels, m, multilabel=self.multilabel,
                            support_reference=support_reference, return_info=True)
                        if info['skipped_reason'] == 'no_positive_proto':
                            n_skipped_no_positive_proto += 1
                        else:
                            loss_proto_paired = loss_proto_paired + l
                            n_pp += 1
            else:
                # single-modality client: predict missing m from present source
                if use_missing_client_proto and prototype_manager is not None and labels is not None:
                    if prototype_manager.num_valid_classes(m) < 2:
                        n_skipped_insufficient_classes += 1
                    else:
                        l, info = prototype_manager.compute_contrastive_loss(
                            pred, labels, m, multilabel=self.multilabel,
                            support_reference=support_reference, return_info=True)
                        if info['skipped_reason'] == 'no_positive_proto':
                            n_skipped_no_positive_proto += 1
                        else:
                            loss_proto_missing = loss_proto_missing + l
                            n_pm += 1

        if n_rec > 0:
            loss_rec = loss_rec / n_rec
        if n_pp > 0:
            loss_proto_paired = loss_proto_paired / n_pp
        if n_pm > 0:
            loss_proto_missing = loss_proto_missing / n_pm

        stats = {'n_rec': n_rec, 'n_proto_paired': n_pp, 'n_proto_missing': n_pm,
                 'n_skipped_no_source': n_skipped_no_source,
                 'n_skipped_no_positive_proto': n_skipped_no_positive_proto,
                 'n_skipped_insufficient_classes': n_skipped_insufficient_classes}
        return loss_rec, loss_proto_paired, loss_proto_missing, stats
