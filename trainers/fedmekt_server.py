# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""
FedMEKT Server — Server-side Knowledge Distillation with Proxy Dataset
Simplified implementation based on: FedMEKT (Le et al., Neural Networks 2024)

Key difference from FedDD: FedMEKT does server-side KD using a proxy dataset,
while FedDD does client-side KD between teacher (full modality) and student (masked modality).

Approach: After FedOpt aggregation, pre-compute client model predictions on proxy data,
then distill from client ensemble to global model.
"""
import torch
import torch.nn.functional as F
import copy, logging
import numpy as np
from .server_trainer import Server


class FedMEKTServer(Server):
    """Server with proxy dataset KD (FedMEKT-style)"""

    def __init__(self, args, model, device, criterion, client_ids, proxy_data=None):
        """
        Args:
            proxy_data: tuple of (x_a, x_b, l_a, l_b, y) tensors for KD
        """
        super().__init__(args, model, device, criterion, client_ids)
        self.proxy_data = proxy_data  # pre-loaded proxy dataset on device
        self.server_kd_steps = getattr(args, 'server_kd_steps', 10)
        self.server_kd_lr = getattr(args, 'server_kd_lr', 0.01)
        self.server_kd_T = getattr(args, 'server_kd_T', 3.0)
        self.server_kd_alpha = getattr(args, 'server_kd_alpha', 0.5)

    def server_kd_step(self, client_models, client_weights):
        """
        Server-side KD using pre-computed predictions.
        1. Pre-compute teacher logits: run each client model on proxy data
        2. Do KD: train global model to match aggregated teacher logits
        """
        if self.proxy_data is None or len(client_models) == 0:
            return

        x_a, x_b, l_a, l_b, y = self.proxy_data

        # Step 1: Pre-compute teacher predictions from all client models
        total_weight = sum(w for w in client_weights if w > 0)
        teacher_logits = torch.zeros_like(
            self.global_model(x_a.float(), x_b.float(), l_a.float(), l_b.float())[0]
        ).detach()

        for model_sd, weight in zip(client_models, client_weights):
            if model_sd is None or weight == 0:
                continue
            tmp_model = copy.deepcopy(self.global_model)
            tmp_model.load_state_dict(model_sd)
            tmp_model.eval()
            with torch.no_grad():
                out, _, _ = tmp_model(x_a.float(), x_b.float(), l_a.float(), l_b.float())
            teacher_logits += (weight / total_weight) * out.detach()
            del tmp_model

        # Step 2: Server KD — train global model to match teacher on proxy data
        self.global_model.train()
        kd_optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            lr=self.server_kd_lr,
            momentum=0.9
        )

        for _ in range(self.server_kd_steps):
            student_out, _, _ = self.global_model(x_a.float(), x_b.float(), l_a.float(), l_b.float())

            # Task loss
            task_loss = F.cross_entropy(student_out, y)

            # KD loss
            kd_loss = F.kl_div(
                F.log_softmax(student_out / self.server_kd_T, dim=1),
                F.softmax(teacher_logits / self.server_kd_T, dim=1),
                reduction='batchmean'
            ) * (self.server_kd_T ** 2)

            total_loss = (1 - self.server_kd_alpha) * task_loss + self.server_kd_alpha * kd_loss

            kd_optimizer.zero_grad()
            total_loss.backward()
            kd_optimizer.step()

        logging.info(f'Server KD done (alpha={self.server_kd_alpha}, T={self.server_kd_T}, steps={self.server_kd_steps})')
