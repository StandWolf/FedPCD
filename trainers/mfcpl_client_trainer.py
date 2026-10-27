# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""
MFCPL Client Trainer
Handles local training with prototype-based losses (CMPR, CMPC, CMA)

Reference: Hong et al., "Multimodal Federated Cross-Prototype Learning", 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .evaluation import EvalMetric


class MFCPLClient:
    """MFCPL Client for local training with prototype-based learning"""

    def __init__(
        self,
        args,
        device,
        criterion,
        dataloader,
        model,
        label_dict=None,
        num_class=None,
        global_prototypes=None,
        alpha_reg=1.0,
        alpha_con=2.0,
        alpha_align=0.1,
        temperature=0.1,
        client_id=None
    ):
        """
        Args:
            args: Configuration arguments
            device: CUDA/CPU device
            criterion: Loss criterion
            dataloader: Local data loader
            model: Client model
            label_dict: Label distribution
            num_class: Number of classes
            global_prototypes: Global prototypes from server
            alpha_reg: Weight for CMPR loss
            alpha_con: Weight for CMPC loss
            alpha_align: Weight for CMA loss
            temperature: Temperature for contrastive loss
        """
        self.args = args
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.model = model.to(device)
        self.client_id = client_id
        
        # Dataset type (multilabel or not)
        self.multilabel = True if args.dataset == 'ptb-xl' else False

        # MFCPL hyperparameters
        self.alpha_reg = alpha_reg
        self.alpha_con = alpha_con
        self.alpha_align = alpha_align
        self.temperature = temperature

        # Global prototypes
        self.num_classes = num_class if num_class else 6
        self.projection_dim = getattr(args, 'projection_dim', 64)

        if global_prototypes is not None:
            self.global_prototypes = global_prototypes.to(self.device)
        else:
            self.global_prototypes = torch.zeros(
                self.num_classes, self.projection_dim
            ).to(self.device)

        # Local prototype storage
        self.local_prototypes = torch.zeros(
            self.num_classes, self.projection_dim
        ).to(self.device)
        self.local_prototype_counts = torch.zeros(self.num_classes).to(self.device)
        self.prototype_buffer = [[] for _ in range(self.num_classes)]

    def update_weights(self):
        """Perform local training"""
        self.model.train()

        # Optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=1e-5
        )

        # Reset prototype buffers
        self.prototype_buffer = [[] for _ in range(self.num_classes)]
        self.local_prototype_counts = torch.zeros(self.num_classes).to(self.device)

        # Local training
        for local_epoch in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                optimizer.zero_grad()
                self.model.zero_grad()

                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)

                    # Forward pass
                    outputs, r_shared, z_modal_list = self.model(
                        x_a.float(), x_b.float(), l_a, l_b
                    )

                    # Compute MFCPL loss
                    total_loss, loss_dict = self._compute_mfcpl_loss(
                        outputs, y, r_shared, z_modal_list, l_a, l_b
                    )

                    # Backward pass
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    optimizer.step()

                    # Collect prototypes
                    self._collect_prototypes(r_shared, y)

        # Finalize local prototypes
        self._finalize_local_prototypes()

        # Collect results for server aggregation
        self.result = {
            'sample': len(self.dataloader.dataset),
        }

        # Set for save_train_updates
        self.client_prototypes = {self.client_id: self.local_prototypes}
        self.prototype_counts = {self.client_id: self.local_prototype_counts}

        return total_loss.item()

    def _compute_mfcpl_loss(self, outputs, labels, r_shared, z_modal_list, l_a, l_b):
        """
        Compute MFCPL total loss with three components:
        1. CMPR (Cross-Modal Prototype Regularization)
        2. CMPC (Cross-Modal Prototype Contrast)
        3. CMA (Cross-Modal Alignment)
        """
        # 1. Classification loss
        if not self.multilabel:
            ce_loss = F.cross_entropy(outputs, labels)
        else:
            ce_loss = self.criterion(outputs, labels)

        # 2. CMPR loss: L2 distance between shared representation and class prototypes
        cmp_r_loss = self._compute_cmpr_loss(r_shared, labels)

        # 3. CMPC loss: Contrastive learning between modal representations and prototypes
        cmp_c_loss = self._compute_cmpc_loss(z_modal_list, labels, l_a, l_b)

        # 4. CMA loss: Alignment between different modalities
        cma_loss = self._compute_cma_loss(z_modal_list, l_a, l_b)

        # Total loss
        total_loss = (
            ce_loss +
            self.alpha_reg * cmp_r_loss +
            self.alpha_con * cmp_c_loss +
            self.alpha_align * cma_loss
        )

        loss_dict = {
            'ce_loss': ce_loss.item(),
            'cmp_r_loss': cmp_r_loss.item(),
            'cmp_c_loss': cmp_c_loss.item(),
            'cma_loss': cma_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict

    def _compute_cmpr_loss(self, r_shared, labels):
        """
        CMPR: Cross-Modal Prototype Regularization
        Minimize L2 distance between shared representation and class prototype.
        For multilabel: weighted average of positive class prototypes.
        """
        if self.multilabel:
            batch_size = labels.size(0)
            pos_mask = labels > 0
            n_pos = pos_mask.sum(dim=1, keepdim=True).float()
            # Weighted prototype: average of all positive class prototypes
            pos_protos = self.global_prototypes.unsqueeze(0).expand(batch_size, -1, -1)
            weights = pos_mask.float() / (n_pos + 1e-8)
            target_protos = (weights.unsqueeze(-1) * pos_protos).sum(dim=1)
            cmp_r_loss = F.mse_loss(r_shared, target_protos)
        else:
            batch_prototypes = self.global_prototypes[labels]
            cmp_r_loss = F.mse_loss(r_shared, batch_prototypes)

        return cmp_r_loss

    def _compute_cmpc_loss(self, z_modal_list, labels, l_a, l_b):
        """
        CMPC: Cross-Modal Prototype Contrast
        Contrastive learning between modality-specific representations and prototypes
        L_CMPC = -log[exp(s(z_m', p_k)/τ) / sum_{p} exp(s(z_m', p)/τ)]
        """
        if not z_modal_list:
            return torch.tensor(0.0, device=self.device)

        batch_size = labels.size(0)
        cmp_c_loss = torch.tensor(0.0, device=self.device)

        num_available_modalities = 0

        for idx, (z_modal, missing_flag) in enumerate(zip(z_modal_list, [l_a, l_b])):
            # Skip if modality is missing (flag=0)
            # Actually we still compute for missing modalities with zero padding
            # to maintain batch consistency
            if missing_flag.sum() == 0:
                continue

            # Compute cosine similarity
            # z_modal: (batch_size, projection_dim)
            # global_prototypes: (num_classes, projection_dim)
            similarities = F.cosine_similarity(
                z_modal.unsqueeze(1),  # (batch, 1, projection_dim)
                self.global_prototypes.unsqueeze(0),  # (1, num_classes, projection_dim)
                dim=-1
            )  # (batch_size, num_classes)

            # Temperature scaling
            similarities = similarities / self.temperature

            # Contrastive loss
            log_probs = F.log_softmax(similarities, dim=1)

            if self.multilabel:
                pos_mask = labels > 0
                n_pos = pos_mask.sum(dim=1).float()
                positive_loss = -(log_probs * pos_mask.float()).sum(dim=1) / (n_pos + 1e-8)
            else:
                positive_loss = -log_probs[torch.arange(batch_size), labels]

            # Mean loss
            cmp_c_loss += positive_loss.mean()
            num_available_modalities += 1

        if num_available_modalities > 0:
            cmp_c_loss /= num_available_modalities

        return cmp_c_loss

    def _compute_cma_loss(self, z_modal_list, l_a, l_b):
        """
        CMA: Cross-Modal Alignment
        Minimize L2 distance between different modality representations
        L_CMA = sum_{(m1,m2) in available_pairs} ||z_m1' - z_m2'||_2^2
        """
        if len(z_modal_list) < 2:
            return torch.tensor(0.0, device=self.device)

        cma_loss = torch.tensor(0.0, device=self.device)
        num_pairs = 0

        # Check which modalities are available
        modalities = []
        if l_a.sum() > 0:
            modalities.append(0)  # ACC index
        if l_b.sum() > 0:
            modalities.append(1)  # Gyro index

        # Only compute if at least 2 modalities available in the batch
        if len(modalities) < 2:
            return torch.tensor(0.0, device=self.device)

        # Compute alignment for each available pair
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                idx_i, idx_j = modalities[i], modalities[j]

                # Only compute for samples where both modalities are available
                mask = (l_a > 0) & (l_b > 0)

                if mask.sum() > 0:
                    z_i = z_modal_list[idx_i][mask]
                    z_j = z_modal_list[idx_j][mask]

                    cma_loss += F.mse_loss(z_i, z_j)
                    num_pairs += 1

        if num_pairs > 0:
            cma_loss /= num_pairs

        return cma_loss

    def _collect_prototypes(self, r_shared, labels):
        """
        Collect shared representations for prototype computation.
        For multilabel, a sample contributes to all its positive classes.
        """
        if self.multilabel:
            for cls in range(self.num_classes):
                cls_mask = labels[:, cls] > 0
                if cls_mask.any():
                    cls_embeddings = r_shared[cls_mask]
                    self.prototype_buffer[cls].append(cls_embeddings)
                    self.local_prototype_counts[cls] += cls_mask.sum().item()
        else:
            for cls in range(self.num_classes):
                cls_mask = (labels == cls)
                if cls_mask.any():
                    cls_embeddings = r_shared[cls_mask]
                    self.prototype_buffer[cls].append(cls_embeddings)
                    self.local_prototype_counts[cls] += cls_mask.sum().item()

    def _finalize_local_prototypes(self):
        """
        Compute final local prototypes by averaging collected embeddings
        """
        for cls in range(self.num_classes):
            if len(self.prototype_buffer[cls]) > 0:
                # Concatenate all embeddings and average
                all_embeddings = torch.cat(self.prototype_buffer[cls], dim=0)
                self.local_prototypes[cls] = all_embeddings.mean(dim=0)
            else:
                self.local_prototypes[cls] = torch.zeros(
                    self.projection_dim
                ).to(self.device)

    def get_parameters(self):
        """Get current model parameters"""
        return self.model.state_dict()

    def get_model_result(self):
        """Get training results"""
        return self.result

    def update_global_prototypes(self, new_prototypes):
        """Update global prototypes from server"""
        self.global_prototypes = new_prototypes.to(self.device)

    def get_local_prototypes(self):
        """Get local prototypes for server aggregation"""
        return self.local_prototypes.clone(), self.local_prototype_counts.clone()

    def inference(self, dataloader):
        """Run inference on dataloader"""
        self.model.eval()

        eval_metric = EvalMetric(self.multilabel)

        with torch.no_grad():
            for batch_data in dataloader:
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)

                    outputs, _, _ = self.model(x_a.float(), x_b.float(), l_a, l_b)
                else:
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)

                    outputs, _, _ = self.model(x.float(), l)

                # Compute loss
                if not self.multilabel:
                    outputs = torch.log_softmax(outputs, dim=1)
                loss = self.criterion(outputs, y)

                # Save results
                if not self.multilabel:
                    eval_metric.append_classification_results(y, outputs, loss)
                else:
                    eval_metric.append_multilabel_results(y, outputs, loss)

        # Get summary
        if not self.multilabel:
            self.result = eval_metric.classification_summary()
        else:
            self.result = eval_metric.multilabel_summary()

        return self.result
