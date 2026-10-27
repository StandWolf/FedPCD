# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""
MFCPL Model Architecture
Multimodal Federated Cross-Prototype Learning

Reference: Hong et al., "Multimodal Federated Cross-Prototype Learning", 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """Single modality encoder: 1D Conv + GRU"""

    def __init__(self, in_channels=3, hidden_dim=128, dropout=0.1):
        super(ModalityEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout)
        )
        self.gru = nn.GRU(128, hidden_dim, batch_first=True, dropout=dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, seq_len)
        Returns:
            output: (batch_size, hidden_dim)
        """
        # Conv layers
        x = self.conv_layers(x)  # (batch, 128, seq_len/2)

        # Transpose for GRU: (batch_size, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        out, _ = self.gru(x)

        # Take last time step
        return out[:, -1, :]


class SelfAttentionFusion(nn.Module):
    """Self-attention based multimodal fusion"""

    def __init__(self, hidden_dim=128, num_modalities=2):
        super(SelfAttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.Tanh(),
            nn.Linear(512, num_modalities)  # Attention weights for each modality
        )

    def forward(self, modal_embeddings):
        """
        Args:
            modal_embeddings: list of tensors, each (batch_size, hidden_dim)
        Returns:
            fused_embedding: (batch_size, hidden_dim)
        """
        if len(modal_embeddings) == 1:
            return modal_embeddings[0]

        # Stack: (batch_size, num_modalities, hidden_dim)
        modal_stack = torch.stack(modal_embeddings, dim=1)

        # Compute attention weights
        attention_weights = self.attention(modal_stack)  # (batch, num_modalities, num_modalities)
        attention_weights = F.softmax(attention_weights.mean(dim=2), dim=1)  # (batch, num_modalities)

        # Weighted fusion
        attention_weights = attention_weights.unsqueeze(-1)  # (batch, num_modalities, 1)
        fused = (modal_stack * attention_weights).sum(dim=1)  # (batch, hidden_dim)

        return fused


class MultimodalFusionNet(nn.Module):
    """Complete multimodal fusion network with encoders, fusion, and classifiers"""

    def __init__(self, num_classes=6, num_modalities=2, hidden_dim=128,
                 projection_dim=64, dropout=0.1,
                 in_channels_a=3, in_channels_b=3):
        super(MultimodalFusionNet, self).__init__()

        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim

        # Modality encoders
        self.acc_encoder = ModalityEncoder(in_channels=in_channels_a, hidden_dim=hidden_dim, dropout=dropout)
        self.gyro_encoder = ModalityEncoder(in_channels=in_channels_b, hidden_dim=hidden_dim, dropout=dropout)

        # Self-attention fusion
        self.fusion = SelfAttentionFusion(hidden_dim, num_modalities)

        # Classifier — uses self-attention fusion output z_shared (hidden_dim), not concatenation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

        # Projection layers for MFCPL
        self.projector_shared = nn.Linear(hidden_dim, projection_dim)
        self.projector_modal = nn.Linear(hidden_dim, projection_dim)

    def forward(self, acc_data=None, gyro_data=None, return_features=False):
        """
        Args:
            acc_data: (batch_size, seq_len, channels) or None
            gyro_data: (batch_size, seq_len, channels) or None
            return_features: whether to return intermediate features
        Returns:
            logits: (batch_size, num_classes)
            r_shared: (batch_size, projection_dim)
            z_modal_list: list of (batch_size, projection_dim)
        """
        modal_embeddings = []

        # Encode each available modality
        # Data format is (batch, seq_len, channels), need to transpose to (batch, channels, seq_len)
        if acc_data is not None:
            acc_data = acc_data.transpose(1, 2)  # (batch, seq_len, channels) -> (batch, channels, seq_len)
            acc_emb = self.acc_encoder(acc_data)
            modal_embeddings.append(acc_emb)

        if gyro_data is not None:
            gyro_data = gyro_data.transpose(1, 2)  # (batch, seq_len, channels) -> (batch, channels, seq_len)
            gyro_emb = self.gyro_encoder(gyro_data)
            modal_embeddings.append(gyro_emb)

        # Skip connection for single modality
        if len(modal_embeddings) == 0:
            raise ValueError("At least one modality must be provided")

        # Self-attention fusion
        if len(modal_embeddings) == 2:
            # Concatenate for fusion input
            modal_stack = torch.stack(modal_embeddings, dim=1)  # (batch, 2, 128)
            attention_weights = self.fusion.attention(modal_stack)  # (batch, 2, 2)
            attention_weights = F.softmax(attention_weights.mean(dim=2), dim=1)  # (batch, 2)

            # Weighted sum
            attention_weights = attention_weights.unsqueeze(-1)
            z_shared = (modal_stack * attention_weights).sum(dim=1)  # (batch, 128)

            # Self-attention fusion output is the classifier input
            z_fused = z_shared
        else:
            # Single modality case
            z_shared = modal_embeddings[0]
            z_fused = z_shared

        # Projection for shared representation
        r_shared = self.projector_shared(z_fused)

        # Projection for modality-specific representations
        z_modal_list = [self.projector_modal(emb) for emb in modal_embeddings]

        # Classification
        logits = self.classifier(z_fused)

        if return_features:
            return logits, r_shared, z_modal_list, z_fused, modal_embeddings

        return logits, r_shared, z_modal_list


class MFCPLModel(nn.Module):
    """Wrapper for MFCPL model to match existing framework interface"""

    def __init__(self, num_classes=6, projection_dim=64, dropout=0.1,
                 in_channels_a=3, in_channels_b=3, hidden_dim=128):
        super(MFCPLModel, self).__init__()
        self.base_model = MultimodalFusionNet(
            num_classes=num_classes,
            num_modalities=2,
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            dropout=dropout,
            in_channels_a=in_channels_a,
            in_channels_b=in_channels_b
        )
        self.num_classes = num_classes

    def forward(self, x_a, x_b, l_a, l_b):
        """
        Match the interface of existing multimodal models
        Args:
            x_a: ACC data (batch_size, 3, 128)
            x_b: Gyro data (batch_size, 3, 128)
            l_a: ACC missing flag (batch_size,)
            l_b: Gyro missing flag (batch_size,)
        Returns:
            outputs: logits (batch_size, num_classes)
            shared_emb: shared embedding (for compatibility)
            feats: feature dictionary
        """
        batch_size = x_a.size(0)

        # Handle missing modalities with zero padding
        acc_data = x_a.clone()
        gyro_data = x_b.clone()

        # Apply zero padding for missing modalities
        for i in range(batch_size):
            if l_a[i] == 0:  # ACC missing
                acc_data[i] = torch.zeros_like(acc_data[i])
            if l_b[i] == 0:  # Gyro missing
                gyro_data[i] = torch.zeros_like(gyro_data[i])

        # Forward pass
        logits, r_shared, z_modal_list = self.base_model(acc_data, gyro_data)

        # Return features in compatible format
        # z_modal_list is a list of modal-specific features
        return logits, r_shared, z_modal_list

    def get_embeddings(self, x_a, x_b, l_a, l_b):
        """Get embeddings for prototype computation"""
        logits, r_shared, z_modal_list = self.forward(x_a, x_b, l_a, l_b)
        return r_shared, z_modal_list
