# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""
MFCPL Missing Modality Simulator
Client-level modality missing simulation using Bernoulli distribution

Reference: Hong et al., "Multimodal Federated Cross-Prototype Learning", 2024
"""

import torch
import numpy as np
from typing import Dict, Tuple


class MFCPLMissingSimulator:
    """
    Simulates client-level modality missing using Bernoulli distribution

    Missing rate q determines:
    - P(missing) = q
    - P(available) = 1 - q
    """

    def __init__(self, num_modalities=2, missing_rate=0.5):
        """
        Args:
            num_modalities: Number of modalities (default: 2 for ACC+Gyro)
            missing_rate: q parameter, probability of modality being missing
        """
        self.num_modalities = num_modalities
        self.missing_rate = missing_rate
        self.keep_prob = 1.0 - missing_rate

    def generate_client_mask(self, num_clients):
        """
        Generate modality availability masks for all clients
        This is done ONCE at the beginning, creating fixed client types

        Client types:
        - Dual-modality: both ACC and Gyro available
        - ACC-only: only ACC available
        - Gyro-only: only Gyro available

        Returns:
            client_masks: dict {client_id: mask_tensor}
                mask_tensor: (num_modalities,) tensor with 1=available, 0=missing
        """
        client_masks = {}

        for client_id in range(num_clients):
            if self.missing_rate == 1.0:
                # Complete cross-modal: each client has exactly one modality
                mask = torch.zeros(self.num_modalities, dtype=torch.float32)
                random_modality = np.random.randint(0, self.num_modalities)
                mask[random_modality] = 1.0
            else:
                # Bernoulli distribution
                mask = torch.bernoulli(
                    torch.full((self.num_modalities,), self.keep_prob)
                ).float()

                # Ensure at least one modality is available
                if mask.sum() == 0:
                    mask[np.random.randint(0, self.num_modalities)] = 1.0

            client_masks[client_id] = mask

        return client_masks

    def apply_missing_to_batch(self, x_a, x_b, client_mask):
        """
        Apply zero-filling to missing modalities in a batch

        Args:
            x_a: ACC data (batch_size, 3, 128)
            x_b: Gyro data (batch_size, 3, 128)
            client_mask: (2,) tensor [acc_available, gyro_available]

        Returns:
            x_a_modified, x_b_modified: Data with missing modalities zero-filled
        """
        x_a_modified = x_a.clone()
        x_b_modified = x_b.clone()

        # Apply zero-filling if modality is missing
        if client_mask[0] == 0:  # ACC missing
            x_a_modified = torch.zeros_like(x_a_modified)

        if client_mask[1] == 0:  # Gyro missing
            x_b_modified = torch.zeros_like(x_b_modified)

        return x_a_modified, x_b_modified

    def get_client_type(self, client_mask):
        """
        Get client type string from mask

        Args:
            client_mask: (num_modalities,) tensor

        Returns:
            client_type: str ('dual', 'acc_only', 'gyro_only')
        """
        if client_mask[0] == 1 and client_mask[1] == 1:
            return 'dual'
        elif client_mask[0] == 1 and client_mask[1] == 0:
            return 'acc_only'
        elif client_mask[0] == 0 and client_mask[1] == 1:
            return 'gyro_only'
        else:
            return 'none'  # Should not happen

    def analyze_client_distribution(self, client_masks):
        """
        Analyze the distribution of client types

        Args:
            client_masks: dict {client_id: mask_tensor}

        Returns:
            distribution: dict with counts of each client type
        """
        distribution = {
            'dual': 0,
            'acc_only': 0,
            'gyro_only': 0,
            'total': len(client_masks)
        }

        for client_id, mask in client_masks.items():
            client_type = self.get_client_type(mask)
            distribution[client_type] += 1

        return distribution


def print_missing_simulation_stats(client_masks, missing_rate):
    """
    Print statistics about modality missing simulation

    Args:
        client_masks: dict {client_id: mask_tensor}
        missing_rate: q parameter used
    """
    simulator = MFCPLMissingSimulator(missing_rate=missing_rate)
    distribution = simulator.analyze_client_distribution(client_masks)

    print("\n" + "="*60)
    print(f"MFCPL Missing Modality Simulation (q={missing_rate})")
    print("="*60)
    print(f"Total clients: {distribution['total']}")
    print(f"Dual-modality clients: {distribution['dual']} ({distribution['dual']/distribution['total']*100:.1f}%)")
    print(f"ACC-only clients: {distribution['acc_only']} ({distribution['acc_only']/distribution['total']*100:.1f}%)")
    print(f"Gyro-only clients: {distribution['gyro_only']} ({distribution['gyro_only']/distribution['total']*100:.1f}%)")
    print("="*60 + "\n")


# Example usage and testing
if __name__ == "__main__":
    # Test with different missing rates
    for missing_rate in [0.0, 0.5, 0.7, 0.8, 1.0]:
        simulator = MFCPLMissingSimulator(num_modalities=2, missing_rate=missing_rate)
        client_masks = simulator.generate_client_mask(num_clients=105)

        print_missing_simulation_stats(client_masks, missing_rate)

        # Test batch application
        x_a = torch.randn(16, 3, 128)  # batch of 16
        x_b = torch.randn(16, 3, 128)

        # Apply for each client type
        for mask_type, mask in [
            ('dual', torch.tensor([1.0, 1.0])),
            ('acc_only', torch.tensor([1.0, 0.0])),
            ('gyro_only', torch.tensor([0.0, 1.0]))
        ]:
            x_a_mod, x_b_mod = simulator.apply_missing_to_batch(x_a, x_b, mask)
            print(f"{mask_type} - ACC sum: {x_a_mod.sum().item():.2f}, Gyro sum: {x_b_mod.sum().item():.2f}")
