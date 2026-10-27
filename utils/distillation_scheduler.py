# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
import numpy as np


class DynamicDistillationScheduler:
    def __init__(self, total_epochs, init_T=2.0, min_T=1.0, alpha_max=0.5,
                 strategy='linear', gamma=0.5):
        """Distillation weight and temperature schedule.

        :param total_epochs: total number of communication rounds R
        :param init_T: initial temperature
        :param min_T: final temperature
        :param alpha_max: maximum distillation weight
        :param strategy: ['linear', 'cosine', 'triangle', 'fixed']
        :param gamma: missing-rate coefficient rho, alpha = alpha_base * (1 - rho * m_k)
        """
        self.total_epochs = total_epochs
        self.init_T = init_T
        self.min_T = min_T
        self.alpha_max = alpha_max
        self.strategy = strategy
        self.gamma = gamma

    def get_params(self, epoch, missing_rate):
        """Return (alpha, T) for the current round and client missing rate."""
        if self.strategy == 'fixed':
            alpha = self.alpha_max * max(0.0, 1.0 - self.gamma * missing_rate)
            return alpha, self.init_T

        progress = epoch / self.total_epochs
        if self.strategy == 'cosine':
            alpha_base = self.alpha_max * (1 - np.cos(np.pi * progress)) / 2
        elif self.strategy == 'triangle':
            if progress <= 0.5:
                alpha_base = self.alpha_max * 2.0 * progress
            else:
                alpha_base = self.alpha_max * 2.0 * (1.0 - progress)
        else:  # linear
            alpha_base = self.alpha_max * progress

        alpha = alpha_base * max(0.0, 1.0 - self.gamma * missing_rate)
        T = self.init_T - (self.init_T - self.min_T) * progress
        return alpha, T
