"""Group Distributionally Robust Optimisation loss (NV3).

Optimises the worst-performing group (e.g. depressed men) by maintaining an
exponentiated-gradient distribution ``q`` over groups and weighting per-group
losses by ``q``.
"""

from __future__ import annotations

import torch
from torch import nn


class GroupDROLoss(nn.Module):
    """Online Group-DRO with exponentiated weight updates.

    Args:
        n_groups: Number of groups ``G``.
        step: Step size ``eta`` for the exponentiated update of ``q``.
    """

    def __init__(self, n_groups: int, step: float = 0.01) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.step = step
        # Registered as a buffer so it moves with .to(device)/state_dict.
        self.register_buffer("q", torch.ones(n_groups) / n_groups)

    def forward(self, per_sample_loss: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """Compute the Group-DRO objective for one batch.

        Args:
            per_sample_loss: ``(N,)`` per-sample losses (e.g. BCE, reduction
                ``none``).
            group_ids: ``(N,)`` integer group id per sample in ``[0, G)``.

        Returns:
            Scalar Group-DRO loss ``sum_g q_g * mean_loss_g``.
        """
        device = per_sample_loss.device
        q = self.q.to(device)
        g_losses = []
        for g in range(self.n_groups):
            mask = group_ids == g
            if mask.any():
                g_losses.append(per_sample_loss[mask].mean())
            else:
                g_losses.append(per_sample_loss.new_zeros(()))
        g_loss = torch.stack(g_losses)                      # (G,)

        # Exponentiated-gradient update on the (detached) group losses.
        q = q * torch.exp(self.step * g_loss.detach())
        q = q / q.sum()
        self.q = q.detach()
        return (q * g_loss).sum()
