"""Optional intermodal consistency loss (symmetric KL between modality logits).

Encourages per-modality predictions to agree. Disabled by default
(``delta = 0``) per the design; provided for completeness/ablation.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _binary_kl(p_logit: torch.Tensor, q_logit: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between two Bernoulli distributions given by logits."""
    p = torch.sigmoid(p_logit)
    q = torch.sigmoid(q_logit)
    eps = 1e-6
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)

    def kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * (a / b).log() + (1 - a) * ((1 - a) / (1 - b)).log()

    return 0.5 * (kl(p, q) + kl(q, p)).mean()


def intermodal_consistency(modality_logits: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mean pairwise symmetric-KL consistency across modality logit streams.

    Args:
        modality_logits: Sequence of ``(N,)`` logit tensors, one per modality.

    Returns:
        Scalar consistency loss (``0`` if fewer than two modalities given).
    """
    if len(modality_logits) < 2:
        return modality_logits[0].new_zeros(()) if modality_logits else torch.zeros(())
    total = modality_logits[0].new_zeros(())
    count = 0
    for i in range(len(modality_logits)):
        for j in range(i + 1, len(modality_logits)):
            total = total + _binary_kl(modality_logits[i], modality_logits[j])
            count += 1
    return total / max(count, 1)
