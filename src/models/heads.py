"""Prediction heads: depression logit and selective (abstain) gate."""

from __future__ import annotations

import torch
from torch import nn


class DepressionHead(nn.Module):
    """Linear depression head producing a single logit per input.

    Args:
        d: Input embedding dimension.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return ``(B,)`` logits."""
        return self.fc(z).squeeze(-1)


class SelectiveHead(nn.Module):
    """Selective-prediction gate ``g in [0,1]`` (1 = predict, 0 = abstain).

    Trained jointly with a coverage penalty so the model can decline to predict
    on low-confidence cases.

    Args:
        d: Input embedding dimension.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return ``(B,)`` gate values in ``[0, 1]``."""
        return torch.sigmoid(self.fc(z)).squeeze(-1)
