"""Gradient Reversal Layer (core of the invariance objective, NV3)."""

from __future__ import annotations

import torch
from torch import nn


class _GradRev(torch.autograd.Function):
    """Identity forward, sign-flipped (and scaled) gradient backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        # Reverse and scale the gradient flowing back into the encoder.
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Functional gradient reversal."""
    return _GradRev.apply(x, lambd)


class GradientReversal(nn.Module):
    """Module wrapper around :func:`grad_reverse`.

    ``lambd`` is typically ramped up over training following
    ``lambd = 2 / (1 + exp(-gamma * p)) - 1`` where ``p`` is training progress.

    Args:
        lambd: Initial reversal strength.
    """

    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambd)
