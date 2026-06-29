"""Invariant Risk Minimisation penalty (NV3, optional)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def irm_penalty(logits: torch.Tensor, y: torch.Tensor, dummy_w: torch.Tensor) -> torch.Tensor:
    """IRMv1 gradient-norm penalty for a binary classifier.

    The penalty is the squared gradient of the risk w.r.t. a constant dummy
    classifier ``dummy_w`` applied to the logits; a small penalty means the
    representation is (approximately) simultaneously optimal across environments.

    Args:
        logits: ``(N,)`` predicted logits.
        y: ``(N,)`` binary targets.
        dummy_w: A scalar tensor with ``requires_grad=True`` (value ``1.0``).

    Returns:
        Scalar IRM penalty (add to the total loss with a small coefficient).
    """
    loss = F.binary_cross_entropy_with_logits(logits * dummy_w, y.float())
    grad = torch.autograd.grad(loss, dummy_w, create_graph=True)[0]
    return grad.pow(2).sum()
