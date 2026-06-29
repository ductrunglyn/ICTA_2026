"""Tests for the gradient reversal layer."""

import torch

from src.models.grl import GradientReversal, grad_reverse


def test_forward_is_identity():
    x = torch.randn(4, 3)
    out = GradientReversal(0.7)(x)
    assert torch.allclose(out, x)


def test_backward_reverses_and_scales_gradient():
    x = torch.randn(5, 2, requires_grad=True)
    lambd = 2.0
    y = grad_reverse(x, lambd).sum()
    y.backward()
    # d(sum x)/dx = 1 everywhere; GRL flips sign and scales by lambd.
    assert torch.allclose(x.grad, torch.full_like(x.grad, -lambd))


def test_lambda_zero_blocks_gradient():
    x = torch.randn(3, requires_grad=True)
    grad_reverse(x, 0.0).sum().backward()
    assert torch.allclose(x.grad, torch.zeros_like(x.grad))
