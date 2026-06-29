"""Modality encoders, attention pooling and segment->bag pooling."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class AttentionPool(nn.Module):
    """Masked attention pooling of a frame sequence ``(N,T,d_in)`` -> ``(N,d_out)``.

    Args:
        d_in: Input feature dimension.
        d_out: Output (pooled) dimension.
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.attn = nn.Linear(d_out, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            x: ``(N, T, d_in)`` frame features.
            mask: ``(N, T)`` boolean mask (``True`` = valid frame).

        Returns:
            ``(N, d_out)`` pooled representation.
        """
        h = torch.tanh(self.proj(x))           # (N, T, d_out)
        a = self.attn(h).squeeze(-1)           # (N, T)
        if mask is not None:
            a = a.masked_fill(~mask, -1e4)
        w = torch.softmax(a, dim=1).unsqueeze(-1)  # (N, T, 1)
        return (w * h).sum(dim=1)              # (N, d_out)


class ModalityEncoder(nn.Module):
    """Encode one modality into a ``d``-dimensional vector.

    * Sequence modalities (audio/acoustic/visual): BiLSTM + :class:`AttentionPool`.
    * Static modality (text ``[CLS]``): MLP.

    Args:
        in_dim: Input feature dimension.
        d: Output embedding dimension.
        seq: If ``True`` treat input as a frame sequence ``(N,T,in_dim)``;
            otherwise as ``(N,in_dim)``.
    """

    def __init__(self, in_dim: int, d: int, seq: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.seq = seq
        self.drop = nn.Dropout(dropout)
        if seq:
            self.rnn = nn.LSTM(in_dim, d // 2, batch_first=True, bidirectional=True)
            self.pool = AttentionPool(d, d)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, d), nn.GELU(), nn.LayerNorm(d)
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return ``(N, d)`` embeddings."""
        if self.seq:
            h, _ = self.rnn(x)            # (N, T, d)
            return self.drop(self.pool(h, mask))     # (N, d)
        return self.drop(self.mlp(x))    # (N, d)


def pool_segments_to_bags(
    z_seg: torch.Tensor,
    seg2bag: torch.Tensor,
    n_bags: int,
    attn: nn.Module,
) -> torch.Tensor:
    """Attention-pool segment embeddings into bag embeddings.

    A learned linear scorer produces per-segment logits; segments belonging to
    the same bag are softmax-normalised and combined. Replaces the hard mean of
    the prior work.

    Args:
        z_seg: ``(N_seg, d)`` segment embeddings.
        seg2bag: ``(N_seg,)`` mapping segment -> bag index in ``[0, n_bags)``.
        n_bags: Number of bags ``B``.
        attn: ``Linear(d, 1)`` scoring module.

    Returns:
        ``(B, d)`` bag embeddings. Empty bags map to a zero vector.

    Note:
        Implemented with a numerically-stable scatter softmax so it is fully
        vectorised (no Python loop over bags).
    """
    d = z_seg.size(1)
    scores = attn(z_seg).squeeze(-1)                       # (N_seg,)

    # Scatter-softmax: subtract per-bag max for stability, then normalise.
    bag_max = z_seg.new_full((n_bags,), float("-inf"))
    bag_max = bag_max.scatter_reduce(0, seg2bag, scores, reduce="amax", include_self=True)
    bag_max = torch.where(torch.isinf(bag_max), torch.zeros_like(bag_max), bag_max)
    shifted = scores - bag_max[seg2bag]
    exp = torch.exp(shifted)                               # (N_seg,)
    denom = z_seg.new_zeros(n_bags).scatter_add(0, seg2bag, exp)
    denom = denom.clamp_min(1e-12)
    weights = (exp / denom[seg2bag]).unsqueeze(-1)         # (N_seg, 1)

    z_bag = z_seg.new_zeros(n_bags, d)
    z_bag = z_bag.scatter_add(
        0, seg2bag.unsqueeze(-1).expand(-1, d), weights * z_seg
    )
    return z_bag
