"""Domain adversary predicting corpus and gender from segment embeddings (NV3).

Placed *after* a Gradient Reversal Layer so that minimising the adversary's
loss forces the encoder to *erase* corpus/gender information from ``z_seg``.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .grl import GradientReversal


class DomainAdversary(nn.Module):
    """Predict ``corpus_id`` and ``gender_id`` from ``z_seg`` after GRL.

    Args:
        d: Embedding dimension of ``z_seg``.
        n_corpus: Number of corpora.
        n_gender: Number of gender classes.
        lambd: Initial gradient-reversal strength.
    """

    def __init__(self, d: int, n_corpus: int, n_gender: int = 2, lambd: float = 1.0) -> None:
        super().__init__()
        self.grl = GradientReversal(lambd)
        self.corpus_clf = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_corpus)
        )
        self.gender_clf = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_gender)
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            z: ``(N, d)`` segment embeddings.

        Returns:
            Tuple of ``(corpus_logits (N,n_corpus), gender_logits (N,n_gender))``.
        """
        z = self.grl(z)
        return self.corpus_clf(z), self.gender_clf(z)
