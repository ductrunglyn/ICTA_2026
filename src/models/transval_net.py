"""TransValNet: assembled two-level (segment + bag) depression model.

Per segment, each modality is encoded, fused (static MLP — fusion is
deliberately simple), then:

* a segment depression head produces auxiliary logits (probe + regulariser);
* segments are attention-pooled into bags;
* a bag depression head produces the main logit, plus a selective gate;
* a GRL + domain adversary erase corpus/gender from ``z_seg``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from ..data.dataset import MODALITY_ORDER
from .adversary import DomainAdversary
from .encoders import ModalityEncoder, pool_segments_to_bags
from .heads import DepressionHead, SelectiveHead

# Input dims per modality (frozen-backbone feature sizes).
IN_DIMS = {"audio": 1024, "acoustic": 79, "text": 768, "visual": 50}


class TransValNet(nn.Module):
    """Full TransVal-Dep network.

    Args:
        d: Shared embedding dimension.
        n_corpus: Number of corpora (for the adversary).
        n_gender: Number of gender classes.
        use_adv: Enable the domain adversary (NV3).
        use_visual: Whether the visual modality is used (kept for config
            symmetry; the visual branch always exists and is masked when
            absent so missing-modality corpora still run).
    """

    def __init__(
        self,
        d: int = 128,
        n_corpus: int = 3,
        n_gender: int = 2,
        use_adv: bool = True,
        use_visual: bool = True,
        dropout: float = 0.0,
        in_dims: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.d = d
        self.use_visual = use_visual
        # Per-modality input dims (override e.g. acoustic for openSMILE eGeMAPS).
        dims = dict(IN_DIMS)
        if in_dims:
            dims.update(in_dims)
        self.in_dims = dims
        self.enc = nn.ModuleDict(
            {
                "audio": ModalityEncoder(dims["audio"], d, seq=True, dropout=dropout),
                "acoustic": ModalityEncoder(dims["acoustic"], d, seq=True, dropout=dropout),
                "text": ModalityEncoder(dims["text"], d, seq=False, dropout=dropout),
                "visual": ModalityEncoder(dims["visual"], d, seq=True, dropout=dropout),
            }
        )
        self.fuse = nn.Sequential(
            nn.Linear(len(MODALITY_ORDER) * d, d), nn.GELU(), nn.LayerNorm(d),
            nn.Dropout(dropout),
        )
        self.bag_attn = nn.Linear(d, 1)
        self.dep_seg = DepressionHead(d)
        self.dep_bag = DepressionHead(d)
        self.selective = SelectiveHead(d)
        self.adv: Optional[DomainAdversary] = (
            DomainAdversary(d, n_corpus, n_gender) if use_adv else None
        )

    def encode_segments(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode + fuse all modalities into ``z_seg (N_seg, d)``."""
        embeds = []
        for m in MODALITY_ORDER:
            x = batch[m]
            if m == "text":
                embeds.append(self.enc[m](x))
            else:
                embeds.append(self.enc[m](x, batch[f"{m}_mask"]))
        # Stack -> (N_seg, n_mod, d), zero out absent modalities, then fuse.
        E = torch.stack(embeds, dim=1)                       # (N_seg, M, d)
        E = E * batch["modality_mask"].unsqueeze(-1)         # mask missing
        return self.fuse(E.flatten(1))                       # (N_seg, d)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the full forward pass.

        Args:
            batch: Output of :func:`src.data.dataset.collate_bags`.

        Returns:
            Dict with ``logit_bag (B,)``, ``logit_seg (N_seg,)``,
            ``gate_bag (B,)``, ``z_seg``, ``z_bag`` and (if ``use_adv``)
            ``corpus_logit``/``gender_logit``.
        """
        z_seg = self.encode_segments(batch)
        n_bags = batch["bag_labels"].size(0)
        z_bag = pool_segments_to_bags(
            z_seg, batch["seg2bag"], n_bags, self.bag_attn
        )
        out: Dict[str, torch.Tensor] = {
            "logit_bag": self.dep_bag(z_bag),
            "logit_seg": self.dep_seg(z_seg),
            "gate_bag": self.selective(z_bag),
            "z_seg": z_seg,
            "z_bag": z_bag,
        }
        if self.adv is not None:
            corpus_logit, gender_logit = self.adv(z_seg)
            out["corpus_logit"] = corpus_logit
            out["gender_logit"] = gender_logit
        return out
