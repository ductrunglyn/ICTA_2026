"""MIL bag dataset and bag-level collate (participant = bag, segment = instance).

Labels live at the *bag* (participant) level; segments are unlabelled
instances. The collate function flattens every bag's segments into a single
segment batch, records ``seg2bag`` so the model can re-pool segments into bags,
builds per-modality padding masks and a ``modality_mask`` (modalities are never
imputed — missing ones are masked out).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .features import ACOUSTIC_DIM, AUDIO_DIM, TEXT_DIM, VISUAL_DIM

# Fixed modality ordering shared with the model's encoder/modality_mask.
MODALITY_ORDER: List[str] = ["audio", "acoustic", "text", "visual"]
SEQ_MODALITIES = {"audio", "acoustic", "visual"}
MODALITY_DIM = {
    "audio": AUDIO_DIM,
    "acoustic": ACOUSTIC_DIM,
    "text": TEXT_DIM,
    "visual": VISUAL_DIM,
}
N_GENDER = 2


@dataclass
class Bag:
    """A participant-level bag of segment feature dicts.

    Attributes:
        participant_id: Owning participant.
        label: Bag-level binary label.
        group_id: ``(corpus, gender)`` group id for Group-DRO.
        segments: List of per-segment feature dicts. Each dict has numpy arrays
            (or ``None``) for ``audio/acoustic/text/visual`` plus integer
            ``qtype``, ``corpus_id`` and ``gender_id``.
    """

    participant_id: str
    label: int
    group_id: int
    segments: List[Dict[str, object]] = field(default_factory=list)


def make_group_id(corpus_id: int, gender_id: int, n_gender: int = N_GENDER) -> int:
    """Combine corpus and gender into a single Group-DRO group id."""
    return int(corpus_id) * n_gender + int(gender_id)


class BagDataset(Dataset):
    """Dataset over participant bags.

    Args:
        bags: Sequence of :class:`Bag` objects (or equivalent dicts).
    """

    def __init__(self, bags: Sequence[Bag]) -> None:
        self.bags: List[Bag] = [
            b if isinstance(b, Bag) else Bag(**b) for b in bags
        ]

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> Bag:
        return self.bags[idx]


def _to_seq_tensor(arr: Optional[np.ndarray], dim: int) -> torch.Tensor:
    """Convert a sequence modality array to a tensor, with a zero placeholder."""
    if arr is None:
        return torch.zeros(1, dim, dtype=torch.float32)
    t = torch.as_tensor(np.asarray(arr), dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


def collate_bags(bags: Sequence[Bag]) -> Dict[str, torch.Tensor]:
    """Collate a list of bags into a flat segment batch.

    Args:
        bags: Bags returned by :class:`BagDataset`.

    Returns:
        Dict of tensors (see module docstring / blueprint section 3.6):
        ``audio (N_seg,T,1024)``, ``audio_mask``, ``acoustic``,
        ``acoustic_mask``, ``text (N_seg,768)``, ``visual``, ``visual_mask``,
        ``modality_mask (N_seg,4)``, ``seg2bag (N_seg,)``, ``bag_labels (B,)``,
        ``group_ids (B,)``, ``corpus_ids (N_seg,)``, ``gender_ids (N_seg,)``,
        ``qtypes (N_seg,)``.
    """
    seq_lists: Dict[str, List[torch.Tensor]] = {m: [] for m in SEQ_MODALITIES}
    text_list: List[torch.Tensor] = []
    modality_mask: List[List[float]] = []
    seg2bag: List[int] = []
    corpus_ids: List[int] = []
    gender_ids: List[int] = []
    qtypes: List[int] = []
    bag_labels: List[int] = []
    group_ids: List[int] = []

    for b_idx, bag in enumerate(bags):
        bag_labels.append(int(bag.label))
        group_ids.append(int(bag.group_id))
        for seg in bag.segments:
            present = []
            for m in MODALITY_ORDER:
                val = seg.get(m)
                present.append(1.0 if val is not None else 0.0)
                if m in SEQ_MODALITIES:
                    seq_lists[m].append(_to_seq_tensor(val, MODALITY_DIM[m]))
            txt = seg.get("text")
            if txt is None:
                text_list.append(torch.zeros(TEXT_DIM, dtype=torch.float32))
            else:
                text_list.append(torch.as_tensor(np.asarray(txt), dtype=torch.float32))
            modality_mask.append(present)
            seg2bag.append(b_idx)
            corpus_ids.append(int(seg.get("corpus_id", 0)))
            gender_ids.append(int(seg.get("gender_id", 0)))
            qtypes.append(int(seg.get("qtype", 0)))

    out: Dict[str, torch.Tensor] = {}
    for m in ("audio", "acoustic", "visual"):
        tensors = seq_lists[m]
        lengths = torch.tensor([t.size(0) for t in tensors], dtype=torch.long)
        padded = pad_sequence(tensors, batch_first=True)  # (N_seg, T_max, F)
        t_max = padded.size(1)
        mask = torch.arange(t_max)[None, :] < lengths[:, None]  # (N_seg, T_max)
        out[m] = padded
        out[f"{m}_mask"] = mask
    out["text"] = torch.stack(text_list, dim=0)
    out["modality_mask"] = torch.tensor(modality_mask, dtype=torch.float32)
    out["seg2bag"] = torch.tensor(seg2bag, dtype=torch.long)
    out["bag_labels"] = torch.tensor(bag_labels, dtype=torch.long)
    out["group_ids"] = torch.tensor(group_ids, dtype=torch.long)
    out["corpus_ids"] = torch.tensor(corpus_ids, dtype=torch.long)
    out["gender_ids"] = torch.tensor(gender_ids, dtype=torch.long)
    out["qtypes"] = torch.tensor(qtypes, dtype=torch.long)
    return out
