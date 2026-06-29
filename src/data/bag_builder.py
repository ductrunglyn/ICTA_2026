"""Build MIL bags from a participant manifest, a segments manifest and cache.

Bridges the on-disk feature cache (``data/interim/features/<seg_id>.pt``) and
the in-memory :class:`Bag` structures consumed by the dataset/training code.
Also supports restricting to a subset of modalities (e.g. acoustic-only for E0).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dataset import Bag, MODALITY_ORDER, make_group_id
from .features import FeatureCache
from .splitter import add_corpus_id


class BagBuilder:
    """Assemble :class:`Bag` objects on demand for given participants.

    Args:
        manifest_df: Participant-level manifest (``participant_id``, ``label``,
            ``gender``, ``corpus``/``corpus_id``).
        segments_df: Segment-level manifest (``participant_id``, ``seg_id``,
            ``qtype``).
        feature_cache: Cache to load per-segment feature dicts from.
        modalities: Subset of modalities to keep; others are masked to ``None``.
            ``None`` keeps all modalities.
        n_gender: Number of gender classes (for group ids).
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        segments_df: pd.DataFrame,
        feature_cache: FeatureCache,
        modalities: Optional[Iterable[str]] = None,
        n_gender: int = 2,
    ) -> None:
        man = manifest_df.copy()
        if "corpus_id" not in man.columns:
            man = add_corpus_id(man)
        self.manifest = man.set_index("participant_id")
        self.segments = segments_df
        self.cache = feature_cache
        self.modalities = set(modalities) if modalities is not None else set(MODALITY_ORDER)
        self.n_gender = n_gender
        self._segs_by_pid = {
            pid: grp for pid, grp in segments_df.groupby("participant_id")
        }

    def _segment_feature(self, seg_row: pd.Series, corpus_id: int, gender_id: int) -> Dict[str, object]:
        feat = self.cache.load(seg_row["seg_id"])
        seg: Dict[str, object] = {}
        for m in MODALITY_ORDER:
            val = feat.get(m)
            seg[m] = val if (m in self.modalities and val is not None) else None
        seg["qtype"] = int(seg_row.get("qtype", feat.get("qtype", 0)))
        seg["corpus_id"] = int(corpus_id)
        seg["gender_id"] = int(gender_id)
        return seg

    def build(self, participant_ids: Sequence[str]) -> List[Bag]:
        """Return a list of bags for the requested participants.

        Participants with no cached segments are skipped with no error so the
        provider stays robust to partial feature extraction.
        """
        bags: List[Bag] = []
        for pid in participant_ids:
            if pid not in self.manifest.index or pid not in self._segs_by_pid:
                continue
            row = self.manifest.loc[pid]
            corpus_id = int(row["corpus_id"])
            gender_id = int(row.get("gender", 0))
            segs = [
                self._segment_feature(s, corpus_id, gender_id)
                for _, s in self._segs_by_pid[pid].iterrows()
            ]
            if not segs:
                continue
            bags.append(
                Bag(
                    participant_id=pid,
                    label=int(row["label"]),
                    group_id=make_group_id(corpus_id, gender_id, self.n_gender),
                    segments=segs,
                )
            )
        return bags

    def n_groups(self) -> int:
        """Number of distinct ``(corpus, gender)`` groups for Group-DRO."""
        n_corpus = int(self.manifest["corpus_id"].max()) + 1
        return n_corpus * self.n_gender
