"""Leakage-free, participant-level cross-validation splitter (NV1).

Two evaluation regimes are provided in one class:

* ``pooled``: stratified K-fold over pooled corpora (statistical power).
* ``loco`` : leave-one-corpus-out (genuine transfer test).

The unit of splitting is always the *participant* so that no participant
appears in more than one test fold and calibrators/thresholds are only ever
fit on training participants.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Canonical, stable ordering of corpora -> integer ids.
CORPUS_ORDER: List[str] = ["daic", "eatd", "androids", "edaic", "cmdc", "modma"]


def add_corpus_id(df: pd.DataFrame, corpus_col: str = "corpus") -> pd.DataFrame:
    """Return a copy of ``df`` with an integer ``corpus_id`` column.

    Unknown corpora are appended deterministically after the canonical list so
    the mapping is stable for a given manifest.

    Args:
        df: Manifest with a ``corpus`` column.
        corpus_col: Name of the corpus column.

    Returns:
        Copy of ``df`` with an added ``corpus_id`` column.
    """
    out = df.copy()
    known = {c: i for i, c in enumerate(CORPUS_ORDER)}
    extras = sorted(set(out[corpus_col]) - set(known))
    for i, c in enumerate(extras):
        known[c] = len(CORPUS_ORDER) + i
    out["corpus_id"] = out[corpus_col].map(known).astype(int)
    return out


class LeakageFreeSplitter:
    """Participant-level CV splitter, stratified by ``(label, corpus)``.

    Args:
        manifest_df: Participant-level manifest. Must contain
            ``participant_id``, ``label`` and ``corpus`` (``corpus_id`` is
            derived if absent).
        n_folds: Number of folds for the pooled regime.
        seed: Random seed controlling the fold shuffle.
        mode: ``"pooled"`` for stratified K-fold, ``"loco"`` for
            leave-one-corpus-out.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        n_folds: int = 5,
        seed: int = 0,
        mode: str = "pooled",
    ) -> None:
        if mode not in {"pooled", "loco"}:
            raise ValueError(f"Unknown mode '{mode}' (expected pooled|loco)")
        df = manifest_df.reset_index(drop=True)
        if "corpus_id" not in df.columns:
            df = add_corpus_id(df)
        self.df = df
        self.k = n_folds
        self.seed = seed
        self.mode = mode

    # -- public API ---------------------------------------------------------
    def folds(self) -> Iterator[Tuple[List[str], List[str]]]:
        """Yield ``(train_ids, test_ids)`` participant-id lists per fold."""
        if self.mode == "pooled":
            yield from self._pooled_folds()
        else:
            yield from self._loco_folds()

    def n_splits(self) -> int:
        """Number of folds for the configured mode."""
        if self.mode == "pooled":
            return self.k
        return int(self.df["corpus_id"].nunique())

    # -- regimes ------------------------------------------------------------
    def _pooled_folds(self) -> Iterator[Tuple[List[str], List[str]]]:
        # Stratify on a joint key so both positive rate and corpus mix are
        # preserved across folds.
        key = self.df["label"].astype(int) * 10 + self.df["corpus_id"].astype(int)
        skf = StratifiedKFold(self.k, shuffle=True, random_state=self.seed)
        pid = self.df["participant_id"].to_numpy()
        for tr, te in skf.split(self.df, key):
            yield pid[tr].tolist(), pid[te].tolist()

    def _loco_folds(self) -> Iterator[Tuple[List[str], List[str]]]:
        pid = self.df["participant_id"].to_numpy()
        corpus = self.df["corpus_id"].to_numpy()
        for held in sorted(self.df["corpus_id"].unique()):
            test_mask = corpus == held
            yield pid[~test_mask].tolist(), pid[test_mask].tolist()

    # -- safety checks ------------------------------------------------------
    def assert_no_leakage(self) -> None:
        """Validate that no participant is shared across two test folds.

        Raises:
            AssertionError: If any participant appears in more than one test
                fold (which would constitute leakage).
        """
        seen: set = set()
        for _, test_ids in self.folds():
            overlap = seen.intersection(test_ids)
            assert not overlap, f"Participant leakage across folds: {overlap}"
            seen.update(test_ids)
