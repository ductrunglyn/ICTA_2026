"""Extraction and normalisation of confound metadata (NV2).

Confounds (gender, age, interview length, comorbidity) are pulled from the
participant manifest and standardised so the :mod:`eval.confound_eval` module
can residualise model logits against them.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# Columns treated as confounds for residualisation / stratification.
CONFOUND_COLUMNS: List[str] = [
    "gender",
    "age",
    "interview_len_s",
    "comorbidity_ptsd",
]


class ConfoundExtractor:
    """Build a standardised confound matrix from a participant manifest.

    Continuous columns are z-scored; binary/categorical columns are passed
    through. Missing columns are skipped, missing values are mean/zero-filled.

    Args:
        columns: Confound columns to use (defaults to :data:`CONFOUND_COLUMNS`).
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = list(columns) if columns is not None else list(CONFOUND_COLUMNS)
        self.means_: dict = {}
        self.stds_: dict = {}
        self.used_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "ConfoundExtractor":
        """Learn normalisation statistics from a manifest."""
        self.used_ = [c for c in self.columns if c in df.columns]
        for col in self.used_:
            vals = pd.to_numeric(df[col], errors="coerce")
            self.means_[col] = float(np.nanmean(vals)) if vals.notna().any() else 0.0
            std = float(np.nanstd(vals)) if vals.notna().any() else 0.0
            self.stds_[col] = std if std > 1e-8 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return a ``(n, k)`` standardised confound matrix.

        Continuous columns (more than two unique values) are z-scored using the
        statistics learned in :meth:`fit`; binary columns are mean-imputed only.
        """
        if not self.used_:
            return np.zeros((len(df), 0), dtype=np.float64)
        cols = []
        for col in self.used_:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            vals = np.where(np.isnan(vals), self.means_[col], vals)
            n_unique = len(np.unique(vals))
            if n_unique > 2:  # continuous -> z-score
                vals = (vals - self.means_[col]) / self.stds_[col]
            cols.append(vals)
        return np.stack(cols, axis=1)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience wrapper around :meth:`fit` then :meth:`transform`."""
        return self.fit(df).transform(df)


def length_bins(interview_len_s: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Bin interview lengths into quantile groups for stratified AUC reports.

    Args:
        interview_len_s: Array of interview durations.
        n_bins: Number of quantile bins.

    Returns:
        Integer bin id per participant.
    """
    x = np.asarray(interview_len_s, dtype=float)
    valid = ~np.isnan(x)
    bins = np.zeros_like(x, dtype=int)
    if valid.sum() == 0:
        return bins
    quantiles = np.quantile(x[valid], np.linspace(0, 1, n_bins + 1)[1:-1])
    bins[valid] = np.digitize(x[valid], quantiles)
    return bins
