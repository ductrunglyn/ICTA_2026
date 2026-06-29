"""Statistics with sufficient power: bootstrap CIs, TOST, seed aggregation (NV5)."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float]]:
    """Participant-level bootstrap confidence interval for a metric.

    Args:
        y: Binary labels (one entry per participant).
        p: Predicted scores (one entry per participant).
        metric: Callable ``metric(y, p) -> float``.
        n: Number of bootstrap resamples.
        level: Confidence level.
        seed: RNG seed.

    Returns:
        Tuple ``(mean, (lo, hi))``.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    idx0 = np.arange(len(y))
    vals: List[float] = []
    for _ in range(n):
        s = rng.choice(idx0, len(idx0), replace=True)  # participant-level resample
        try:
            vals.append(float(metric(y[s], p[s])))
        except ValueError:
            # Skip resamples with a single class present.
            continue
    if not vals:
        return float("nan"), (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(np.mean(vals)), (float(lo), float(hi))


def tost_equivalence(diffs: Sequence[float], eps: float) -> float:
    """Two One-Sided Tests for equivalence within margin ``+/- eps``.

    ``diffs`` is the per-(fold x seed) metric difference ``A - B``. The null is
    ``|Delta| >= eps``; a returned p-value ``< 0.05`` lets us conclude
    *equivalence* (not merely "no significant difference").

    Args:
        diffs: Paired metric differences.
        eps: Pre-registered equivalence margin.

    Returns:
        The larger of the two one-sided p-values.
    """
    diffs = np.asarray(diffs, dtype=np.float64).reshape(-1)
    n = len(diffs)
    if n < 2:
        return float("nan")
    m = diffs.mean()
    sd = diffs.std(ddof=1)
    se = sd / np.sqrt(n) if sd > 0 else 1e-12
    t_low = (m - (-eps)) / se
    p_low = 1 - student_t.cdf(t_low, n - 1)
    t_high = (eps - m) / se
    p_high = 1 - student_t.cdf(t_high, n - 1)
    return float(max(p_low, p_high))


def aggregate_seeds(
    per_run: Sequence[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Aggregate metric dicts across folds x seeds.

    Args:
        per_run: List of metric dicts (one per fold/seed run).

    Returns:
        Mapping ``metric -> {"mean","std","n"}`` ignoring ``nan`` entries.
    """
    keys = set().union(*(d.keys() for d in per_run)) if per_run else set()
    out: Dict[str, Dict[str, float]] = {}
    for k in sorted(keys):
        vals = np.array([d[k] for d in per_run if k in d], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        else:
            out[k] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
            }
    return out
