"""Core evaluation metrics.

AUC-ROC is the primary, threshold-independent metric. Calibration (ECE,
Brier), sensitivity/specificity and partial AUC are also reported.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC, returning ``nan`` when only one class is present."""
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error with equal-width probability bins.

    Args:
        y: Binary labels.
        p: Predicted probabilities.
        n_bins: Number of bins over ``[0, 1]``.

    Returns:
        Weighted mean absolute gap between confidence and accuracy.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=False)
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score (mean squared error of probabilistic predictions)."""
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return float(brier_score_loss(y, p))


def partial_auc(y: np.ndarray, p: np.ndarray, max_fpr: float = 0.2) -> float:
    """Standardised partial AUC restricted to ``fpr <= max_fpr``.

    Uses scikit-learn's McClish-corrected partial AUC.
    """
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p, max_fpr=max_fpr))


def sensitivity_specificity(
    y: np.ndarray, yhat: np.ndarray
) -> Dict[str, float]:
    """Compute sensitivity (recall) and specificity from hard predictions."""
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)
    tp = float(((yhat == 1) & (y == 1)).sum())
    tn = float(((yhat == 0) & (y == 0)).sum())
    fp = float(((yhat == 1) & (y == 0)).sum())
    fn = float(((yhat == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {"sensitivity": sens, "specificity": spec}


def f1_precision_recall(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    """F1, precision and recall from hard predictions."""
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)
    return {
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
    }


def compute_all_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float = 0.5,
    max_fpr: float = 0.2,
) -> Dict[str, float]:
    """Compute the full primary metric suite for one prediction set.

    Args:
        y: Binary labels.
        p: Calibrated probabilities.
        threshold: Decision threshold for the hard-label metrics.
        max_fpr: Upper FPR bound for partial AUC.

    Returns:
        Dict of metric name -> value.
    """
    yhat = (np.asarray(p, dtype=np.float64).reshape(-1) >= threshold).astype(int)
    metrics: Dict[str, float] = {
        "auc": roc_auc(y, p),
        "partial_auc": partial_auc(y, p, max_fpr=max_fpr),
        "ece": expected_calibration_error(y, p),
        "brier": brier_score(y, p),
    }
    metrics.update(f1_precision_recall(y, yhat))
    metrics.update(sensitivity_specificity(y, yhat))
    return metrics
