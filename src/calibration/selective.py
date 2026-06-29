"""Selective prediction: risk-coverage analysis and an abstain wrapper (NV4)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import f1_score


def risk_coverage_curve(
    y: np.ndarray, p: np.ndarray, gate: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the risk-coverage curve.

    Cases are admitted in decreasing order of the selective ``gate`` (most
    confident first). At each coverage level the risk is ``1 - F1`` over the
    admitted set.

    Args:
        y: Binary labels.
        p: Calibrated probabilities.
        gate: Selective gate values (higher = more confident to predict).
        threshold: Decision threshold for ``p``.

    Returns:
        Tuple ``(coverage, risk)`` arrays.
    """
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    gate = np.asarray(gate, dtype=np.float64).reshape(-1)
    order = np.argsort(-gate)
    n = len(order)
    cov, risk = [], []
    for k in range(1, n + 1):
        idx = order[:k]
        yhat = (p[idx] >= threshold).astype(int)
        cov.append(k / n)
        if len(np.unique(y[idx])) < 2:
            # F1 undefined with a single class present; treat risk as error rate.
            risk.append(float((yhat != y[idx]).mean()))
        else:
            risk.append(1.0 - f1_score(y[idx], yhat))
    return np.asarray(cov), np.asarray(risk)


def aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    coverage = np.asarray(coverage, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64)
    if len(coverage) < 2:
        return float(risk.mean()) if len(risk) else 0.0
    # np.trapezoid (NumPy>=2.0) with a fallback to the legacy np.trapz name.
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(risk, coverage))


def f1_at_coverage(
    y: np.ndarray, p: np.ndarray, gate: np.ndarray, coverage: float, threshold: float = 0.5
) -> float:
    """F1 over the most-confident fraction ``coverage`` of cases."""
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    gate = np.asarray(gate, dtype=np.float64).reshape(-1)
    n = len(y)
    k = max(1, int(round(coverage * n)))
    idx = np.argsort(-gate)[:k]
    yhat = (p[idx] >= threshold).astype(int)
    if len(np.unique(y[idx])) < 2:
        return float((yhat == y[idx]).mean())
    return float(f1_score(y[idx], yhat))


class SelectivePredictor:
    """Wrap calibrated probabilities + gate into accept/abstain decisions.

    Args:
        threshold: Decision threshold on probabilities.
        gate_threshold: Gate value above which a prediction is made; below it
            the model abstains (returns ``-1``).
    """

    def __init__(self, threshold: float = 0.5, gate_threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.gate_threshold = gate_threshold

    def predict(self, p: np.ndarray, gate: np.ndarray) -> np.ndarray:
        """Return predictions in ``{0, 1}`` or ``-1`` for abstentions."""
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        gate = np.asarray(gate, dtype=np.float64).reshape(-1)
        yhat = (p >= self.threshold).astype(int)
        yhat[gate < self.gate_threshold] = -1
        return yhat

    def coverage(self, gate: np.ndarray) -> float:
        """Fraction of cases the predictor does not abstain on."""
        gate = np.asarray(gate, dtype=np.float64).reshape(-1)
        return float((gate >= self.gate_threshold).mean())
