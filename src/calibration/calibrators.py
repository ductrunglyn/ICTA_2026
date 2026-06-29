"""Post-hoc probability calibration (Platt / Isotonic), fit on train-fold only.

Calibration is never learned inside the network. After a fold is trained, the
calibrator is fit on the *inner-validation* logits (a held-out slice of the
training fold) and applied to the test fold. The test fold is never touched
during fitting.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


class ProbabilityCalibrator:
    """Platt scaling or isotonic regression calibrator.

    Args:
        method: ``"platt"`` (logistic on logits) or ``"isotonic"`` (monotone on
            probabilities).
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in {"platt", "isotonic"}:
            raise ValueError(f"Unknown calibration method '{method}'")
        self.method = method
        self.model = None

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        """Fit on uncalibrated logits and binary labels (train-fold only)."""
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        y = np.asarray(y).reshape(-1)
        if self.method == "platt":
            self.model = LogisticRegression(C=1e6, solver="lbfgs").fit(
                logits.reshape(-1, 1), y
            )
        else:
            p = _sigmoid(logits)
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Map logits -> calibrated probabilities in ``[0, 1]``."""
        if self.model is None:
            raise RuntimeError("Calibrator must be fit before transform().")
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        if self.method == "platt":
            return self.model.predict_proba(logits.reshape(-1, 1))[:, 1]
        return self.model.predict(_sigmoid(logits))

    def fit_transform(self, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(logits, y).transform(logits)


def choose_threshold(
    y: np.ndarray, p: np.ndarray, strategy: str = "youden_inner"
) -> float:
    """Select a decision threshold on inner-validation probabilities.

    Args:
        y: Binary labels.
        p: Calibrated probabilities.
        strategy: ``"youden_inner"`` (maximise sensitivity+specificity-1) or
            ``"fixed"`` (return ``0.5``).

    Returns:
        Chosen probability threshold.
    """
    if strategy == "fixed":
        return 0.5
    from sklearn.metrics import roc_curve

    y = np.asarray(y).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    youden = tpr - fpr
    return float(thr[int(np.argmax(youden))])
