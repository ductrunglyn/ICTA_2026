"""Confound-controlled evaluation (NV2 — the headline contribution).

Reports group-conditioned AUC, residualised AUC (logit regressed on
confounds), and the specificity-gap = AUC(depression) - AUC(distress-proxy).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


class ConfoundEvaluator:
    """Confound analysis utilities.

    All methods are static-style (stateless) and operate on numpy arrays so
    they can be reused across folds/seeds.
    """

    def partial_auc_by_group(
        self, y: np.ndarray, p: np.ndarray, group: np.ndarray
    ) -> Dict[int, float]:
        """AUC computed separately within each confound group.

        Args:
            y: Binary labels.
            p: Predicted scores/probabilities.
            group: Integer group id per sample (gender / corpus / length-bin).

        Returns:
            Mapping ``group_id -> AUC``.
        """
        y = np.asarray(y).reshape(-1)
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        group = np.asarray(group).reshape(-1)
        return {
            int(g): _safe_auc(y[group == g], p[group == g])
            for g in np.unique(group)
        }

    def residualized_auc(
        self, y: np.ndarray, logit: np.ndarray, C: np.ndarray
    ) -> float:
        """AUC of the residual after regressing logits on confounds.

        A large drop relative to the raw AUC indicates the model is exploiting
        confounds rather than depression signal.

        Args:
            y: Binary labels.
            logit: Uncalibrated logits.
            C: ``(n, k)`` confound matrix (gender, age, length, ...).

        Returns:
            AUC computed on the residualised logits.
        """
        y = np.asarray(y).reshape(-1)
        logit = np.asarray(logit, dtype=np.float64).reshape(-1)
        C = np.asarray(C, dtype=np.float64)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        if C.shape[1] == 0:
            return _safe_auc(y, logit)
        pred = LinearRegression().fit(C, logit).predict(C)
        residual = logit - pred
        return _safe_auc(y, residual)

    def specificity_gap(
        self, y_dep: np.ndarray, y_distress: np.ndarray, p: np.ndarray
    ) -> float:
        """``AUC(depression) - AUC(distress-proxy)``.

        ``y_distress`` marks general distress (e.g. PTSD/anxiety or high score
        below the depression threshold). A gap near zero means the model tracks
        distress, not depression specifically.

        Args:
            y_dep: Depression labels.
            y_distress: Distress-proxy labels.
            p: Predicted scores.

        Returns:
            The specificity gap.
        """
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        return _safe_auc(np.asarray(y_dep).reshape(-1), p) - _safe_auc(
            np.asarray(y_distress).reshape(-1), p
        )

    def evaluate(
        self,
        y: np.ndarray,
        p: np.ndarray,
        logit: np.ndarray,
        C: np.ndarray,
        groups: Dict[str, np.ndarray],
    ) -> Dict[str, object]:
        """Run the full confound report.

        Args:
            y: Binary labels.
            p: Calibrated probabilities.
            logit: Uncalibrated logits (for residualisation).
            C: Confound matrix.
            groups: Mapping ``name -> group_id_array`` (e.g. ``{"gender": ...}``).

        Returns:
            Nested dict with overall AUC, residualised AUC, and per-group AUCs.
        """
        report: Dict[str, object] = {
            "auc": _safe_auc(np.asarray(y).reshape(-1), np.asarray(p).reshape(-1)),
            "residualized_auc": self.residualized_auc(y, logit, C),
        }
        for name, g in groups.items():
            report[f"auc_by_{name}"] = self.partial_auc_by_group(y, p, g)
        return report
