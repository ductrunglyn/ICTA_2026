"""Evaluation: metrics, confound analysis, statistics, validity probe."""

from .metrics import (
    roc_auc,
    expected_calibration_error,
    brier_score,
    partial_auc,
    sensitivity_specificity,
    f1_precision_recall,
    compute_all_metrics,
)
from .confound_eval import ConfoundEvaluator
from .stats import bootstrap_ci, tost_equivalence, aggregate_seeds
from .probe import QuestionTypeProbe

__all__ = [
    "roc_auc",
    "expected_calibration_error",
    "brier_score",
    "partial_auc",
    "sensitivity_specificity",
    "f1_precision_recall",
    "compute_all_metrics",
    "ConfoundEvaluator",
    "bootstrap_ci",
    "tost_equivalence",
    "aggregate_seeds",
    "QuestionTypeProbe",
]
