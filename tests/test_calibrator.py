"""Tests for probability calibration and selective prediction."""

import numpy as np

from src.calibration.calibrators import ProbabilityCalibrator, choose_threshold
from src.calibration.selective import aurc, risk_coverage_curve


def _toy_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    logits = rng.normal(loc=(y * 2.0 - 1.0), scale=1.0)  # signal + noise
    return y, logits


def test_isotonic_outputs_valid_probabilities():
    y, logits = _toy_data()
    cal = ProbabilityCalibrator("isotonic").fit(logits, y)
    p = cal.transform(logits)
    assert p.shape == logits.shape
    assert np.all((p >= 0) & (p <= 1))


def test_platt_outputs_valid_probabilities():
    y, logits = _toy_data()
    p = ProbabilityCalibrator("platt").fit_transform(logits, y)
    assert np.all((p >= 0) & (p <= 1))


def test_choose_threshold_in_unit_interval():
    y, logits = _toy_data()
    p = ProbabilityCalibrator("isotonic").fit_transform(logits, y)
    thr = choose_threshold(y, p, "youden_inner")
    assert 0.0 <= thr <= 1.0


def test_risk_coverage_curve_shapes():
    y, logits = _toy_data()
    p = 1 / (1 + np.exp(-logits))
    gate = np.abs(p - 0.5)  # confidence proxy
    cov, risk = risk_coverage_curve(y, p, gate)
    assert len(cov) == len(risk) == len(y)
    assert cov[-1] == 1.0
    assert 0.0 <= aurc(cov, risk) <= 1.0
