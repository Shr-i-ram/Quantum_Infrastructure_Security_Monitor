"""Unit tests for V0.8 statistical drift detectors and distance metrics."""

import numpy as np
import pytest

from qism.drift.detectors import (
    AggregationStrategy,
    DriftDetectorConfig,
    DriftMetric,
    StatisticalDriftDetector,
    compute_ks_distance,
    compute_psi_distance,
    compute_wasserstein_distance,
)


def test_distance_metrics_identical_distributions():
    """Verify statistical distances are zero or negligible for identical distributions."""
    rng = np.random.RandomState(42)
    data = rng.normal(loc=0.0, scale=1.0, size=200)

    ks_stat, ks_pval = compute_ks_distance(data, data)
    assert np.isclose(ks_stat, 0.0)
    assert ks_pval >= 0.99

    w1_stat, _ = compute_wasserstein_distance(data, data)
    assert np.isclose(w1_stat, 0.0)

    psi_stat, _ = compute_psi_distance(data, data)
    assert np.isclose(psi_stat, 0.0)


def test_distance_metrics_shifted_distribution():
    """Verify statistical distances clearly detect distribution shifts."""
    rng = np.random.RandomState(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=150)
    curr = rng.normal(loc=3.5, scale=1.0, size=150)  # Distinct shift of 3.5 sigma

    ks_stat, ks_pval = compute_ks_distance(ref, curr)
    assert ks_stat > 0.80
    assert ks_pval < 1e-4

    w1_stat, _ = compute_wasserstein_distance(ref, curr)
    assert w1_stat > 2.0  # Normalized distance

    psi_stat, _ = compute_psi_distance(ref, curr)
    assert psi_stat > 1.0


def test_multivariate_drift_detector_clean_vs_drift():
    """Verify StatisticalDriftDetector identifies multivariate shift and dominant features."""
    rng = np.random.RandomState(42)
    n, d = 100, 4
    feature_names = ["qber", "signal_gain", "temp", "vibration"]

    ref_data = rng.normal(loc=0.0, scale=1.0, size=(n, d))
    curr_clean = rng.normal(loc=0.0, scale=1.0, size=(n, d))

    cfg = DriftDetectorConfig(
        metric=DriftMetric.WASSERSTEIN,
        aggregation=AggregationStrategy.MEAN,
        threshold=0.50,
        feature_names=feature_names,
    )
    detector = StatisticalDriftDetector(cfg)

    # 1. Clean comparison
    clean_rep = detector.compute_drift(ref_data, curr_clean)
    assert not clean_rep.is_drift
    assert clean_rep.drift_score < 0.50

    # 2. Shift specifically applied to 'temp' (column 2)
    curr_drift = curr_clean.copy()
    curr_drift[:, 2] += 4.0

    drift_rep = detector.compute_drift(ref_data, curr_drift)
    assert drift_rep.is_drift
    assert drift_rep.drift_score > 0.50
    assert "temp" in drift_rep.dominant_features
    assert drift_rep.dominant_features[0] == "temp"


def test_drift_threshold_calibration():
    """Verify empirical threshold calibration from reference bootstraps."""
    rng = np.random.RandomState(42)
    ref_data = rng.normal(loc=0.0, scale=1.0, size=(200, 3))

    detector = StatisticalDriftDetector(DriftDetectorConfig(threshold=0.20))
    calibrated_th = detector.calibrate_threshold_from_reference(
        ref_data, window_size=20, quantile=0.95, num_bootstrap=50
    )
    assert calibrated_th >= 0.05
    assert detector.threshold == calibrated_th
