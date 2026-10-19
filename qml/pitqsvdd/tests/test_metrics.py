"""Tests for evaluation metrics and threshold calibration."""

import numpy as np
import pytest

from qml.pitqsvdd.evaluation.event_metrics import compute_event_level_metrics
from qml.pitqsvdd.evaluation.metrics import compute_point_metrics
from qml.pitqsvdd.evaluation.thresholding import ThresholdCalibrator


def test_threshold_calibrator_and_point_metrics():
    """Verify threshold calibration on normal data and metric calculations."""
    calibrator = ThresholdCalibrator(percentile=90.0)
    normal_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    th = calibrator.fit(normal_scores)
    assert 0.8 < th <= 1.0

    # Test scoring
    test_scores = np.array([0.2, 0.5, 0.95, 1.5])
    preds = calibrator.apply(test_scores)
    y_true = np.array([0, 0, 1, 1])

    metrics = compute_point_metrics(y_true, preds, test_scores)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auroc"] == 1.0


def test_event_level_metrics():
    """Verify incident recall and delay calculation."""
    labels = [0, 0, 1, 1, 0, 1, 1]
    preds = [0, 0, 0, 1, 0, 1, 0]  # First incident detected with delay=1, second with delay=0
    meta = [{"scenario_id": "scen_1"} for _ in range(len(labels))]

    res = compute_event_level_metrics(preds, labels, meta)
    assert res["total_attack_incidents"] == 2
    assert res["detected_attack_incidents"] == 2
    assert res["incident_recall"] == 1.0
    assert res["mean_detection_delay_windows"] == 0.5
