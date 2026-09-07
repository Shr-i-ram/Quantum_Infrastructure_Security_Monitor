"""Unit tests for classification metrics, AUROC/AUPRC, and attack-instance detection delay (Requirements 65–81, 252–254)."""

import numpy as np
import pandas as pd
import pytest

from qism.ml.evaluation.instance import AttackInstanceEvaluator
from qism.ml.evaluation.metrics import calculate_classification_metrics, calculate_continuous_curve_metrics


def test_hand_calculated_classification_metrics():
    """Classification metrics match hand-calculated ground truth values."""
    # TP=8, FP=2, FN=2, TN=88
    y_true = np.array([1]*8 + [0]*2 + [1]*2 + [0]*88)
    y_pred = np.array([1]*8 + [1]*2 + [0]*2 + [0]*88)

    m = calculate_classification_metrics(y_true, y_pred)
    assert np.isclose(m["precision"], 8.0 / 10.0)
    assert np.isclose(m["recall"], 8.0 / 10.0)
    assert np.isclose(m["f1"], 0.80)
    assert np.isclose(m["fpr"], 2.0 / 90.0)
    assert np.isclose(m["specificity"], 88.0 / 90.0)
    assert m["confusion_matrix"]["tp"] == 8
    assert m["confusion_matrix"]["fp"] == 2
    assert m["confusion_matrix"]["fn"] == 2
    assert m["confusion_matrix"]["tn"] == 88


def test_continuous_auroc_and_auprc():
    """AUROC and AUPRC correctly compute from continuous scores."""
    # Perfect separation
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0])

    curve = calculate_continuous_curve_metrics(y_true, scores)
    assert np.isclose(curve["auroc"], 1.0)
    assert np.isclose(curve["auprc"], 1.0)


def test_attack_instance_detection_delay_calculation():
    """AttackInstanceEvaluator calculates detection delay on known synthetic attack intervals."""
    # Simulation of 10 windows, 1000 pulses each (repetition rate 1 MHz -> 1 ms per window)
    # Windows 0-2: Normal
    # Windows 3-6: Attack (pulses 3000 to 6999)
    # Windows 7-9: Normal
    # Alarm raised at Window 4 (pulse 4000) -> Delay should be 1000 pulses = 0.001 s
    df = pd.DataFrame({
        "scenario_id": ["scen_01"] * 10,
        "run_id": ["run_01"] * 10,
        "window_id": list(range(10)),
        "window_start_pulse": [i * 1000 for i in range(10)],
        "attack_active": [False, False, False, True, True, True, True, False, False, False],
        "alarm":         [0,     0,     0,     0,    1,    1,    0,    0,     0,     0],
        "attack_type":   ["none"]*3 + ["intercept_resend"]*4 + ["none"]*3,
        "attack_severity": [0.0]*3 + [0.75]*4 + [0.0]*3,
    })

    evaluator = AttackInstanceEvaluator(repetition_rate_hz=1_000_000.0)
    res = evaluator.evaluate_instances(df, alarm_col="alarm")

    assert res["total_attack_episodes"] == 1
    assert res["detected_attack_episodes"] == 1
    assert res["missed_attack_episodes"] == 0
    assert np.isclose(res["episode_detection_rate"], 1.0)

    # Delay: pulse 4000 - pulse 3000 = 1000 pulses = 0.001 seconds
    assert np.isclose(res["detection_delay_summary"]["median_delay_s"], 0.001)
