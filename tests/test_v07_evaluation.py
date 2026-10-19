"""Unit tests for V0.7 temporal evaluation framework."""

import numpy as np
import pandas as pd
import pytest

from qism.temporal.evaluation import TemporalEvaluator


def test_temporal_evaluator_metrics():
    """Verify sample and event metrics computation."""
    evaluator = TemporalEvaluator(repetition_rate_hz=1_000_000.0, default_pulses_per_window=2000)

    # 10 windows: 5 normal, 5 attack
    scores = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 2.5, 3.0, 2.8, 3.5, 3.2])
    raw_alarms = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    filtered_alarms = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # 1-step persistence delay
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    y_meta_df = pd.DataFrame({
        "scenario_id": ["scen_01"] * 10,
        "run_id": ["run_01"] * 10,
        "attack_type": ["none"] * 5 + ["intercept_resend"] * 5,
        "attack_severity": [0.0] * 5 + [1.0] * 5,
        "window_start_pulse": [i * 2000 for i in range(10)],
        "window_end_pulse": [(i + 1) * 2000 - 1 for i in range(10)],
    })

    summary = evaluator.evaluate(
        scores=scores,
        raw_alarms=raw_alarms,
        filtered_alarms=filtered_alarms,
        y_true_active=y_true,
        y_meta_df=y_meta_df,
    )

    # AUROC should be 1.0 (perfect separation)
    assert np.isclose(summary.curve_metrics["auroc"], 1.0)
    # Episode recall should be 1.0 (the attack episode was detected)
    assert summary.total_attack_episodes == 1
    assert summary.detected_attack_episodes == 1
    assert np.isclose(summary.episode_detection_rate, 1.0)
    # Detection delay: filtered alarm fired at index 6, attack began at index 5 -> delay = 1 window = 2000 pulses = 0.002s
    assert summary.detection_delay_summary["median_delay_pulses"] == 2000
    assert np.isclose(summary.detection_delay_summary["median_delay_s"], 0.002)
    # Zero false alarms in normal windows
    assert summary.false_alarm_episodes == 0
