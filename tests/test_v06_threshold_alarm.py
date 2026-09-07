"""Unit tests for threshold calibration and operational alarm state machines (Requirements 48–52, 167–170, 251)."""

import numpy as np
import pytest

from qism.ml.thresholding import AlarmState, AlarmStateMachine, ThresholdCalibrator


def test_threshold_calibrator_quantiles():
    """Threshold calibrator sets threshold to match the requested false positive rate."""
    scores = np.linspace(0.0, 100.0, 1001)  # 0, 0.1, 0.2, ..., 100.0
    cal = ThresholdCalibrator(target_fpr=0.05)
    res = cal.calibrate(scores)

    # 95th percentile of [0, 100] is 95.0
    assert np.isclose(res["threshold"], 95.0, atol=0.2)
    assert np.isclose(res["empirical_val_fpr"], 0.05, atol=0.01)


def test_alarm_state_machine_persistence_and_hysteresis():
    """AlarmStateMachine enforces K-consecutive persistence and hysteresis recovery."""
    tau_high = 10.0
    tau_low = 5.0
    sm = AlarmStateMachine(tau_high=tau_high, tau_low=tau_low, consecutive_k=3)

    # Window 0: normal
    st0, alm0 = sm.step(2.0)
    assert st0 == AlarmState.NORMAL and not alm0

    # Window 1: spike 1 -> SUSPECT (1/3)
    st1, alm1 = sm.step(12.0)
    assert st1 == AlarmState.SUSPECT and not alm1

    # Window 2: spike 2 -> SUSPECT (2/3)
    st2, alm2 = sm.step(14.0)
    assert st2 == AlarmState.SUSPECT and not alm2

    # Window 3: spike 3 -> ALARM (3/3 reached!)
    st3, alm3 = sm.step(13.0)
    assert st3 == AlarmState.ALARM and alm3

    # Window 4: score drops to 7.0 (between low and high) -> remains in ALARM due to hysteresis
    st4, alm4 = sm.step(7.0)
    assert st4 == AlarmState.ALARM and alm4

    # Window 5: score drops below tau_low (4.0) -> recovers to NORMAL
    st5, alm5 = sm.step(4.0)
    assert st5 == AlarmState.NORMAL and not alm5
