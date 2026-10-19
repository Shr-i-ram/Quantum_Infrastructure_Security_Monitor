"""Unit tests for V0.7 temporal alarm state machine, persistence, and hysteresis."""

import numpy as np
import pytest

from qism.temporal.alarm import AlarmConfig, AlarmState, TemporalAlarmStateMachine


def test_alarm_persistence_k_of_n():
    """Verify K-of-N persistence suppresses isolated noise spikes."""
    cfg = AlarmConfig(threshold=10.0, k_consecutive=2, n_window=3, cooldown_steps=0)
    sm = TemporalAlarmStateMachine(cfg)

    # 1. Single spike: normal -> suspect, but not alarm
    out1 = sm.update(score=12.0)
    assert out1.raw_alarm == 1
    assert out1.filtered_alarm == 0
    assert out1.state == AlarmState.SUSPECT

    # Return to normal
    out2 = sm.update(score=5.0)
    assert out2.raw_alarm == 0
    assert out2.filtered_alarm == 0
    assert out2.state == AlarmState.NORMAL

    # 2. Two consecutive spikes: triggers ALARM
    out3 = sm.update(score=12.0)
    assert out3.filtered_alarm == 0
    assert out3.state == AlarmState.SUSPECT

    out4 = sm.update(score=14.0)
    assert out4.filtered_alarm == 1
    assert out4.state == AlarmState.ALARM


def test_alarm_hysteresis_and_cooldown():
    """Verify dual threshold hysteresis and cooldown prevent alarm chattering."""
    cfg = AlarmConfig(
        threshold=10.0,
        recovery_threshold=5.0,
        k_consecutive=1,
        n_window=1,
        cooldown_steps=2,
    )
    sm = TemporalAlarmStateMachine(cfg)

    # Trigger alarm
    out = sm.update(12.0)
    assert out.state == AlarmState.ALARM
    assert out.filtered_alarm == 1

    # Score drops below trigger (10.0) but above recovery (5.0) -> still ALARM
    out = sm.update(7.0)
    assert out.state == AlarmState.ALARM
    assert out.filtered_alarm == 1

    # Score drops below recovery (5.0) -> transitions to COOLDOWN
    out = sm.update(4.0)
    assert out.state == AlarmState.COOLDOWN
    assert out.filtered_alarm == 0

    # In cooldown for 2 steps (counter = 2 -> 1 -> 0)
    out = sm.update(12.0)  # Suppressed by cooldown step 1
    assert out.state == AlarmState.COOLDOWN
    assert out.filtered_alarm == 0

    out = sm.update(4.0)   # Suppressed by cooldown step 2
    assert out.state == AlarmState.COOLDOWN
    assert out.filtered_alarm == 0

    # Cooldown complete -> back to NORMAL
    out = sm.update(4.0)
    assert out.state == AlarmState.NORMAL

