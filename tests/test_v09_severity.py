"""Tests for QISM V0.9 deterministic severity assessment and explainability."""

import pytest
from qism.monitor.severity import SeverityEngine, SeverityThresholdConfig
from qism.monitor.types import DetectorSignal, IncidentState, SeverityLevel


def _make_signal(name: str, subsystem: str, norm_val: float) -> DetectorSignal:
    return DetectorSignal(
        signal_name=name,
        subsystem=subsystem,
        raw_value=norm_val * 2.0,
        normalized_value=norm_val,
        threshold=1.0,
        is_anomalous=norm_val >= 0.50,
        timestamp_s=0.0,
    )


def test_severity_levels_and_explanations():
    engine = SeverityEngine()

    # 1. Low evidence -> INFO
    sev_info, expl_info, rec_info = engine.evaluate(
        evidence_score=0.15,
        signals=[],
        persistence_steps=1,
        affected_features=[],
    )
    assert sev_info == SeverityLevel.INFO
    assert len(expl_info["reasons"]) > 0

    # 2. Medium evidence -> MEDIUM
    signals_med = [_make_signal("qber_dev", "quantum_channel", 0.60)]
    sev_med, expl_med, rec_med = engine.evaluate(
        evidence_score=0.60,
        signals=signals_med,
        persistence_steps=2,
        affected_features=["qber"],
    )
    assert sev_med == SeverityLevel.MEDIUM
    assert "quantum_channel" in expl_med["subsystems_affected"]

    # 3. High evidence + persistence -> HIGH
    signals_high = [
        _make_signal("qber_dev", "quantum_channel", 0.85),
        _make_signal("gain_dev", "optical_receiver", 0.75),
    ]
    sev_high, expl_high, rec_high = engine.evaluate(
        evidence_score=0.80,
        signals=signals_high,
        persistence_steps=4,
        affected_features=["qber", "signal_gain"],
    )
    assert sev_high == SeverityLevel.HIGH

    # 4. Critical evidence across multiple subsystems -> CRITICAL
    signals_crit = [
        _make_signal("qber_dev", "quantum_channel", 0.95),
        _make_signal("timing_dev", "synchronization", 0.95),
    ]
    sev_crit, expl_crit, rec_crit = engine.evaluate(
        evidence_score=0.95,
        signals=signals_crit,
        persistence_steps=5,
        affected_features=["qber", "clock_drift_rate"],
    )
    assert sev_crit == SeverityLevel.CRITICAL
    assert "critical" in expl_crit["reasons"][0].lower()


def test_advisory_recommendation_non_executing():
    """Verify that response recommendations are strictly non-executing advisory guidance."""
    engine = SeverityEngine()
    signals = [_make_signal("qber_dev", "quantum_channel", 0.90)]
    _, _, rec = engine.evaluate(
        evidence_score=0.90,
        signals=signals,
        persistence_steps=3,
        affected_features=["qber"],
    )
    # Crucial security guarantee: automated execution MUST be False
    assert rec.is_automated_execution is False
    assert len(rec.suggested_actions) > 0
    assert rec.advisory_text != ""
