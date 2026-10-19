"""Tests for QISM V0.9 incident lifecycle management, correlation, and deduplication."""

import pytest
from qism.monitor.incident import IncidentManager, IncidentConfig
from qism.monitor.types import (
    AlarmRecord,
    DetectorSignal,
    IncidentState,
    SeverityLevel,
)


def _make_dummy_signal(name: str, subsystem: str, norm_val: float) -> DetectorSignal:
    return DetectorSignal(
        signal_name=name,
        subsystem=subsystem,
        raw_value=norm_val * 2.0,
        normalized_value=norm_val,
        threshold=1.0,
        is_anomalous=norm_val >= 0.50,
        timestamp_s=0.0,
    )


def test_incident_opening_and_alert_generation():
    mgr = IncidentManager(IncidentConfig(min_evidence_to_open=0.50))
    signals = [_make_dummy_signal("anom", "ml", 0.80)]
    alarm = AlarmRecord(
        alarm_id="ALM-1",
        source="detector",
        timestamp_s=0.010,
        severity=SeverityLevel.HIGH,
        is_active=True,
        persistence_count=1,
        message="Active alarm",
    )

    inc, alert = mgr.process_step(
        timestamp_s=0.010,
        evidence_score=0.80,
        signals=signals,
        alarms=[alarm],
        affected_features=["qber"],
        active_model_version="v1",
        active_reference_version="ref1",
    )

    assert inc is not None
    assert inc.incident_id.startswith("INC-")
    assert alert is not None
    assert alert.incident_id == inc.incident_id
    assert mgr.active_incident is not None


def test_alert_deduplication_during_ongoing_incident():
    """Verify that multiple continuing anomalous steps do NOT emit spam alerts."""
    mgr = IncidentManager(IncidentConfig(min_evidence_to_open=0.50))
    signals = [_make_dummy_signal("anom", "ml", 0.70)]

    # Step 1: Open incident -> emits Alert 1
    inc, alert1 = mgr.process_step(
        timestamp_s=0.010,
        evidence_score=0.70,
        signals=signals,
        alarms=[],
        affected_features=["qber"],
        active_model_version="v1",
        active_reference_version="ref1",
    )
    assert alert1 is not None

    # Step 2: Persistence confirms incident -> transitions from OPENING to ACTIVE
    inc, alert2 = mgr.process_step(
        timestamp_s=0.012,
        evidence_score=0.70,
        signals=signals,
        alarms=[],
        affected_features=["qber"],
        active_model_version="v1",
        active_reference_version="ref1",
    )
    assert inc.state == IncidentState.ACTIVE
    assert inc.duration_s == pytest.approx(0.002)

    # Step 3: Continued anomaly in ACTIVE state at same severity -> Deduplicated (alert is None)
    inc, alert3 = mgr.process_step(
        timestamp_s=0.014,
        evidence_score=0.70,
        signals=signals,
        alarms=[],
        affected_features=["qber"],
        active_model_version="v1",
        active_reference_version="ref1",
    )
    assert alert3 is None

    # Step 4: Continued anomaly in ACTIVE state at same severity -> Deduplicated (alert is None)
    inc, alert4 = mgr.process_step(
        timestamp_s=0.016,
        evidence_score=0.70,
        signals=signals,
        alarms=[],
        affected_features=["qber"],
        active_model_version="v1",
        active_reference_version="ref1",
    )
    assert alert4 is None


def test_incident_resolution_lifecycle():
    """Verify clean state transition to RESOLVED after benign patience elapsed."""
    mgr = IncidentManager(IncidentConfig(min_evidence_to_open=0.50, resolution_patience_steps=3))
    signals = [_make_dummy_signal("anom", "ml", 0.80)]

    # Step 1: Anomaly opens incident
    mgr.process_step(
        timestamp_s=0.010,
        evidence_score=0.80,
        signals=signals,
        alarms=[],
        affected_features=["qber"],
        active_model_version="v1",
        active_reference_version="ref1",
    )
    assert mgr.active_incident is not None

    # 3 consecutive benign steps
    for step in range(1, 4):
        t = 0.010 + step * 0.002
        mgr.process_step(
            timestamp_s=t,
            evidence_score=0.10,
            signals=[],
            alarms=[],
            affected_features=[],
            active_model_version="v1",
            active_reference_version="ref1",
        )

    # Incident should now be resolved and archived
    assert mgr.active_incident is None
    assert len(mgr.resolved_incidents) == 1
    assert mgr.resolved_incidents[0].state == IncidentState.RESOLVED
    assert mgr.resolved_incidents[0].end_time_s == pytest.approx(0.016)
