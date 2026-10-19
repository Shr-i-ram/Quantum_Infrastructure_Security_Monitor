"""Tests for structured alert dispatching and reproducible JSONL audit logging."""

import json
from pathlib import Path
import pytest
from qism.monitor.alerts import (
    AuditLogger,
    ConsoleNotifier,
    FileNotifier,
    JSONNotifier,
)
from qism.monitor.types import (
    Alert,
    AuditEventType,
    IncidentState,
    SeverityLevel,
)


def _make_dummy_alert(alt_id: str = "ALT-0001", inc_id: str = "INC-0001") -> Alert:
    return Alert(
        alert_id=alt_id,
        incident_id=inc_id,
        timestamp_s=0.010,
        sequence_id=5,
        severity=SeverityLevel.HIGH,
        incident_state=IncidentState.ACTIVE,
        evidence_score=0.82,
        anomaly_score=3.4,
        drift_score=0.1,
        affected_features=["qber"],
        subsystems_involved=["quantum_channel"],
        duration_s=0.004,
        model_version="model_v001",
        reference_version="ref_v001",
        explanation_summary="High severity QBER deviation detected",
        explanation_details={"reasons": ["QBER deviated by 4.2x"]},
        recommendation_summary="Inspect optical channel",
    )


def test_notification_backends(tmp_path: Path):
    alt = _make_dummy_alert()

    # 1. Console Notifier
    con = ConsoleNotifier()
    assert con.dispatch(alt) is True
    assert con.dispatched_count == 1

    # 2. File Notifier
    log_file = tmp_path / "alerts.log"
    fil = FileNotifier(log_file)
    assert fil.dispatch(alt) is True
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "ALT-0001" in content

    # 3. JSON Notifier
    json_file = tmp_path / "alerts.jsonl"
    jsn = JSONNotifier(json_file)
    assert jsn.dispatch(alt) is True
    assert json_file.exists()
    lines = json_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["alert_id"] == "ALT-0001"


def test_audit_logger_serialization(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(audit_path)

    logger.log_event(
        event_type=AuditEventType.TELEMETRY_RECEIVED,
        component="ingestion",
        timestamp_s=0.002,
        message="Received valid telemetry",
    )
    logger.log_event(
        event_type=AuditEventType.ANOMALY_DETECTED,
        component="ml_engine",
        timestamp_s=0.004,
        incident_id="INC-0001",
        severity=SeverityLevel.HIGH,
        anomaly_score=3.5,
        evidence_score=0.85,
        message="Anomaly detected",
    )

    assert len(logger.events) == 2
    assert audit_path.exists()
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    # Query by incident
    inc_events = logger.get_events_for_incident("INC-0001")
    assert len(inc_events) == 1
    assert inc_events[0].event_type == AuditEventType.ANOMALY_DETECTED
