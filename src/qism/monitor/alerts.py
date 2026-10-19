"""Structured alerts, notification backends, and immutable audit logging for QISM V0.9.

Provides:
1. Pluggable notification backends (ConsoleNotifier, FileNotifier, JSONNotifier) with safe dry-run defaults
2. Append-only, machine-readable JSONL audit trail tracking all security-relevant decisions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from qism.monitor.types import (
    Alert,
    AuditEvent,
    AuditEventType,
    Incident,
    SeverityLevel,
)


class NotificationBackend(ABC):
    """Abstract interface for dispatching operational security alerts."""

    @abstractmethod
    def dispatch(self, alert: Alert) -> bool:
        """Deliver an alert through the configured channel. Returns True on success."""
        pass


class ConsoleNotifier(NotificationBackend):
    """Prints formatted operational alerts to the console."""

    def __init__(self, prefix: str = "[QISM-ALERT]"):
        self.prefix = prefix
        self.dispatched_count = 0

    def dispatch(self, alert: Alert) -> bool:
        sev = alert.severity.value.upper()
        print(
            f"{self.prefix} {sev} | Inc: {alert.incident_id} | State: {alert.incident_state.value} | "
            f"Evidence: {alert.evidence_score:.2f} | Subsystems: {','.join(alert.subsystems_involved)}"
        )
        self.dispatched_count += 1
        return True


class FileNotifier(NotificationBackend):
    """Appends human-readable alerts to a log file."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.dispatched_count = 0

    def dispatch(self, alert: Alert) -> bool:
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(
                f"[{alert.timestamp_s:.3f}s] [{alert.severity.value.upper()}] "
                f"AlertID={alert.alert_id} IncidentID={alert.incident_id} State={alert.incident_state.value} "
                f"Evidence={alert.evidence_score:.3f} Model={alert.model_version}\n"
                f"  Summary: {alert.explanation_summary}\n"
                f"  Recommendation: {alert.recommendation_summary}\n\n"
            )
        self.dispatched_count += 1
        return True


class JSONNotifier(NotificationBackend):
    """Appends structured JSON objects to a JSONL file."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.dispatched_count = 0

    def dispatch(self, alert: Alert) -> bool:
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")
        self.dispatched_count += 1
        return True


class AuditLogger:
    """Maintains an append-only, reproducible JSONL event log of all system transitions."""

    def __init__(self, log_filepath: str | Path | None = None):
        self.log_filepath = Path(log_filepath) if log_filepath else None
        if self.log_filepath:
            self.log_filepath.parent.mkdir(parents=True, exist_ok=True)
            # Create/truncate file
            with open(self.log_filepath, "w", encoding="utf-8") as f:
                pass
        self.events: list[AuditEvent] = []
        self._event_counter = 0

    def log_event(
        self,
        event_type: AuditEventType,
        component: str,
        timestamp_s: float,
        run_id: str = "default_run",
        session_id: str = "default_session",
        incident_id: str | None = None,
        alert_id: str | None = None,
        severity: SeverityLevel | None = None,
        anomaly_score: float | None = None,
        drift_score: float | None = None,
        evidence_score: float | None = None,
        model_version: str = "v1.0.0",
        reference_version: str = "ref_v1.0.0",
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Record a structured event into the audit trail."""
        self._event_counter += 1
        evt_id = f"EVT-{self._event_counter:06d}"

        evt = AuditEvent(
            event_id=evt_id,
            timestamp_s=timestamp_s,
            event_type=event_type,
            component=component,
            run_id=run_id,
            session_id=session_id,
            incident_id=incident_id,
            alert_id=alert_id,
            severity=severity,
            anomaly_score=anomaly_score,
            drift_score=drift_score,
            evidence_score=evidence_score,
            model_version=model_version,
            reference_version=reference_version,
            message=message,
            details=details or {},
        )
        self.events.append(evt)

        if self.log_filepath:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(evt.to_json() + "\n")

        return evt

    def get_events_for_incident(self, incident_id: str) -> list[AuditEvent]:
        """Retrieve all audit events correlated with a specific incident ID."""
        return [e for e in self.events if e.incident_id == incident_id]

    def export_json(self, output_path: str | Path) -> None:
        """Export full audit log to a JSON array file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.events], f, indent=2)

    def reset(self) -> None:
        """Clear in-memory events."""
        self.events.clear()
        self._event_counter = 0
