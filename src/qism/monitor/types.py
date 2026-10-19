"""Type definitions, enumerations, and data classes for QISM V0.9.

Defines the conceptual hierarchy:
    TelemetryRecord -> DetectorSignal -> AlarmRecord -> Incident -> Alert
along with AuditEvent, ResponseRecommendation, and Orchestration interfaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any


class IncidentState(str, Enum):
    """Lifecycle state machine for correlated security incidents."""
    OPENING = "opening"
    ACTIVE = "active"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    CLOSED = "closed"


class SeverityLevel(str, Enum):
    """Deterministic, explainable security severity prioritization."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        ranks = {
            SeverityLevel.INFO: 0,
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4,
        }
        return ranks[self]


class AuditEventType(str, Enum):
    """Machine-readable audit trail event categories."""
    TELEMETRY_RECEIVED = "telemetry_received"
    ANOMALY_DETECTED = "anomaly_detected"
    DRIFT_DETECTED = "drift_detected"
    ALARM_OPENED = "alarm_opened"
    ALARM_ESCALATED = "alarm_escalated"
    INCIDENT_OPENED = "incident_opened"
    INCIDENT_UPDATED = "incident_updated"
    INCIDENT_RESOLVED = "incident_resolved"
    ADAPTATION_STARTED = "adaptation_started"
    ADAPTATION_ACCEPTED = "adaptation_accepted"
    ADAPTATION_REJECTED = "adaptation_rejected"
    MODEL_PROMOTED = "model_promoted"
    MODEL_ROLLED_BACK = "model_rolled_back"


class FusionMethod(str, Enum):
    """Signal fusion strategy."""
    WEIGHTED = "weighted"
    RULE_BASED = "rule_based"
    PERSISTENCE_AWARE = "persistence_aware"


@dataclass
class TelemetryRecord:
    """Raw/processed physical observation from the QKD system."""
    timestamp_s: float
    sequence_id: int
    subsystem: str
    features: dict[str, float]
    run_id: str = "default_run"
    session_id: str = "default_session"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TelemetryRecord:
        return cls(
            timestamp_s=float(data["timestamp_s"]),
            sequence_id=int(data.get("sequence_id", 0)),
            subsystem=str(data.get("subsystem", "composite")),
            features={str(k): float(v) for k, v in data.get("features", {}).items()},
            run_id=str(data.get("run_id", "default_run")),
            session_id=str(data.get("session_id", "default_session")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class DetectorSignal:
    """Normalized evidence produced by a detector or feature deviation."""
    signal_name: str
    subsystem: str
    raw_value: float
    normalized_value: float  # In [0.0, 1.0]
    threshold: float
    is_anomalous: bool
    timestamp_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlarmRecord:
    """A threshold/persistence-based detector state."""
    alarm_id: str
    source: str
    timestamp_s: float
    severity: SeverityLevel
    is_active: bool
    persistence_count: int
    message: str
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["severity"] = self.severity.value
        return res


@dataclass
class ResponseRecommendation:
    """Advisory, non-executing operator response suggestion."""
    advisory_text: str
    rationale: str
    severity: SeverityLevel
    target_subsystem: str
    suggested_actions: list[str] = field(default_factory=list)
    is_automated_execution: bool = False  # Strictly False per V0.9 specification

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["severity"] = self.severity.value
        return res


@dataclass
class Incident:
    """Correlated collection of alarms/signals representing one security-relevant episode."""
    incident_id: str
    start_time_s: float
    end_time_s: float | None
    duration_s: float
    state: IncidentState
    severity: SeverityLevel
    peak_severity: SeverityLevel
    evidence_score: float  # In [0.0, 1.0]
    peak_evidence_score: float
    constituent_alarms: list[AlarmRecord]
    contributing_signals: list[DetectorSignal]
    affected_features: list[str]
    subsystems_involved: list[str]
    explanation: dict[str, Any]
    recommendation: ResponseRecommendation | None
    active_model_version: str
    active_reference_version: str
    alarm_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.duration_s,
            "state": self.state.value,
            "severity": self.severity.value,
            "peak_severity": self.peak_severity.value,
            "evidence_score": float(self.evidence_score),
            "peak_evidence_score": float(self.peak_evidence_score),
            "constituent_alarms": [a.to_dict() for a in self.constituent_alarms],
            "contributing_signals": [s.to_dict() for s in self.contributing_signals],
            "affected_features": list(self.affected_features),
            "subsystems_involved": list(self.subsystems_involved),
            "explanation": dict(self.explanation),
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
            "active_model_version": self.active_model_version,
            "active_reference_version": self.active_reference_version,
            "alarm_count": self.alarm_count,
        }


@dataclass
class Alert:
    """An operationally structured notification describing an active incident."""
    alert_id: str
    incident_id: str
    timestamp_s: float
    sequence_id: int
    severity: SeverityLevel
    incident_state: IncidentState
    evidence_score: float
    anomaly_score: float
    drift_score: float
    affected_features: list[str]
    subsystems_involved: list[str]
    duration_s: float
    model_version: str
    reference_version: str
    explanation_summary: str
    explanation_details: dict[str, Any]
    recommendation_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "incident_id": self.incident_id,
            "timestamp_s": self.timestamp_s,
            "sequence_id": self.sequence_id,
            "severity": self.severity.value,
            "incident_state": self.incident_state.value,
            "evidence_score": float(self.evidence_score),
            "anomaly_score": float(self.anomaly_score),
            "drift_score": float(self.drift_score),
            "affected_features": list(self.affected_features),
            "subsystems_involved": list(self.subsystems_involved),
            "duration_s": self.duration_s,
            "model_version": self.model_version,
            "reference_version": self.reference_version,
            "explanation_summary": self.explanation_summary,
            "explanation_details": dict(self.explanation_details),
            "recommendation_summary": self.recommendation_summary,
        }


@dataclass
class AuditEvent:
    """Structured, immutable audit record for research and regulatory reproducibility."""
    event_id: str
    timestamp_s: float
    event_type: AuditEventType
    component: str
    run_id: str
    session_id: str
    incident_id: str | None
    alert_id: str | None
    severity: SeverityLevel | None
    anomaly_score: float | None
    drift_score: float | None
    evidence_score: float | None
    model_version: str
    reference_version: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["event_type"] = self.event_type.value
        res["severity"] = self.severity.value if self.severity else None
        return res

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class MonitorStepResult:
    """Per-step result produced by QISMMonitor.update()."""
    step_index: int
    timestamp_s: float
    telemetry_valid: bool
    anomaly_score: float
    anomaly_threshold: float
    is_anomaly: bool
    drift_score: float
    drift_threshold: float
    is_drift: bool
    evidence_score: float
    active_alarms: list[AlarmRecord]
    active_incidents: list[Incident]
    new_alert: Alert | None
    current_severity: SeverityLevel
    adaptation_state: str
    model_version: str
    reference_version: str
    processing_latency_ms: float
    signals: list[DetectorSignal] = field(default_factory=list)


@dataclass
class MonitorRunResult:
    """Result of full batch or sequential run."""
    total_steps: int
    total_duration_s: float
    step_results: list[MonitorStepResult]
    all_incidents: list[Incident]
    all_alerts: list[Alert]
    audit_events: list[AuditEvent]
    metrics: dict[str, Any] = field(default_factory=dict)
