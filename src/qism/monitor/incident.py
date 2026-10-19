"""Incident lifecycle management, correlation, and alert deduplication for QISM V0.9.

Transforms individual point alarms and signals into cohesive, correlated incidents:
- Correlates alarms occurring within temporal proximity
- Manages state machine: OPENING -> ACTIVE -> ESCALATED -> RESOLVED
- Deduplicates repetitive alarms to eliminate operator alert fatigue
- Calculates operational alert compression ratio
"""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid

from qism.monitor.severity import SeverityEngine
from qism.monitor.types import (
    AlarmRecord,
    Alert,
    DetectorSignal,
    Incident,
    IncidentState,
    ResponseRecommendation,
    SeverityLevel,
)


@dataclass
class IncidentConfig:
    """Configuration for temporal correlation, lifecycle, and resolution."""
    correlation_window_s: float = 0.050  # 50 ms temporal correlation horizon
    max_gap_s: float = 0.030             # Max time gap before closing/opening new incident
    min_evidence_to_open: float = 0.45   # Threshold on unified evidence score to open incident
    resolution_patience_steps: int = 5   # Consecutive benign steps to declare RESOLVED
    escalation_threshold: float = 0.75   # Evidence score threshold to ESCALATE
    emit_alert_on_state_change_only: bool = True  # Deduplication policy


class IncidentManager:
    """Manages active and resolved security incidents across telemetry streams."""

    def __init__(
        self,
        config: IncidentConfig | None = None,
        severity_engine: SeverityEngine | None = None,
    ):
        self.config = config or IncidentConfig()
        self.severity_engine = severity_engine or SeverityEngine()

        self.active_incident: Incident | None = None
        self.resolved_incidents: list[Incident] = []
        self.all_emitted_alerts: list[Alert] = []

        self._benign_counter: int = 0
        self._persistence_counter: int = 0
        self._incident_counter: int = 0
        self._alert_counter: int = 0
        self.total_raw_alarms: int = 0

    def process_step(
        self,
        timestamp_s: float,
        evidence_score: float,
        signals: list[DetectorSignal],
        alarms: list[AlarmRecord],
        affected_features: list[str],
        active_model_version: str,
        active_reference_version: str,
        sequence_id: int = 0,
        drift_active: bool = False,
    ) -> tuple[Incident | None, Alert | None]:
        """Update incident state for a single temporal step.

        Returns:
            current_incident: The active or newly resolved incident (if any)
            new_alert: A newly emitted Alert object, or None if deduplicated
        """
        cfg = self.config
        self.total_raw_alarms += len(alarms)

        has_active_alarm = any(a.is_active for a in alarms)
        evidence_exceeds = evidence_score >= cfg.min_evidence_to_open

        new_alert: Alert | None = None

        if has_active_alarm or evidence_exceeds:
            # Active anomaly or alarm condition
            self._benign_counter = 0
            self._persistence_counter += 1

            if self.active_incident is None:
                # 1. OPEN NEW INCIDENT
                self._incident_counter += 1
                inc_id = f"INC-{self._incident_counter:04d}"

                # Evaluate severity and rationale
                sev, expl, rec = self.severity_engine.evaluate(
                    evidence_score=evidence_score,
                    signals=signals,
                    persistence_steps=self._persistence_counter,
                    affected_features=affected_features,
                    incident_state=IncidentState.OPENING,
                    drift_active=drift_active,
                )

                subsystems = sorted(list({s.subsystem for s in signals if s.is_anomalous}))

                self.active_incident = Incident(
                    incident_id=inc_id,
                    start_time_s=timestamp_s,
                    end_time_s=None,
                    duration_s=0.0,
                    state=IncidentState.ACTIVE if self._persistence_counter > 1 else IncidentState.OPENING,
                    severity=sev,
                    peak_severity=sev,
                    evidence_score=evidence_score,
                    peak_evidence_score=evidence_score,
                    constituent_alarms=list(alarms),
                    contributing_signals=list(signals),
                    affected_features=list(affected_features),
                    subsystems_involved=subsystems,
                    explanation=expl,
                    recommendation=rec,
                    active_model_version=active_model_version,
                    active_reference_version=active_reference_version,
                    alarm_count=len(alarms),
                )

                # Emit Alert on opening
                new_alert = self._create_alert(self.active_incident, timestamp_s, sequence_id)
                self.all_emitted_alerts.append(new_alert)

            else:
                # 2. UPDATE ONGOING INCIDENT (CORRELATION & MERGING)
                inc = self.active_incident
                inc.duration_s = max(0.0, timestamp_s - inc.start_time_s)
                inc.evidence_score = evidence_score
                inc.alarm_count += len(alarms)
                inc.constituent_alarms.extend(alarms)

                # Merge affected features and subsystems
                for f in affected_features:
                    if f not in inc.affected_features:
                        inc.affected_features.append(f)
                for s in signals:
                    if s.is_anomalous and s.subsystem not in inc.subsystems_involved:
                        inc.subsystems_involved.append(s.subsystem)

                if evidence_score > inc.peak_evidence_score:
                    inc.peak_evidence_score = evidence_score

                prev_sev = inc.severity
                prev_state = inc.state

                # Check for state transition
                if evidence_score >= cfg.escalation_threshold:
                    inc.state = IncidentState.ESCALATED
                else:
                    inc.state = IncidentState.ACTIVE

                # Re-evaluate severity
                sev, expl, rec = self.severity_engine.evaluate(
                    evidence_score=evidence_score,
                    signals=signals,
                    persistence_steps=self._persistence_counter,
                    affected_features=inc.affected_features,
                    incident_state=inc.state,
                    drift_active=drift_active,
                )
                inc.severity = sev
                inc.explanation = expl
                inc.recommendation = rec

                if sev.rank > inc.peak_severity.rank:
                    inc.peak_severity = sev

                # Alert Deduplication: Emit alert only if escalated or severity increased
                severity_increased = sev.rank > prev_sev.rank
                state_changed = inc.state != prev_state

                if severity_increased or state_changed:
                    new_alert = self._create_alert(inc, timestamp_s, sequence_id)
                    self.all_emitted_alerts.append(new_alert)

        else:
            # Benign step
            self._persistence_counter = max(0, self._persistence_counter - 1)
            if self.active_incident is not None:
                self._benign_counter += 1
                self.active_incident.duration_s = max(0.0, timestamp_s - self.active_incident.start_time_s)

                if self._benign_counter >= cfg.resolution_patience_steps:
                    # 3. RESOLVE ACTIVE INCIDENT
                    inc = self.active_incident
                    inc.state = IncidentState.RESOLVED
                    inc.end_time_s = timestamp_s
                    inc.duration_s = max(0.0, timestamp_s - inc.start_time_s)

                    # Update explanation for resolution
                    inc.explanation["resolution_timestamp_s"] = timestamp_s
                    inc.explanation["resolution_duration_s"] = inc.duration_s
                    inc.explanation["reasons"].append(
                        f"Incident resolved after {self._benign_counter} consecutive benign monitoring windows."
                    )

                    # Emit final resolution alert
                    new_alert = self._create_alert(inc, timestamp_s, sequence_id)
                    self.all_emitted_alerts.append(new_alert)

                    self.resolved_incidents.append(inc)
                    self.active_incident = None
                    self._benign_counter = 0

        return self.active_incident, new_alert

    def _create_alert(self, inc: Incident, timestamp_s: float, sequence_id: int) -> Alert:
        """Create a structured Alert object from the incident."""
        self._alert_counter += 1
        alt_id = f"ALT-{self._alert_counter:04d}"

        anom_score = 0.0
        drift_score = 0.0
        for s in inc.contributing_signals:
            if s.signal_name == "temporal_anomaly_model":
                anom_score = s.raw_value
            elif s.signal_name == "statistical_drift":
                drift_score = s.raw_value

        summary = (
            f"[{inc.severity.value.upper()}] Incident {inc.incident_id} ({inc.state.value}): "
            f"Evidence={inc.evidence_score:.2f}, Subsystems={','.join(inc.subsystems_involved)}"
        )
        rec_summary = inc.recommendation.advisory_text if inc.recommendation else "Continue normal monitoring."

        return Alert(
            alert_id=alt_id,
            incident_id=inc.incident_id,
            timestamp_s=timestamp_s,
            sequence_id=sequence_id,
            severity=inc.severity,
            incident_state=inc.state,
            evidence_score=inc.evidence_score,
            anomaly_score=anom_score,
            drift_score=drift_score,
            affected_features=list(inc.affected_features),
            subsystems_involved=list(inc.subsystems_involved),
            duration_s=inc.duration_s,
            model_version=inc.active_model_version,
            reference_version=inc.active_reference_version,
            explanation_summary=summary,
            explanation_details=dict(inc.explanation),
            recommendation_summary=rec_summary,
        )

    def finalize(self, final_timestamp_s: float | None = None) -> list[Incident]:
        """Close any open incident at the conclusion of a monitoring run."""
        if self.active_incident is not None:
            inc = self.active_incident
            inc.state = IncidentState.RESOLVED
            ts = final_timestamp_s if final_timestamp_s is not None else inc.start_time_s + inc.duration_s
            inc.end_time_s = ts
            inc.duration_s = max(0.0, ts - inc.start_time_s)
            self.resolved_incidents.append(inc)
            self.active_incident = None
        return list(self.resolved_incidents)

    @property
    def compression_ratio(self) -> float:
        """Operational Alert Compression Ratio: total raw alarms / total incidents."""
        n_inc = len(self.resolved_incidents) + (1 if self.active_incident else 0)
        if n_inc == 0:
            return 1.0
        return float(self.total_raw_alarms / n_inc)

    def reset(self) -> None:
        """Reset internal incident state."""
        self.active_incident = None
        self.resolved_incidents.clear()
        self.all_emitted_alerts.clear()
        self._benign_counter = 0
        self._persistence_counter = 0
        self._incident_counter = 0
        self._alert_counter = 0
        self.total_raw_alarms = 0
