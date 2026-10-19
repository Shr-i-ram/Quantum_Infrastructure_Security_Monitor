"""Incident management service for QISM Live Demonstration Console.

Translates detector outputs into actionable, persistent security incidents according
to calibrated threshold policies, consecutive-alarm rules, and cooldown grouping.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..config import settings
from ..database import get_incidents, save_incident, update_incident_status

log = logging.getLogger("qism.demo.incident")


class IncidentService:
    """Manages detector alert policy, incident creation, grouping, and persistence."""

    def __init__(self) -> None:
        # Track consecutive anomalous windows per session
        self._consecutive_anomalies: dict[str, int] = {}
        # Track cooldown countdown per session
        self._cooldown_remaining: dict[str, int] = {}
        # Track detection latency per session (windows since attack started to first alert)
        self._attack_start_window: dict[str, int | None] = {}
        self._first_alert_window: dict[str, int | None] = {}

    def reset_session(self, session_id: str) -> None:
        """Reset tracking state for a session."""
        self._consecutive_anomalies.pop(session_id, None)
        self._cooldown_remaining.pop(session_id, None)
        self._attack_start_window.pop(session_id, None)
        self._first_alert_window.pop(session_id, None)

    def evaluate_window(
        self,
        session_id: str,
        window_index: int,
        simulated_time_ms: float,
        inference_result: dict[str, Any],
        telemetry_point: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Evaluate an inference result against the alert policy.

        Returns an incident dict if an incident is declared, otherwise None.
        """
        is_anomaly = bool(inference_result.get("is_anomaly", False))
        anomaly_score = float(inference_result.get("anomaly_score", 0.0))
        threshold = float(inference_result.get("threshold", 0.05))
        model_id = str(inference_result.get("model_id", settings.default_model_id))

        gt_active = bool(telemetry_point.get("ground_truth_attack_active", False))
        gt_type = str(telemetry_point.get("ground_truth_attack_type", "none"))

        # Track ground truth attack start for honest latency computation
        if gt_active and session_id not in self._attack_start_window:
            self._attack_start_window[session_id] = window_index
        elif not gt_active:
            self._attack_start_window.pop(session_id, None)
            self._first_alert_window.pop(session_id, None)

        # Decrement cooldown if active
        cur_cooldown = self._cooldown_remaining.get(session_id, 0)
        if cur_cooldown > 0:
            self._cooldown_remaining[session_id] = cur_cooldown - 1

        if is_anomaly:
            prev_streak = self._consecutive_anomalies.get(session_id, 0)
            cur_streak = prev_streak + 1
            self._consecutive_anomalies[session_id] = cur_streak
        else:
            self._consecutive_anomalies[session_id] = 0
            return None

        # Check threshold policy: required consecutive anomalies
        required_streak = settings.consecutive_anomalies_for_alert
        if cur_streak < required_streak:
            log.debug(f"Anomaly detected but streak {cur_streak} < required {required_streak}")
            return None

        # Check cooldown to avoid redundant alerts during sustained attack
        if self._cooldown_remaining.get(session_id, 0) > 0:
            log.debug(f"Alert suppressed by cooldown ({self._cooldown_remaining[session_id]} windows remaining)")
            return None

        # Calculate severity based on anomaly margin above threshold
        ratio = anomaly_score / max(threshold, 1e-6)
        if ratio >= 2.5:
            severity = "CRITICAL"
        elif ratio >= 1.8:
            severity = "HIGH"
        elif ratio >= 1.2:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Calculate honest detection latency in windows
        detection_latency = 0
        if session_id in self._attack_start_window:
            start_win = self._attack_start_window[session_id]
            if start_win is not None:
                detection_latency = max(0, window_index - start_win)

        # Create incident ID
        incident_id = f"INC-{session_id[:8].upper()}-{window_index:04d}"

        # Build evidence snapshot with observable telemetry and model components
        evidence = {
            "window_index": window_index,
            "simulated_time_ms": simulated_time_ms,
            "qber": telemetry_point.get("qber"),
            "sifted_rate": telemetry_point.get("sifted_rate"),
            "signal_gain": telemetry_point.get("signal_gain"),
            "decoy_gain": telemetry_point.get("decoy_gain"),
            "detector_imbalance": telemetry_point.get("detector_imbalance"),
            "model_components": inference_result.get("components", {}),
            "key_features": {
                k: v for k, v in telemetry_point.get("features", {}).items()
                if any(sub in k for sub in ("qber", "gain", "rate", "mean", "imbalance"))
            },
        }

        incident = {
            "incident_id": incident_id,
            "session_id": session_id,
            "detection_timestamp": time.time(),
            "simulated_time_ms": simulated_time_ms,
            "model_id": model_id,
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "severity": severity,
            "status": "NEW",
            "ground_truth_attack_active": int(gt_active),
            "ground_truth_attack_type": gt_type,
            "detection_latency_windows": detection_latency,
            "evidence_snapshot": evidence,
        }

        # Persist to database
        save_incident(incident)

        # Set cooldown for subsequent windows in this session
        self._cooldown_remaining[session_id] = settings.alert_cooldown_windows

        log.info(
            f"Declared Incident {incident_id}: Score={anomaly_score:.4f} Threshold={threshold:.4f} "
            f"Severity={severity} GroundTruth={gt_type}"
        )
        return incident

    def list_incidents(
        self,
        session_id: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List incidents from the persistent store with optional filtering."""
        return get_incidents(session_id=session_id, severity=severity, status=status, limit=limit)

    def update_status(self, incident_id: str, new_status: str) -> bool:
        """Update workflow status of an incident (NEW, ACKNOWLEDGED, INVESTIGATING, RESOLVED)."""
        valid_statuses = {"NEW", "ACKNOWLEDGED", "INVESTIGATING", "RESOLVED"}
        if new_status.upper() not in valid_statuses:
            raise ValueError(f"Invalid status: {new_status}. Must be one of {valid_statuses}")
        return update_incident_status(incident_id, new_status.upper())


incident_service = IncidentService()
