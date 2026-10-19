"""Deterministic severity assessment and explainability engine for QISM V0.9.

Provides:
1. Deterministic priority ranking (INFO, LOW, MEDIUM, HIGH, CRITICAL)
2. Verifiable machine-generated rationales for every severity assignment
3. Subsystem pattern characterization (without fabricating causal attack attribution)
4. Advisory, non-executing response recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qism.monitor.types import (
    DetectorSignal,
    IncidentState,
    ResponseRecommendation,
    SeverityLevel,
)


@dataclass
class SeverityThresholdConfig:
    """Configurable evidence score thresholds for deterministic severity assignment."""
    th_info: float = 0.20
    th_low: float = 0.35
    th_medium: float = 0.55
    th_high: float = 0.75
    th_critical: float = 0.90
    min_subsystems_for_critical: int = 2
    min_persistence_for_high: int = 3


class SeverityEngine:
    """Evaluates security evidence and produces explainable severity rankings and advisory actions."""

    def __init__(self, config: SeverityThresholdConfig | None = None):
        self.config = config or SeverityThresholdConfig()

    def evaluate(
        self,
        evidence_score: float,
        signals: list[DetectorSignal],
        persistence_steps: int,
        affected_features: list[str],
        incident_state: IncidentState = IncidentState.ACTIVE,
        drift_active: bool = False,
    ) -> tuple[SeverityLevel, dict[str, Any], ResponseRecommendation]:
        """Compute severity level, structured machine explanation, and advisory recommendation.

        Returns:
            severity: Determined SeverityLevel
            explanation: Structured explanation dictionary with explicit rationale bullets
            recommendation: Non-executing advisory recommendation
        """
        cfg = self.config
        anomalous_signals = [s for s in signals if s.is_anomalous]
        subsystems_affected = sorted(list({s.subsystem for s in anomalous_signals}))
        num_subsystems = len(subsystems_affected)

        reasons: list[str] = []
        severity = SeverityLevel.INFO

        # 1. Base threshold evaluation
        if evidence_score >= cfg.th_critical and num_subsystems >= cfg.min_subsystems_for_critical:
            severity = SeverityLevel.CRITICAL
            reasons.append(f"Unified evidence score ({evidence_score:.3f}) exceeds critical threshold ({cfg.th_critical:.2f})")
            reasons.append(f"Multi-subsystem compromise: {num_subsystems} independent subsystems affected ({', '.join(subsystems_affected)})")
        elif evidence_score >= cfg.th_high:
            if persistence_steps >= cfg.min_persistence_for_high or num_subsystems >= 2:
                severity = SeverityLevel.HIGH
                reasons.append(f"High security evidence score ({evidence_score:.3f}) with sustained multi-window persistence ({persistence_steps} steps)")
            else:
                severity = SeverityLevel.MEDIUM
                reasons.append(f"Elevated evidence score ({evidence_score:.3f}) without sustained persistence")
        elif evidence_score >= cfg.th_medium:
            severity = SeverityLevel.MEDIUM
            reasons.append(f"Moderate security evidence score ({evidence_score:.3f}) exceeding medium threshold")
        elif evidence_score >= cfg.th_low:
            severity = SeverityLevel.LOW
            reasons.append(f"Low-level anomaly indication ({evidence_score:.3f}) exceeding operational baseline")
        else:
            severity = SeverityLevel.INFO
            reasons.append(f"Nominal telemetry: evidence score ({evidence_score:.3f}) remains within benign operating limits")

        # 2. Detailed signal explanations
        for sig in anomalous_signals:
            reasons.append(
                f"Signal '{sig.signal_name}' in subsystem '{sig.subsystem}' exceeded threshold "
                f"(raw: {sig.raw_value:.4f}, norm: {sig.normalized_value:.3f}, thresh: {sig.threshold:.4f})"
            )

        # 3. Drift context
        if drift_active:
            reasons.append("Environmental/channel drift actively detected; verified baseline distribution shift")

        # 4. Pattern characterization (Non-causal, objective descriptive observation)
        pattern_desc = self._characterize_telemetry_pattern(anomalous_signals, affected_features)

        # 5. Build structured explanation
        explanation = {
            "evidence_score": float(evidence_score),
            "assigned_severity": severity.value,
            "persistence_steps": int(persistence_steps),
            "subsystems_affected": subsystems_affected,
            "anomalous_signals_count": len(anomalous_signals),
            "reasons": reasons,
            "pattern_characterization": pattern_desc,
            "drift_context": "benign_drift_active" if drift_active else "no_drift",
            "incident_state": incident_state.value,
        }

        # 6. Generate advisory response recommendation
        recommendation = self._generate_advisory_recommendation(
            severity, subsystems_affected, pattern_desc, affected_features
        )

        return severity, explanation, recommendation

    def _characterize_telemetry_pattern(
        self, anomalous_signals: list[DetectorSignal], affected_features: list[str]
    ) -> str:
        """Provide scientific pattern description based on observable evidence."""
        sig_names = {s.signal_name for s in anomalous_signals}
        feats = set(affected_features)

        if "qber_deviation" in sig_names or "qber" in feats:
            if "timing_deviation" in sig_names or "clock_drift_rate" in feats:
                return "Telemetry pattern consistent with simultaneous quantum bit error escalation and timing synchronization perturbation."
            return "Telemetry pattern dominated by quantum bit error rate (QBER) deviation on decoy/signal pulses."
        elif "optical_gain_deviation" in sig_names or "signal_gain" in feats or "vacuum_gain" in feats:
            if "double_click_rate" in feats or "dark_count_rate" in feats:
                return "Telemetry pattern consistent with abnormal detector click rates and optical power flux."
            return "Telemetry pattern characterized by significant optical transmittance / gain attenuation."
        elif "statistical_drift" in sig_names:
            return "Telemetry pattern characterized primarily by smooth distribution shift in optical or environmental baseline."
        elif sig_names:
            return f"Telemetry pattern characterized by isolated deviations in {', '.join(sorted(sig_names))}."
        else:
            return "Telemetry pattern within nominal operational limits."

    def _generate_advisory_recommendation(
        self,
        severity: SeverityLevel,
        subsystems: list[str],
        pattern: str,
        affected_features: list[str],
    ) -> ResponseRecommendation:
        """Formulate non-executing advisory guidance for QKD network operators."""
        if severity == SeverityLevel.CRITICAL:
            text = "URGENT: Review quantum transmission integrity and multi-subsystem telemetry."
            actions = [
                "Operator action: Inspect real-time raw photon detection statistics and coincidence windows.",
                "Operator action: Verify classical authenticated channel and eavesdropping estimation logs.",
                "Operator action: Review optical fiber link integrity for severe intrusive taps or blinding power.",
            ]
        elif severity == SeverityLevel.HIGH:
            text = "ELEVATED ALERT: Investigate persistent telemetry anomaly across key subsystems."
            actions = [
                "Operator action: Cross-reference QBER trends with optical attenuation monitoring.",
                "Operator action: Inspect Alice source intensity stabilization and Bob detector dark rates.",
            ]
        elif severity == SeverityLevel.MEDIUM:
            text = "ADVISORY: Monitor telemetry trends for potential sustained security degradation."
            actions = [
                "Operator action: Observe next 20 transmission windows for recovery or escalation.",
                "Operator action: Check detector environmental sensor logs (temperature, vibration).",
            ]
        elif severity == SeverityLevel.LOW:
            text = "NOTE: Minor telemetry fluctuation observed; continue standard monitoring."
            actions = [
                "Operator action: Log event for periodic audit review.",
            ]
        else:
            text = "STATUS NORMAL: Telemetry within established operational baselines."
            actions = []

        primary_target = subsystems[0] if subsystems else "system_wide"

        return ResponseRecommendation(
            advisory_text=text,
            rationale=f"Assigned severity {severity.value.upper()} based on pattern: {pattern}",
            severity=severity,
            target_subsystem=primary_target,
            suggested_actions=actions,
            is_automated_execution=False,  # Explicit non-execution
        )
