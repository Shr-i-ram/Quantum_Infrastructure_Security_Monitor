"""Safe adaptation gate, shadow validation, and adaptation state machine (V0.8).

Architected around the fundamental security principle:
    "NEVER LEARN AN ATTACK AS NORMAL."

Provides:
- AdaptationState: Explicit operational state machine enum.
- AdaptationPolicy: FIXED, NAIVE, or SAFE mode.
- SafeAdaptationGate: Multi-condition security gate enforcing anomaly suppression, persistence, and cooldown.
- ShadowModelValidator: Parallel validation of candidate models before operational promotion.
- AdaptationStateMachine: Coordinates drift detection, candidate evaluation, promotion, and rollback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Callable

import numpy as np

from qism.drift.detectors import DriftReport
from qism.drift.reference import ReferenceManager, ReferenceStrategy
from qism.drift.registry import AdaptationDecision, AdaptationLogEntry, ModelRegistry, ModelStatus


class AdaptationState(str, Enum):
    """Operational states of the adaptation subsystem."""
    STABLE = "STABLE"                     # Baseline or validated stable monitoring
    DRIFT_SUSPECTED = "DRIFT_SUSPECTED"   # Single/transient drift detected
    ADAPTATION_PENDING = "ADAPTATION_PENDING" # Persistent drift confirmed, gathering clean buffer
    ADAPTING = "ADAPTING"                 # Candidate model instantiated
    VALIDATING = "VALIDATING"             # Candidate running in shadow mode alongside active model
    STABLE_UPDATED = "STABLE_UPDATED"     # Promoted to new active baseline
    SECURITY_ALERT = "SECURITY_ALERT"     # Anomaly or attack detected; adaptation strictly locked
    ROLLBACK = "ROLLBACK"                 # Rollback executed to restore parent model


class AdaptationPolicy(str, Enum):
    """Adaptation operational policy."""
    FIXED = "fixed"     # Zero adaptation (fixed baseline detector)
    NAIVE = "naive"     # Unconstrained immediate adaptation on drift (vulnerable baseline)
    SAFE = "safe"       # Anomaly-gated, shadow-validated, rollback-capable adaptation (recommended)


@dataclass
class AdaptationConfig:
    """Configuration for safe adaptation state machine."""
    policy: AdaptationPolicy = AdaptationPolicy.SAFE
    persistence_k: int = 3          # Require K drift flags
    persistence_n: int = 5          # In the last N windows to confirm persistence
    safe_anomaly_ratio: float = 0.80 # Max anomaly score as fraction of alarm threshold to permit adaptation
    candidate_buffer_size: int = 30 # Samples needed before fitting candidate model
    validation_window: int = 20     # Windows candidate must run in shadow mode before promotion
    cooldown_steps: int = 40        # Steps after promotion before next adaptation can be considered
    max_validation_fpr: float = 0.05 # Maximum permissible false alarm rate during shadow validation
    rollback_enabled: bool = True   # Allow rollback if post-promotion alerts occur
    rollback_patience: int = 15     # Window after promotion where high alerts trigger immediate rollback


@dataclass
class AdaptationStepResult:
    """Result produced at each monitoring step by the adaptation engine."""
    state: AdaptationState
    decision: AdaptationDecision
    active_version: str
    candidate_version: str | None
    is_shadow_active: bool
    candidate_anomaly_score: float | None = None
    candidate_alarm: bool = False
    validation_progress: float = 0.0
    reason: str = ""


class AdaptationStateMachine:
    """Stateful coordinator for safe telemetry adaptation."""

    def __init__(
        self,
        config: AdaptationConfig,
        registry: ModelRegistry,
        reference_manager: ReferenceManager,
        alarm_threshold: float,
        model_builder: Callable[[np.ndarray], Any] | None = None,
    ):
        self.config = config
        self.registry = registry
        self.reference_manager = reference_manager
        self.alarm_threshold = float(alarm_threshold)
        self.safe_anomaly_threshold = float(alarm_threshold * config.safe_anomaly_ratio)
        self.model_builder = model_builder

        # State tracking
        self.state = AdaptationState.STABLE
        self.recent_drift_flags: list[bool] = []
        self.cooldown_remaining: int = 0
        self.validation_steps_completed: int = 0
        self.shadow_scores: list[float] = []
        self.shadow_alarms: list[bool] = []
        self.steps_since_last_promotion: int = 9999
        self.candidate_model: Any | None = None
        self.candidate_threshold: float = 0.0

    def reset(self) -> None:
        """Reset state machine for a new simulation stream."""
        self.state = AdaptationState.STABLE
        self.recent_drift_flags.clear()
        self.cooldown_remaining = 0
        self.validation_steps_completed = 0
        self.shadow_scores.clear()
        self.shadow_alarms.clear()
        self.steps_since_last_promotion = 9999
        self.candidate_model = None
        self.reference_manager.clear_candidate_buffer()

    def step(
        self,
        sample: np.ndarray,
        anomaly_score: float,
        operational_alarm: bool,
        drift_report: DriftReport,
        step_index: int,
        timestamp_s: float | None = None,
    ) -> AdaptationStepResult:
        """Process one monitoring step through the adaptation safety gate.

        Enforces strict causal ordering:
        1. Anomaly & Security Gating: Any high anomaly score immediately aborts adaptation.
        2. Persistence Check: Transient drift fluctuations are dampened.
        3. Shadow Validation: In SAFE mode, candidate models must demonstrate stability without false alarms.
        4. Atomic Promotion & Rollback: State updates are transactional and recorded in registry.
        """
        # Decrement cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        self.steps_since_last_promotion += 1

        # Track recent drift detections for persistence
        self.recent_drift_flags.append(drift_report.is_drift)
        if len(self.recent_drift_flags) > self.config.persistence_n:
            self.recent_drift_flags.pop(0)

        # -------------------------------------------------------------
        # POLICY: FIXED (Never adapt)
        # -------------------------------------------------------------
        if self.config.policy == AdaptationPolicy.FIXED:
            return AdaptationStepResult(
                state=AdaptationState.STABLE,
                decision=AdaptationDecision.NO_DRIFT,
                active_version=self.registry.active_version_id,
                candidate_version=None,
                is_shadow_active=False,
                reason="Adaptation disabled by FIXED policy",
            )

        # -------------------------------------------------------------
        # SECURITY OVERRIDE: Check for high anomaly / active attack
        # -------------------------------------------------------------
        is_security_concern = (
            operational_alarm or (anomaly_score >= self.safe_anomaly_threshold)
        )

        if is_security_concern:
            # If we were attempting to adapt, immediately abort to prevent attack contamination
            if self.state in (
                AdaptationState.DRIFT_SUSPECTED,
                AdaptationState.ADAPTATION_PENDING,
                AdaptationState.ADAPTING,
                AdaptationState.VALIDATING,
            ):
                if self.registry.candidate_version_id:
                    self.registry.reject_candidate(
                        reason=f"Security alert triggered (Score={anomaly_score:.3f} >= SafeThresh={self.safe_anomaly_threshold:.3f})"
                    )
                self.reference_manager.clear_candidate_buffer()
                self.candidate_model = None
                self.validation_steps_completed = 0
                self.state = AdaptationState.SECURITY_ALERT

                entry = AdaptationLogEntry(
                    step_index=step_index,
                    timestamp_s=timestamp_s,
                    active_model_version=self.registry.active_version_id,
                    reference_version=self.reference_manager.current_version_id,
                    drift_score=drift_report.drift_score,
                    drift_threshold=drift_report.threshold,
                    is_drift=drift_report.is_drift,
                    anomaly_score=anomaly_score,
                    anomaly_threshold=self.alarm_threshold,
                    operational_alarm=operational_alarm,
                    adaptation_state=self.state.value,
                    decision=AdaptationDecision.REJECTED_ANOMALY,
                    reason="Aborted adaptation due to elevated anomaly score during drift window",
                )
                self.registry.log_event(entry)

                return AdaptationStepResult(
                    state=self.state,
                    decision=AdaptationDecision.REJECTED_ANOMALY,
                    active_version=self.registry.active_version_id,
                    candidate_version=None,
                    is_shadow_active=False,
                    reason="Security alert locked adaptation",
                )

            # Check post-promotion rollback condition
            if (
                self.config.rollback_enabled
                and self.steps_since_last_promotion <= self.config.rollback_patience
                and self.state == AdaptationState.STABLE_UPDATED
            ):
                # Rollback! The newly promoted model suffered an alarm almost immediately
                restored_rec = self.registry.rollback()
                self.reference_manager.rollback()
                self.state = AdaptationState.ROLLBACK

                entry = AdaptationLogEntry(
                    step_index=step_index,
                    timestamp_s=timestamp_s,
                    active_model_version=self.registry.active_version_id,
                    reference_version=self.reference_manager.current_version_id,
                    drift_score=drift_report.drift_score,
                    drift_threshold=drift_report.threshold,
                    is_drift=drift_report.is_drift,
                    anomaly_score=anomaly_score,
                    anomaly_threshold=self.alarm_threshold,
                    operational_alarm=operational_alarm,
                    adaptation_state=self.state.value,
                    decision=AdaptationDecision.ROLLBACK_EXECUTED,
                    reason="Immediate alarm post-promotion triggered rollback to parent model",
                )
                self.registry.log_event(entry)

                return AdaptationStepResult(
                    state=self.state,
                    decision=AdaptationDecision.ROLLBACK_EXECUTED,
                    active_version=self.registry.active_version_id,
                    candidate_version=None,
                    is_shadow_active=False,
                    reason=f"Rolled back to {restored_rec.version_id}",
                )

        # -------------------------------------------------------------
        # POLICY: NAIVE (Immediate unconstrained adaptation on drift)
        # -------------------------------------------------------------
        if self.config.policy == AdaptationPolicy.NAIVE:
            if drift_report.is_drift and self.cooldown_remaining == 0:
                # Naive adaptation blindly buffers without any safety gating or anomaly check
                self.reference_manager._candidate_buffer.append(sample.flatten())
                cand_samples = self.reference_manager.get_candidate_samples()
                if len(cand_samples) < 5:
                    return AdaptationStepResult(
                        state=AdaptationState.ADAPTING,
                        decision=AdaptationDecision.ADAPTATION_INITIATED,
                        active_version=self.registry.active_version_id,
                        candidate_version=None,
                        is_shadow_active=False,
                        reason="Naive accumulating samples blindly",
                    )
                # Blind promotion
                ref_meta = self.reference_manager.promote_candidate(
                    step_index=step_index, timestamp_s=timestamp_s
                )
                if self.model_builder:
                    new_model = self.model_builder(cand_samples)
                    cand_scores = new_model.score(cand_samples) if hasattr(new_model, "score") else np.zeros(len(cand_samples))
                    new_thresh = float(np.percentile(cand_scores, 95)) if len(cand_scores) > 0 else self.alarm_threshold
                    new_model.set_threshold(new_thresh)
                    rec = self.registry.register_candidate(
                        candidate_model=new_model,
                        candidate_threshold=new_thresh,
                        reference_version_id=ref_meta.version_id,
                        step_index=step_index,
                        timestamp_s=timestamp_s,
                        trigger="NAIVE_UNCONSTRAINED",
                    )
                    self.registry.promote_candidate()
                self.cooldown_remaining = self.config.cooldown_steps
                return AdaptationStepResult(
                    state=AdaptationState.STABLE_UPDATED,
                    decision=AdaptationDecision.PROMOTED,
                    active_version=self.registry.active_version_id,
                    candidate_version=None,
                    is_shadow_active=False,
                    reason="Naive immediate adaptation executed",
                )
            else:
                return AdaptationStepResult(
                    state=AdaptationState.STABLE,
                    decision=AdaptationDecision.NO_DRIFT,
                    active_version=self.registry.active_version_id,
                    candidate_version=None,
                    is_shadow_active=False,
                )

        # -------------------------------------------------------------
        # POLICY: SAFE (Persistence-gated, shadow-validated adaptation)
        # -------------------------------------------------------------
        # Handle current state transitions
        if self.state in (AdaptationState.STABLE, AdaptationState.STABLE_UPDATED, AdaptationState.ROLLBACK, AdaptationState.SECURITY_ALERT):
            if drift_report.is_drift:
                self.state = AdaptationState.DRIFT_SUSPECTED

        if self.state == AdaptationState.DRIFT_SUSPECTED:
            # Check persistence: K of N windows
            k_count = sum(self.recent_drift_flags)
            if k_count >= self.config.persistence_k:
                if self.cooldown_remaining > 0:
                    decision = AdaptationDecision.REJECTED_COOLDOWN
                    reason = f"Drift persistent but in cooldown ({self.cooldown_remaining} steps left)"
                else:
                    self.state = AdaptationState.ADAPTATION_PENDING
                    decision = AdaptationDecision.DRIFT_PERSISTING
                    reason = f"Drift persistent ({k_count}/{len(self.recent_drift_flags)} windows)"
            else:
                decision = AdaptationDecision.DRIFT_DETECTED
                reason = f"Drift detected ({k_count}/{len(self.recent_drift_flags)} windows, need {self.config.persistence_k})"

            return AdaptationStepResult(
                state=self.state,
                decision=decision,
                active_version=self.registry.active_version_id,
                candidate_version=None,
                is_shadow_active=False,
                reason=reason,
            )

        if self.state == AdaptationState.ADAPTATION_PENDING:
            # Accumulate clean candidate samples
            buffered = self.reference_manager.buffer_candidate_sample(
                sample=sample,
                anomaly_score=anomaly_score,
                is_operational_alarm=operational_alarm,
            )
            cand_samples = self.reference_manager.get_candidate_samples()

            if len(cand_samples) >= self.config.candidate_buffer_size:
                # Instantiate candidate model and enter VALIDATING (shadow mode)
                if self.model_builder:
                    self.candidate_model = self.model_builder(cand_samples)
                else:
                    self.candidate_model = self.registry.get_active_model()

                # Calibrate candidate threshold on candidate buffer (at target FPR)
                cand_scores = self.candidate_model.score(cand_samples) if hasattr(self.candidate_model, "score") else np.zeros(len(cand_samples))
                self.candidate_threshold = float(np.percentile(cand_scores, 95)) if len(cand_scores) > 0 else self.alarm_threshold

                # Register in registry
                self.registry.register_candidate(
                    candidate_model=self.candidate_model,
                    candidate_threshold=self.candidate_threshold,
                    reference_version_id="candidate_ref",
                    step_index=step_index,
                    timestamp_s=timestamp_s,
                    trigger="SAFE_GATED_DRIFT",
                )

                self.state = AdaptationState.VALIDATING
                self.validation_steps_completed = 0
                self.shadow_scores.clear()
                self.shadow_alarms.clear()

                return AdaptationStepResult(
                    state=self.state,
                    decision=AdaptationDecision.ADAPTATION_INITIATED,
                    active_version=self.registry.active_version_id,
                    candidate_version=self.registry.candidate_version_id,
                    is_shadow_active=True,
                    reason=f"Candidate buffer complete ({len(cand_samples)} samples). Commencing shadow validation.",
                )
            else:
                return AdaptationStepResult(
                    state=self.state,
                    decision=AdaptationDecision.ADAPTATION_INITIATED,
                    active_version=self.registry.active_version_id,
                    candidate_version=None,
                    is_shadow_active=False,
                    reason=f"Buffering clean telemetry ({len(cand_samples)}/{self.config.candidate_buffer_size})",
                )

        if self.state == AdaptationState.VALIDATING:
            # Shadow evaluation: evaluate sample on candidate model without affecting operational alert
            cand_sample_2d = sample.reshape(1, -1) if sample.ndim == 1 else sample
            cand_score = float(self.candidate_model.score(cand_sample_2d)[0]) if hasattr(self.candidate_model, "score") else 0.0
            cand_alarm = cand_score > self.candidate_threshold

            self.shadow_scores.append(cand_score)
            self.shadow_alarms.append(cand_alarm)
            self.validation_steps_completed += 1

            # Buffer sample into candidate reference as well
            self.reference_manager.buffer_candidate_sample(sample, anomaly_score, operational_alarm)

            val_progress = float(self.validation_steps_completed / self.config.validation_window)

            if self.validation_steps_completed >= self.config.validation_window:
                # Evaluate candidate stability
                shadow_fpr = float(np.mean(self.shadow_alarms)) if self.shadow_alarms else 0.0

                if shadow_fpr <= self.config.max_validation_fpr:
                    # PROMOTION! Candidate successfully validated
                    ref_meta = self.reference_manager.promote_candidate(
                        step_index=step_index,
                        timestamp_s=timestamp_s,
                    )
                    promoted_rec = self.registry.promote_candidate(
                        validation_metrics={"shadow_fpr": shadow_fpr, "mean_shadow_score": float(np.mean(self.shadow_scores))}
                    )
                    self.state = AdaptationState.STABLE_UPDATED
                    self.cooldown_remaining = self.config.cooldown_steps
                    self.steps_since_last_promotion = 0
                    self.candidate_model = None

                    entry = AdaptationLogEntry(
                        step_index=step_index,
                        timestamp_s=timestamp_s,
                        active_model_version=promoted_rec.version_id,
                        reference_version=ref_meta.version_id,
                        drift_score=drift_report.drift_score,
                        drift_threshold=drift_report.threshold,
                        is_drift=drift_report.is_drift,
                        anomaly_score=anomaly_score,
                        anomaly_threshold=self.alarm_threshold,
                        operational_alarm=operational_alarm,
                        adaptation_state=self.state.value,
                        decision=AdaptationDecision.PROMOTED,
                        candidate_version=promoted_rec.version_id,
                        reason=f"Candidate promoted after passing shadow validation (Shadow FPR={shadow_fpr:.3f})",
                    )
                    self.registry.log_event(entry)

                    return AdaptationStepResult(
                        state=self.state,
                        decision=AdaptationDecision.PROMOTED,
                        active_version=promoted_rec.version_id,
                        candidate_version=None,
                        is_shadow_active=False,
                        candidate_anomaly_score=cand_score,
                        candidate_alarm=cand_alarm,
                        validation_progress=1.0,
                        reason="Candidate passed shadow validation and promoted to active",
                    )
                else:
                    # REJECTION! Candidate produced excessive alarms during shadow validation
                    self.registry.reject_candidate(
                        reason=f"Shadow FPR {shadow_fpr:.3f} exceeded max allowable {self.config.max_validation_fpr:.3f}"
                    )
                    self.state = AdaptationState.STABLE
                    self.candidate_model = None
                    self.reference_manager.clear_candidate_buffer()

                    return AdaptationStepResult(
                        state=self.state,
                        decision=AdaptationDecision.REJECTED_VALIDATION_FAILED,
                        active_version=self.registry.active_version_id,
                        candidate_version=None,
                        is_shadow_active=False,
                        candidate_anomaly_score=cand_score,
                        candidate_alarm=cand_alarm,
                        validation_progress=1.0,
                        reason="Candidate rejected due to high shadow false alarm rate",
                    )

            return AdaptationStepResult(
                state=self.state,
                decision=AdaptationDecision.SHADOW_VALIDATING,
                active_version=self.registry.active_version_id,
                candidate_version=self.registry.candidate_version_id,
                is_shadow_active=True,
                candidate_anomaly_score=cand_score,
                candidate_alarm=cand_alarm,
                validation_progress=val_progress,
                reason=f"Shadow validation step {self.validation_steps_completed}/{self.config.validation_window}",
            )

        return AdaptationStepResult(
            state=self.state,
            decision=AdaptationDecision.NO_DRIFT,
            active_version=self.registry.active_version_id,
            candidate_version=None,
            is_shadow_active=False,
        )
