"""Unit tests for V0.8 safe adaptation state machine, gating, and rollback."""

import numpy as np
import pytest

from qism.drift.adaptation import (
    AdaptationConfig,
    AdaptationDecision,
    AdaptationPolicy,
    AdaptationState,
    AdaptationStateMachine,
)
from qism.drift.detectors import DriftMetric, DriftReport
from qism.drift.reference import ReferenceManager, ReferenceStrategy
from qism.drift.registry import ModelRegistry
from qism.temporal.models import RollingZScoreDetector


def _create_mock_drift_report(is_drift: bool, drift_score: float = 0.6) -> DriftReport:
    return DriftReport(
        timestamp_s=0.0,
        step_index=0,
        metric=DriftMetric.WASSERSTEIN,
        drift_score=drift_score if is_drift else 0.1,
        threshold=0.45,
        is_drift=is_drift,
        feature_drift_scores={},
        feature_p_values={},
        dominant_features=[],
        sample_size_ref=20,
        sample_size_curr=20,
    )


def test_safe_adaptation_state_machine_happy_path():
    """Verify clean drift transitions: STABLE -> DRIFT_SUSPECTED -> ADAPTATION_PENDING -> VALIDATING -> STABLE_UPDATED."""
    rng = np.random.RandomState(42)
    initial_data = rng.normal(loc=0.0, scale=1.0, size=(30, 2))

    detector = RollingZScoreDetector(window_length=3)
    detector.fit(initial_data)
    detector.set_threshold(3.0)

    reg = ModelRegistry()
    reg.register_initial_model(detector, 3.0)

    ref_mgr = ReferenceManager(strategy=ReferenceStrategy.ANCHORED, initial_data=initial_data)

    cfg = AdaptationConfig(
        policy=AdaptationPolicy.SAFE,
        persistence_k=2,
        persistence_n=3,
        candidate_buffer_size=10,
        validation_window=5,
        cooldown_steps=10,
    )

    def model_builder(cand):
        m = RollingZScoreDetector(window_length=3)
        m.fit(cand)
        m.set_threshold(3.0)
        return m

    sm = AdaptationStateMachine(cfg, reg, ref_mgr, alarm_threshold=3.0, model_builder=model_builder)

    # 1. Step 0: Clean operation -> STABLE
    r0 = sm.step(np.zeros(2), anomaly_score=0.2, operational_alarm=False, drift_report=_create_mock_drift_report(False), step_index=0)
    assert r0.state == AdaptationState.STABLE

    # 2. Step 1: Single drift flag -> DRIFT_SUSPECTED
    r1 = sm.step(np.ones(2), anomaly_score=0.3, operational_alarm=False, drift_report=_create_mock_drift_report(True), step_index=1)
    assert r1.state == AdaptationState.DRIFT_SUSPECTED

    # 3. Step 2: Second drift flag (K=2 reached) -> ADAPTATION_PENDING
    r2 = sm.step(np.ones(2), anomaly_score=0.4, operational_alarm=False, drift_report=_create_mock_drift_report(True), step_index=2)
    assert r2.state == AdaptationState.ADAPTATION_PENDING

    # 4. Feed clean samples until candidate_buffer_size (10) is satisfied
    for step in range(3, 13):
        res = sm.step(np.ones(2), anomaly_score=0.3, operational_alarm=False, drift_report=_create_mock_drift_report(True), step_index=step)

    # Transition to VALIDATING (shadow mode)
    assert sm.state == AdaptationState.VALIDATING
    assert res.is_shadow_active is True

    # 5. Complete validation_window (5 steps) with clean shadow scores
    for step in range(13, 18):
        res = sm.step(np.ones(2), anomaly_score=0.3, operational_alarm=False, drift_report=_create_mock_drift_report(True), step_index=step)

    # Candidate promoted!
    assert res.state == AdaptationState.STABLE_UPDATED
    assert res.decision == AdaptationDecision.PROMOTED
    assert reg.active_version_id == "model_v002"


def test_safe_adaptation_security_gating_blocks_attack():
    """Verify anomaly spike or alarm during drift strictly aborts adaptation and rejects candidate."""
    rng = np.random.RandomState(42)
    initial_data = rng.normal(loc=0.0, scale=1.0, size=(30, 2))

    detector = RollingZScoreDetector(window_length=3)
    detector.fit(initial_data)
    detector.set_threshold(3.0)

    reg = ModelRegistry()
    reg.register_initial_model(detector, 3.0)
    ref_mgr = ReferenceManager(strategy=ReferenceStrategy.ANCHORED, initial_data=initial_data)

    cfg = AdaptationConfig(
        policy=AdaptationPolicy.SAFE,
        persistence_k=1,
        persistence_n=1,
        candidate_buffer_size=5,
    )
    sm = AdaptationStateMachine(cfg, reg, ref_mgr, alarm_threshold=3.0)

    # Trigger drift
    sm.step(np.ones(2), anomaly_score=0.2, operational_alarm=False, drift_report=_create_mock_drift_report(True), step_index=0)
    assert sm.state == AdaptationState.ADAPTATION_PENDING

    # Suddenly an attack arrives: anomaly score spikes!
    res = sm.step(
        np.array([10.0, 10.0]),
        anomaly_score=4.5,  # Exceeds safe threshold (3.0 * 0.8 = 2.4)
        operational_alarm=True,
        drift_report=_create_mock_drift_report(True),
        step_index=1,
    )

    # Must transition to SECURITY_ALERT and abort adaptation
    assert res.state == AdaptationState.SECURITY_ALERT
    assert res.decision == AdaptationDecision.REJECTED_ANOMALY
    assert reg.active_version_id == "model_v001"  # Retained baseline!
