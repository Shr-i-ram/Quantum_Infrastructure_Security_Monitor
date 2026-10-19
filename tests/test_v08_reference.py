"""Unit tests for V0.8 reference window management and contamination protection."""

import numpy as np
import pytest

from qism.drift.reference import (
    ReferenceManager,
    ReferenceMetadata,
    ReferenceStrategy,
)


def test_reference_manager_initialization():
    """Verify initialization and metadata capture."""
    rng = np.random.RandomState(42)
    initial_data = rng.normal(loc=1.0, scale=0.5, size=(40, 3))

    mgr = ReferenceManager(
        strategy=ReferenceStrategy.ANCHORED,
        window_size=30,
        safe_anomaly_threshold=2.0,
        initial_data=initial_data,
    )

    assert mgr.current_version_id == "ref_v001"
    ref_samples = mgr.get_reference_samples()
    assert len(ref_samples) == 30  # Capped at window_size
    assert ref_samples.shape[1] == 3


def test_reference_contamination_safeguard():
    """Verify high anomaly score or active alarm strictly blocks sample from entering reference."""
    rng = np.random.RandomState(42)
    initial_data = rng.normal(loc=0.0, scale=1.0, size=(20, 2))

    mgr = ReferenceManager(
        strategy=ReferenceStrategy.SLIDING,
        window_size=20,
        safe_anomaly_threshold=2.0,
        initial_data=initial_data,
    )

    # 1. Normal clean sample -> accepted
    clean_sample = np.array([0.1, 0.2])
    accepted = mgr.ingest_sample(clean_sample, anomaly_score=0.5, is_operational_alarm=False, step_index=1)
    assert accepted is True
    assert mgr.contamination_rejections == 0

    # 2. Attack sample with high anomaly score -> REJECTED
    attack_sample = np.array([10.0, 15.0])
    rejected_score = mgr.ingest_sample(attack_sample, anomaly_score=4.5, is_operational_alarm=False, step_index=2)
    assert rejected_score is False
    assert mgr.contamination_rejections == 1

    # 3. Sample arriving during operational alarm -> REJECTED even if score happens to dip
    rejected_alarm = mgr.ingest_sample(clean_sample, anomaly_score=1.2, is_operational_alarm=True, step_index=3)
    assert rejected_alarm is False
    assert mgr.contamination_rejections == 2


def test_reference_promotion_and_rollback():
    """Verify candidate buffer promotion and rollback to parent version."""
    rng = np.random.RandomState(42)
    initial_data = rng.normal(loc=0.0, scale=1.0, size=(20, 2))
    mgr = ReferenceManager(strategy=ReferenceStrategy.ANCHORED, initial_data=initial_data)

    initial_version = mgr.current_version_id
    assert initial_version == "ref_v001"

    # Buffer candidate samples
    for _ in range(15):
        s = rng.normal(loc=2.0, scale=1.0, size=2)
        mgr.buffer_candidate_sample(s, anomaly_score=0.3, is_operational_alarm=False)

    # Promote
    meta = mgr.promote_candidate(step_index=10)
    assert mgr.current_version_id == "ref_v002"
    assert meta.parent_version_id == "ref_v001"
    assert np.allclose(mgr.get_reference_samples().mean(axis=0), [2.0, 2.0], atol=0.5)

    # Rollback
    reverted_meta = mgr.rollback()
    assert mgr.current_version_id == "ref_v001"
    assert np.allclose(mgr.get_reference_samples().mean(axis=0), [0.0, 0.0], atol=0.5)
