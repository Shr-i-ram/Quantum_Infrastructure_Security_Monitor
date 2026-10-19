"""Tests for QISM V0.9 multivariate signal fusion engine."""

import pytest
from qism.monitor.fusion import (
    SignalFusionEngine,
    SignalFusionConfig,
    SignalWeightConfig,
    FusionMethod,
)


def test_fusion_score_normalization():
    engine = SignalFusionEngine()
    # Score 0 -> 0.0
    assert engine.normalize_score(0.0, 2.50) == 0.0
    # Score == threshold -> 0.50
    assert engine.normalize_score(2.50, 2.50) == 0.50
    # Score >= 2*threshold -> 1.00
    assert engine.normalize_score(5.00, 2.50) == 1.00
    # Score > 2*threshold is clamped to 1.0
    assert engine.normalize_score(10.00, 2.50) == 1.00


def test_weighted_fusion_baseline():
    cfg = SignalFusionConfig(
        fusion_method=FusionMethod.WEIGHTED,
        weights=SignalWeightConfig(
            w_anomaly_model=0.50,
            w_qber_deviation=0.50,
            w_gain_deviation=0.0,
            w_timing_deviation=0.0,
            w_drift_score=0.0,
        ),
    )
    engine = SignalFusionEngine(cfg)

    # Moderate anomaly (norm=0.5) and nominal QBER (norm=0.0)
    evidence, signals, breakdown = engine.evaluate(
        anomaly_score=2.50,
        anomaly_threshold=2.50,
        drift_score=0.10,
        drift_threshold=0.40,
        features={"qber": 0.015},
        timestamp_s=0.10,
    )
    assert 0.0 <= evidence <= 1.0
    # Anomaly norm = 0.50, QBER norm = 0.0 -> weighted sum = 0.25
    assert pytest.approx(evidence, abs=0.05) == 0.25


def test_persistence_aware_fusion_scaling():
    cfg = SignalFusionConfig(
        fusion_method=FusionMethod.PERSISTENCE_AWARE,
        persistence_boost_factor=0.20,
        evidence_threshold=0.40,
    )
    engine = SignalFusionEngine(cfg)

    # Step 1: Anomaly occurs
    ev1, _, _ = engine.evaluate(
        anomaly_score=3.50,
        anomaly_threshold=2.50,
        drift_score=0.10,
        drift_threshold=0.40,
        features={"qber": 0.025},
        timestamp_s=0.01,
    )

    # Step 2: Anomaly persists
    ev2, _, _ = engine.evaluate(
        anomaly_score=3.50,
        anomaly_threshold=2.50,
        drift_score=0.10,
        drift_threshold=0.40,
        features={"qber": 0.025},
        timestamp_s=0.02,
    )

    # Persistence boost should increase evidence score
    assert ev2 > ev1


def test_orthogonal_synergy_boost():
    cfg = SignalFusionConfig(
        fusion_method=FusionMethod.RULE_BASED,
        orthogonal_synergy_boost=0.15,
    )
    engine = SignalFusionEngine(cfg)

    # All subsystems deviate simultaneously: ML engine, quantum channel, optical receiver, synchronization
    ev, signals, breakdown = engine.evaluate(
        anomaly_score=5.0,
        anomaly_threshold=2.5,
        drift_score=0.1,
        drift_threshold=0.4,
        features={"qber": 0.25, "signal_gain": 0.005, "clock_drift_rate": 0.50},
        timestamp_s=0.05,
    )
    assert breakdown["synergy_boost_applied"] is True
    assert breakdown["anomalous_subsystems_count"] >= 3
    assert ev >= 0.70
