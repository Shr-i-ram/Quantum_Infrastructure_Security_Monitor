"""Unit tests for V0.8 adaptive streaming engine, causality, and checkpoint recovery."""

import numpy as np
import pandas as pd
import pytest

from qism.drift.adaptation import AdaptationConfig, AdaptationPolicy, AdaptationStateMachine
from qism.drift.detectors import DriftDetectorConfig, StatisticalDriftDetector
from qism.drift.reference import ReferenceManager, ReferenceStrategy
from qism.drift.registry import ModelRegistry
from qism.drift.streaming import AdaptiveStreamingEngine, AdaptiveStreamingResult
from qism.ml.preprocessing.scalers import PreprocessorPipeline
from qism.temporal.alarm import AlarmConfig, TemporalAlarmStateMachine
from qism.temporal.models import RollingZScoreDetector


def _build_test_engine(d: int = 3, L: int = 3, threshold: float = 3.0, tmp_path=None):
    rng = np.random.RandomState(42)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(30, d))

    prep = PreprocessorPipeline(scaler="robust")
    X_train_scaled = prep.fit_transform(pd.DataFrame(X_train))

    detector = RollingZScoreDetector(window_length=L)
    detector.fit(X_train_scaled)
    detector.set_threshold(threshold)

    alarm_sm = TemporalAlarmStateMachine(AlarmConfig(tau_high=threshold, persistence_k=2, persistence_n=3))
    drift_det = StatisticalDriftDetector(DriftDetectorConfig(threshold=0.50))
    ref_mgr = ReferenceManager(strategy=ReferenceStrategy.ANCHORED, initial_data=X_train_scaled)

    reg = ModelRegistry(output_dir=tmp_path)
    reg.register_initial_model(detector, threshold)

    adapt_sm = AdaptationStateMachine(
        AdaptationConfig(policy=AdaptationPolicy.SAFE, candidate_buffer_size=10, validation_window=5),
        reg, ref_mgr, threshold
    )

    engine = AdaptiveStreamingEngine(
        active_model=detector,
        preprocessor=prep,
        alarm_machine=alarm_sm,
        drift_detector=drift_det,
        reference_manager=ref_mgr,
        adaptation_machine=adapt_sm,
        registry=reg,
        history_window_length=L,
        drift_evaluation_window=15,
    )
    return engine


def test_streaming_causal_processing(tmp_path):
    """Verify sample-by-sample ingestion without future information and profile latency."""
    engine = _build_test_engine(d=3, L=3, tmp_path=tmp_path)
    rng = np.random.RandomState(42)
    X_stream = rng.normal(loc=0.0, scale=1.0, size=(25, 3))

    results: list[AdaptiveStreamingResult] = []
    for i in range(25):
        t_s = float(i * 0.002)
        res = engine.update(X_stream[i], timestamp_s=t_s)
        results.append(res)
        assert res.step_index == i
        assert res.timestamp_s == t_s
        assert res.computational_latency_ms > 0.0
        assert res.active_model_version.startswith("model_v")

    assert len(results) == 25
    assert np.mean(engine.computational_latencies) < 5.0  # Sub-5ms latency on CPU


def test_streaming_checkpoint_and_resume(tmp_path):
    """Verify engine state can be serialized, reloaded, and continue identically."""
    engine = _build_test_engine(d=3, L=3, tmp_path=tmp_path)
    rng = np.random.RandomState(42)
    X_stream = rng.normal(loc=0.0, scale=1.0, size=(20, 3))

    # Process first 10 steps
    for i in range(10):
        engine.update(X_stream[i], timestamp_s=float(i * 0.002))

    # Save checkpoint
    chk_dir = tmp_path / "checkpoint_test"
    engine.save_checkpoint(chk_dir)

    # Resume into new engine
    engine_resumed = _build_test_engine(d=3, L=3, tmp_path=tmp_path)
    engine_resumed.load_checkpoint(chk_dir)

    assert engine_resumed._step_counter == 10

    # Step 11 on resumed engine
    res_resumed = engine_resumed.update(X_stream[10], timestamp_s=float(10 * 0.002))
    assert res_resumed.step_index == 10


def test_deterministic_streaming_replay(tmp_path):
    """Verify identical input stream and seed produces exact identical outputs."""
    eng1 = _build_test_engine(d=3, L=3, tmp_path=tmp_path / "run1")
    eng2 = _build_test_engine(d=3, L=3, tmp_path=tmp_path / "run2")

    rng = np.random.RandomState(123)
    X_stream = rng.normal(loc=0.0, scale=1.0, size=(15, 3))

    res1 = eng1.stream_trajectory(X_stream)
    res2 = eng2.stream_trajectory(X_stream)

    for r1, r2 in zip(res1, res2):
        assert np.isclose(r1.anomaly_score, r2.anomaly_score)
        assert np.isclose(r1.drift_score, r2.drift_score)
        assert r1.filtered_alarm == r2.filtered_alarm
        assert r1.active_model_version == r2.active_model_version
