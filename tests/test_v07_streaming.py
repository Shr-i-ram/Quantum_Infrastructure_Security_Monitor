"""Unit tests for V0.7 streaming inference engine and causal ring buffer."""

import numpy as np
import pandas as pd
import pytest

from qism.ml.preprocessing.scalers import PreprocessorPipeline
from qism.temporal.alarm import AlarmConfig, TemporalAlarmStateMachine
from qism.temporal.models import RollingZScoreDetector
from qism.temporal.streaming import StreamingInferenceEngine, StreamingResult


def test_streaming_causal_ingestion():
    """Verify streaming engine ingests samples one by one causally without future information."""
    d = 4
    L = 3
    rng = np.random.RandomState(42)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(40, d))

    preprocessor = PreprocessorPipeline(scaler="robust")
    X_train_scaled = preprocessor.fit_transform(pd.DataFrame(X_train))

    detector = RollingZScoreDetector(window_length=L)
    detector.fit(X_train_scaled)
    detector.set_threshold(3.0)


    alarm_sm = TemporalAlarmStateMachine(AlarmConfig(tau_high=3.0, persistence_k=2, persistence_n=3))

    engine = StreamingInferenceEngine(
        model=detector,
        preprocessor=preprocessor,
        alarm_machine=alarm_sm,
        history_window_length=L,
    )

    # Ingest 10 samples
    X_test = rng.normal(loc=0.0, scale=1.0, size=(10, d))
    results: list[StreamingResult] = []

    for i in range(10):
        t_s = float(i * 0.002)
        res = engine.update(X_test[i], timestamp_s=t_s)
        results.append(res)
        assert res.step_index == i
        assert res.timestamp_s == t_s
        assert res.computational_latency_ms > 0.0
        assert isinstance(res.score, float)
        assert res.raw_alarm in (True, False, 0, 1)

    assert len(results) == 10
    assert len(engine.computational_latencies) == 10


def test_streaming_trajectory_batch():
    """Verify stream_trajectory helper processes sequence consistently."""
    d = 3
    L = 2
    X_train = np.ones((20, d))
    preprocessor = PreprocessorPipeline(scaler="robust")
    preprocessor.fit(pd.DataFrame(X_train))

    detector = RollingZScoreDetector(window_length=L)
    detector.fit(X_train)
    detector.set_threshold(2.0)

    alarm_sm = TemporalAlarmStateMachine(AlarmConfig(tau_high=2.0))
    engine = StreamingInferenceEngine(
        model=detector,
        preprocessor=preprocessor,
        alarm_machine=alarm_sm,
        history_window_length=L,
    )

    X_stream = np.ones((5, d))
    results = engine.stream_trajectory(X_stream)
    assert len(results) == 5
    assert results[-1].step_index == 4
