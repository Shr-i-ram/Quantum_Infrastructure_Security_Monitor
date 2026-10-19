"""Unit tests for V0.8 security safeguards and attack contamination prevention."""

import numpy as np
import pandas as pd
import pytest

from qism.drift.adaptation import AdaptationConfig, AdaptationPolicy, AdaptationStateMachine
from qism.drift.detectors import DriftDetectorConfig, StatisticalDriftDetector
from qism.drift.evaluation import DriftEvaluator
from qism.drift.reference import ReferenceManager, ReferenceStrategy
from qism.drift.registry import ModelRegistry
from qism.drift.scenarios import DriftScenarioGenerator, DriftScenarioType, ScenarioSpec
from qism.drift.streaming import AdaptiveStreamingEngine
from qism.ml.preprocessing.scalers import PreprocessorPipeline
from qism.temporal.alarm import AlarmConfig, TemporalAlarmStateMachine
from qism.temporal.models import RollingZScoreDetector


def test_attack_contamination_safeguard():
    """Verify safe adaptation prevents attack telemetry from entering reference distribution.

    Contrast:
    - Naive adaptation incorporates attack into its reference, resulting in contamination.
    - Safe adaptation suppresses candidate updates during attack windows and records zero contamination.
    """
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.CONTAMINATION_STRESS,
        drift_start_step=20,
        drift_end_step=50,
        attack_type="intercept_resend",
        total_steps=80,
        seed=42,
    )
    df_stream, feature_cols = DriftScenarioGenerator(spec).generate_stream()

    X_train_raw = df_stream[feature_cols].iloc[:20].values

    # 1. Setup Naive Engine
    prep_naive = PreprocessorPipeline(scaler="robust")
    X_train_scaled_naive = prep_naive.fit_transform(pd.DataFrame(X_train_raw, columns=feature_cols))
    det_naive = RollingZScoreDetector(window_length=3)
    det_naive.fit(X_train_scaled_naive)
    det_naive.set_threshold(3.0)

    alarm_sm_naive = TemporalAlarmStateMachine(AlarmConfig(tau_high=3.0))
    drift_det_naive = StatisticalDriftDetector(DriftDetectorConfig(threshold=0.40, feature_names=feature_cols))
    ref_mgr_naive = ReferenceManager(strategy=ReferenceStrategy.ANCHORED, initial_data=X_train_scaled_naive)
    reg_naive = ModelRegistry()
    reg_naive.register_initial_model(det_naive, 3.0)

    def model_builder_naive(cand):
        m = RollingZScoreDetector(window_length=3)
        m.fit(cand)
        m.set_threshold(3.0)
        return m

    adapt_sm_naive = AdaptationStateMachine(
        AdaptationConfig(policy=AdaptationPolicy.NAIVE, cooldown_steps=10),
        reg_naive, ref_mgr_naive, 3.0, model_builder=model_builder_naive
    )

    engine_naive = AdaptiveStreamingEngine(
        active_model=det_naive,
        preprocessor=prep_naive,
        alarm_machine=alarm_sm_naive,
        drift_detector=drift_det_naive,
        reference_manager=ref_mgr_naive,
        adaptation_machine=adapt_sm_naive,
        registry=reg_naive,
        history_window_length=3,
        drift_evaluation_window=15,
        feature_names=feature_cols,
    )

    res_naive = engine_naive.stream_trajectory(df_stream[feature_cols], timestamps_s=df_stream["timestamp"].values)

    # 2. Setup Safe Engine
    prep_safe = PreprocessorPipeline(scaler="robust")
    X_train_scaled_safe = prep_safe.fit_transform(pd.DataFrame(X_train_raw, columns=feature_cols))
    det_safe = RollingZScoreDetector(window_length=3)
    det_safe.fit(X_train_scaled_safe)
    det_safe.set_threshold(3.0)

    alarm_sm_safe = TemporalAlarmStateMachine(AlarmConfig(tau_high=3.0, persistence_k=2, persistence_n=3))
    drift_det_safe = StatisticalDriftDetector(DriftDetectorConfig(threshold=0.40, feature_names=feature_cols))
    ref_mgr_safe = ReferenceManager(strategy=ReferenceStrategy.ANCHORED, initial_data=X_train_scaled_safe)
    reg_safe = ModelRegistry()
    reg_safe.register_initial_model(det_safe, 3.0)

    def model_builder(cand):
        m = RollingZScoreDetector(window_length=3)
        m.fit(cand)
        m.set_threshold(3.0)
        return m

    adapt_sm_safe = AdaptationStateMachine(
        AdaptationConfig(policy=AdaptationPolicy.SAFE, candidate_buffer_size=15, validation_window=10, cooldown_steps=20),
        reg_safe, ref_mgr_safe, 3.0, model_builder=model_builder
    )

    engine_safe = AdaptiveStreamingEngine(
        active_model=det_safe,
        preprocessor=prep_safe,
        alarm_machine=alarm_sm_safe,
        drift_detector=drift_det_safe,
        reference_manager=ref_mgr_safe,
        adaptation_machine=adapt_sm_safe,
        registry=reg_safe,
        history_window_length=3,
        drift_evaluation_window=15,
        feature_names=feature_cols,
    )

    res_safe = engine_safe.stream_trajectory(df_stream[feature_cols], timestamps_s=df_stream["timestamp"].values)

    evaluator = DriftEvaluator()
    m_naive = evaluator.evaluate(res_naive, df_stream, "naive_eval", "contamination_stress", "naive")
    m_safe = evaluator.evaluate(res_safe, df_stream, "safe_eval", "contamination_stress", "safe")

    # Safe adaptation MUST retain superior attack recall and zero contamination
    assert m_safe.recall > m_naive.recall, (
        f"Safe recall ({m_safe.recall:.3f}) should strictly exceed naive recall ({m_naive.recall:.3f})"
    )
    assert m_safe.contamination_events == 0, "Safe adaptation must have zero attack contamination events!"
