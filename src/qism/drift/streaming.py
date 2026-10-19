"""Adaptive online streaming inference engine with drift tracking and safe adaptation (V0.8).

Extends the V0.7 streaming architecture with:
1. Online multivariate statistical drift detection (KS, Wasserstein, PSI)
2. Configurable reference window management (fixed, sliding, anchored)
3. Safe adaptation state machine with anomaly suppression and shadow validation
4. Model/reference versioning registry and rollback capabilities
5. Real-time per-step latency profiling (sub-millisecond streaming)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from qism.drift.adaptation import (
    AdaptationConfig,
    AdaptationDecision,
    AdaptationPolicy,
    AdaptationState,
    AdaptationStateMachine,
    AdaptationStepResult,
)
from qism.drift.detectors import (
    DriftDetectorConfig,
    DriftMetric,
    DriftReport,
    StatisticalDriftDetector,
)
from qism.drift.reference import ReferenceManager, ReferenceStrategy
from qism.drift.registry import ModelRegistry, ModelStatus
from qism.ml.models.base import AnomalyDetector
from qism.ml.preprocessing.scalers import PreprocessorPipeline
from qism.temporal.alarm import AlarmConfig, OperationalState, TemporalAlarmStateMachine


@dataclass
class AdaptiveStreamingResult:
    """Comprehensive output produced at each sequential streaming step."""
    step_index: int
    timestamp_s: float | None
    anomaly_score: float
    anomaly_threshold: float
    raw_alarm: bool
    filtered_alarm: bool
    operational_state: OperationalState
    drift_score: float
    drift_threshold: float
    is_drift: bool
    dominant_drift_feature: str
    adaptation_state: AdaptationState
    adaptation_decision: AdaptationDecision
    active_model_version: str
    reference_version: str
    candidate_version: str | None
    computational_latency_ms: float
    reason: str = ""


class AdaptiveStreamingEngine:
    """Stateful, causal adaptive streaming inference engine."""

    def __init__(
        self,
        active_model: AnomalyDetector,
        preprocessor: PreprocessorPipeline,
        alarm_machine: TemporalAlarmStateMachine,
        drift_detector: StatisticalDriftDetector,
        reference_manager: ReferenceManager,
        adaptation_machine: AdaptationStateMachine,
        registry: ModelRegistry,
        history_window_length: int = 5,
        drift_evaluation_window: int = 20,
        feature_names: list[str] | None = None,
    ):
        self.active_model = active_model
        self.preprocessor = preprocessor
        self.alarm_machine = alarm_machine
        self.drift_detector = drift_detector
        self.reference_manager = reference_manager
        self.adaptation_machine = adaptation_machine
        self.registry = registry

        self.history_window_length = max(1, history_window_length)
        self.drift_window_length = max(10, drift_evaluation_window)
        self.feature_names = feature_names

        # Causal historical buffers
        self._anomaly_ring_buffer: deque[np.ndarray] = deque(maxlen=self.history_window_length)
        self._drift_ring_buffer: deque[np.ndarray] = deque(maxlen=self.drift_window_length)

        self._step_counter: int = 0
        self.computational_latencies: list[float] = []

    def reset(self) -> None:
        """Reset internal engine state for a new independent simulation stream."""
        self._anomaly_ring_buffer.clear()
        self._drift_ring_buffer.clear()
        self._step_counter = 0
        self.alarm_machine.reset()
        self.adaptation_machine.reset()
        self.computational_latencies.clear()

    def update(
        self,
        sample: np.ndarray | list[float] | dict[str, float],
        timestamp_s: float | None = None,
        scenario_id: str = "default",
        run_id: str = "default",
    ) -> AdaptiveStreamingResult:
        """Ingest a single telemetry observation X_t causally in real time."""
        t_start = time.perf_counter()

        # 1. Format raw sample into 2D array (1, d)
        if isinstance(sample, dict):
            if self.feature_names:
                sample_arr = np.array([sample.get(f, 0.0) for f in self.feature_names], dtype=np.float64)
            else:
                sample_arr = np.array(list(sample.values()), dtype=np.float64)
        else:
            sample_arr = np.asarray(sample, dtype=np.float64)

        if sample_arr.ndim == 1:
            sample_2d = sample_arr.reshape(1, -1)
        else:
            sample_2d = sample_arr

        # 2. Scale sample using preprocessor
        scaled_sample = self.preprocessor.transform(sample_2d)[0]  # (d,)

        # 3. Append to causal buffers
        self._anomaly_ring_buffer.append(scaled_sample)
        self._drift_ring_buffer.append(scaled_sample)

        # 4. Construct anomaly input sequence W_t of length L
        current_len = len(self._anomaly_ring_buffer)
        if current_len < self.history_window_length:
            pad_count = self.history_window_length - current_len
            first_obs = np.asarray(self._anomaly_ring_buffer[0], dtype=np.float32)
            pad = np.repeat(first_obs[np.newaxis, :], pad_count, axis=0)
            window_tensor = np.vstack([pad, list(self._anomaly_ring_buffer)])
        else:
            window_tensor = np.array(self._anomaly_ring_buffer, dtype=np.float32)

        # 5. Anomaly evaluation via active model
        # Use active model from registry if available
        curr_active_model = self.registry.get_active_model() or self.active_model
        if hasattr(curr_active_model, "seq_len") or "sequence" in curr_active_model.__class__.__name__.lower() or "temporal" in curr_active_model.__class__.__name__.lower():
            input_3d = window_tensor.reshape(1, self.history_window_length, -1)
            score_val = float(curr_active_model.score(input_3d)[0])
        else:
            score_val = float(curr_active_model.score(scaled_sample.reshape(1, -1))[0])

        active_threshold = getattr(curr_active_model, "threshold", self.alarm_machine.tau_high) or self.alarm_machine.tau_high
        self.alarm_machine.tau_high = float(active_threshold)
        if self.alarm_machine.tau_low is not None:
            self.alarm_machine.tau_low = float(active_threshold * 0.90)

        # 6. Evaluate operational alarm state machine (persistence & hysteresis)
        alarm_rec = self.alarm_machine.step(
            score=score_val,
            window_idx=self._step_counter,
            timestamp_s=timestamp_s,
            scenario_id=scenario_id,
            run_id=run_id,
        )

        # 7. Evaluate statistical drift detector against active reference
        ref_samples = self.reference_manager.get_reference_samples()
        curr_drift_samples = np.array(self._drift_ring_buffer, dtype=np.float64)

        drift_report = self.drift_detector.compute_drift(
            reference_samples=ref_samples,
            current_samples=curr_drift_samples,
            step_index=self._step_counter,
            timestamp_s=timestamp_s,
        )

        dominant_feat = drift_report.dominant_features[0] if drift_report.dominant_features else "none"

        # 8. Safe adaptation gating & state machine
        adapt_res = self.adaptation_machine.step(
            sample=scaled_sample,
            anomaly_score=score_val,
            operational_alarm=alarm_rec.filtered_alarm,
            drift_report=drift_report,
            step_index=self._step_counter,
            timestamp_s=timestamp_s,
        )

        # 9. Reference ingestion (sliding mode update with anti-contamination safeguard)
        self.reference_manager.ingest_sample(
            sample=scaled_sample,
            anomaly_score=score_val,
            is_operational_alarm=alarm_rec.filtered_alarm,
            step_index=self._step_counter,
            timestamp_s=timestamp_s,
        )

        t_end = time.perf_counter()
        comp_latency_ms = (t_end - t_start) * 1000.0
        self.computational_latencies.append(comp_latency_ms)

        result = AdaptiveStreamingResult(
            step_index=self._step_counter,
            timestamp_s=timestamp_s,
            anomaly_score=score_val,
            anomaly_threshold=active_threshold,
            raw_alarm=alarm_rec.raw_alarm,
            filtered_alarm=alarm_rec.filtered_alarm,
            operational_state=alarm_rec.state,
            drift_score=drift_report.drift_score,
            drift_threshold=drift_report.threshold,
            is_drift=drift_report.is_drift,
            dominant_drift_feature=dominant_feat,
            adaptation_state=adapt_res.state,
            adaptation_decision=adapt_res.decision,
            active_model_version=self.registry.active_version_id,
            reference_version=self.reference_manager.current_version_id,
            candidate_version=self.registry.candidate_version_id,
            computational_latency_ms=comp_latency_ms,
            reason=adapt_res.reason,
        )

        self._step_counter += 1
        return result

    def stream_trajectory(
        self,
        X_trajectory: np.ndarray | pd.DataFrame,
        timestamps_s: np.ndarray | list[float] | None = None,
        scenario_id: str = "default",
        run_id: str = "default",
    ) -> list[AdaptiveStreamingResult]:
        """Stream an entire continuous trajectory sequentially through the engine."""
        self.reset()
        if isinstance(X_trajectory, pd.DataFrame):
            X_arr = X_trajectory.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X_trajectory, dtype=np.float64)

        results: list[AdaptiveStreamingResult] = []
        n_steps = len(X_arr)

        for i in range(n_steps):
            t_s = float(timestamps_s[i]) if timestamps_s is not None else None
            res = self.update(X_arr[i], timestamp_s=t_s, scenario_id=scenario_id, run_id=run_id)
            results.append(res)

        return results

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Persist complete streaming state, reference, and registry to disk."""
        chk_dir = Path(checkpoint_path)
        chk_dir.mkdir(parents=True, exist_ok=True)

        self.registry.save_checkpoint(chk_dir / "registry_checkpoint.json")

        state = {
            "step_counter": self._step_counter,
            "anomaly_ring_buffer": [arr.tolist() for arr in self._anomaly_ring_buffer],
            "drift_ring_buffer": [arr.tolist() for arr in self._drift_ring_buffer],
            "active_version_id": self.registry.active_version_id,
            "reference_version_id": self.reference_manager.current_version_id,
            "adaptation_state": self.adaptation_machine.state.value,
            "reference_strategy": self.reference_manager.strategy.value,
        }
        with open(chk_dir / "engine_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Reload streaming state, reference, and registry from disk."""
        chk_dir = Path(checkpoint_path)
        with open(chk_dir / "engine_state.json", "r", encoding="utf-8") as f:
            state = json.load(f)

        self._step_counter = state["step_counter"]
        self._anomaly_ring_buffer = deque([np.array(x, dtype=np.float32) for x in state["anomaly_ring_buffer"]], maxlen=self.history_window_length)
        self._drift_ring_buffer = deque([np.array(x, dtype=np.float64) for x in state["drift_ring_buffer"]], maxlen=self.drift_window_length)
        self.adaptation_machine.state = AdaptationState(state["adaptation_state"])
