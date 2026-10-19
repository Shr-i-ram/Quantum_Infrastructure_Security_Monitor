"""Streaming causal online inference engine with ring buffering and latency profiling (V0.7).

Simulates real-time telemetry streaming:
- Sequential sample-by-sample ingestion: update(sample) -> StreamingResult
- Fixed-capacity circular ring buffer maintaining causal history horizon W_t = [X_{t-L+1}, ..., X_t]
- Guarantees zero access to future observations
- Explicit distinction between:
  * Detection latency: physical delay from attack onset to operational alarm (seconds/pulses)
  * Computational latency: CPU/GPU execution time required to process a single sample (milliseconds)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any
import numpy as np
import pandas as pd

from qism.ml.models.base import AnomalyDetector

from qism.ml.preprocessing.scalers import PreprocessorPipeline
from qism.temporal.alarm import AlarmConfig, AlarmTransitionRecord, OperationalState, TemporalAlarmStateMachine


@dataclass(frozen=True)
class StreamingConfig:
    """Configuration for causal streaming online inference."""
    window_length: int = 5
    feature_dim: int = 0
    warmup_steps: int = 5


@dataclass
class StreamingResult:

    """Result of an incremental streaming telemetry update."""
    step_index: int
    score: float
    raw_alarm: bool
    filtered_alarm: bool
    operational_state: OperationalState
    computational_latency_ms: float
    feature_contributions: np.ndarray | None = None
    timestamp_s: float | None = None
    scenario_id: str = "default"
    run_id: str = "default"


class StreamingInferenceEngine:
    """Stateful, causal online streaming inference engine."""

    def __init__(
        self,
        model: AnomalyDetector,
        preprocessor: PreprocessorPipeline,
        alarm_machine: TemporalAlarmStateMachine,
        history_window_length: int = 5,
        feature_names: list[str] | None = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.alarm_machine = alarm_machine
        self.window_length = max(1, history_window_length)
        self.feature_names = feature_names

        # Causal historical ring buffer
        self._history_buffer: deque[np.ndarray] = deque(maxlen=self.window_length)
        self._step_counter: int = 0
        self.computational_latencies: list[float] = []

    def reset(self) -> None:
        """Reset internal streaming state for a new independent telemetry stream."""
        self._history_buffer.clear()
        self._step_counter = 0
        self.alarm_machine.reset()
        self.computational_latencies.clear()

    def update(
        self,
        sample: np.ndarray | list[float] | dict[str, float],
        timestamp_s: float | None = None,
        scenario_id: str = "default",
        run_id: str = "default",
    ) -> StreamingResult:
        """Process a single incoming telemetry vector X_t causally in real-time.

        Args:
            sample: 1D array of length d (raw observable telemetry features).
            timestamp_s: Simulated timestamp of the sample.
            scenario_id: Scenario identifier.
            run_id: Simulation run identifier.

        Returns:
            StreamingResult containing score, alarm states, and computational latency.
        """
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

        # 2. Apply preprocessing (fit strictly on historical normal baseline)
        scaled_sample = self.preprocessor.transform(sample_2d)[0]  # 1D array of length d

        # 3. Append to causal ring buffer
        self._history_buffer.append(scaled_sample)

        # 4. Construct causal input sequence W_t of length L
        current_len = len(self._history_buffer)
        if current_len < self.window_length:
            # Causal left-padding with first observation of buffer
            pad_count = self.window_length - current_len
            first_obs = np.asarray(self._history_buffer[0], dtype=np.float32)
            pad = np.repeat(first_obs[np.newaxis, :], pad_count, axis=0)
            window_tensor = np.vstack([pad, list(self._history_buffer)])  # (L, d)
        else:
            window_tensor = np.array(self._history_buffer, dtype=np.float32)  # (L, d)


        # 5. Evaluate model score
        # Models accept either 2D (batch, d) or 3D (batch, L, d)
        feat_contrib = None
        has_score_decomp = hasattr(self.model, "score_decomposition")

        if has_score_decomp:
            input_3d = window_tensor.reshape(1, self.window_length, -1)
            scores_batch, contribs_batch = self.model.score_decomposition(input_3d)
            score_val = float(scores_batch[0])
            feat_contrib = contribs_batch[0]
        else:
            # Fallback to standard model.score()
            if hasattr(self.model, "seq_len") or "sequence" in self.model.__class__.__name__.lower() or "temporal" in self.model.__class__.__name__.lower():
                input_3d = window_tensor.reshape(1, self.window_length, -1)
                score_val = float(self.model.score(input_3d)[0])
            else:
                # Point-based model: evaluate current scaled observation
                score_val = float(self.model.score(scaled_sample.reshape(1, -1))[0])

        # 6. Evaluate alarm state machine
        alarm_rec = self.alarm_machine.step(
            score=score_val,
            window_idx=self._step_counter,
            timestamp_s=timestamp_s,
            scenario_id=scenario_id,
            run_id=run_id,
        )

        t_end = time.perf_counter()
        comp_latency_ms = (t_end - t_start) * 1000.0
        self.computational_latencies.append(comp_latency_ms)

        res = StreamingResult(
            step_index=self._step_counter,
            score=score_val,
            raw_alarm=alarm_rec.raw_alarm,
            filtered_alarm=alarm_rec.filtered_alarm,
            operational_state=alarm_rec.state,
            computational_latency_ms=comp_latency_ms,
            feature_contributions=feat_contrib,
            timestamp_s=timestamp_s,
            scenario_id=scenario_id,
            run_id=run_id,
        )

        self._step_counter += 1
        return res

    def stream_trajectory(
        self,
        X_trajectory: np.ndarray | pd.DataFrame,
        timestamps_s: np.ndarray | list[float] | None = None,
        scenario_id: str = "default",
        run_id: str = "default",
    ) -> list[StreamingResult]:
        """Stream an entire continuous trajectory sequentially through the engine."""
        self.reset()
        if isinstance(X_trajectory, pd.DataFrame):
            X_arr = X_trajectory.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X_trajectory, dtype=np.float64)

        results: list[StreamingResult] = []
        n_steps = len(X_arr)

        for i in range(n_steps):
            t_s = float(timestamps_s[i]) if timestamps_s is not None else None
            r = self.update(X_arr[i], timestamp_s=t_s, scenario_id=scenario_id, run_id=run_id)
            results.append(r)

        return results
