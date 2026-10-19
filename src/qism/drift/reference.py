"""Reference window management and contamination protection (V0.8).

Manages baseline and operating reference distributions:
- FIXED: Frozen initial reference distribution (security baseline).
- SLIDING: Slowly moving window over trusted normal observations.
- ANCHORED: Stationary reference that updates only after validated adaptation (safest).
- Contamination Safeguard: Strictly prevents anomalous or attack telemetry from entering reference.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any

import numpy as np


class ReferenceStrategy(str, Enum):
    """Reference window operational strategy."""
    FIXED = "fixed"
    SLIDING = "sliding"
    ANCHORED = "anchored"


@dataclass
class ReferenceMetadata:
    """Provenance and metadata for a reference distribution snapshot."""
    version_id: str
    parent_version_id: str | None
    created_at_step: int
    created_at_time_s: float | None
    sample_count: int
    feature_dim: int
    strategy: ReferenceStrategy
    mean_vector: list[float]
    std_vector: list[float]
    contamination_rejections: int = 0


class ReferenceManager:
    """Manages baseline and adaptive reference distributions with contamination gating."""

    def __init__(
        self,
        strategy: ReferenceStrategy = ReferenceStrategy.ANCHORED,
        window_size: int = 50,
        safe_anomaly_threshold: float = 1.0,
        initial_data: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ):
        self.strategy = strategy
        self.window_size = max(10, window_size)
        self.safe_anomaly_threshold = float(safe_anomaly_threshold)
        self.feature_names = feature_names

        # Active reference state
        self._reference_data: np.ndarray | None = None
        self._sliding_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._candidate_buffer: list[np.ndarray] = []

        # Versioning and lineage
        self.current_version_id: str = "ref_v001"
        self._version_history: dict[str, tuple[np.ndarray, ReferenceMetadata]] = {}
        self.contamination_rejections: int = 0
        self.updates_count: int = 0

        if initial_data is not None:
            self.initialize_reference(initial_data, version_id="ref_v001")

    def initialize_reference(
        self,
        baseline_data: np.ndarray,
        version_id: str = "ref_v001",
        step_index: int = 0,
        timestamp_s: float | None = 0.0,
    ) -> ReferenceMetadata:
        """Initialize primary reference distribution from trusted calibration telemetry."""
        arr = np.asarray(baseline_data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        # Retain up to window_size samples
        if len(arr) > self.window_size:
            arr = arr[-self.window_size :]

        self._reference_data = arr.copy()
        self._sliding_buffer.clear()
        for row in self._reference_data:
            self._sliding_buffer.append(row)

        self.current_version_id = version_id
        meta = self._build_metadata(version_id, None, step_index, timestamp_s, self._reference_data)
        self._version_history[version_id] = (self._reference_data.copy(), meta)
        return meta

    def get_reference_samples(self) -> np.ndarray:
        """Retrieve current active reference samples."""
        if self._reference_data is None:
            raise RuntimeError("Reference distribution has not been initialized.")
        if self.strategy == ReferenceStrategy.SLIDING:
            return np.array(self._sliding_buffer, dtype=np.float64)
        return self._reference_data

    def ingest_sample(
        self,
        sample: np.ndarray,
        anomaly_score: float,
        is_operational_alarm: bool,
        step_index: int,
        timestamp_s: float | None = None,
    ) -> bool:
        """Attempt to ingest incoming sample into sliding or candidate reference buffer.

        CRITICAL SECURITY INVARIANT:
        If anomaly_score >= safe_anomaly_threshold OR is_operational_alarm is True:
        Sample is strictly rejected from reference updates to prevent attack poisoning.

        Returns:
            True if sample was accepted as clean normal telemetry, False if rejected.
        """
        sample_arr = np.asarray(sample, dtype=np.float64).flatten()

        # Security contamination gate
        if is_operational_alarm or anomaly_score >= self.safe_anomaly_threshold:
            self.contamination_rejections += 1
            return False

        if self.strategy == ReferenceStrategy.SLIDING:
            self._sliding_buffer.append(sample_arr)
            return True

        # In ANCHORED or FIXED mode, samples accumulate in candidate buffer only when requested
        return True

    def buffer_candidate_sample(
        self,
        sample: np.ndarray,
        anomaly_score: float,
        is_operational_alarm: bool,
    ) -> bool:
        """Buffer a sample strictly for candidate adaptation validation.

        Returns False if rejected by contamination safeguard.
        """
        sample_arr = np.asarray(sample, dtype=np.float64).flatten()
        if is_operational_alarm or anomaly_score >= self.safe_anomaly_threshold:
            self.contamination_rejections += 1
            return False
        self._candidate_buffer.append(sample_arr)
        return True

    def clear_candidate_buffer(self) -> None:
        """Clear candidate accumulation buffer."""
        self._candidate_buffer.clear()

    def get_candidate_samples(self) -> np.ndarray:
        """Retrieve accumulated candidate samples."""
        if not self._candidate_buffer:
            return np.empty((0, self._reference_data.shape[1] if self._reference_data is not None else 0))
        return np.array(self._candidate_buffer, dtype=np.float64)

    def promote_candidate(
        self,
        candidate_data: np.ndarray | None = None,
        new_version_id: str | None = None,
        step_index: int = 0,
        timestamp_s: float | None = None,
    ) -> ReferenceMetadata:
        """Promote validated candidate samples to become active reference."""
        if candidate_data is None:
            if not self._candidate_buffer:
                raise ValueError("No candidate samples available for promotion.")
            candidate_data = np.array(self._candidate_buffer, dtype=np.float64)

        if len(candidate_data) < 5:
            raise ValueError(f"Insufficient candidate samples ({len(candidate_data)}) for reference promotion.")

        parent_id = self.current_version_id
        self.updates_count += 1
        new_vid = new_version_id or f"ref_v{self.updates_count + 1:03d}"

        # Bound to window_size
        if len(candidate_data) > self.window_size:
            candidate_data = candidate_data[-self.window_size :]

        self._reference_data = candidate_data.copy()
        self._sliding_buffer.clear()
        for row in self._reference_data:
            self._sliding_buffer.append(row)

        self.current_version_id = new_vid
        meta = self._build_metadata(new_vid, parent_id, step_index, timestamp_s, self._reference_data)
        self._version_history[new_vid] = (self._reference_data.copy(), meta)
        self._candidate_buffer.clear()
        return meta

    def rollback(self, target_version_id: str | None = None) -> ReferenceMetadata:
        """Rollback active reference to a previously validated version."""
        if not self._version_history:
            raise RuntimeError("No version history available for rollback.")

        if target_version_id is None:
            # Revert to immediate parent
            curr_meta = self._version_history[self.current_version_id][1]
            target_version_id = curr_meta.parent_version_id
            if target_version_id is None:
                # If no parent, stay at current (initial)
                target_version_id = self.current_version_id

        if target_version_id not in self._version_history:
            raise KeyError(f"Target version {target_version_id} not found in reference history.")

        data, meta = self._version_history[target_version_id]
        self._reference_data = data.copy()
        self._sliding_buffer.clear()
        for row in self._reference_data:
            self._sliding_buffer.append(row)
        self.current_version_id = target_version_id
        self._candidate_buffer.clear()
        return meta

    def _build_metadata(
        self,
        vid: str,
        parent_id: str | None,
        step: int,
        t_s: float | None,
        data: np.ndarray,
    ) -> ReferenceMetadata:
        return ReferenceMetadata(
            version_id=vid,
            parent_version_id=parent_id,
            created_at_step=step,
            created_at_time_s=t_s,
            sample_count=len(data),
            feature_dim=data.shape[1] if data.ndim > 1 else 1,
            strategy=self.strategy,
            mean_vector=[float(x) for x in np.mean(data, axis=0)],
            std_vector=[float(x) for x in np.std(data, axis=0)],
            contamination_rejections=self.contamination_rejections,
        )
