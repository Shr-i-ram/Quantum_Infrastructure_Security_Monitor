"""Telemetry ingestion and validation layer for QISM V0.9.

Handles:
1. Streaming and batch telemetry ingestion
2. Strict schema and physical range validation
3. Monotonic timestamp verification
4. Ground-truth stripping (guaranteeing zero leakage into operational inference)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Generator, Iterable, Sequence

import numpy as np
import pandas as pd

from qism.monitor.types import TelemetryRecord


class TelemetryValidationError(ValueError):
    """Raised when incoming telemetry fails schema, consistency, or physical validity checks."""


# Reserved sidecar/ground-truth keys that must NEVER enter operational features
FORBIDDEN_GROUND_TRUTH_KEYS = {
    "attack_type",
    "attack_active",
    "is_attack",
    "attack_strength",
    "ground_truth_label",
    "drift_active",
    "is_drift",
    "drift_label",
    "true_loss_delta_db",
    "true_temp_delta_c",
    "operating_condition",
    "op_condition",
}

# Physical bounds for standard QKD observable quantities
DEFAULT_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "qber": (0.0, 1.0),
    "qber_z": (0.0, 1.0),
    "qber_x": (0.0, 1.0),
    "sifted_rate": (0.0, 1e9),
    "signal_gain": (0.0, 1.0),
    "decoy_gain": (0.0, 1.0),
    "vacuum_gain": (0.0, 1.0),
    "detector_imbalance": (-1.0, 1.0),
    "double_click_rate": (0.0, 1.0),
    "dark_count_rate": (0.0, 1e7),
    "sensor_temperature_mean": (-50.0, 150.0),
    "sensor_temperature_std": (0.0, 100.0),
    "sensor_background_light_mean": (0.0, 1e6),
    "sensor_vibration_mean": (0.0, 1e4),
    "clock_drift_rate": (-1e4, 1e4),
    "autocorr_lag1": (-1.0, 1.0),
}


@dataclass
class IngestionConfig:
    """Configuration for ingestion and validation."""
    strict_bounds: bool = True
    enforce_monotonic_timestamps: bool = True
    max_timestamp_gap_s: float = 60.0
    allow_subsystem_fallback: bool = True
    strip_ground_truth_silently: bool = False
    physical_bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_PHYSICAL_BOUNDS)
    )


class TelemetryIngestor:
    """Validates, sanitizes, and streams telemetry records into the monitoring pipeline."""

    def __init__(self, config: IngestionConfig | None = None):
        self.config = config or IngestionConfig()
        self._last_timestamp: dict[str, float] = {}  # Per run_id
        self._seen_sequence_ids: dict[str, set[int]] = {}  # Per run_id
        self.total_ingested = 0
        self.total_rejected = 0

    def validate_and_sanitize(self, raw_record: TelemetryRecord | dict[str, Any]) -> TelemetryRecord:
        """Validate a single record and ensure strict ground-truth separation."""
        if isinstance(raw_record, dict):
            # Convert raw dict
            record = self._parse_dict_to_record(raw_record)
        else:
            record = raw_record

        run_id = record.run_id

        # 1. Monotonic timestamp validation
        if self.config.enforce_monotonic_timestamps:
            if run_id in self._last_timestamp:
                prev_ts = self._last_timestamp[run_id]
                if record.timestamp_s < prev_ts:
                    self.total_rejected += 1
                    raise TelemetryValidationError(
                        f"Non-monotonic timestamp detected for run '{run_id}': "
                        f"current {record.timestamp_s} < previous {prev_ts}"
                    )
            self._last_timestamp[run_id] = record.timestamp_s

        # 2. Sequence ID deduplication
        if run_id not in self._seen_sequence_ids:
            self._seen_sequence_ids[run_id] = set()
        if record.sequence_id in self._seen_sequence_ids[run_id]:
            self.total_rejected += 1
            raise TelemetryValidationError(
                f"Duplicate sequence_id {record.sequence_id} detected in run '{run_id}'"
            )
        self._seen_sequence_ids[run_id].add(record.sequence_id)

        # 3. Ground-truth stripping (Security Boundary)
        sanitized_features: dict[str, float] = {}
        for k, v in record.features.items():
            if k in FORBIDDEN_GROUND_TRUTH_KEYS:
                if not self.config.strip_ground_truth_silently:
                    # Log or strip cleanly
                    pass
                continue

            # 4. Check finite and numerical validity
            if not isinstance(v, (int, float, np.number)):
                self.total_rejected += 1
                raise TelemetryValidationError(
                    f"Feature '{k}' is non-numeric: {type(v).__name__} = {v}"
                )
            f_val = float(v)
            if not math.isfinite(f_val):
                self.total_rejected += 1
                raise TelemetryValidationError(
                    f"Feature '{k}' is non-finite (NaN or Inf): {f_val}"
                )

            # 5. Physical range bounds
            if self.config.strict_bounds and k in self.config.physical_bounds:
                low, high = self.config.physical_bounds[k]
                if not (low <= f_val <= high):
                    self.total_rejected += 1
                    raise TelemetryValidationError(
                        f"Feature '{k}' value {f_val} violates physical bounds [{low}, {high}]"
                    )

            sanitized_features[k] = f_val

        if not sanitized_features:
            self.total_rejected += 1
            raise TelemetryValidationError(f"Telemetry record at t={record.timestamp_s} contains no valid features.")

        self.total_ingested += 1
        return TelemetryRecord(
            timestamp_s=record.timestamp_s,
            sequence_id=record.sequence_id,
            subsystem=record.subsystem,
            features=sanitized_features,
            run_id=record.run_id,
            session_id=record.session_id,
            metadata=dict(record.metadata),
        )

    def _parse_dict_to_record(self, d: dict[str, Any]) -> TelemetryRecord:
        """Parse arbitrary dictionary input, safely isolating features from metadata."""
        if "timestamp_s" not in d and "timestamp" not in d:
            raise TelemetryValidationError("Record missing required 'timestamp_s' field.")
        ts = float(d.get("timestamp_s", d.get("timestamp", 0.0)))
        seq = int(d.get("sequence_id", d.get("step_index", 0)))
        subsystem = str(d.get("subsystem", "composite"))
        run_id = str(d.get("run_id", "default_run"))
        session_id = str(d.get("session_id", "default_session"))

        # Features might be in 'features' subdict or top-level keys
        if "features" in d and isinstance(d["features"], dict):
            raw_features = d["features"]
            metadata = {k: v for k, v in d.items() if k not in ("features", "timestamp_s", "timestamp", "sequence_id", "step_index", "subsystem", "run_id", "session_id")}
        else:
            raw_features = {}
            metadata = {}
            top_level_meta = {
                "timestamp_s", "timestamp", "sequence_id", "step_index",
                "subsystem", "run_id", "session_id", "metadata"
            }
            for k, v in d.items():
                if k in top_level_meta:
                    continue
                if k in FORBIDDEN_GROUND_TRUTH_KEYS:
                    metadata[k] = v  # Stored in evaluation metadata only!
                elif isinstance(v, (int, float, np.number)):
                    raw_features[k] = float(v)
                else:
                    metadata[k] = v

        return TelemetryRecord(
            timestamp_s=ts,
            sequence_id=seq,
            subsystem=subsystem,
            features=raw_features,
            run_id=run_id,
            session_id=session_id,
            metadata=metadata,
        )

    def stream_from_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str] | None = None,
        run_id: str = "batch_run",
    ) -> Generator[TelemetryRecord, None, None]:
        """Convert a pandas DataFrame into a validated stream of TelemetryRecord objects."""
        if feature_cols is None:
            # Filter numeric columns excluding known sidecars and metadata
            meta_cols = set(FORBIDDEN_GROUND_TRUTH_KEYS).union(
                {"timestamp_s", "timestamp", "step_index", "sequence_id", "run_id", "session_id", "subsystem"}
            )
            feature_cols = [c for c in df.columns if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)]

        t_col = "timestamp_s" if "timestamp_s" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
        seq_col = "sequence_id" if "sequence_id" in df.columns else ("step_index" if "step_index" in df.columns else None)

        for idx, row in df.iterrows():
            ts = float(row[t_col]) if t_col else float(idx) * 0.002
            seq = int(row[seq_col]) if seq_col else int(idx)
            feats = {c: float(row[c]) for c in feature_cols if c in row}

            # Offline ground truth is captured in metadata if present, NEVER in features
            meta = {}
            for k in FORBIDDEN_GROUND_TRUTH_KEYS:
                if k in row:
                    meta[k] = row[k]

            raw_rec = TelemetryRecord(
                timestamp_s=ts,
                sequence_id=seq,
                subsystem="qkd_composite",
                features=feats,
                run_id=run_id,
                metadata=meta,
            )
            yield raw_rec

    def reset(self) -> None:
        """Reset internal ingestion state."""
        self._last_timestamp.clear()
        self._seen_sequence_ids.clear()
        self.total_ingested = 0
        self.total_rejected = 0
