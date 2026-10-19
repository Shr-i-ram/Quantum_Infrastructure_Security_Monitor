"""Temporal window construction, boundary enforcement, and multi-strategy labeling (V0.7).

Converts 2D telemetry time-series into strictly causal sliding windows W_t = [X_{t-L+1}, ..., X_t]
without bridging across scenario or run boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
import numpy as np
import pandas as pd


class LabelingStrategy(str, Enum):
    """Supported labeling strategies for temporal windows."""
    TERMINAL = "terminal"     # State at final timestep t (causal online)
    ANY = "any"               # True if ANY sample in the window is anomalous
    MAJORITY = "majority"     # True if fraction >= threshold (default 0.5)
    FRACTION = "fraction"     # Store continuous attack overlap ratio r in [0, 1]


@dataclass(frozen=True)
class TemporalWindowConfig:
    """Configuration for temporal window construction."""
    window_length: int = 5      # Number of telemetry steps L in each sequence
    stride: int = 1             # Stride between consecutive windows
    min_window_length: int = 2  # Minimum required length (if partial/padded)
    causal_padding: bool = True # If True, pad start of sequence with initial observation
    labeling_strategy: LabelingStrategy = LabelingStrategy.TERMINAL
    majority_threshold: float = 0.5


@dataclass
class WindowMetadata:
    """Provenance and boundary metadata for a generated temporal window."""
    scenario_id: str
    run_id: str
    start_index: int
    end_index: int
    start_pulse: int | None = None
    end_pulse: int | None = None
    start_time_s: float | None = None
    end_time_s: float | None = None
    regime: str = "nominal"
    attack_fraction: float = 0.0
    attack_type: str = "none"
    attack_phase: str = "normal"
    mean_severity: float = 0.0
    is_anomalous: bool = False


class TemporalWindowBuilder:
    """Constructs boundary-aware temporal window tensors and metadata sidecars."""

    def __init__(self, config: TemporalWindowConfig | None = None):
        self.config = config or TemporalWindowConfig()

    def build_windows(
        self,
        X_df: pd.DataFrame,
        y_df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, list[WindowMetadata], np.ndarray]:
        """Convert telemetry dataframe into 3D window tensor and metadata.

        Args:
            X_df: 2D numerical observable feature DataFrame (shape N x d).
            y_df: Metadata and ground-truth sidecar DataFrame (shape N x k).

        Returns:
            X_windows: 3D numpy array of shape (n_windows, L, d).
            metadata: List of WindowMetadata objects of length n_windows.
            y_labels: 1D binary label array of length n_windows (0 or 1).
        """
        if isinstance(X_df, np.ndarray):
            if X_df.size == 0:
                d = X_df.shape[1] if X_df.ndim == 2 else 0
                return (
                    np.zeros((0, self.config.window_length, d), dtype=np.float32),
                    [],
                    np.zeros(0, dtype=np.int64),
                )
            cols = [f"f_{i}" for i in range(X_df.shape[1])]
            X_df = pd.DataFrame(X_df, columns=cols)
        elif X_df.empty:
            d = X_df.shape[1] if X_df.ndim == 2 else 0
            return (
                np.zeros((0, self.config.window_length, d), dtype=np.float32),
                [],
                np.zeros(0, dtype=np.int64),
            )

        L = self.config.window_length
        stride = max(1, self.config.stride)
        feature_cols = list(X_df.columns)
        d = len(feature_cols)


        # Identify run boundary grouping columns
        # In QISM, unique trajectory identity requires BOTH scenario_id and run_id
        has_meta = y_df is not None and not y_df.empty
        if has_meta and "scenario_id" in y_df.columns and "run_id" in y_df.columns:
            group_keys = list(zip(y_df["scenario_id"].astype(str), y_df["run_id"].astype(str)))
        elif "scenario_id" in X_df.columns and "run_id" in X_df.columns:
            group_keys = list(zip(X_df["scenario_id"].astype(str), X_df["run_id"].astype(str)))
        else:
            group_keys = [("default_scenario", "default_run")] * len(X_df)

        # Find contiguous trajectory boundaries
        trajectories: list[tuple[tuple[str, str], list[int]]] = []
        current_key = None
        current_indices: list[int] = []

        for idx, key in enumerate(group_keys):
            if key != current_key:
                if current_indices:
                    trajectories.append((current_key, current_indices))
                current_key = key
                current_indices = [idx]
            else:
                current_indices.append(idx)
        if current_indices:
            trajectories.append((current_key, current_indices))

        all_windows: list[np.ndarray] = []
        all_metadata: list[WindowMetadata] = []
        all_labels: list[int] = []

        X_values = X_df.to_numpy(dtype=np.float32)

        for (scen_id, run_id), indices in trajectories:
            n_samples = len(indices)
            if n_samples == 0:
                continue

            traj_X = X_values[indices]

            # Extract ground-truth series for this trajectory if available
            traj_atk_active: np.ndarray | None = None
            traj_atk_type: np.ndarray | None = None
            traj_atk_phase: np.ndarray | None = None
            traj_atk_sev: np.ndarray | None = None
            traj_start_pulse: np.ndarray | None = None
            traj_end_pulse: np.ndarray | None = None

            if has_meta:
                sub_y = y_df.iloc[indices]
                if "attack_active" in sub_y.columns:
                    traj_atk_active = sub_y["attack_active"].to_numpy(dtype=bool)
                elif "ground_truth_attack_active" in sub_y.columns:
                    traj_atk_active = sub_y["ground_truth_attack_active"].to_numpy(dtype=bool)

                if "attack_type" in sub_y.columns:
                    traj_atk_type = sub_y["attack_type"].to_numpy(dtype=str)
                if "attack_phase" in sub_y.columns:
                    traj_atk_phase = sub_y["attack_phase"].to_numpy(dtype=str)
                if "attack_severity" in sub_y.columns:
                    traj_atk_sev = sub_y["attack_severity"].to_numpy(dtype=float)
                if "window_start_pulse" in sub_y.columns:
                    traj_start_pulse = sub_y["window_start_pulse"].to_numpy()
                if "window_end_pulse" in sub_y.columns:
                    traj_end_pulse = sub_y["window_end_pulse"].to_numpy()

            # Generate windows within this single trajectory
            # For causal streaming, window t ends at index t and looks back L steps
            if self.config.causal_padding:
                # Every observation in the trajectory generates a window
                window_end_positions = range(0, n_samples, stride)
            else:
                # Only valid full windows (no padding)
                if n_samples < L:
                    continue
                window_end_positions = range(L - 1, n_samples, stride)

            for end_pos in window_end_positions:
                start_pos = end_pos - L + 1
                if start_pos < 0:
                    # Causal padding with initial observation
                    pad_count = -start_pos
                    valid_sub = traj_X[0 : end_pos + 1]
                    padding = np.repeat(valid_sub[:1], pad_count, axis=0)
                    window_tensor = np.vstack([padding, valid_sub])
                    active_slice_start = 0
                else:
                    window_tensor = traj_X[start_pos : end_pos + 1]
                    active_slice_start = start_pos

                # Ground truth labeling across the active slice
                if traj_atk_active is not None:
                    atk_sub = traj_atk_active[active_slice_start : end_pos + 1]
                    n_atk = int(np.sum(atk_sub))
                    frac = float(n_atk / len(atk_sub)) if len(atk_sub) > 0 else 0.0
                    terminal_atk = bool(traj_atk_active[end_pos])
                    any_atk = bool(n_atk > 0)
                    majority_atk = bool(frac >= self.config.majority_threshold)

                    # Determine binary label according to configured strategy
                    strat = self.config.labeling_strategy
                    if strat == LabelingStrategy.TERMINAL:
                        is_anom = terminal_atk
                    elif strat == LabelingStrategy.ANY:
                        is_anom = any_atk
                    elif strat == LabelingStrategy.MAJORITY:
                        is_anom = majority_atk
                    else:  # FRACTION or other
                        is_anom = majority_atk

                    # Determine dominant attack type & phase
                    if n_atk > 0 and traj_atk_type is not None:
                        type_sub = traj_atk_type[active_slice_start : end_pos + 1]
                        valid_types = [t for t in type_sub if t != "none"]
                        dom_type = valid_types[-1] if valid_types else "none"
                    else:
                        dom_type = "none"

                    dom_phase = str(traj_atk_phase[end_pos]) if traj_atk_phase is not None else "normal"
                    mean_sev = float(np.mean(traj_atk_sev[active_slice_start : end_pos + 1])) if traj_atk_sev is not None else 0.0
                else:
                    frac = 0.0
                    is_anom = False
                    dom_type = "none"
                    dom_phase = "normal"
                    mean_sev = 0.0

                s_pulse = int(traj_start_pulse[active_slice_start]) if traj_start_pulse is not None else None
                e_pulse = int(traj_end_pulse[end_pos]) if traj_end_pulse is not None else None

                meta = WindowMetadata(
                    scenario_id=scen_id,
                    run_id=run_id,
                    start_index=active_slice_start,
                    end_index=end_pos,
                    start_pulse=s_pulse,
                    end_pulse=e_pulse,
                    attack_fraction=frac,
                    attack_type=dom_type,
                    attack_phase=dom_phase,
                    mean_severity=mean_sev,
                    is_anomalous=is_anom,
                )

                all_windows.append(window_tensor)
                all_metadata.append(meta)
                all_labels.append(1 if is_anom else 0)

        if not all_windows:
            return (
                np.zeros((0, L, d), dtype=np.float32),
                [],
                np.zeros(0, dtype=np.int64),
            )

        X_out = np.stack(all_windows, axis=0).astype(np.float32)
        y_out = np.array(all_labels, dtype=np.int64)

        return X_out, all_metadata, y_out

    @staticmethod
    def to_metadata_dataframe(metadata_list: list[WindowMetadata]) -> pd.DataFrame:
        """Convert list of WindowMetadata objects into a pandas DataFrame."""
        if not metadata_list:
            return pd.DataFrame()
        records = [
            {
                "scenario_id": m.scenario_id,
                "run_id": m.run_id,
                "start_index": m.start_index,
                "end_index": m.end_index,
                "start_pulse": m.start_pulse,
                "end_pulse": m.end_pulse,
                "start_time_s": m.start_time_s,
                "end_time_s": m.end_time_s,
                "regime": m.regime,
                "attack_fraction": m.attack_fraction,
                "attack_type": m.attack_type,
                "attack_phase": m.attack_phase,
                "attack_severity": m.mean_severity,
                "mean_severity": m.mean_severity,
                "is_anomalous": m.is_anomalous,
                "window_start_pulse": m.start_pulse,
                "window_end_pulse": m.end_pulse,
            }
            for m in metadata_list
        ]
        return pd.DataFrame(records)

