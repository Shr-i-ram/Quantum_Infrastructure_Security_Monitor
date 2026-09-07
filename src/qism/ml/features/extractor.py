"""Feature extraction, causal rolling window transforms, and strict zero-leakage guards (Requirements 24–31, 39–45, 112, 243, 244)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qism.ml.features.definitions import (
    GROUND_TRUTH_FIELDS,
    GROUND_TRUTH_PREFIXES,
    METADATA_FIELDS,
    get_feature_set,
    is_ground_truth_field,
)


class FeatureExtractor:
    """Extracts observable telemetry features while guaranteeing zero ground-truth leakage."""

    def __init__(
        self,
        feature_set: str = "full",
        add_temporal_features: bool = False,
        rolling_window_length: int = 3,
    ):
        self.feature_set_name = feature_set
        self.base_feature_names = list(get_feature_set(feature_set))
        self.add_temporal_features = add_temporal_features
        self.rolling_window_length = rolling_window_length

        # Assert no ground-truth fields are in the requested feature set
        for f in self.base_feature_names:
            if is_ground_truth_field(f):
                raise ValueError(f"Ground-truth leakage! Field '{f}' is forbidden from observable features.")
            if f in METADATA_FIELDS:
                raise ValueError(f"Metadata field '{f}' cannot be used as a numerical feature.")

    def get_feature_names(self) -> list[str]:
        """Return full list of extracted feature column names."""
        names = list(self.base_feature_names)
        if self.add_temporal_features:
            for bf in self.base_feature_names:
                names.append(f"{bf}_rolling_mean_{self.rolling_window_length}")
                names.append(f"{bf}_rolling_std_{self.rolling_window_length}")
                names.append(f"{bf}_diff_1")
        return names

    def extract(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract (X_observable, y_ground_truth_and_metadata).

        Guarantees:
        - X contains ONLY observable telemetry features.
        - Zero columns with ground-truth prefixes or forbidden names exist in X.
        - Causal temporal transforms strictly use past observations (x <= t).
        """
        if df.empty:
            empty_x = pd.DataFrame(columns=self.get_feature_names())
            empty_y = pd.DataFrame()
            return empty_x, empty_y

        # 1. Verify and extract base features
        feature_dict: dict[str, Any] = {}
        for f in self.base_feature_names:
            if f in df.columns:
                feature_dict[f] = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=np.float64)
            else:
                feature_dict[f] = np.full(len(df), np.nan, dtype=np.float64)

        base_df = pd.DataFrame(feature_dict, index=df.index)

        # 2. Add causal rolling temporal features if requested
        if self.add_temporal_features:
            L = self.rolling_window_length
            group_col = df["run_id"] if "run_id" in df.columns else None
            temporal_dict: dict[str, Any] = {}
            for bf in self.base_feature_names:
                col_series = base_df[bf]
                if group_col is not None:
                    rm = col_series.groupby(group_col).rolling(window=L, min_periods=1).mean().reset_index(level=0, drop=True)
                    rstd = col_series.groupby(group_col).rolling(window=L, min_periods=1).std(ddof=0).fillna(0.0).reset_index(level=0, drop=True)
                    diff = col_series.groupby(group_col).diff().fillna(0.0)
                else:
                    rm = col_series.rolling(window=L, min_periods=1).mean()
                    rstd = col_series.rolling(window=L, min_periods=1).std(ddof=0).fillna(0.0)
                    diff = col_series.diff().fillna(0.0)

                temporal_dict[f"{bf}_rolling_mean_{L}"] = rm
                temporal_dict[f"{bf}_rolling_std_{L}"] = rstd
                temporal_dict[f"{bf}_diff_1"] = diff

            temp_df = pd.DataFrame(temporal_dict, index=df.index)
            base_df = pd.concat([base_df, temp_df], axis=1)

        # 3. Final leakage verification on X
        for col in base_df.columns:
            if is_ground_truth_field(col):
                raise RuntimeError(f"CRITICAL LEAKAGE: Ground-truth column '{col}' detected in feature matrix X!")
            if col in METADATA_FIELDS:
                raise RuntimeError(f"CRITICAL LEAKAGE: Metadata column '{col}' detected in feature matrix X!")

        # 4. Extract ground truth and metadata sidecar
        y_cols = {}
        for meta in METADATA_FIELDS:
            if meta in df.columns:
                y_cols[meta] = df[meta]

        # Extract attack ground truth
        if "ground_truth_attack_active" in df.columns:
            y_cols["attack_active"] = df["ground_truth_attack_active"].astype(bool)
        elif "attack_active" in df.columns:
            y_cols["attack_active"] = df["attack_active"].astype(bool)
        else:
            y_cols["attack_active"] = pd.Series(False, index=df.index)

        if "ground_truth_attack_type" in df.columns:
            y_cols["attack_type"] = df["ground_truth_attack_type"].astype(str)
        elif "attack_type" in df.columns:
            y_cols["attack_type"] = df["attack_type"].astype(str)
        else:
            y_cols["attack_type"] = pd.Series("none", index=df.index)

        if "ground_truth_mean_severity" in df.columns:
            y_cols["attack_severity"] = pd.to_numeric(df["ground_truth_mean_severity"], errors="coerce").fillna(0.0)
        elif "attack_strength" in df.columns:
            y_cols["attack_severity"] = pd.to_numeric(df["attack_strength"], errors="coerce").fillna(0.0)
        else:
            y_cols["attack_severity"] = pd.Series(0.0, index=df.index)

        if "ground_truth_attack_phase" in df.columns:
            y_cols["attack_phase"] = df["ground_truth_attack_phase"].astype(str)
        else:
            y_cols["attack_phase"] = pd.Series("normal", index=df.index)

        y_df = pd.DataFrame(y_cols, index=df.index)

        return base_df, y_df
