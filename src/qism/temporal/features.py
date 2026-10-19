"""Temporal feature engineering with strict causal guarantees and boundary isolation (V0.7).

Calculates physics-informed time-series features:
- First difference: Delta x_t = x_t - x_{t-1}
- Second difference: Delta^2 x_t = Delta x_t - Delta x_{t-1}
- Rolling mean, variance, and standard deviation over causal horizon L
- Exponentially weighted moving average (EWMA)
- Rate of change: r_t = (x_t - x_{t-k}) / (|x_{t-k}| + eps)
- Robust rolling statistics (median and MAD)
- Cross-stream physics correlation ratios

All transformations strictly respect (scenario_id, run_id) trajectory boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from qism.ml.features.definitions import (
    GROUND_TRUTH_FIELDS,
    GROUND_TRUTH_PREFIXES,
    METADATA_FIELDS,
    is_ground_truth_field,
)


@dataclass
class TemporalFeatureConfig:
    """Configuration for temporal feature generation."""
    window_length: int = 3
    alpha_ewma: float = 0.3
    rate_of_change_lag: int = 1
    include_first_difference: bool = True
    include_second_difference: bool = True
    include_rolling_mean: bool = True
    include_rolling_std: bool = True
    include_rolling_var: bool = False
    include_ewma: bool = True
    include_rate_of_change: bool = True
    include_robust_mad: bool = False
    include_cross_correlations: bool = True

    def __init__(
        self,
        window_length: int = 3,
        alpha_ewma: float = 0.3,
        rate_of_change_lag: int = 1,
        include_first_difference: bool = True,
        include_second_difference: bool = True,
        include_rolling_mean: bool = True,
        include_rolling_std: bool = True,
        include_rolling_var: bool = False,
        include_ewma: bool = True,
        include_rate_of_change: bool = True,
        include_robust_mad: bool = False,
        include_cross_correlations: bool = True,
        rolling_windows: list[int] | None = None,
        compute_diff1: bool | None = None,
        compute_diff2: bool | None = None,
        compute_rolling_mean: bool | None = None,
        compute_rolling_std: bool | None = None,
        compute_ewma: bool | None = None,
        compute_rate_of_change: bool | None = None,
        compute_physical_couplings: bool | None = None,
        **kwargs: Any,
    ):
        if rolling_windows and len(rolling_windows) > 0:
            window_length = rolling_windows[0]
        if compute_diff1 is not None:
            include_first_difference = compute_diff1
        if compute_diff2 is not None:
            include_second_difference = compute_diff2
        if compute_rolling_mean is not None:
            include_rolling_mean = compute_rolling_mean
        if compute_rolling_std is not None:
            include_rolling_std = compute_rolling_std
        if compute_ewma is not None:
            include_ewma = compute_ewma
        if compute_rate_of_change is not None:
            include_rate_of_change = compute_rate_of_change
        if compute_physical_couplings is not None:
            include_cross_correlations = compute_physical_couplings

        self.window_length = int(window_length)
        self.alpha_ewma = float(alpha_ewma)
        self.rate_of_change_lag = int(rate_of_change_lag)
        self.include_first_difference = bool(include_first_difference)
        self.include_second_difference = bool(include_second_difference)
        self.include_rolling_mean = bool(include_rolling_mean)
        self.include_rolling_std = bool(include_rolling_std)
        self.include_rolling_var = bool(include_rolling_var)
        self.include_ewma = bool(include_ewma)
        self.include_rate_of_change = bool(include_rate_of_change)
        self.include_robust_mad = bool(include_robust_mad)
        self.include_cross_correlations = bool(include_cross_correlations)


class TemporalFeatureEngine:
    """Computes causal temporal features strictly bounded by trajectory."""

    def __init__(self, config: TemporalFeatureConfig | None = None):
        self.config = config or TemporalFeatureConfig()

    def fit(self, X_df: pd.DataFrame, y: Any = None) -> TemporalFeatureEngine:
        return self

    def transform(self, X_df: pd.DataFrame, group_df: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.extract_temporal_features(X_df, group_df)

    def fit_transform(self, X_df: pd.DataFrame, group_df: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.extract_temporal_features(X_df, group_df)


    def get_feature_names(self, base_features: list[str]) -> list[str]:
        """Generate deterministic names for all extracted temporal features."""
        cfg = self.config
        L = cfg.window_length
        names = list(base_features)

        for f in base_features:
            if cfg.include_first_difference:
                names.append(f"{f}_diff_1")
            if cfg.include_second_difference:
                names.append(f"{f}_diff_2")
            if cfg.include_rolling_mean:
                names.append(f"{f}_rolling_mean_{L}")
            if cfg.include_rolling_std:
                names.append(f"{f}_rolling_std_{L}")
            if cfg.include_rolling_var:
                names.append(f"{f}_rolling_var_{L}")
            if cfg.include_ewma:
                names.append(f"{f}_ewma_{cfg.alpha_ewma:.2f}")
            if cfg.include_rate_of_change:
                names.append(f"{f}_roc_{cfg.rate_of_change_lag}")
            if cfg.include_robust_mad:
                names.append(f"{f}_rolling_mad_{L}")

        if cfg.include_cross_correlations:
            # Physical cross-stream coupling proxies if features exist
            if "qber" in base_features and "signal_gain" in base_features:
                names.append("cross_qber_to_gain_ratio")
            if "detector_imbalance" in base_features and "double_click_rate" in base_features:
                names.append("cross_imbalance_double_click_product")
            if "sensor_temperature_mean" in base_features and "dark_count_rate" in base_features:
                names.append("cross_temp_dark_count_ratio")

        return names

    def extract_temporal_features(
        self,
        X_df: pd.DataFrame,
        group_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute causal temporal features on observable feature matrix X_df.

        Guarantees:
        - Strict trajectory isolation: never differences or averages across runs.
        - Strict causality: values at time t depend only on historical observations (<= t).
        - No NaNs or infinities in output (safely imputed with causal defaults).
        """
        if X_df.empty:
            return pd.DataFrame(columns=self.get_feature_names(list(X_df.columns)))

        cfg = self.config
        L = cfg.window_length
        # Select numeric observable base features
        base_features = [
            f for f in X_df.columns
            if f not in METADATA_FIELDS
            and not is_ground_truth_field(f)
            and pd.api.types.is_numeric_dtype(X_df[f])
        ]



        # Build composite trajectory identifier
        if group_df is not None and "scenario_id" in group_df.columns and "run_id" in group_df.columns:
            traj_id = group_df["scenario_id"].astype(str) + "___" + group_df["run_id"].astype(str)
        elif "scenario_id" in X_df.columns and "run_id" in X_df.columns:
            traj_id = X_df["scenario_id"].astype(str) + "___" + X_df["run_id"].astype(str)
        else:
            traj_id = pd.Series(["default_traj"] * len(X_df), index=X_df.index)

        out_features: dict[str, np.ndarray] = {}

        # Retain base features
        for f in base_features:
            out_features[f] = X_df[f].to_numpy(dtype=np.float64)

        # Process trajectory by trajectory to prevent boundary leakage
        unique_trajs = traj_id.unique()

        # Pre-allocate output arrays
        n_samples = len(X_df)
        feature_buffers: dict[str, np.ndarray] = {}

        for f in base_features:
            if cfg.include_first_difference:
                feature_buffers[f"{f}_diff_1"] = np.zeros(n_samples, dtype=np.float64)
            if cfg.include_second_difference:
                feature_buffers[f"{f}_diff_2"] = np.zeros(n_samples, dtype=np.float64)
            if cfg.include_rolling_mean:
                feature_buffers[f"{f}_rolling_mean_{L}"] = np.zeros(n_samples, dtype=np.float64)
            if cfg.include_rolling_std:
                feature_buffers[f"{f}_rolling_std_{L}"] = np.zeros(n_samples, dtype=np.float64)
            if cfg.include_rolling_var:
                feature_buffers[f"{f}_rolling_var_{L}"] = np.zeros(n_samples, dtype=np.float64)
            if cfg.include_ewma:
                feature_buffers[f"{f}_ewma_{cfg.alpha_ewma:.2f}"] = np.zeros(n_samples, dtype=np.float64)
            if cfg.include_rate_of_change:
                feature_buffers[f"{f}_roc_{cfg.rate_of_change_lag}"] = np.zeros(n_samples, dtype=np.float64)
            if cfg.include_robust_mad:
                feature_buffers[f"{f}_rolling_mad_{L}"] = np.zeros(n_samples, dtype=np.float64)

        # Vectorized per-trajectory causal computations
        for ut in unique_trajs:
            mask = (traj_id == ut).to_numpy()
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            sub_df = X_df.iloc[indices]

            for f in base_features:
                s = sub_df[f].astype(np.float64)

                # 1. First difference
                d1 = s.diff().fillna(0.0).to_numpy()
                if cfg.include_first_difference:
                    feature_buffers[f"{f}_diff_1"][indices] = d1

                # 2. Second difference
                if cfg.include_second_difference:
                    d2 = pd.Series(d1).diff().fillna(0.0).to_numpy()
                    feature_buffers[f"{f}_diff_2"][indices] = d2

                # 3. Rolling mean
                rm = s.rolling(window=L, min_periods=1).mean().to_numpy()
                if cfg.include_rolling_mean:
                    feature_buffers[f"{f}_rolling_mean_{L}"][indices] = rm

                # 4. Rolling std & var
                rvar = s.rolling(window=L, min_periods=1).var(ddof=0).fillna(0.0).to_numpy()
                rstd = np.sqrt(np.maximum(0.0, rvar))
                if cfg.include_rolling_std:
                    feature_buffers[f"{f}_rolling_std_{L}"][indices] = rstd
                if cfg.include_rolling_var:
                    feature_buffers[f"{f}_rolling_var_{L}"][indices] = rvar

                # 5. EWMA
                if cfg.include_ewma:
                    ewma = s.ewm(alpha=cfg.alpha_ewma, adjust=False).mean().to_numpy()
                    feature_buffers[f"{f}_ewma_{cfg.alpha_ewma:.2f}"][indices] = ewma

                # 6. Rate of change
                if cfg.include_rate_of_change:
                    k = cfg.rate_of_change_lag
                    past = s.shift(k).fillna(s.iloc[0]).to_numpy()
                    curr = s.to_numpy()
                    roc = (curr - past) / (np.abs(past) + 1e-6)
                    feature_buffers[f"{f}_roc_{k}"][indices] = np.clip(roc, -100.0, 100.0)

                # 7. Robust rolling MAD
                if cfg.include_robust_mad:
                    # Rolling median
                    rmed = s.rolling(window=L, min_periods=1).median()
                    # Deviation from rolling median
                    dev = (s - rmed).abs()
                    mad = dev.rolling(window=L, min_periods=1).median().fillna(0.0).to_numpy()
                    feature_buffers[f"{f}_rolling_mad_{L}"][indices] = mad

        # Cross-stream physical couplings
        if cfg.include_cross_correlations:
            if "qber" in base_features and "signal_gain" in base_features:
                qber_val = X_df["qber"].to_numpy()
                gain_val = X_df["signal_gain"].to_numpy()
                ratio = qber_val / (np.maximum(1e-5, gain_val))
                feature_buffers["cross_qber_to_gain_ratio"] = np.clip(ratio, 0.0, 100.0)

            if "detector_imbalance" in base_features and "double_click_rate" in base_features:
                imb = X_df["detector_imbalance"].to_numpy()
                dbl = X_df["double_click_rate"].to_numpy()
                feature_buffers["cross_imbalance_double_click_product"] = np.abs(imb) * dbl

            if "sensor_temperature_mean" in base_features and "dark_count_rate" in base_features:
                temp = X_df["sensor_temperature_mean"].to_numpy()
                dark = X_df["dark_count_rate"].to_numpy()
                feature_buffers["cross_temp_dark_count_ratio"] = dark / np.maximum(1.0, temp)

        out_features.update(feature_buffers)
        result_df = pd.DataFrame(out_features, index=X_df.index)

        # Zero leakage and integrity assertion
        for col in result_df.columns:
            if is_ground_truth_field(col) or col in METADATA_FIELDS:
                raise RuntimeError(f"CRITICAL LEAKAGE: Forbidden field '{col}' present in temporal feature matrix!")

        return result_df
