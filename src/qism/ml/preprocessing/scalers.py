"""Feature preprocessing pipeline with strict fit-on-train isolation (Requirements 32–35, 133–137, 140, 245)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class PreprocessorPipeline:
    """Preprocesses features: missing value imputation and scaling.

    Strictly guarantees:
    - Statistics (mean, std, median, IQR) are fit ONLY on normal training data.
    - Constant / zero-variance features are safely handled without producing NaNs or infinities.
    - Exact pipeline state can be saved to and loaded from JSON.
    """

    def __init__(
        self,
        scaler: str = "standard",
        impute_strategy: str = "median",
        clip_sigma: float = 10.0,
    ):
        self.scaler_type = scaler  # "standard", "robust", "none"
        self.impute_strategy = impute_strategy
        self.clip_sigma = clip_sigma

        self.feature_names: list[str] = []
        self.impute_values_: np.ndarray | None = None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> PreprocessorPipeline:
        """Fit preprocessing statistics strictly on training data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self.feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        n_samples, n_features = X_arr.shape

        # 1. Compute imputation statistics
        impute_vals = np.zeros(n_features, dtype=np.float64)
        for j in range(n_features):
            col = X_arr[:, j]
            valid = col[~np.isnan(col) & ~np.isinf(col)]
            if valid.size == 0:
                impute_vals[j] = 0.0
            elif self.impute_strategy == "median":
                impute_vals[j] = float(np.median(valid))
            elif self.impute_strategy == "mean":
                impute_vals[j] = float(np.mean(valid))
            else:
                impute_vals[j] = 0.0

        self.impute_values_ = impute_vals

        # Impute missing in temporary array to compute scaling parameters
        X_clean = np.copy(X_arr)
        for j in range(n_features):
            mask = np.isnan(X_clean[:, j]) | np.isinf(X_clean[:, j])
            X_clean[mask, j] = self.impute_values_[j]

        # 2. Compute scaling parameters
        if self.scaler_type == "standard":
            mean = np.mean(X_clean, axis=0)
            std = np.std(X_clean, axis=0)
            # Protect against zero variance
            std[std < 1e-8] = 1.0
            self.center_ = mean
            self.scale_ = std
        elif self.scaler_type == "robust":
            median = np.median(X_clean, axis=0)
            q25 = np.percentile(X_clean, 25, axis=0)
            q75 = np.percentile(X_clean, 75, axis=0)
            iqr = q75 - q25
            iqr[iqr < 1e-8] = 1.0
            self.center_ = median
            self.scale_ = iqr
        else:
            self.center_ = np.zeros(n_features, dtype=np.float64)
            self.scale_ = np.ones(n_features, dtype=np.float64)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply fitted preprocessing to X without recalculating statistics."""
        if not self.is_fitted:
            raise RuntimeError("PreprocessorPipeline must be fit before transform.")

        if isinstance(X, pd.DataFrame):
            # Verify feature alignment
            if self.feature_names and list(X.columns) != self.feature_names:
                # Reorder or select columns matching fitted feature_names
                missing = [f for f in self.feature_names if f not in X.columns]
                if missing:
                    raise ValueError(f"Transform DataFrame is missing fitted features: {missing[:5]}")
                X = X[self.feature_names]
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        X_out = np.copy(X_arr)

        # 1. Impute
        for j in range(X_out.shape[1]):
            mask = np.isnan(X_out[:, j]) | np.isinf(X_out[:, j])
            X_out[mask, j] = self.impute_values_[j]

        # 2. Scale
        X_out = (X_out - self.center_) / self.scale_

        # 3. Clip extreme outliers if configured
        if self.clip_sigma > 0:
            X_out = np.clip(X_out, -self.clip_sigma, self.clip_sigma)

        return X_out

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit on X and transform X."""
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        """Serialize preprocessor state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "scaler_type": self.scaler_type,
            "impute_strategy": self.impute_strategy,
            "clip_sigma": self.clip_sigma,
            "feature_names": self.feature_names,
            "impute_values": self.impute_values_.tolist() if self.impute_values_ is not None else None,
            "center": self.center_.tolist() if self.center_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
            "is_fitted": self.is_fitted,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> PreprocessorPipeline:
        """Load fitted preprocessor state from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        pipe = cls(
            scaler=state["scaler_type"],
            impute_strategy=state["impute_strategy"],
            clip_sigma=state["clip_sigma"],
        )
        pipe.feature_names = state["feature_names"]
        pipe.impute_values_ = np.array(state["impute_values"], dtype=np.float64) if state["impute_values"] is not None else None
        pipe.center_ = np.array(state["center"], dtype=np.float64) if state["center"] is not None else None
        pipe.scale_ = np.array(state["scale"], dtype=np.float64) if state["scale"] is not None else None
        pipe.is_fitted = state["is_fitted"]
        return pipe
