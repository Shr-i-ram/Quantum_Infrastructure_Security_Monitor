"""Feature transformations and classical-to-quantum mappings for PI-TQSVDD.

Enforces strict zero-leakage conventions:
- Transformers are fitted strictly on the normal training partition.
- Handles missing or unpopulated features through train-fitted imputation.
- Supports compact PCA, high-dimensional PCA for data reuploading,
  trainable neural bottlenecks, domain feature-group partitioning,
  and fixed random projections.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn

from qism.ml.features.definitions import (
    DETECTOR_ONLY_FEATURES,
    ENVIRONMENTAL_ONLY_FEATURES,
    FULL_FEATURES,
    QKD_ONLY_FEATURES,
    TIMING_CHANNEL_FEATURES,
)


FEATURE_FAMILIES = {
    "qkd_error": list(QKD_ONLY_FEATURES),
    "detector_dynamics": list(DETECTOR_ONLY_FEATURES),
    "environmental": list(ENVIRONMENTAL_ONLY_FEATURES),
    "timing_channel": list(TIMING_CHANNEL_FEATURES),
}


class TrainableClassicalBottleneck(nn.Module):
    """Trainable classical bottleneck mapping 48 observable features to m dimensions."""

    def __init__(self, in_features: int = 48, hidden_dim: int = 24, out_dim: int = 4, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Maps to [-1, 1], then rescales to [0, pi] for angle embedding
        out = self.net(x)
        return (out + 1.0) * (np.pi / 2.0)


class PITQSVDDFeatureTransformer:
    """Master feature transformation manager for PI-TQSVDD."""

    def __init__(
        self,
        strategy: Literal["pca", "learned_bottleneck", "feature_group", "highdim_pca", "random_proj"] = "pca",
        n_qubits: int = 4,
        bottleneck_hidden_dim: int = 24,
        highdim_components: int = 16,
        seed: int = 42,
    ):
        self.strategy = strategy
        self.n_qubits = n_qubits
        self.bottleneck_hidden_dim = bottleneck_hidden_dim
        self.highdim_components = highdim_components
        self.seed = seed

        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.angle_scaler: MinMaxScaler | None = None
        self.pca: PCA | None = None
        self.highdim_pca: PCA | None = None
        self.random_proj_matrix: np.ndarray | None = None
        self.bottleneck_model: TrainableClassicalBottleneck | None = None
        self.valid_cols: list[str] = []
        self.group_transformers: dict[str, dict[str, Any]] = {}
        self.is_fitted: bool = False

    def fit(self, X_train_df: pd.DataFrame) -> PITQSVDDFeatureTransformer:
        """Fit all preprocessing transformers exclusively on normal training data."""
        # 1. Identify columns with non-trivial data
        valid_cols = [c for c in X_train_df.columns if not X_train_df[c].isna().all()]
        if not valid_cols:
            valid_cols = list(X_train_df.columns)
        self.valid_cols = valid_cols

        X_clean = X_train_df[self.valid_cols].copy()

        # 2. Impute and standardize
        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X_clean)

        self.scaler = StandardScaler()
        X_std = self.scaler.fit_transform(X_imp)

        # 3. Strategy-specific fitting
        if self.strategy == "pca":
            n_comp = min(self.n_qubits, X_std.shape[1], X_std.shape[0])
            self.pca = PCA(n_components=n_comp, random_state=self.seed)
            X_pca = self.pca.fit_transform(X_std)
            self.angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
            self.angle_scaler.fit(X_pca)

        elif self.strategy == "highdim_pca":
            n_comp = min(self.highdim_components, X_std.shape[1], X_std.shape[0])
            self.highdim_pca = PCA(n_components=n_comp, random_state=self.seed)
            X_hpca = self.highdim_pca.fit_transform(X_std)
            self.angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
            self.angle_scaler.fit(X_hpca)

        elif self.strategy == "learned_bottleneck":
            self.bottleneck_model = TrainableClassicalBottleneck(
                in_features=X_std.shape[1],
                hidden_dim=self.bottleneck_hidden_dim,
                out_dim=self.n_qubits,
                seed=self.seed,
            )

        elif self.strategy == "feature_group":
            # Fit an independent imputer and scaler for each physical family
            self.group_transformers = {}
            for fam_name, fam_cols in FEATURE_FAMILIES.items():
                cols_present = [c for c in fam_cols if c in X_train_df.columns]
                if not cols_present:
                    continue
                sub_df = X_train_df[cols_present].copy()
                sub_valid = [c for c in cols_present if not sub_df[c].isna().all()]
                if not sub_valid:
                    sub_valid = cols_present
                sub_df = sub_df[sub_valid]

                imp = SimpleImputer(strategy="median")
                imp_data = imp.fit_transform(sub_df)
                sc = StandardScaler()
                sc_data = sc.fit_transform(imp_data)
                # Compress each family to 1 or 2 components
                n_grp_comp = max(1, self.n_qubits // len(FEATURE_FAMILIES))
                n_grp_comp = min(n_grp_comp, sc_data.shape[1], sc_data.shape[0])
                grp_pca = PCA(n_components=n_grp_comp, random_state=self.seed)
                grp_proj = grp_pca.fit_transform(sc_data)
                asc = MinMaxScaler(feature_range=(0.0, np.pi))
                asc.fit(grp_proj)

                self.group_transformers[fam_name] = {
                    "valid_cols": sub_valid,
                    "imputer": imp,
                    "scaler": sc,
                    "pca": grp_pca,
                    "angle_scaler": asc,
                }

        elif self.strategy == "random_proj":
            rng = np.random.RandomState(self.seed)
            self.random_proj_matrix = rng.randn(X_std.shape[1], self.n_qubits) / np.sqrt(self.n_qubits)
            X_proj = np.dot(X_std, self.random_proj_matrix)
            self.angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
            self.angle_scaler.fit(X_proj)

        else:
            raise ValueError(f"Unknown feature strategy '{self.strategy}'")

        self.is_fitted = True
        return self

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        """Transform arbitrary partition using pre-fitted transformers."""
        if not self.is_fitted:
            raise RuntimeError("Transformer must be fitted before calling transform().")

        if self.strategy == "feature_group":
            group_pieces = []
            for fam_name in FEATURE_FAMILIES.keys():
                if fam_name not in self.group_transformers:
                    continue
                tinfo = self.group_transformers[fam_name]
                sub_cols = tinfo["valid_cols"]
                sub_df = X_df[sub_cols].copy()
                imp_data = tinfo["imputer"].transform(sub_df)
                sc_data = tinfo["scaler"].transform(imp_data)
                grp_proj = tinfo["pca"].transform(sc_data)
                grp_scaled = tinfo["angle_scaler"].transform(grp_proj)
                group_pieces.append(grp_scaled)
            if group_pieces:
                res = np.concatenate(group_pieces, axis=1)
                # If dimension doesn't match n_qubits, adjust smoothly
                if res.shape[1] > self.n_qubits:
                    res = res[:, :self.n_qubits]
                elif res.shape[1] < self.n_qubits:
                    pad = np.zeros((res.shape[0], self.n_qubits - res.shape[1]))
                    res = np.concatenate([res, pad], axis=1)
                return np.clip(res, 0.0, np.pi)
            return np.zeros((len(X_df), self.n_qubits))

        # Standard pipeline for other strategies
        X_clean = X_df[self.valid_cols].copy()
        X_imp = self.imputer.transform(X_clean)
        X_std = self.scaler.transform(X_imp)

        if self.strategy == "pca":
            X_pca = self.pca.transform(X_std)
            # Pad if n_comp < n_qubits
            if X_pca.shape[1] < self.n_qubits:
                pad = np.zeros((X_pca.shape[0], self.n_qubits - X_pca.shape[1]))
                X_pca = np.concatenate([X_pca, pad], axis=1)
            X_out = self.angle_scaler.transform(X_pca[:, :self.n_qubits])
            return np.clip(X_out, 0.0, np.pi)

        elif self.strategy == "highdim_pca":
            X_hpca = self.highdim_pca.transform(X_std)
            X_out = self.angle_scaler.transform(X_hpca)
            return np.clip(X_out, 0.0, np.pi)

        elif self.strategy == "learned_bottleneck":
            with torch.no_grad():
                t_in = torch.tensor(X_std, dtype=torch.float32)
                t_out = self.bottleneck_model(t_in).numpy()
            return np.clip(t_out, 0.0, np.pi)

        elif self.strategy == "random_proj":
            X_proj = np.dot(X_std, self.random_proj_matrix)
            X_out = self.angle_scaler.transform(X_proj)
            return np.clip(X_out, 0.0, np.pi)

        raise ValueError(f"Unknown strategy '{self.strategy}'")

    def save(self, path: str | Path) -> None:
        """Save transformer to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: str | Path) -> PITQSVDDFeatureTransformer:
        """Load transformer from disk."""
        return joblib.load(path)
