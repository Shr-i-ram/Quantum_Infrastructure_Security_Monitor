"""Data loading, feature extraction, and split management for QML experiments (Review 2).

Enforces strict zero-leakage conventions:
- Target labels (attack_active, attack_type, etc.) and metadata (scenario_id, run_id, etc.)
  are stripped from the input feature space.
- Preprocessing scalers and dimensionality reduction (PCA) are fitted strictly on the
  training partition and applied to validation, test, and generalization sets.
- Features are scaled to [0, pi] for angle embedding into quantum rotation gates.
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

from qism.ml.data.loader import MLDatasetLoader
from qism.ml.features.extractor import FeatureExtractor


PHYSICAL_4_FEATURES = ("qber", "sifted_rate", "signal_gain", "detector_imbalance")


class QMLDatasetManager:
    """Manages QML dataset preparation, representations, and anti-leakage scaling."""

    def __init__(
        self,
        dataset_root: str | Path = "data/v0.5_small/v05_small_dataset",
        representation: Literal["pca", "physical"] = "pca",
        n_qubits: int = 4,
        master_seed: int = 42,
    ):
        self.dataset_root = Path(dataset_root)
        self.representation = representation
        self.n_qubits = n_qubits
        self.master_seed = master_seed

        self.imputer: SimpleImputer | None = None
        self.std_scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.angle_scaler: MinMaxScaler | None = None

        self._load_and_partition()

    def _load_and_partition(self) -> None:
        """Load scenarios and create clean, zero-leakage scenario-level partitions."""
        loader = MLDatasetLoader(self.dataset_root, master_seed=self.master_seed)
        extractor = FeatureExtractor(feature_set="full", add_temporal_features=False)

        # Collect scenario records and their extracted features
        scenarios_data: list[dict[str, Any]] = []
        for split in ["train", "val", "test", "generalization"]:
            for sc, df in loader.iter_scenario_telemetry(split):
                sc_id = sc["scenario_id"]
                X_df, y_df = extractor.extract(df)
                scenarios_data.append({
                    "scenario_id": sc_id,
                    "original_split": split,
                    "dataset_split": sc.get("dataset_split", sc.get("split")),
                    "X": X_df,
                    "y": y_df,
                })

        # Organize by scenario ID
        by_scen = {s["scenario_id"]: s for s in scenarios_data}

        # Supervised scenario assignments:
        # Train: 3 normal + 3 attack scenarios
        train_scens = ["normal_0000", "normal_0001", "normal_0002", "attack_0000", "attack_0001", "attack_0002"]
        # Val: 1 normal + 1 attack scenario
        val_scens = ["normal_0003", "attack_0003"]
        # Test: 2 normal + 2 attack scenarios
        test_scens = ["normal_0004", "normal_0005", "attack_0004", "attack_0005"]
        # Generalization (OOD): 4 generalization scenarios
        gen_scens = ["generalization_0000", "generalization_0001", "generalization_0002", "generalization_0003"]

        def _concat_scens(scen_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
            X_list = [by_scen[sid]["X"] for sid in scen_ids if sid in by_scen]
            y_list = [by_scen[sid]["y"] for sid in scen_ids if sid in by_scen]
            X_concat = pd.concat(X_list, ignore_index=True) if X_list else pd.DataFrame()
            y_concat = pd.concat(y_list, ignore_index=True) if y_list else pd.DataFrame()
            return X_concat, y_concat

        self.raw_train_X, self.raw_train_y = _concat_scens(train_scens)
        self.raw_val_X, self.raw_val_y = _concat_scens(val_scens)
        self.raw_test_X, self.raw_test_y = _concat_scens(test_scens)
        self.raw_gen_X, self.raw_gen_y = _concat_scens(gen_scens)

        # Fit preprocessors strictly on raw_train_X
        self._fit_preprocessors(self.raw_train_X)

        # Transform all partitions
        self.X_train = self._transform(self.raw_train_X)
        self.y_train = self.raw_train_y["attack_active"].astype(int).to_numpy()

        self.X_val = self._transform(self.raw_val_X)
        self.y_val = self.raw_val_y["attack_active"].astype(int).to_numpy()

        self.X_test = self._transform(self.raw_test_X)
        self.y_test = self.raw_test_y["attack_active"].astype(int).to_numpy()

        self.X_gen = self._transform(self.raw_gen_X)
        self.y_gen = self.raw_gen_y["attack_active"].astype(int).to_numpy()

    def _fit_preprocessors(self, X_train_df: pd.DataFrame) -> None:
        """Fit imputation, normalization, PCA (if applicable), and angle scaling strictly on training set."""
        if self.representation == "physical":
            selected_df = X_train_df[list(PHYSICAL_4_FEATURES)].copy()
            self.imputer = SimpleImputer(strategy="median")
            X_imp = self.imputer.fit_transform(selected_df)
            self.angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
            self.angle_scaler.fit(X_imp)
        elif self.representation == "pca":
            # Drop features that are entirely NaN across training set
            valid_cols = [c for c in X_train_df.columns if not X_train_df[c].isna().all()]
            X_clean = X_train_df[valid_cols].copy()
            self.valid_cols = valid_cols
            self.imputer = SimpleImputer(strategy="median")
            X_imp = self.imputer.fit_transform(X_clean)
            self.std_scaler = StandardScaler()
            X_std = self.std_scaler.fit_transform(X_imp)
            self.pca = PCA(n_components=self.n_qubits, random_state=self.master_seed)
            X_pca = self.pca.fit_transform(X_std)
            self.angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
            self.angle_scaler.fit(X_pca)
        else:
            raise ValueError(f"Unknown representation '{self.representation}'")

    def _transform(self, X_df: pd.DataFrame) -> np.ndarray:
        """Transform arbitrary partition using pre-fitted transformers."""
        if self.representation == "physical":
            selected_df = X_df[list(PHYSICAL_4_FEATURES)].copy()
            X_imp = self.imputer.transform(selected_df)
            X_out = self.angle_scaler.transform(X_imp)
        elif self.representation == "pca":
            X_clean = X_df[self.valid_cols].copy()
            X_imp = self.imputer.transform(X_clean)
            X_std = self.std_scaler.transform(X_imp)
            X_pca = self.pca.transform(X_std)
            X_out = self.angle_scaler.transform(X_pca)
        return np.clip(X_out, 0.0, np.pi)

    def get_torch_data(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return PyTorch float tensors for train, val, test, generalization."""
        return {
            "train": (torch.tensor(self.X_train, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.float32)),
            "val": (torch.tensor(self.X_val, dtype=torch.float32), torch.tensor(self.y_val, dtype=torch.float32)),
            "test": (torch.tensor(self.X_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.float32)),
            "gen": (torch.tensor(self.X_gen, dtype=torch.float32), torch.tensor(self.y_gen, dtype=torch.float32)),
        }

    def get_numpy_data(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return NumPy arrays for train, val, test, generalization."""
        return {
            "train": (self.X_train, self.y_train),
            "val": (self.X_val, self.y_val),
            "test": (self.X_test, self.y_test),
            "gen": (self.X_gen, self.y_gen),
        }

    def get_metadata(self) -> dict[str, pd.DataFrame]:
        """Return metadata DataFrames (scenario_id, attack_type, etc.) for forensic tracing."""
        return {
            "train": self.raw_train_y,
            "val": self.raw_val_y,
            "test": self.raw_test_y,
            "gen": self.raw_gen_y,
        }

    def save_artifacts(self, output_dir: str | Path) -> None:
        """Persist transformers and metadata to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "representation": self.representation,
            "n_qubits": self.n_qubits,
            "master_seed": self.master_seed,
            "train_samples": len(self.X_train),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "gen_samples": len(self.X_gen),
            "pca_explained_variance_ratio": self.pca.explained_variance_ratio_.tolist() if self.pca else None,
            "total_explained_variance": float(np.sum(self.pca.explained_variance_ratio_)) if self.pca else None,
        }
        with open(out / f"data_meta_{self.representation}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        joblib.dump({
            "imputer": self.imputer,
            "std_scaler": self.std_scaler,
            "pca": self.pca,
            "angle_scaler": self.angle_scaler,
        }, out / f"transformers_{self.representation}.joblib")
