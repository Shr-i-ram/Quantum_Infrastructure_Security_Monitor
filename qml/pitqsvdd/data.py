"""Data loading, scenario partitioning, and temporal dataset management for PI-TQSVDD.

Enforces:
- Normal-only training: Training partition strictly contains legitimate QKD scenarios.
- Zero test leakage: Transformations fitted exclusively on normal training data.
- Causal temporal windowing: Sliding windows strictly isolated within scenarios.
- Rich metadata tracking: Provenance sidecars for forensic tracing and per-attack metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from qism.ml.data.loader import MLDatasetLoader
from qism.ml.features.extractor import FeatureExtractor
from qml.pitqsvdd.config import DataConfig
from qml.pitqsvdd.feature_transforms import PITQSVDDFeatureTransformer
from qml.pitqsvdd.temporal_windows import TemporalWindowBuilder


class PITQSVDDDatasetManager:
    """Manages end-to-end dataset preparation and temporal sequence batching."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.transformer = PITQSVDDFeatureTransformer(
            strategy=config.representation,
            n_qubits=config.n_qubits,
            seed=config.master_seed,
        )
        self.window_builder = TemporalWindowBuilder(
            sequence_length=config.sequence_length,
            stride=config.temporal_stride,
        )

        self.scenarios_raw: dict[str, list[dict[str, Any]]] = {}
        self.scenarios_transformed: dict[str, list[np.ndarray]] = {}
        self.scenarios_metadata: dict[str, list[pd.DataFrame]] = {}
        self.temporal_data: dict[str, dict[str, Any]] = {}

        self._load_and_partition()

    def _load_and_partition(self) -> None:
        """Load scenarios from MLDatasetLoader and partition into clean splits."""
        loader = MLDatasetLoader(self.dataset_root, master_seed=self.config.master_seed)
        extractor = FeatureExtractor(feature_set="full", add_temporal_features=False)

        # Collect scenario records
        by_scen: dict[str, dict[str, Any]] = {}
        splits_to_load = [s for s in ["train", "val", "test", "generalization", "stress", "calibration"] if s in loader.partitions]
        for split in splits_to_load:
            for sc, df in loader.iter_scenario_telemetry(split):
                sc_id = sc["scenario_id"]
                if sc_id in by_scen:
                    continue
                X_df, y_df = extractor.extract(df)
                by_scen[sc_id] = {
                    "scenario": sc,
                    "X": X_df,
                    "y": y_df,
                }

        # Check if research dataset naming scheme is present
        if any(s.startswith("train_") for s in by_scen.keys()):
            self.partitions_scen_ids = {
                "train": [s["scenario_id"] for s in loader.get_scenarios("train") if s["scenario_id"] in by_scen],
                "val": [s["scenario_id"] for s in loader.get_scenarios("val") if s["scenario_id"] in by_scen],
                "test": [s["scenario_id"] for s in loader.get_scenarios("test") if s["scenario_id"] in by_scen],
                "gen": [s["scenario_id"] for s in loader.get_scenarios("generalization") if s["scenario_id"] in by_scen],
                "stress": [s["scenario_id"] for s in loader.get_scenarios("stress") if s["scenario_id"] in by_scen],
            }
        else:
            # Legacy scenario partition definitions
            train_scens = ["normal_0000", "normal_0001", "normal_0002"]
            val_scens = ["normal_0003", "attack_0000"]
            test_scens = ["normal_0004", "normal_0005", "attack_0001", "attack_0002", "attack_0003", "attack_0004", "attack_0005"]
            gen_scens = ["generalization_0000", "generalization_0001", "generalization_0002", "generalization_0003"]
            self.partitions_scen_ids = {
                "train": [s for s in train_scens if s in by_scen],
                "val": [s for s in val_scens if s in by_scen],
                "test": [s for s in test_scens if s in by_scen],
                "gen": [s for s in gen_scens if s in by_scen],
            }

        # Fit transformer exclusively on normal training scenarios
        train_X_list = [by_scen[s]["X"] for s in self.partitions_scen_ids["train"]]
        if not train_X_list:
            raise RuntimeError("No training scenarios found in dataset!")
        train_X_concat = pd.concat(train_X_list, ignore_index=True)
        self.transformer.fit(train_X_concat)

        # Transform all scenarios and build causal temporal windows
        for split_name, scen_ids in self.partitions_scen_ids.items():
            scen_trans_list = []
            scen_meta_list = []
            for sid in scen_ids:
                sc_data = by_scen[sid]
                X_arr = self.transformer.transform(sc_data["X"])
                scen_trans_list.append(X_arr)
                scen_meta_list.append(sc_data["y"])

            self.scenarios_transformed[split_name] = scen_trans_list
            self.scenarios_metadata[split_name] = scen_meta_list

            # Construct causal temporal windows
            self.temporal_data[split_name] = self.window_builder.build_dataset_windows(
                scen_trans_list, scen_meta_list
            )

    def get_dataloader(self, split: str, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
        """Return PyTorch DataLoader for a given partition."""
        if split not in self.temporal_data:
            raise ValueError(f"Unknown split '{split}'. Available: {list(self.temporal_data.keys())}")

        d = self.temporal_data[split]
        dataset = TensorDataset(
            d["windows"],
            d["targets"],
            d["has_target"],
            d["labels"],
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_raw_metadata(self, split: str) -> list[dict[str, Any]]:
        """Return list of metadata dicts corresponding to decision steps in a partition."""
        return self.temporal_data[split]["metadata"]

    def save_artifacts(self, output_dir: str | Path) -> None:
        """Persist transformers and dataset manifest to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.transformer.save(out / f"transformer_{self.config.representation}.joblib")
        manifest = {
            "dataset_root": str(self.dataset_root),
            "representation": self.config.representation,
            "n_qubits": self.config.n_qubits,
            "sequence_length": self.config.sequence_length,
            "temporal_stride": self.config.temporal_stride,
            "master_seed": self.config.master_seed,
            "scenario_partitions": self.partitions_scen_ids,
            "sample_counts": {
                split: len(self.temporal_data[split]["labels"])
                for split in self.temporal_data
            },
            "attack_counts": {
                split: int(self.temporal_data[split]["labels"].sum().item())
                for split in self.temporal_data
            },
        }
        with open(out / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
