"""ML dataset loader for streaming, chunked, and scenario-level partitioned data (Requirements 141–144)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow.parquet as pq

from qism.ml.data.splits import SplitManager


class MLDatasetLoader:
    """Dedicated ML dataset loader providing scenario-level split access."""

    def __init__(
        self,
        dataset_root: str | Path,
        master_seed: int = 42,
        train_val_split_ratio: float = 0.70,
    ):
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        self.manifest_path = self.dataset_root / "dataset_manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing dataset_manifest.json in {self.dataset_root}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.scenario_manifest_path = self.dataset_root / "scenarios" / "scenario_manifest.json"
        if not self.scenario_manifest_path.exists():
            raise FileNotFoundError(f"Missing scenario_manifest.json in {self.dataset_root}/scenarios")

        with open(self.scenario_manifest_path, "r", encoding="utf-8") as f:
            self.scenario_manifest = json.load(f)

        self.split_manager = SplitManager(
            master_seed=master_seed,
            train_val_split_ratio=train_val_split_ratio,
        )
        self.partitions = self.split_manager.partition_scenarios(self.scenario_manifest)
        self._scenario_file_index: dict[str, list[Path]] = {}
        self._build_file_index()

    def _build_file_index(self) -> None:
        """Index all telemetry parquet files by scenario_id once for O(1) retrieval."""
        telemetry_files = list(self.dataset_root.rglob("telemetry.parquet"))
        for tf in telemetry_files:
            try:
                # Read just scenario_id column
                meta = pq.read_table(tf, columns=["scenario_id"])
                scens = set(meta["scenario_id"].to_pylist())
                for s in scens:
                    self._scenario_file_index.setdefault(s, []).append(tf)
            except Exception:
                continue

    def get_scenarios(self, split: str) -> list[dict[str, Any]]:
        """Return scenario metadata records for a given split partition."""
        if split not in self.partitions:
            raise ValueError(f"Unknown split '{split}'. Available: {list(self.partitions.keys())}")
        return self.partitions[split]

    def iter_scenario_telemetry(self, split: str) -> Iterator[tuple[dict[str, Any], pd.DataFrame]]:
        """Yield (scenario_metadata, telemetry_dataframe) lazily without loading all in RAM."""
        scenarios = self.get_scenarios(split)
        for sc in scenarios:
            scen_id = sc["scenario_id"]
            df = self._load_scenario_telemetry(scen_id)
            if df is not None and not df.empty:
                yield sc, df

    def load_split_dataframe(self, split: str) -> pd.DataFrame:
        """Load all telemetry rows for a given split partition into a single DataFrame."""
        frames = []
        for _, df in self.iter_scenario_telemetry(split):
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _load_scenario_telemetry(self, scenario_id: str) -> pd.DataFrame | None:
        """Find and load telemetry parquet for a given scenario using the pre-built index."""
        matching_files = self._scenario_file_index.get(scenario_id, [])
        if not matching_files:
            return None
        matching_tables = []
        for tf in matching_files:
            try:
                table = pq.read_table(tf)
                scen_col = table["scenario_id"].to_pylist()
                mask = [s == scenario_id for s in scen_col]
                filtered = table.filter(mask)
                matching_tables.append(filtered.to_pandas())
            except Exception:
                continue

        if not matching_tables:
            return None
        return pd.concat(matching_tables, ignore_index=True)

    def summary(self) -> dict[str, Any]:
        """Return dataset metadata summary."""
        return {
            "dataset_id": self.manifest.get("dataset_id"),
            "schema_version": self.manifest.get("dataset_schema_version"),
            "simulator_version": self.manifest.get("simulator_version"),
            "total_scenarios": len(self.scenario_manifest.get("scenarios", [])),
            "split_scenario_counts": {k: len(v) for k, v in self.partitions.items()},
        }
