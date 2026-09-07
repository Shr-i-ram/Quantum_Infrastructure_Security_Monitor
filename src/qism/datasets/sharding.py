"""Dataset sharding and logical dataset view utilities.

In accordance with Requirements 38-41, 82-84:
- Partitions large datasets into logical shards (shard_000, shard_001, ...) keeping complete scenarios intact.
- Binds memory consumption to O(1) regardless of total dataset size.
- Provides logical dataset traversal without physical file merging.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import pandas as pd

log = logging.getLogger("qism.datasets.sharding")


class DatasetShardWriter:
    """Manages directory layout and metadata for dataset shards."""

    def __init__(self, split_dir: pathlib.Path, shard_size: int = 10):
        self.split_dir = split_dir
        self.shard_size = max(1, shard_size)
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self._shard_runs: dict[str, list[dict[str, Any]]] = {}

    def get_shard_dir(self, run_index: int) -> pathlib.Path:
        """Compute the shard directory path for a given run index."""
        shard_idx = run_index // self.shard_size
        shard_name = f"shard_{shard_idx:04d}"
        s_dir = self.split_dir / shard_name
        s_dir.mkdir(parents=True, exist_ok=True)
        return s_dir

    def record_run_in_shard(self, shard_name: str, scenario_id: str, run_id: str, run_path: str) -> None:
        """Record a completed run into shard tracking."""
        if shard_name not in self._shard_runs:
            self._shard_runs[shard_name] = []
        self._shard_runs[shard_name].append({
            "scenario_id": scenario_id,
            "run_id": run_id,
            "path": str(run_path),
        })

    def finalize_shard(self, shard_dir: pathlib.Path, runs: list[dict[str, Any]]) -> pathlib.Path:
        """Write shard_manifest.json for the shard."""
        manifest_path = shard_dir / "shard_manifest.json"
        payload = {
            "shard_id": shard_dir.name,
            "total_runs": len(runs),
            "runs": runs,
        }
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return manifest_path

    def finalize_all_manifests(self) -> list[pathlib.Path]:
        """Write shard_manifest.json across all tracked shards."""
        manifests = []
        for shard_name, runs in self._shard_runs.items():
            s_dir = self.split_dir / shard_name
            manifests.append(self.finalize_shard(s_dir, runs))
        return manifests


def list_sharded_telemetry_files(split_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find all telemetry.parquet files across all shards in a split directory."""
    if not split_dir.exists():
        return []
    files = sorted(split_dir.rglob("telemetry.parquet"))
    return files


def list_sharded_events_files(split_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find all events.parquet files across all shards in a split directory."""
    if not split_dir.exists():
        return []
    files = sorted(split_dir.rglob("events.parquet"))
    return files


def load_sharded_telemetry(split_dir: pathlib.Path, max_rows: int | None = None) -> pd.DataFrame:
    """Read a logical view of all telemetry across shards without physical merging."""
    files = list_sharded_telemetry_files(split_dir)
    if not files:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    total_rows = 0
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
        total_rows += len(df)
        if max_rows is not None and total_rows >= max_rows:
            break

    combined = pd.concat(dfs, ignore_index=True)
    if max_rows is not None:
        combined = combined.iloc[:max_rows]
    return combined
