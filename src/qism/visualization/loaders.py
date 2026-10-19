"""Data ingestion and validation utilities for QISM visualization.

Loads Parquet files (events and telemetry), YAML configurations, and JSON manifests
from sharded dataset directory structures with memory-bounded streaming/sampling.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Sequence

import pandas as pd
import pyarrow.parquet as pq
import yaml

log = logging.getLogger("qism.visualization.loaders")


def load_dataset_manifest(dataset_dir: pathlib.Path) -> dict[str, Any]:
    """Load dataset-level manifest if available."""
    manifest_file = dataset_dir / "dataset_manifest.json"
    if not manifest_file.exists():
        log.warning("Dataset manifest not found at %s", manifest_file)
        return {}
    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scenario_manifest(dataset_dir: pathlib.Path) -> dict[str, Any]:
    """Load population scenario manifest if available."""
    scen_file = dataset_dir / "scenarios" / "scenario_manifest.json"
    if not scen_file.exists():
        log.warning("Scenario manifest not found at %s", scen_file)
        return {}
    with open(scen_file, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_runs(dataset_dir: pathlib.Path, split: str | None = None) -> list[dict[str, Any]]:
    """Scan dataset directory and return metadata for all run directories found."""
    runs: list[dict[str, Any]] = []
    splits = [split] if split else ["calibration", "normal", "attack", "generalization"]

    for sp in splits:
        sp_dir = dataset_dir / sp
        if not sp_dir.is_dir():
            continue

        for run_path in sorted(sp_dir.glob("**/run_*")):
            if not run_path.is_dir():
                continue
            ev_path = run_path / "events.parquet"
            tel_path = run_path / "telemetry.parquet"
            cfg_path = run_path / "config.yaml"
            man_path = run_path / "manifest.json"

            runs.append({
                "split": sp,
                "run_id": run_path.name,
                "path": run_path,
                "has_events": ev_path.exists(),
                "has_telemetry": tel_path.exists(),
                "has_config": cfg_path.exists(),
                "has_manifest": man_path.exists(),
            })

    return runs


def load_run_telemetry(run_dir: pathlib.Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load telemetry dataframe for a single run."""
    tel_path = run_dir / "telemetry.parquet"
    if not tel_path.exists():
        raise FileNotFoundError(f"Telemetry not found at {tel_path}")
    
    if columns is not None:
        avail_cols = pq.read_schema(tel_path).names
        valid_cols = [c for c in columns if c in avail_cols]
        missing = set(columns) - set(avail_cols)
        if missing:
            log.warning("Requested columns %s missing in %s", missing, tel_path)
        return pq.read_table(tel_path, columns=valid_cols).to_pandas()
    
    return pq.read_table(tel_path).to_pandas()


def load_run_events(
    run_dir: pathlib.Path,
    max_pulses: int | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load events dataframe for a single run, optionally sampled up to max_pulses."""
    ev_path = run_dir / "events.parquet"
    if not ev_path.exists():
        raise FileNotFoundError(f"Events not found at {ev_path}")

    avail_cols = pq.read_schema(ev_path).names
    valid_cols = [c for c in columns if c in avail_cols] if columns else avail_cols

    table = pq.read_table(ev_path, columns=valid_cols)
    if max_pulses is not None and table.num_rows > max_pulses:
        table = table.slice(0, max_pulses)

    return table.to_pandas()


def load_run_config(run_dir: pathlib.Path) -> dict[str, Any]:
    """Load config.yaml for a run."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_run_manifest(run_dir: pathlib.Path) -> dict[str, Any]:
    """Load manifest.json for a run."""
    man_path = run_dir / "manifest.json"
    if not man_path.exists():
        return {}
    with open(man_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_telemetry(
    dataset_dir: pathlib.Path,
    split: str,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate telemetry across all runs in a given split."""
    runs = discover_runs(dataset_dir, split=split)
    dfs: list[pd.DataFrame] = []

    for r in runs:
        if r["has_telemetry"]:
            try:
                df = load_run_telemetry(r["path"], columns=columns)
                df["split"] = split
                df["run_id"] = r["run_id"]
                dfs.append(df)
            except Exception as e:
                log.warning("Failed to load telemetry from %s: %s", r["path"], e)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def load_all_telemetry(
    dataset_dir: pathlib.Path,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load concatenated telemetry across all partitions (calibration, normal, attack, generalization)."""
    splits = ["calibration", "normal", "attack", "generalization"]
    dfs = [load_split_telemetry(dataset_dir, sp, columns=columns) for sp in splits]
    valid_dfs = [df for df in dfs if not df.empty]
    if not valid_dfs:
        return pd.DataFrame()
    return pd.concat(valid_dfs, ignore_index=True)


def get_representative_run(
    dataset_dir: pathlib.Path,
    split: str,
    attack_type: str | None = None,
) -> pathlib.Path | None:
    """Find a representative run directory matching split and optional attack_type."""
    runs = discover_runs(dataset_dir, split=split)
    for r in runs:
        if not r["has_telemetry"] or not r["has_events"]:
            continue
        if attack_type is None:
            return r["path"]

        cfg = load_run_config(r["path"])
        atk_cfg = cfg.get("attack", {})
        scenarios = atk_cfg.get("scenarios", [])
        scen_types = [s.get("type") for s in scenarios if isinstance(s, dict)]
        top_type = atk_cfg.get("type")
        if attack_type in scen_types or attack_type == top_type:
            return r["path"]

    return runs[0]["path"] if runs else None


def validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    context: str = "",
) -> list[str]:
    """Check that dataframe contains required columns; returns list of missing columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning(
            "Missing columns %s in %s (available: %s)",
            missing,
            context or "DataFrame",
            list(df.columns)[:10],
        )
    return missing
