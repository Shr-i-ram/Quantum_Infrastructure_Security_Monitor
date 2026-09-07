"""Tests for QISM V0.5 dataset sharding, shard manifests, and reading (Requirements 82–84)."""

import pathlib
import pandas as pd
import pytest

from qism.config import QKDConfig, Scenario, apply_overrides, default_config
from qism.datasets.sharding import (
    DatasetShardWriter,
    list_sharded_telemetry_files,
    load_sharded_telemetry,
)
from qism.execution.job import SimulationJob
from qism.execution.worker import execute_simulation_job


def test_dataset_shard_writer_partitioning(tmp_path):
    writer = DatasetShardWriter(tmp_path / "split_normal", shard_size=3)

    # 3 runs per shard
    s0 = writer.get_shard_dir(0)
    s1 = writer.get_shard_dir(1)
    s2 = writer.get_shard_dir(2)
    s3 = writer.get_shard_dir(3)

    assert s0.name == "shard_0000"
    assert s1.name == "shard_0000"
    assert s2.name == "shard_0000"
    assert s3.name == "shard_0001"


def test_shard_execution_and_reading(tmp_path):
    base = default_config()
    cfg = apply_overrides(base, {
        "protocol.n_pulses": 2000,
        "telemetry.window_size": 1000,
        "simulation.chunk_size": 1000,
        "simulation.checkpoint_interval": 1000,
        "attack.enabled": False,
    })
    scen = Scenario(scenario_id="shard_scen", infrastructure_id="dev_shard", seed=321, config=cfg)

    split_dir = tmp_path / "sharded_dataset"
    writer = DatasetShardWriter(split_dir, shard_size=2)

    # Execute 3 runs -> Shard 0 (runs 0, 1), Shard 1 (run 2)
    for run_idx in range(3):
        shard_dir = writer.get_shard_dir(run_idx)
        run_id = f"run_{run_idx:03d}"
        job = SimulationJob(
            scenario=scen,
            run_id=run_id,
            seed=100 + run_idx,
            output_dir=shard_dir,
            checkpoint_dir=shard_dir / "checkpoints",
            shard_id=shard_dir.name,
        )
        res = execute_simulation_job(job)
        assert res.is_success
        writer.record_run_in_shard(shard_dir.name, scen.scenario_id, run_id, str(shard_dir))

    # Finalize manifests
    writer.finalize_all_manifests()

    shard0_manifest = split_dir / "shard_0000" / "shard_manifest.json"
    shard1_manifest = split_dir / "shard_0001" / "shard_manifest.json"
    assert shard0_manifest.exists()
    assert shard1_manifest.exists()

    # Discover sharded telemetry files
    files = list_sharded_telemetry_files(split_dir)
    assert len(files) == 3

    # Load unified sharded telemetry DataFrame
    df = load_sharded_telemetry(split_dir)
    assert df is not None
    assert len(df) >= 3  # At least 1 window per run
    assert set(df["run_id"].unique()) == {"run_000", "run_001", "run_002"}
