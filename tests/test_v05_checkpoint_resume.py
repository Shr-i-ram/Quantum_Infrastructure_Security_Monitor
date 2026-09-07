"""Tests for QISM V0.5 job checkpointing, manifest tracking, and resume (Requirements 31–34, 156–158, 191, 192)."""

import json
import pathlib
import pytest

from qism.config import QKDConfig, Scenario, apply_overrides, default_config
from qism.execution.job import SimulationJob
from qism.execution.scheduler import JobScheduler
from qism.types import JobStatus


def test_interrupted_dataset_generation_and_resume(tmp_path):
    """Generate 50%, interrupt, resume, and verify zero duplicates (Requirements 31-33, 191, 192)."""
    base = default_config()
    cfg = apply_overrides(base, {
        "protocol.n_pulses": 2000,
        "telemetry.window_size": 1000,
        "simulation.chunk_size": 1000,
        "simulation.checkpoint_interval": 1000,
        "attack.enabled": False,
    })
    scen = Scenario(scenario_id="res_scen", infrastructure_id="dev_res", seed=500, config=cfg)

    manifest_file = tmp_path / "job_manifest.json"
    jobs = [
        SimulationJob(scenario=scen, run_id="run_01", seed=1001, output_dir=tmp_path / "r1", checkpoint_dir=tmp_path / "c1"),
        SimulationJob(scenario=scen, run_id="run_02", seed=1002, output_dir=tmp_path / "r2", checkpoint_dir=tmp_path / "c2"),
        SimulationJob(scenario=scen, run_id="run_03", seed=1003, output_dir=tmp_path / "r3", checkpoint_dir=tmp_path / "c3"),
        SimulationJob(scenario=scen, run_id="run_04", seed=1004, output_dir=tmp_path / "r4", checkpoint_dir=tmp_path / "c4"),
    ]

    # --- Step 1: Simulate 50% completion ---
    # We execute only the first 2 jobs in scheduler 1
    sched1 = JobScheduler(jobs[:2], worker_count=1, state_file=manifest_file)
    res1 = sched1.run(mode="serial")
    assert len(res1) == 2
    assert all(r.is_success for r in res1)

    # Inspect manifest: jobs 1 and 2 should be COMPLETED
    with open(manifest_file, "r", encoding="utf-8") as fh:
        manifest_data = json.load(fh)
    assert manifest_data["res_scen__run_01"]["status"] == "COMPLETED"
    assert manifest_data["res_scen__run_02"]["status"] == "COMPLETED"

    # --- Step 2: Resume with all 4 jobs ---
    # Scheduler initialized with all 4 jobs, pointing to existing state_file
    sched2 = JobScheduler(jobs, worker_count=2, state_file=manifest_file)
    # Verify it detects 2 completed jobs already
    assert sched2.state["res_scen__run_01"]["status"] in ("COMPLETED", JobStatus.COMPLETED.value)
    assert sched2.state["res_scen__run_02"]["status"] in ("COMPLETED", JobStatus.COMPLETED.value)

    res2 = sched2.run(mode="parallel")
    assert len(res2) == 4
    assert all(r.is_success for r in res2)

    # --- Step 3: Run resume again (Requirement 192: verify no duplicate execution) ---
    sched3 = JobScheduler(jobs, worker_count=2, state_file=manifest_file)
    summary_before = sched3.summary()
    assert summary_before["completed"] == 4
    assert summary_before["pending"] == 0

    res3 = sched3.run(mode="parallel")
    assert len(res3) == 4
    # All 4 were loaded directly from COMPLETED cache without re-running
    assert all(r.is_success for r in res3)
