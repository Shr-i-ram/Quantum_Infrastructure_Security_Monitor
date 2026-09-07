"""Tests for QISM V0.5 parallel execution, worker fault isolation, and retries (Requirements 26–30, 72–74, 159, 190)."""

import pathlib
import pytest

from qism.config import QKDConfig, Scenario, apply_overrides, default_config
from qism.execution.job import SimulationJob
from qism.execution.scheduler import JobScheduler
from qism.execution.worker import execute_simulation_job
from qism.types import JobStatus


def test_worker_captures_exceptions_without_crashing(tmp_path):
    """A single bad job fails cleanly and returns error metadata (Requirements 29, 71)."""
    base = default_config()
    # Create an invalid config that will fail physical validation in RunManager
    bad_cfg = apply_overrides(base, {
        "protocol.n_pulses": 2000,
        "telemetry.window_size": 1000,
    })
    # Force negative efficiency directly
    bad_cfg.detectors.detector_0.efficiency = -0.5

    scen = Scenario(scenario_id="bad_scen", infrastructure_id="dev_bad", seed=999, config=bad_cfg)
    job = SimulationJob(
        scenario=scen,
        run_id="run_bad",
        seed=123,
        output_dir=tmp_path / "bad",
        checkpoint_dir=tmp_path / "bad_ckpt",
    )

    res = execute_simulation_job(job)
    assert res.status == JobStatus.FAILED
    assert res.error_type is not None
    assert res.stack_trace is not None
    assert not res.is_success


def test_scheduler_fault_tolerance(tmp_path):
    """Scheduler executes good jobs even when one job in the batch fails (Requirements 29, 30, 159, 190)."""
    base = default_config()
    good_cfg = apply_overrides(base, {
        "protocol.n_pulses": 2000,
        "telemetry.window_size": 1000,
        "simulation.chunk_size": 1000,
        "simulation.checkpoint_interval": 1000,
        "attack.enabled": False,
    })
    bad_cfg = apply_overrides(base, {
        "protocol.n_pulses": 2000,
        "telemetry.window_size": 1000,
    })
    bad_cfg.detectors.detector_0.efficiency = -0.5  # Will cause failure

    scen_good = Scenario(scenario_id="scen_good", infrastructure_id="dev_good", seed=1, config=good_cfg)
    scen_bad = Scenario(scenario_id="scen_bad", infrastructure_id="dev_bad", seed=2, config=bad_cfg)

    jobs = [
        SimulationJob(scenario=scen_good, run_id="run_01", seed=101, output_dir=tmp_path / "g1", checkpoint_dir=tmp_path / "g1_ckpt"),
        SimulationJob(scenario=scen_bad, run_id="run_02", seed=102, output_dir=tmp_path / "b1", checkpoint_dir=tmp_path / "b1_ckpt"),
        SimulationJob(scenario=scen_good, run_id="run_03", seed=103, output_dir=tmp_path / "g2", checkpoint_dir=tmp_path / "g2_ckpt"),
    ]

    manifest_file = tmp_path / "job_manifest.json"
    scheduler = JobScheduler(jobs, worker_count=2, max_retries=1, state_file=manifest_file)
    results = scheduler.run(mode="parallel")

    assert len(results) == 3
    good_res = [r for r in results if r.scenario_id == "scen_good"]
    bad_res = [r for r in results if r.scenario_id == "scen_bad"]

    assert all(r.is_success for r in good_res)
    assert len(bad_res) == 1
    assert bad_res[0].status == JobStatus.FAILED

    # Check manifest state tracking
    summary = scheduler.summary()
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["pending"] == 0
