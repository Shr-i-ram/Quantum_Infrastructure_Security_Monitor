"""Tests for QISM V0.5 seed hierarchy, determinism, and serial/parallel equivalence (Requirements 23–25, 78–81, 100–103, 188, 189)."""

import pathlib
import shutil
import numpy as np
import pytest

from qism.config import QKDConfig, Scenario, apply_overrides, default_config
from qism.execution.job import SimulationJob
from qism.execution.scheduler import JobScheduler
from qism.execution.worker import execute_simulation_job
from qism.random import (
    derive_dataset_seed,
    derive_run_seed,
    derive_scenario_seed,
)


def test_seed_hierarchy_uniqueness():
    master_seed = 20260501
    d_seed1 = derive_dataset_seed(master_seed, "dataset_A")
    d_seed2 = derive_dataset_seed(master_seed, "dataset_B")
    assert d_seed1 != d_seed2

    s_seed1 = derive_scenario_seed(d_seed1, "scen_0001")
    s_seed2 = derive_scenario_seed(d_seed1, "scen_0002")
    assert s_seed1 != s_seed2

    r_seed1 = derive_run_seed(s_seed1, "run_0001")
    r_seed2 = derive_run_seed(s_seed1, "run_0002")
    assert r_seed1 != r_seed2


def test_simulation_run_determinism(tmp_path):
    base = default_config()
    overrides = {
        "protocol.n_pulses": 5000,
        "telemetry.window_size": 2500,
        "simulation.chunk_size": 2500,
        "simulation.checkpoint_interval": 2500,
        "attack.enabled": False,
    }
    cfg = apply_overrides(base, overrides)
    scen = Scenario(scenario_id="det_scen", infrastructure_id="dev_det", seed=777, config=cfg)

    # Run 1
    job1 = SimulationJob(
        scenario=scen,
        run_id="run_001",
        seed=424242,
        output_dir=tmp_path / "run1",
        checkpoint_dir=tmp_path / "run1_ckpt",
    )
    res1 = execute_simulation_job(job1)
    assert res1.is_success

    # Run 2 with identical seed and run_id
    job2 = SimulationJob(
        scenario=scen,
        run_id="run_001",
        seed=424242,
        output_dir=tmp_path / "run2",
        checkpoint_dir=tmp_path / "run2_ckpt",
    )
    res2 = execute_simulation_job(job2)
    assert res2.is_success

    # Metrics and events must match identically
    assert res1.events_generated == res2.events_generated
    assert res1.metrics.get("raw_detections") == res2.metrics.get("raw_detections")
    assert np.isclose(res1.metrics.get("sifted_qber", 0.0), res2.metrics.get("sifted_qber", 0.0), atol=1e-7)


def test_state_isolation_across_runs(tmp_path):
    """Scenario A -> Scenario B -> Scenario A produces identical results (Requirements 100-103)."""
    base = default_config()
    cfgA = apply_overrides(base, {
        "protocol.n_pulses": 4000,
        "telemetry.window_size": 2000,
        "channel.distance_km": 15.0,
        "detectors.detector_0.efficiency": 0.20,
        "attack.enabled": False,
    })
    scenA = Scenario(scenario_id="scen_A", infrastructure_id="dev_A", seed=1001, config=cfgA)

    cfgB = apply_overrides(base, {
        "protocol.n_pulses": 4000,
        "telemetry.window_size": 2000,
        "channel.distance_km": 40.0,
        "detectors.detector_0.efficiency": 0.10,
        "attack.enabled": False,
    })
    scenB = Scenario(scenario_id="scen_B", infrastructure_id="dev_B", seed=2002, config=cfgB)

    # 1. Run Scenario A (first time)
    jobA1 = SimulationJob(scenario=scenA, run_id="run_0", seed=999, output_dir=tmp_path / "A1", checkpoint_dir=tmp_path / "A1_ckpt")
    resA1 = execute_simulation_job(jobA1)
    assert resA1.is_success

    # 2. Run Scenario B
    jobB = SimulationJob(scenario=scenB, run_id="run_0", seed=888, output_dir=tmp_path / "B", checkpoint_dir=tmp_path / "B_ckpt")
    resB = execute_simulation_job(jobB)
    assert resB.is_success

    # 3. Run Scenario A (second time with same seed)
    jobA2 = SimulationJob(scenario=scenA, run_id="run_0", seed=999, output_dir=tmp_path / "A2", checkpoint_dir=tmp_path / "A2_ckpt")
    resA2 = execute_simulation_job(jobA2)
    assert resA2.is_success

    # Verify A1 and A2 are bit-identical despite B having run in between
    assert resA1.metrics.get("raw_detections") == resA2.metrics.get("raw_detections")
    assert np.isclose(resA1.metrics.get("sifted_qber", 0.0), resA2.metrics.get("sifted_qber", 0.0), atol=1e-7)


def test_serial_vs_parallel_equivalence(tmp_path):
    """Executing jobs with workers=1 vs workers=2 produces identical per-job results (Requirements 78, 79, 189)."""
    base = default_config()
    cfg = apply_overrides(base, {
        "protocol.n_pulses": 3000,
        "telemetry.window_size": 1500,
        "simulation.chunk_size": 1500,
        "simulation.checkpoint_interval": 1500,
        "attack.enabled": False,
    })
    scen1 = Scenario(scenario_id="par_scen1", infrastructure_id="dev_1", seed=111, config=cfg)
    scen2 = Scenario(scenario_id="par_scen2", infrastructure_id="dev_2", seed=222, config=cfg)

    # Create 2 jobs for serial execution
    serial_out = tmp_path / "serial"
    jobs_serial = [
        SimulationJob(scenario=scen1, run_id="run_s1", seed=1001, output_dir=serial_out / "scen1", checkpoint_dir=serial_out / "ckpt1"),
        SimulationJob(scenario=scen2, run_id="run_s2", seed=2002, output_dir=serial_out / "scen2", checkpoint_dir=serial_out / "ckpt2"),
    ]
    sched_serial = JobScheduler(jobs_serial, worker_count=1, state_file=serial_out / "job_manifest.json")
    results_serial = sched_serial.run(mode="serial")
    assert all(r.is_success for r in results_serial)

    # Create 2 jobs for parallel execution with identical seeds
    par_out = tmp_path / "parallel"
    jobs_par = [
        SimulationJob(scenario=scen1, run_id="run_s1", seed=1001, output_dir=par_out / "scen1", checkpoint_dir=par_out / "ckpt1"),
        SimulationJob(scenario=scen2, run_id="run_s2", seed=2002, output_dir=par_out / "scen2", checkpoint_dir=par_out / "ckpt2"),
    ]
    sched_par = JobScheduler(jobs_par, worker_count=2, state_file=par_out / "job_manifest.json")
    results_par = sched_par.run(mode="parallel")
    assert all(r.is_success for r in results_par)

    # Map by job_id and assert equivalence
    res_map_ser = {r.job_id: r for r in results_serial}
    res_map_par = {r.job_id: r for r in results_par}

    for jid in res_map_ser:
        r_s = res_map_ser[jid]
        r_p = res_map_par[jid]
        assert r_s.events_generated == r_p.events_generated
        assert r_s.metrics.get("raw_detections") == r_p.metrics.get("raw_detections")
        assert np.isclose(r_s.metrics.get("sifted_qber", 0.0), r_p.metrics.get("sifted_qber", 0.0), atol=1e-7)
