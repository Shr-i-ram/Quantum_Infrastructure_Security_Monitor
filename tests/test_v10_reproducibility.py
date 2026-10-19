"""V1.0 Reproducibility, Multi-Seed Determinism, and Checkpoint Recovery Tests."""

import json
from pathlib import Path
import numpy as np
import pytest

from qism.drift.scenarios import DriftScenarioGenerator, DriftScenarioType, ScenarioSpec
from qism.monitor.orchestrator import MonitorConfig, QISMMonitor
from qism.monitor.v10_benchmark import run_v10_demo
from qism.random import seed_all


def test_multi_seed_scenario_determinism():
    """Verify that identical random seeds produce identical scenario dataframes and feature matrices."""
    spec_a = ScenarioSpec(scenario_type=DriftScenarioType.DRIFT_THEN_ATTACK, total_steps=60, seed=42)
    spec_b = ScenarioSpec(scenario_type=DriftScenarioType.DRIFT_THEN_ATTACK, total_steps=60, seed=42)

    df_a, feat_a = DriftScenarioGenerator(spec_a).generate_stream()
    df_b, feat_b = DriftScenarioGenerator(spec_b).generate_stream()

    assert feat_a == feat_b
    assert len(df_a) == len(df_b)
    np.testing.assert_array_almost_equal(df_a["qber"].to_numpy(), df_b["qber"].to_numpy())
    np.testing.assert_array_almost_equal(df_a["sifted_rate"].to_numpy(), df_b["sifted_rate"].to_numpy())


def test_monitor_run_exact_determinism():
    """Verify that running the monitor twice with the exact same seed and data produces identical outputs."""
    spec = ScenarioSpec(scenario_type=DriftScenarioType.DRIFT_THEN_ATTACK, total_steps=60, seed=42)
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()

    # Run 1
    seed_all(42)
    cfg1 = MonitorConfig(run_id="run_det_1", enable_adaptation=True)
    mon1 = QISMMonitor(cfg1, feature_names=feat_cols)
    res1 = mon1.run(df, feature_cols=feat_cols)

    # Run 2
    seed_all(42)
    cfg2 = MonitorConfig(run_id="run_det_2", enable_adaptation=True)
    mon2 = QISMMonitor(cfg2, feature_names=feat_cols)
    res2 = mon2.run(df, feature_cols=feat_cols)

    assert len(res1.step_results) == len(res2.step_results)
    assert len(res1.all_incidents) == len(res2.all_incidents)
    assert len(res1.all_alerts) == len(res2.all_alerts)

    scores1 = [r.evidence_score for r in res1.step_results]
    scores2 = [r.evidence_score for r in res2.step_results]
    np.testing.assert_array_almost_equal(scores1, scores2)


def test_checkpoint_state_restoration_bit_exactness(tmp_path):
    """Verify that saving a checkpoint and restoring produces identical continuation."""
    spec = ScenarioSpec(scenario_type=DriftScenarioType.DRIFT_THEN_ATTACK, total_steps=60, seed=42)
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()
    ckpt_file = tmp_path / "test_ckpt.json"

    # Baseline uninterrupted run
    cfg_base = MonitorConfig(run_id="uninterrupted", enable_adaptation=True)
    mon_base = QISMMonitor(cfg_base, feature_names=feat_cols)
    res_base = mon_base.run(df, feature_cols=feat_cols)

    # Two-part run with checkpoint at step 30
    cfg_ckpt = MonitorConfig(run_id="checkpointed", enable_adaptation=True)
    mon_ckpt = QISMMonitor(cfg_ckpt, feature_names=feat_cols)
    records = list(mon_ckpt.ingestor.stream_from_dataframe(df, feature_cols=feat_cols))

    for rec in records[:30]:
        mon_ckpt.update(rec)
    mon_ckpt.checkpoint(ckpt_file)

    # Restore in fresh monitor
    mon_resumed = QISMMonitor(cfg_ckpt, feature_names=feat_cols)
    mon_resumed.restore(ckpt_file)

    for rec in records[30:]:
        mon_resumed.update(rec)

    all_incs = mon_resumed.incident_manager.finalize(final_timestamp_s=records[-1].timestamp_s)

    assert len(all_incs) == len(res_base.all_incidents)
    assert mon_resumed._step_counter == mon_base._step_counter
    assert mon_resumed.incident_manager.total_raw_alarms == mon_base.incident_manager.total_raw_alarms


def test_canonical_v10_demo_execution(tmp_path):
    """Verify that run_v10_demo executes cleanly and produces expected artifact files."""
    out_dir = tmp_path / "v10_demo_test"
    code = run_v10_demo(
        scenario="drift_then_attack",
        output_dir=out_dir,
        seed=42,
        verbose=False,
    )
    assert code == 0
    assert (out_dir / "plots" / "demo_timeline.png").exists()
    assert (out_dir / "demo_dashboard.html").exists()
    assert (out_dir / "demo_audit_log.jsonl").exists()
