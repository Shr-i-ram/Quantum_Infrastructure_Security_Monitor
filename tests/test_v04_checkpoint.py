"""Checkpoint and recovery tests for V0.4 with dynamic environment + active attack."""

import numpy as np
import pandas as pd
import pytest

from qism.config import AttackScenarioConfig, default_config
from qism.simulation.runner import RunManager


def _read_events(run_dir) -> pd.DataFrame:
    events_path = run_dir / "events.parquet"
    return pd.read_parquet(events_path)


def test_v04_checkpoint_resume_under_drift_and_attack(tmp_path):
    """Resumed simulation under dynamic environment drift AND active attack matches uninterrupted run."""
    cfg_base = default_config()
    cfg_base.simulation.master_seed = 424242
    cfg_base.protocol.n_pulses = 3000
    cfg_base.simulation.chunk_size = 500
    cfg_base.simulation.checkpoint_interval = 500

    # Enable detector dynamics
    cfg_base.detectors.dead_time.enabled = True
    cfg_base.detectors.dead_time.tau_dead_s = 1e-7
    cfg_base.detectors.afterpulse.enabled = True
    cfg_base.detectors.afterpulse.probability = 0.05

    # Enable V0.4 dynamic environment
    cfg_base.environment.type = "dynamic"
    cfg_base.environment.temperature.baseline = 24.0
    cfg_base.environment.temperature.theta = 1.0
    cfg_base.environment.temperature.sigma = 0.3
    cfg_base.environment.clock_drift.baseline = 50.0  # 50 ppm

    # Enable V0.3 attack
    cfg_base.attack.enabled = True
    cfg_base.attack.scenarios = [
        AttackScenarioConfig(
            scenario_id="atk_v04_test",
            type="intercept_resend",
            start_pulse=800,
            end_pulse=2200,
            severity=0.8,
            onset_pulses=200,
            termination_pulses=200,
            params={"fraction": 1.0},
        )
    ]

    # --- Run A: Uninterrupted ---
    run_dir_a = tmp_path / "run_uninterrupted"
    ckpt_dir_a = tmp_path / "ckpt_uninterrupted"
    mgr_a = RunManager(cfg_base, "run_v04_chk", output_root=run_dir_a, checkpoint_root=ckpt_dir_a)
    summary_a = mgr_a.run()
    assert summary_a.status.value == "COMPLETED"
    df_a = _read_events(mgr_a.run_dir)

    # --- Run B: Interrupted at 1500 pulses (inside active attack and drift) ---
    run_dir_b = tmp_path / "run_resumed"
    ckpt_dir_b = tmp_path / "ckpt_resumed"
    mgr_b = RunManager(cfg_base, "run_v04_chk", output_root=run_dir_b, checkpoint_root=ckpt_dir_b)
    summary_b1 = mgr_b.run(max_pulses=1500)
    assert summary_b1.pulses_completed == 1500

    # Resume to completion
    summary_b2 = mgr_b.run(resume=True, max_pulses=3000)
    assert summary_b2.status.value == "COMPLETED"
    assert summary_b2.pulses_completed == 3000
    df_b = _read_events(mgr_b.run_dir)

    # Verify identical length and pulse IDs
    assert len(df_a) == len(df_b)
    np.testing.assert_array_equal(df_a["pulse_id"], df_b["pulse_id"])

    # Verify identical attack states
    np.testing.assert_array_equal(df_a["attack_active"], df_b["attack_active"])
    np.testing.assert_array_equal(df_a["attack_phase"], df_b["attack_phase"])

    # Verify identical detector clicks
    np.testing.assert_array_equal(df_a["detector_0_click"], df_b["detector_0_click"])
    np.testing.assert_array_equal(df_a["detector_1_click"], df_b["detector_1_click"])
    np.testing.assert_array_equal(df_a["click_type"], df_b["click_type"])

    # Verify identical environmental telemetry
    np.testing.assert_allclose(df_a["clock_offset_s"], df_b["clock_offset_s"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(df_a["true_temperature_c"], df_b["true_temperature_c"], rtol=1e-9, atol=1e-12)
