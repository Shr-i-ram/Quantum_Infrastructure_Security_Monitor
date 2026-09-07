"""Test exact bit-identical resumption across checkpoint boundaries during an active attack."""

import pandas as pd
import pytest

from qism.config import AttackScenarioConfig, default_config
from qism.simulation.runner import RunManager


def test_checkpoint_resume_under_active_attack(tmp_path):
    """Run an attack-aware simulation with a checkpoint mid-attack, resume, and verify bit-identical output."""
    total_pulses = 12000
    chunk_size = 2000

    sc = AttackScenarioConfig(
        scenario_id="ckpt_ir",
        type="intercept_resend",
        severity=0.8,
        start_pulse=3000,
        end_pulse=9000,
        onset_pulses=1000,
        termination_pulses=1000,
        params={"fraction": 1.0},
    )

    base = default_config()
    base.protocol.n_pulses = total_pulses
    base.telemetry.window_size = chunk_size
    base.simulation.chunk_size = chunk_size
    base.simulation.checkpoint_interval = chunk_size
    base.simulation.master_seed = 998877
    base.attack.enabled = True
    base.attack.scenarios = [sc]
    base.simulation.scenario_id = "test_ckpt_attack"

    dir_golden = tmp_path / "golden"
    dir_resume = tmp_path / "resume"
    ckpt_golden = tmp_path / "ckpt_golden"
    ckpt_resume = tmp_path / "ckpt_resume"

    # 1. Uninterrupted run (golden reference)
    mgr_golden = RunManager(
        base,
        "run_ckpt",
        output_root=dir_golden,
        checkpoint_root=ckpt_golden,
    )
    res_golden = mgr_golden.run(fresh=True)
    assert res_golden.pulses_completed == total_pulses

    df_events_gold = mgr_golden.writer.read_final_events()
    df_telem_gold = mgr_golden.writer.read_final_telemetry()

    # 2. Interrupted run: simulate first 6000 pulses (active attack phase: 3000..9000)
    cfg_part = base.model_copy(deep=True)
    cfg_part.protocol.n_pulses = 6000

    mgr_part = RunManager(
        cfg_part,
        "run_ckpt",
        output_root=dir_resume,
        checkpoint_root=ckpt_resume,
    )
    res_part = mgr_part.run(fresh=True)
    assert res_part.pulses_completed == 6000

    # 3. Resume the interrupted run to the full 12000 pulses
    cfg_full = base.model_copy(deep=True)
    cfg_full.protocol.n_pulses = total_pulses

    mgr_full = RunManager(
        cfg_full,
        "run_ckpt",
        output_root=dir_resume,
        checkpoint_root=ckpt_resume,
    )
    res_full = mgr_full.run(resume=True)
    assert res_full.pulses_completed == total_pulses

    df_events_resumed = mgr_full.writer.read_final_events()
    df_telem_resumed = mgr_full.writer.read_final_telemetry()

    # Verify bit-identical results between golden uninterrupted and resumed run
    pd.testing.assert_frame_equal(df_events_gold, df_events_resumed)
    pd.testing.assert_frame_equal(df_telem_gold, df_telem_resumed)
