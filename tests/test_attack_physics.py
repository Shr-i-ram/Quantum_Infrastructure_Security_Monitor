"""Statistical physical tests for all 6 QISM V0.3 attack models."""

import numpy as np
import pytest

from qism.config import AttackScenarioConfig, default_config
from qism.simulation.runner import RunManager


@pytest.fixture
def base_config():
    cfg = default_config()
    cfg.protocol.n_pulses = 30000
    cfg.telemetry.window_size = 5000
    cfg.simulation.chunk_size = 5000
    cfg.simulation.checkpoint_interval = 5000
    return cfg


def test_intercept_resend_physics(base_config, tmp_path):
    """Verify that Intercept-Resend causally drives QBER toward 25% on intercepted pulses."""
    sc = AttackScenarioConfig(
        scenario_id="phys_ir",
        type="intercept_resend",
        severity=1.0,
        start_pulse=10000,
        end_pulse=25000,
        onset_pulses=1000,
        termination_pulses=1000,
        params={"fraction": 1.0},
    )
    cfg = base_config.model_copy(deep=True)
    cfg.attack.enabled = True
    cfg.attack.scenarios = [sc]
    cfg.simulation.scenario_id = "phys_ir"

    mgr = RunManager(cfg, "run_ir", output_root=tmp_path / "data", checkpoint_root=tmp_path / "ckpt")
    mgr.run(fresh=True)
    df = mgr.writer.read_final_telemetry()

    # Window 0 (0..5000) is baseline normal
    baseline_qber = float(df.iloc[0]["qber"])
    assert baseline_qber < 0.08, f"Baseline QBER should be low, got {baseline_qber}"

    # Window 3 (15000..20000) is fully active IR attack
    active_qber = float(df.iloc[3]["qber"])
    # Quantum mechanics: 50% Eve basis match -> 0% error, 50% Eve basis mismatch -> 50% error -> 25% net (with noise: 18%..40%)
    assert active_qber > 0.18, f"Active IR QBER should exceed 18%, got {active_qber}"
    assert active_qber < 0.40, f"Active IR QBER should be around 25%, got {active_qber}"


def test_channel_loss_physics(base_config, tmp_path):
    """Verify that Channel Loss reduces detection counts and signal gain proportionally."""
    sc = AttackScenarioConfig(
        scenario_id="phys_loss",
        type="channel_loss",
        severity=1.0,
        start_pulse=10000,
        end_pulse=25000,
        onset_pulses=1000,
        termination_pulses=1000,
        params={"delta_L_max_db": 6.0},
    )
    cfg = base_config.model_copy(deep=True)
    cfg.attack.enabled = True
    cfg.attack.scenarios = [sc]
    cfg.simulation.scenario_id = "phys_loss"

    mgr = RunManager(cfg, "run_loss", output_root=tmp_path / "data", checkpoint_root=tmp_path / "ckpt")
    mgr.run(fresh=True)
    df = mgr.writer.read_final_telemetry()

    baseline_gain = float(df.iloc[0]["signal_gain"])
    active_gain = float(df.iloc[3]["signal_gain"])

    # 6 dB extra loss -> factor of 10^(-0.6) ~ 0.251 of baseline transmittance
    assert active_gain < 0.40 * baseline_gain, (
        f"Active gain {active_gain} should drop by ~75% from baseline {baseline_gain}"
    )


def test_source_manipulation_physics(base_config, tmp_path):
    """Verify that source manipulation modulates photon statistics and detection rates."""
    sc_inc = AttackScenarioConfig(
        scenario_id="phys_src_inc",
        type="source_manipulation",
        severity=1.0,
        start_pulse=10000,
        end_pulse=25000,
        onset_pulses=1000,
        termination_pulses=1000,
        params={"delta_mu": 0.4, "direction": "increase"},
    )
    cfg_inc = base_config.model_copy(deep=True)
    cfg_inc.attack.enabled = True
    cfg_inc.attack.scenarios = [sc_inc]

    mgr_inc = RunManager(cfg_inc, "run_src_inc", output_root=tmp_path / "data", checkpoint_root=tmp_path / "ckpt")
    mgr_inc.run(fresh=True)
    df_inc = mgr_inc.writer.read_final_telemetry()

    baseline_count = float(df_inc.iloc[0]["total_count"])
    active_count = float(df_inc.iloc[3]["total_count"])
    assert active_count > 1.25 * baseline_count, (
        f"Source intensity increase should elevate detections: active {active_count} vs baseline {baseline_count}"
    )


def test_detector_manipulation_physics(base_config, tmp_path):
    """Verify that detector efficiency mismatch creates strong detector imbalance."""
    sc_det = AttackScenarioConfig(
        scenario_id="phys_det",
        type="detector_manipulation",
        severity=1.0,
        start_pulse=10000,
        end_pulse=25000,
        onset_pulses=1000,
        termination_pulses=1000,
        params={"mode": "mismatch", "delta_eta": 0.8, "target_detector": 0},
    )
    cfg = base_config.model_copy(deep=True)
    cfg.attack.enabled = True
    cfg.attack.scenarios = [sc_det]

    mgr = RunManager(cfg, "run_det", output_root=tmp_path / "data", checkpoint_root=tmp_path / "ckpt")
    mgr.run(fresh=True)
    df = mgr.writer.read_final_telemetry()

    baseline_imb = float(df.iloc[0]["detector_imbalance"])
    active_imb = float(df.iloc[3]["detector_imbalance"])

    # Suppressing D0 means (D0 - D1)/(D0 + D1) becomes strongly negative
    assert active_imb < -0.4, f"D0 suppression should drive imbalance negative, got {active_imb}"
    assert active_imb < baseline_imb - 0.25, f"Attack should significantly lower imbalance below baseline"


def test_timing_attack_physics(base_config, tmp_path):
    """Verify that timing attack causes measurable arrival time displacement."""
    shift_s = 200.0e-12  # 200 ps
    sc_time = AttackScenarioConfig(
        scenario_id="phys_time",
        type="timing",
        severity=1.0,
        start_pulse=10000,
        end_pulse=25000,
        onset_pulses=1000,
        termination_pulses=1000,
        params={"delta_t_max_s": shift_s},
    )
    cfg = base_config.model_copy(deep=True)
    cfg.attack.enabled = True
    cfg.attack.scenarios = [sc_time]

    mgr = RunManager(cfg, "run_time", output_root=tmp_path / "data", checkpoint_root=tmp_path / "ckpt")
    mgr.run(fresh=True)
    df = mgr.writer.read_final_telemetry()

    baseline_off = float(df.iloc[0]["mean_arrival_offset"])
    active_off = float(df.iloc[3]["mean_arrival_offset"])

    # Mean offset should shift by ~200 ps
    assert abs(baseline_off) < 60e-12, f"Baseline timing offset near 0, got {baseline_off}"
    assert active_off > 100e-12, f"Active timing offset should shift toward 200 ps, got {active_off*1e12} ps"


def test_pns_attack_physics(base_config, tmp_path):
    """Verify that PNS attack disrupts multi-photon yields and gain ratios."""
    sc_pns = AttackScenarioConfig(
        scenario_id="phys_pns",
        type="pns",
        severity=1.0,
        start_pulse=10000,
        end_pulse=25000,
        onset_pulses=1000,
        termination_pulses=1000,
        params={"single_photon_suppression": 0.8},
    )
    cfg = base_config.model_copy(deep=True)
    cfg.attack.enabled = True
    cfg.attack.scenarios = [sc_pns]

    mgr = RunManager(cfg, "run_pns", output_root=tmp_path / "data", checkpoint_root=tmp_path / "ckpt")
    mgr.run(fresh=True)
    df = mgr.writer.read_final_telemetry()

    baseline_gain_ratio = float(df.iloc[0]["signal_gain"] / max(df.iloc[0]["decoy_gain"], 1e-9))
    active_gain_ratio = float(df.iloc[3]["signal_gain"] / max(df.iloc[3]["decoy_gain"], 1e-9))

    # Splitting multi-photons alters the relative gains of signal vs decoy pulses
    assert abs(active_gain_ratio - baseline_gain_ratio) > 0.3, (
        f"PNS attack should perturb signal/decoy gain ratio: active {active_gain_ratio} vs baseline {baseline_gain_ratio}"
    )
