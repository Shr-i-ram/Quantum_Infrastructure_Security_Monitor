"""V0.1 regression — the simulator with all V0.2 detector dynamics disabled
(dead time off, afterpulsing off) must reproduce the V0.1 baseline statistics
and emit neutral V0.2 schema columns; enabling dynamics at the default
operating point must be only a small perturbation of the V0.1 results.
"""

from __future__ import annotations

import numpy as np
import pytest

from helpers import make_config, make_legacy_config, run_events, run_windows


def test_legacy_mode_v01_baseline_gain() -> None:
    """The V0.1-documented signal gain (~0.0187 at the default operating
    point) is reproduced exactly by the legacy (vectorized) engine."""
    cfg = make_legacy_config()
    ev = run_events(cfg, 200_000, seed=20)
    sig = ev[ev["intensity_class"] == "signal"]
    q_obs = (sig["detector_0_click"] | sig["detector_1_click"]).mean()
    eta_ch = 10.0 ** (-(0.20 * 25.0 + 1.0) / 10.0)
    p_dark = -np.expm1(-1000.0 * cfg.receiver.effective_detection_interval_s)
    q_theory = 1.0 - (1.0 - (1.0 - np.exp(-0.5 * eta_ch * 0.15))) * (1.0 - p_dark) ** 2
    assert q_obs == pytest.approx(q_theory, rel=0.03)
    assert q_obs == pytest.approx(0.01866, rel=0.05)  # V0.1 documented value


def test_legacy_mode_v01_qber() -> None:
    """V0.1 QBER ~= p_mis = sin^2(1 deg)."""
    cfg = make_legacy_config()
    ev = run_events(cfg, 300_000, seed=21)
    sifted = ev["alice_basis"].to_numpy() == ev["bob_basis"].to_numpy()
    single = ev["detector_0_click"].to_numpy() != ev["detector_1_click"].to_numpy()
    key = sifted & single
    n_e = np.sum(ev["bit_error"].to_numpy(dtype=np.float64) == 1.0)
    qber = n_e / key.sum()
    assert qber == pytest.approx(np.sin(np.deg2rad(1.0)) ** 2, abs=0.001)


def test_legacy_mode_emits_neutral_v02_columns() -> None:
    """V0.1 event records carry neutral V0.2 detector-dynamics values so the
    schema is identical in both modes."""
    cfg = make_legacy_config()
    ev = run_events(cfg, 5_000, seed=22)
    assert (ev["detector_0_state"] == "ready").all()
    assert (ev["detector_1_state"] == "ready").all()
    assert (ev["detector_0_dead_time_remaining_s"] == 0.0).all()
    assert (ev["detector_1_dead_time_remaining_s"] == 0.0).all()
    assert not ev["dead_time_loss"].any()
    assert (ev["detector_0_candidate_count"] == 0).all()
    assert (ev["detector_1_candidate_count"] == 0).all()
    assert not ev["afterpulse_candidate"].any()
    assert not ev["afterpulse_detected"].any()


def test_dynamics_are_small_perturbation_at_default_operating_point() -> None:
    """At the default operating point (tau ~ 50 ns << T_rep = 1 us, p_ap =
    0.01) the stateful engine's signal-class detector rates stay within a few
    percent of V0.1. (Signal-class rates are used — all-class means mix the
    three intensity classes and add noise.)"""
    n = 300_000
    legacy = run_events(make_legacy_config(), n, seed=23)
    stateful = run_events(make_config(), n, seed=23)
    for ev in (legacy, stateful):
        assert (ev["intensity_class"] == "signal").sum() > 0
    for col in ("detector_0_click", "detector_1_click"):
        sig_legacy = legacy[legacy["intensity_class"] == "signal"]
        sig_stateful = stateful[stateful["intensity_class"] == "signal"]
        r_legacy = sig_legacy[col].mean()
        r_stateful = sig_stateful[col].mean()
        assert r_legacy > 0
        # statistical noise at 300k pulses is ~1%; allow 5% for the small
        # dead-time/afterpulse effects plus sampling
        assert abs(r_stateful - r_legacy) / r_legacy < 0.05

    def qber_of(ev) -> float:
        sifted = ev["alice_basis"].to_numpy() == ev["bob_basis"].to_numpy()
        single = ev["detector_0_click"].to_numpy() != ev["detector_1_click"].to_numpy()
        key = sifted & single
        n_e = np.sum(ev["bit_error"].to_numpy(dtype=np.float64) == 1.0)
        return n_e / key.sum()

    assert abs(qber_of(stateful) - qber_of(legacy)) < 0.0015


def test_window_features_identical_schema_both_modes() -> None:
    """The window feature set is identical whether dynamics are on or off."""
    legacy = run_windows(make_legacy_config(), 30_000, seed=24)
    stateful = run_windows(make_config(), 30_000, seed=24)
    assert set(legacy.columns) == set(stateful.columns)
    for col in (
        "dead_time_loss_rate",
        "detector_0_busy_fraction",
        "effective_dead_time",
        "afterpulse_rate_ground_truth",
        "fano_factor",
    ):
        assert col in legacy.columns


def test_legacy_checkpoint_roundtrip() -> None:
    """Checkpoint/resume still works in the V0.1-compatible mode."""
    import pathlib

    import pyarrow.parquet as pq

    from qism.simulation.runner import RunManager

    tmp = pathlib.Path(__file__).resolve().parent / "_tmp_v01_ckpt"
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    cfg = make_legacy_config(
        **{
            "protocol.n_pulses": 1000,
            "simulation.chunk_size": 500,
            "simulation.checkpoint_interval": 500,
            "simulation.master_seed": 42,
        }
    )
    mgr = RunManager(cfg, "v01_reg", output_root=tmp / "data", checkpoint_root=tmp / "ckpt")
    mgr.run(max_pulses=500)
    mgr.run(resume=True, max_pulses=1000)
    events = pq.read_table(mgr.run_dir / "events.parquet").to_pandas()
    assert len(events) == 1000
    assert events["pulse_id"].tolist() == list(range(1000))
    assert (events["detector_0_state"] == "ready").all()
    shutil.rmtree(tmp, ignore_errors=True)
