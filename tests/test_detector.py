"""TEST 6–9 — detector efficiency, dark counts, timing jitter, symmetry; plus
V0.2 detector dynamics: READY/DEAD state machine, nonparalyzable dead time,
recovery, afterpulsing, rate-dependent effective dead time, busy fraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helpers import make_config, make_legacy_config, run_events
from qism.detector.dark_counts import dark_click_probability
from qism.detector.detector import Detector
from qism.random import make_rng
from qism.types import DetectorState


def _make_detector(config, index=0) -> Detector:
    cfg = config.detectors.detector_0 if index == 0 else config.detectors.detector_1
    return Detector(
        cfg,
        index=index,
        gate_width_s=config.receiver.gate_width_s,
        detection_interval_s=config.receiver.effective_detection_interval_s,
        dead_time=config.detectors.dead_time,
        afterpulse=config.detectors.afterpulse,
    )


# --------------------------------------------------------------------------- #
# Core physics (V0.1 relations, independent of the state machine)
# --------------------------------------------------------------------------- #
def test_signal_click_probability_formula() -> None:
    cfg = make_legacy_config()
    det = _make_detector(cfg)
    assert det.signal_click_probability(np.array([0]))[0] == pytest.approx(0.0)
    assert det.signal_click_probability(np.array([1]))[0] == pytest.approx(0.15)
    assert det.signal_click_probability(np.array([2]))[0] == pytest.approx(1 - 0.85**2)
    assert det.signal_click_probability(np.array([10]))[0] == pytest.approx(
        1 - (1 - 0.15) ** 10, rel=1e-9
    )


def test_dark_click_probability() -> None:
    cfg = make_legacy_config()
    det = _make_detector(cfg)
    rate, dt = 1000.0, cfg.receiver.effective_detection_interval_s
    assert det.dark_click_probability() == pytest.approx(-np.expm1(-rate * dt), rel=1e-12)
    assert dark_click_probability(1000.0, 1e-9) == pytest.approx(1e-6, rel=1e-6)


def test_detector_state_api() -> None:
    """V0.2 state machine: READY -> (click) -> DEAD -> (recovery) -> READY."""
    cfg = make_config()  # V0.2 defaults: dead time enabled
    det = _make_detector(cfg)
    assert det.get_state() is DetectorState.READY
    assert det.is_available(0.0) is True
    assert det.dead_time_enabled is True
    assert det.effective_dead_time_s > 0.0

    # an accepted avalanche puts the detector DEAD until t + tau_eff
    tau = det.effective_dead_time_s
    det.register_click(1.0)
    assert det.last_click_time == pytest.approx(1.0)
    assert det.get_state() is DetectorState.DEAD
    assert det.dead_time_remaining(1.0 + tau / 2) == pytest.approx(tau / 2)
    assert det.is_available(1.0) is False
    assert det.is_available(1.0 + tau / 2) is False
    assert det.is_available(1.0 + tau) is True  # recovery at t + tau
    assert det.get_state() is DetectorState.READY
    assert det.dead_time_remaining(2.0 + tau) == pytest.approx(0.0)

    params = det.get_parameters()
    assert params["efficiency"] == pytest.approx(0.15)
    assert params["dead_time_enabled"] is True
    assert params["dead_time_model"] == "rate_dependent"


def test_dead_time_disabled_is_always_ready() -> None:
    cfg = make_legacy_config()
    det = _make_detector(cfg)
    assert det.dead_time_enabled is False
    det.register_click(1.0)
    assert det.get_state() is DetectorState.READY
    assert det.is_available(1.5) is True
    assert det.effective_dead_time_s == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Nonparalyzable dead time
# --------------------------------------------------------------------------- #
def test_nonparalyzable_dead_time_does_not_extend() -> None:
    """A candidate arriving during the dead interval is lost and must NOT
    extend the dead period (the central nonparalyzable property)."""
    cfg = make_config(
        **{
            "detectors.dead_time.model": "fixed",
            "detectors.dead_time.tau_dead_s": 50e-9,
            "detectors.afterpulse.enabled": False,
        }
    )
    det = _make_detector(cfg)
    rng = make_rng(1)
    tau = det.effective_dead_time_s
    assert tau == pytest.approx(50e-9)

    # first click at t=0 -> DEAD until 50 ns
    res1 = det.offer(0.0, rng, kind="probe", p_click=1.0)
    assert res1.clicked
    assert det.dead_until == pytest.approx(50e-9)

    # a candidate inside the dead interval is rejected and does not extend it
    res2 = det.offer(20e-9, rng, kind="probe", p_click=1.0)
    assert not res2.available
    assert res2.loss_reason == "dead_time"
    assert det.dead_until == pytest.approx(50e-9)  # NOT extended
    assert det.last_click_time == pytest.approx(0.0)  # NOT reset
    assert det.total_dead_rejected == 1

    # a candidate at exactly the recovery boundary is accepted
    res3 = det.offer(50e-9, rng, kind="probe", p_click=1.0)
    assert res3.available
    assert res3.clicked
    assert det.total_clicks == 2


def test_dead_time_loss_counter() -> None:
    cfg = make_config(
        **{
            "detectors.dead_time.model": "fixed",
            "detectors.dead_time.tau_dead_s": 100e-9,
            "detectors.afterpulse.enabled": False,
        }
    )
    det = _make_detector(cfg)
    rng = make_rng(2)
    det.offer(0.0, rng, kind="probe", p_click=1.0)
    for t in (10e-9, 30e-9, 70e-9):
        r = det.offer(t, rng, kind="probe", p_click=1.0)
        assert not r.available
    assert det.total_dead_rejected == 3
    assert det.total_clicks == 1


# --------------------------------------------------------------------------- #
# Recovery / busy fraction
# --------------------------------------------------------------------------- #
def test_busy_fraction_from_state_intervals() -> None:
    """busy_time() equals the union of the dead intervals; nonparalyzable
    dead time produces disjoint intervals, so busy = N_clicks * tau."""
    cfg = make_config(
        **{
            "detectors.dead_time.model": "fixed",
            "detectors.dead_time.tau_dead_s": 50e-9,
            "detectors.afterpulse.enabled": False,
        }
    )
    det = _make_detector(cfg)
    rng = make_rng(3)
    for t in (0.0, 100e-9, 200e-9, 300e-9):
        det.offer(t, rng, kind="probe", p_click=1.0)
    assert det.total_clicks == 4
    busy = det.busy_time(0.0, 400e-9)
    assert busy == pytest.approx(4 * 50e-9)
    assert busy == pytest.approx(det.total_clicks * det.effective_dead_time_s)


def test_prune_history_before() -> None:
    cfg = make_config(
        **{
            "detectors.dead_time.model": "fixed",
            "detectors.dead_time.tau_dead_s": 50e-9,
            "detectors.afterpulse.enabled": False,
        }
    )
    det = _make_detector(cfg)
    rng = make_rng(4)
    det.offer(0.0, rng, kind="probe", p_click=1.0)
    det.offer(200e-9, rng, kind="probe", p_click=1.0)
    assert det.busy_time(0.0, 300e-9) == pytest.approx(100e-9)
    det.prune_history_before(100e-9)
    assert det.busy_time(0.0, 300e-9) == pytest.approx(50e-9)  # first interval dropped


# --------------------------------------------------------------------------- #
# Afterpulsing
# --------------------------------------------------------------------------- #
def test_afterpulse_scheduling_and_delay() -> None:
    """A primary avalanche schedules an afterpulse with probability p_ap and an
    exponential delay; the afterpulse does not generate a second generation."""
    cfg = make_config(
        **{
            "detectors.dead_time.enabled": False,
            "detectors.afterpulse.enabled": True,
            "detectors.afterpulse.probability": 0.5,  # enough samples for a short test
            "detectors.afterpulse.tau_afterpulse_s": 100e-9,
        }
    )
    det = _make_detector(cfg)
    rng = make_rng(5)
    delays = []
    for _ in range(2000):
        det.offer(0.0, rng, kind="probe", p_click=1.0)
        while True:
            entry = det.pop_next_pending()
            if entry is None:
                break
            delays.append(entry.delay_s)
            # offering the afterpulse must NOT schedule a second generation
            res = det.offer(entry.t, rng, kind="afterpulse", p_click=1.0)
            assert res.clicked
            assert det.pending_afterpulse_count() == 0
    assert len(delays) > 0
    # empirical fraction of avalanches that produced an afterpulse ~ p_ap
    frac = len(delays) / 2000.0
    assert 0.35 < frac < 0.65
    # mean delay ~ tau_ap (exponential)
    assert np.mean(delays) == pytest.approx(100e-9, rel=0.15)


def test_afterpulse_lost_during_dead_time() -> None:
    """An afterpulse arriving while the detector is still DEAD is lost and
    does not extend the dead period."""
    cfg = make_config(
        **{
            "detectors.dead_time.model": "fixed",
            "detectors.dead_time.tau_dead_s": 200e-9,
            "detectors.afterpulse.enabled": True,
            "detectors.afterpulse.probability": 1.0,  # always schedule
            "detectors.afterpulse.tau_afterpulse_s": 1e-9,  # tiny delay -> inside dead time
        }
    )
    det = _make_detector(cfg)
    rng = make_rng(6)
    res = det.offer(0.0, rng, kind="probe", p_click=1.0)
    assert res.clicked
    # the afterpulse lands at ~1 ns while the detector is DEAD until 200 ns
    entry = det.pending_front()
    assert entry is not None
    assert entry.t < 200e-9
    res2 = det.offer(entry.t, rng, kind="afterpulse", p_click=1.0)
    assert not res2.available
    assert res2.loss_reason == "dead_time"
    assert det.dead_until == pytest.approx(200e-9)  # not extended
    assert det.total_clicks == 1


# --------------------------------------------------------------------------- #
# Rate-dependent effective dead time
# --------------------------------------------------------------------------- #
def test_rate_dependent_tau_eff() -> None:
    cfg = make_config()  # defaults: rate_dependent, tau_0=50ns, delta=50ns, lambda_c=1MHz
    det = _make_detector(cfg)
    assert det.dead_time_cfg.model == "rate_dependent"
    assert det.effective_dead_time_s == pytest.approx(50e-9)  # initial: tau_0
    assert det.tau_eff_at_rate(0.0) == pytest.approx(50e-9)
    # at lambda = lambda_c: tau_eff = 50 + 50 * (1e6/(1e6+1e6)) = 75 ns
    assert det.tau_eff_at_rate(1e6) == pytest.approx(75e-9)
    assert det.tau_eff_at_rate(1e12) == pytest.approx(100e-9, rel=1e-6)
    # set_candidate_rate updates the effective dead time used by new avalanches
    det.set_candidate_rate(3e6)
    assert det.effective_dead_time_s == pytest.approx(50e-9 + 50e-9 * 3.0 / 4.0)
    # fixed model ignores the rate
    cfg2 = make_config(
        **{
            "detectors.dead_time.model": "fixed",
            "detectors.dead_time.tau_dead_s": 80e-9,
        }
    )
    det2 = _make_detector(cfg2)
    assert det2.tau_eff_at_rate(1e12) == pytest.approx(80e-9)
    assert det2.effective_dead_time_s == pytest.approx(80e-9)


# --------------------------------------------------------------------------- #
# V0.1 statistics regression (dynamics disabled -> vectorized engine)
# --------------------------------------------------------------------------- #
def test_detector_efficiency_increases_detection() -> None:
    """TEST 6: increasing eta_d increases the detection probability."""
    n = 100_000
    low = run_events(
        make_legacy_config(
            **{
                "detectors.detector_0.efficiency": 0.05,
                "detectors.detector_1.efficiency": 0.05,
                "detectors.detector_0.dark_count_rate_hz": 0.0,
                "detectors.detector_1.dark_count_rate_hz": 0.0,
            }
        ),
        n,
        seed=1,
    )
    high = run_events(
        make_legacy_config(
            **{
                "detectors.detector_0.efficiency": 0.30,
                "detectors.detector_1.efficiency": 0.30,
                "detectors.detector_0.dark_count_rate_hz": 0.0,
                "detectors.detector_1.dark_count_rate_hz": 0.0,
            }
        ),
        n,
        seed=1,
    )
    rate_low = np.mean(low["detector_0_click"] | low["detector_1_click"])
    rate_high = np.mean(high["detector_0_click"] | high["detector_1_click"])
    assert rate_high > 3 * rate_low


def test_dark_count_increases_vacuum_detection() -> None:
    """TEST 7: increasing lambda_dark increases the vacuum detection rate.

    Vacuum gain is ~1e-6 at the default operating point, so this test uses a
    wider detection interval (100 ns) to make the effect measurable in a
    short run — the model relation P_dark = 1-exp(-lambda*Delta_t) is what is
    being validated.
    """
    n = 200_000
    base = {"receiver.detection_interval_s": 1e-7}
    low = run_events(
        make_legacy_config(
            **{
                **base,
                "detectors.detector_0.dark_count_rate_hz": 1e4,
                "detectors.detector_1.dark_count_rate_hz": 1e4,
            }
        ),
        n,
        seed=2,
    )
    high = run_events(
        make_legacy_config(
            **{
                **base,
                "detectors.detector_0.dark_count_rate_hz": 1e6,
                "detectors.detector_1.dark_count_rate_hz": 1e6,
            }
        ),
        n,
        seed=2,
    )
    vac_low = low[low["intensity_class"] == "vacuum"]
    vac_high = high[high["intensity_class"] == "vacuum"]
    rate_low = np.mean(vac_low["detector_0_click"] | vac_low["detector_1_click"])
    rate_high = np.mean(vac_high["detector_0_click"] | vac_high["detector_1_click"])
    assert rate_low > 0
    assert rate_high > 20 * rate_low
    # matches theory 1-(1-p_dark)^2
    p = -np.expm1(-1e6 * 1e-7)
    assert rate_high == pytest.approx(1 - (1 - p) ** 2, rel=0.05)


def test_timing_jitter_std() -> None:
    """TEST 8: empirical arrival-offset std matches sigma_t."""
    cfg = make_legacy_config(
        **{
            "detectors.detector_0.dark_count_rate_hz": 0.0,
            "detectors.detector_1.dark_count_rate_hz": 0.0,
        }
    )
    ev = run_events(cfg, 200_000, seed=3)
    offs = ev["timing_offset_s"].dropna().to_numpy()
    assert len(offs) > 1000
    sigma = cfg.detectors.detector_0.timing_jitter_s
    assert abs(np.std(offs, ddof=0) - sigma) / sigma < 0.05


def test_detector_symmetry() -> None:
    """TEST 9: symmetric detectors give imbalance statistically centred at 0."""
    cfg = make_legacy_config()
    ev = run_events(cfg, 200_000, seed=4)
    ws = cfg.telemetry.window_size
    imbalances = []
    for w in range(len(ev) // ws):
        sub = ev.iloc[w * ws : (w + 1) * ws]
        c0 = int(sub["detector_0_click"].sum())
        c1 = int(sub["detector_1_click"].sum())
        imbalances.append((c0 - c1) / (c0 + c1))
    imbalances = np.array(imbalances)
    assert abs(imbalances.mean()) < 0.05


def test_detector_per_call_api() -> None:
    """The event-level detect() API matches the vectorized click model.

    Uses the V0.1-compatible mode (dynamics disabled): detect() is the
    stateless per-pulse convenience API.
    """
    cfg = make_legacy_config()
    det = _make_detector(cfg)
    rng = make_rng(99)
    clicked = 0
    for _ in range(20_000):
        n_photons = int(rng.poisson(0.5))
        c, source = det.detect(n_photons, rng, 0.0)
        if c:
            clicked += 1
    # P(click) ~ 1 - exp(-mu*eta_ch*eta_d) is not applicable here (no channel);
    # instead verify the click fraction is in a sane range for eta=0.15, mu=0.5:
    # ~1 - exp(-0.5*0.15) = 0.072 plus dark (~2e-6)
    frac = clicked / 20_000
    assert 0.04 < frac < 0.12


# --------------------------------------------------------------------------- #
# V0.2 stateful engine: events carry dynamics telemetry
# --------------------------------------------------------------------------- #
def test_stateful_engine_emits_dynamics_columns() -> None:
    cfg = make_config()  # dynamics enabled
    ev = run_events(cfg, 20_000, seed=7)
    assert "detector_0_state" in ev.columns
    assert (ev["detector_0_state"].isin(["ready", "dead"])).all()
    assert (ev["detector_1_state"].isin(["ready", "dead"])).all()
    assert (ev["detector_0_dead_time_remaining_s"] >= 0).all()
    assert (ev["detector_1_dead_time_remaining_s"] >= 0).all()
    assert (ev["detector_0_candidate_count"] >= 0).all()
    assert (ev["detector_1_candidate_count"] >= 0).all()
    # afterpulse ground-truth columns exist and are boolean where required
    assert ev["afterpulse_candidate"].dtype == bool
    assert ev["afterpulse_detected"].dtype == bool


def test_stateful_engine_click_origins() -> None:
    cfg = make_config()
    ev = run_events(cfg, 50_000, seed=8)
    valid = {
        "none",
        "d0_signal",
        "d0_dark",
        "d1_signal",
        "d1_dark",
        "d0_signal_dark",
        "d1_signal_dark",
        "d0_afterpulse",
        "d1_afterpulse",
        "double_signal",
        "double_dark",
        "double_mixed",
        "double_mixed_afterpulse",
    }
    assert set(ev["click_type"].unique()) <= valid


def test_stateful_dead_time_suppresses_counts() -> None:
    """With a long dead time, detector counts drop vs. the stateless engine.

    Dark-only detection (mu_signal=0) with per-pulse click probability
    p_dark ~ 0.3 and tau_dead = 2 * T_rep: a click suppresses the next two
    pulses' candidates on the same detector, so the observed count must drop
    well below the stateless count (nonparalyzable model: p_eff = p/(1+2p)).
    """
    common = {
        "detectors.dead_time.model": "fixed",
        "detectors.afterpulse.enabled": False,
        "source.mu_signal": 0.0,
        "source.mu_decoy": 0.0,
        "receiver.detection_interval_s": 1e-7,
        "detectors.detector_0.dark_count_rate_hz": 3.57e6,
        "detectors.detector_1.dark_count_rate_hz": 3.57e6,
    }
    # p_dark = 1-exp(-3.57e6*1e-7) ~ 0.30 per pulse per detector
    stateless = run_events(make_legacy_config(**common), 100_000, seed=9)
    stateful = run_events(
        make_config(**{**common, "detectors.dead_time.tau_dead_s": 2000e-9}), 100_000, seed=9
    )
    c_stateless = stateless["detector_0_click"].sum() + stateless["detector_1_click"].sum()
    c_stateful = stateful["detector_0_click"].sum() + stateful["detector_1_click"].sum()
    assert c_stateless > 1000
    # p_eff = 0.30 / (1 + 2*0.30) ~ 0.19 -> count drops to ~62% of stateless
    assert c_stateful < 0.7 * c_stateless
    # and the stateful run records dead-time losses
    assert int(stateful["detector_0_dead_rejected_count"].sum()) > 0


def test_stateful_engine_deterministic() -> None:
    cfg = make_config()
    a = run_events(cfg, 10_000, seed=11)
    b = run_events(cfg, 10_000, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_simulate_pulse_unavailable_with_dynamics() -> None:
    from qism.random import derive_run_seed
    from qism.simulation.simulator import Simulator

    cfg = make_config()  # dynamics enabled
    sim = Simulator(cfg, "t", seed=derive_run_seed(1, "t"))
    with pytest.raises(NotImplementedError):
        sim.simulate_pulse(0)
