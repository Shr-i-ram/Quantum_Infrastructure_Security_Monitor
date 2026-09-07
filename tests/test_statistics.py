"""Analytical relations: gains, yields, decoy residual, QBER vs misalignment."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import make_config, run_events
from qism.validation.analytical import (
    gain_total,
    gain_vacuum,
    misalignment_probability,
    q_mu_pred,
)

N = 200_000


def test_signal_gain_matches_coherent_state_theory() -> None:
    cfg = make_config()
    ev = run_events(cfg, N, seed=20)
    sig = ev[ev["intensity_class"] == "signal"]
    clicked = sig["detector_0_click"] | sig["detector_1_click"]
    q_obs = clicked.mean()
    eta_ch = 10.0 ** (-(0.20 * 25.0 + 1.0) / 10.0)
    p_dark = -np.expm1(-1000.0 * cfg.receiver.effective_detection_interval_s)
    q_theory = gain_total(0.5, eta_ch, 0.15, p_dark, p_dark)
    assert q_obs == pytest.approx(q_theory, rel=0.03)


def test_decoy_gain_matches_theory() -> None:
    cfg = make_config()
    ev = run_events(cfg, N, seed=21)
    dec = ev[ev["intensity_class"] == "decoy"]
    q_obs = (dec["detector_0_click"] | dec["detector_1_click"]).mean()
    eta_ch = 10.0 ** (-6.0 / 10.0)
    p_dark = -np.expm1(-1000.0 * cfg.receiver.effective_detection_interval_s)
    q_theory = gain_total(0.1, eta_ch, 0.15, p_dark, p_dark)
    # ~30k decoy pulses -> ~110 clicks; Poisson noise is ~9%, use rel=0.15
    assert q_obs == pytest.approx(q_theory, rel=0.15)


def test_vacuum_gain_matches_dark_only_theory() -> None:
    """Vacuum gain is tiny at the default operating point (~2e-6); use a wider
    detection interval so the effect is measurable and validate the model
    relation P_dark = 1 - exp(-lambda * Delta_t)."""
    cfg = make_config(
        **{
            "receiver.detection_interval_s": 1e-7,
            "detectors.detector_0.dark_count_rate_hz": 1e4,
            "detectors.detector_1.dark_count_rate_hz": 1e4,
        }
    )
    ev = run_events(cfg, 4_000_000, seed=22)
    vac = ev[ev["intensity_class"] == "vacuum"]
    q_obs = (vac["detector_0_click"] | vac["detector_1_click"]).mean()
    p_dark = -np.expm1(-1e4 * 1e-7)
    assert q_obs == pytest.approx(gain_vacuum(p_dark, p_dark), rel=0.15)
    assert q_obs > 0


def test_decoy_residual_small() -> None:
    """Q_mu_pred = sum_n P(n|mu) Y_n reconstructs the observed gain."""
    cfg = make_config()
    ev = run_events(cfg, N, seed=23)
    clicked = ev["detector_0_click"].to_numpy() | ev["detector_1_click"].to_numpy()
    pn = ev["photon_number"].to_numpy()
    yields = {}
    for n in range(0, cfg.source.max_photon_number + 1):
        mask = pn == n
        n_sent = int(mask.sum())
        yields[n] = clicked[mask].sum() / n_sent if n_sent else float("nan")
    for cls, mu in (("signal", 0.5), ("decoy", 0.1), ("vacuum", 0.0)):
        mask = ev["intensity_class"].to_numpy() == cls
        q_obs = clicked[mask].sum() / mask.sum()
        q_pred = q_mu_pred(yields, mu, cfg.source.max_photon_number)
        assert abs(q_obs - q_pred) < 1e-3, f"{cls}: |R| too large: {q_obs - q_pred}"


def test_yield_zero_equals_vacuum_dark_rate() -> None:
    """Y_0 (click | zero photons) is the dark-only detection probability."""
    cfg = make_config(
        **{
            "receiver.detection_interval_s": 1e-7,
            "detectors.detector_0.dark_count_rate_hz": 1e4,
            "detectors.detector_1.dark_count_rate_hz": 1e4,
        }
    )
    ev = run_events(cfg, N, seed=24)
    pn = ev["photon_number"].to_numpy()
    clicked = ev["detector_0_click"].to_numpy() | ev["detector_1_click"].to_numpy()
    mask = pn == 0
    y0 = clicked[mask].sum() / mask.sum()
    p_dark = -np.expm1(-1e4 * 1e-7)
    assert y0 == pytest.approx(gain_vacuum(p_dark, p_dark), rel=0.15)
    assert y0 > 0


def test_qber_increases_with_misalignment() -> None:
    """QBER increases with misalignment and matches p_mis = sin^2(theta).

    Uses a low-loss channel and 300k pulses so enough sifted key bits are
    collected to resolve small angles.
    """
    cfg_base = {"channel.distance_km": 0.0, "source.mu_signal": 1.0}
    qbers = []
    for m in (0.5, 1.0, 2.0, 3.0, 5.0):
        cfg_m = make_config(**{**cfg_base, "channel.misalignment_deg": m})
        ev = run_events(cfg_m, 300_000, seed=25)
        sifted = ev["alice_basis"].to_numpy() == ev["bob_basis"].to_numpy()
        single = ev["detector_0_click"].to_numpy() != ev["detector_1_click"].to_numpy()
        key = sifted & single
        n_e = np.sum(ev["bit_error"].to_numpy(dtype=np.float64) == 1.0)
        qbers.append(n_e / key.sum())
    assert qbers[0] < qbers[1] < qbers[2] < qbers[3] < qbers[4]
    # dominant term: QBER ~ p_mis (statistical error shrinks with angle)
    for m, q in zip((2.0, 3.0, 5.0), qbers[2:]):
        assert q == pytest.approx(misalignment_probability(m), abs=0.003)


def test_photon_statistics_features() -> None:
    from qism.telemetry.features import window_features

    cfg = make_config()
    ev = run_events(cfg, 20_000, seed=26)
    feats = window_features(
        ev,
        window_size=cfg.telemetry.window_size,
        max_photon_number=cfg.source.max_photon_number,
        mu_signal=cfg.source.mu_signal,
        mu_decoy=cfg.source.mu_decoy,
        mu_vacuum=cfg.source.mu_vacuum,
    )
    # vacuum fraction across all classes (Poisson mixture) is not 0
    assert 0.05 < feats["vacuum_fraction"] < 0.9
    assert feats["mean_photon_number"] > 0.3
    # mean photon number is the mixture mean
    mu = 0.8 * 0.5 + 0.15 * 0.1 + 0.05 * 0.0
    assert feats["mean_photon_number"] == pytest.approx(mu, abs=0.01)


# --------------------------------------------------------------------------- #
# V0.2 detector-dynamics analytical validation (free-running probe)
# --------------------------------------------------------------------------- #
def _probe_detector(**overrides):
    from qism.config import AfterpulseConfig, DeadTimeConfig, DetectorConfig
    from qism.detector.detector import Detector

    defaults = dict(
        dead_time_enabled=True,
        dead_time_model="fixed",
        tau_dead_s=50e-9,
        tau_0_s=50e-9,
        delta_tau_s=50e-9,
        lambda_c_hz=1e6,
        afterpulse_enabled=False,
        p_afterpulse=0.0,
        tau_afterpulse_s=100e-9,
    )
    defaults.update(overrides)
    dcfg = DetectorConfig(
        label="probe", efficiency=1.0, dark_count_rate_hz=0.0, timing_jitter_s=0.0
    )
    dt = DeadTimeConfig(
        enabled=defaults["dead_time_enabled"],
        model=defaults["dead_time_model"],
        tau_dead_s=defaults["tau_dead_s"],
        tau_0_s=defaults["tau_0_s"],
        delta_tau_s=defaults["delta_tau_s"],
        lambda_c_hz=defaults["lambda_c_hz"],
    )
    ap = AfterpulseConfig(
        enabled=defaults["afterpulse_enabled"],
        probability=defaults["p_afterpulse"],
        tau_afterpulse_s=defaults["tau_afterpulse_s"],
    )
    return Detector(dcfg, index=0, dead_time=dt, afterpulse=ap)


def test_dead_time_observed_rate_matches_theory() -> None:
    """V0.2: r_obs = lambda / (1 + lambda*tau) for a nonparalyzable detector
    driven by a Poisson probe (no gating, no afterpulses)."""
    from qism.detector.probe import run_probe
    from qism.random import make_rng
    from qism.validation.analytical import observed_rate_nonparalyzable

    lam, tau = 2e5, 50e-9
    det = _probe_detector(tau_dead_s=tau)
    run = run_probe(det, make_rng(31), rate_hz=lam, duration_s=0.5)
    theory = observed_rate_nonparalyzable(lam, tau)
    assert run.observed_rate_hz == pytest.approx(theory, rel=0.05)
    # accepted clicks on the same detector are never closer than tau
    gaps = np.diff(run.click_times)
    if gaps.size:
        assert gaps.min() >= tau * 0.999


def test_interarrival_mean_matches_theory() -> None:
    """V0.2: E[dt] = tau + 1/lambda and std(dt) = 1/lambda for the ideal
    nonparalyzable model."""
    from qism.detector.probe import run_probe
    from qism.random import make_rng
    from qism.validation.analytical import (
        interarrival_mean_nonparalyzable,
        interarrival_std_nonparalyzable,
    )

    lam, tau = 1e5, 50e-9
    det = _probe_detector(tau_dead_s=tau)
    run = run_probe(det, make_rng(32), rate_hz=lam, duration_s=1.5)
    gaps = np.diff(run.click_times)
    gaps = gaps[gaps > 0]
    assert gaps.size > 500
    assert np.mean(gaps) == pytest.approx(interarrival_mean_nonparalyzable(lam, tau), rel=0.08)
    if gaps.size > 1:
        assert np.std(gaps, ddof=1) == pytest.approx(interarrival_std_nonparalyzable(lam), rel=0.15)


def test_afterpulse_delay_exponential() -> None:
    """V0.2: afterpulse delays are exponential with mean tau_AP."""
    from qism.detector.probe import run_probe
    from qism.random import make_rng

    tau_ap = 100e-9
    det = _probe_detector(
        dead_time_enabled=False,
        afterpulse_enabled=True,
        p_afterpulse=0.2,
        tau_afterpulse_s=tau_ap,
    )
    run = run_probe(det, make_rng(33), rate_hz=2e4, duration_s=1.5)
    assert run.n_afterpulse_clicks > 100
    assert np.mean(run.afterpulse_delays) == pytest.approx(tau_ap, rel=0.15)


def test_rate_dependent_tau_eff_observed() -> None:
    """V0.2: the inferred effective dead time tracks tau_eff(lambda)."""
    from qism.detector.probe import run_probe
    from qism.random import make_rng
    from qism.validation.analytical import tau_eff_rate_dependent

    lam = 3e5
    tau_0, delta_tau, lambda_c = 50e-9, 50e-9, 1e6
    det = _probe_detector(
        dead_time_model="rate_dependent",
        tau_0_s=tau_0,
        delta_tau_s=delta_tau,
        lambda_c_hz=lambda_c,
    )
    run = run_probe(det, make_rng(34), rate_hz=lam, duration_s=1.0, rate_window_s=0.25)
    r_obs = run.observed_rate_hz
    tau_est = (lam / r_obs - 1.0) / lam
    theory = tau_eff_rate_dependent(lam, tau_0, delta_tau, lambda_c)
    assert tau_est == pytest.approx(theory, rel=0.2)
    # and the effect is clearly larger than the baseline tau_0
    assert tau_est > tau_0 * 1.1


def test_busy_fraction_matches_clicks_times_tau() -> None:
    """V0.2: busy fraction = N_clicks * tau / T_obs (disjoint dead intervals)."""
    from qism.detector.probe import run_probe
    from qism.random import make_rng

    lam, tau = 1e5, 50e-9
    det = _probe_detector(tau_dead_s=tau)
    run = run_probe(det, make_rng(35), rate_hz=lam, duration_s=1.0)
    t_obs = run.duration_s
    expected = len(run.click_times) * tau / t_obs
    busy = det.busy_time(0.0, t_obs) / t_obs
    assert busy == pytest.approx(expected, rel=0.02)
    assert 0.0 < busy < 0.05  # ~1% at this operating point


def test_stateful_window_features_sane() -> None:
    """V0.2: QKD-engine window features (dead-time loss, busy fraction,
    inter-arrival, Fano, autocorrelation) are in their physical ranges."""
    from qism.telemetry.features import window_features

    cfg = make_config()  # dynamics enabled
    ev = run_events(cfg, 40_000, seed=36)
    feats = window_features(
        ev,
        window_size=cfg.telemetry.window_size,
        max_photon_number=cfg.source.max_photon_number,
        mu_signal=cfg.source.mu_signal,
        mu_decoy=cfg.source.mu_decoy,
        mu_vacuum=cfg.source.mu_vacuum,
        repetition_rate_hz=cfg.protocol.repetition_rate_hz,
        interarrival_histogram_bins=cfg.telemetry.interarrival_histogram_bins,
        interarrival_histogram_scale=cfg.telemetry.interarrival_histogram_scale,
        fano_sub_bins=cfg.telemetry.fano_sub_bins,
    )
    # dead-time loss rate is a probability (may be NaN with no candidates)
    if not np.isnan(feats["dead_time_loss_rate"]):
        assert 0.0 <= feats["dead_time_loss_rate"] <= 1.0
    assert 0.0 <= feats["detector_0_busy_fraction"] <= 1.0
    assert 0.0 <= feats["detector_1_busy_fraction"] <= 1.0
    assert feats["detector_0_busy_fraction"] + feats["detector_0_ready_fraction"] == pytest.approx(
        1.0
    )
    # effective dead time at the default operating point ~ 50-60 ns
    assert 45e-9 < feats["effective_dead_time"] < 120e-9
    # inter-arrival percentiles are ordered
    assert feats["detector_0_p05_interarrival"] <= feats["detector_0_p50_interarrival"]
    assert feats["detector_0_p50_interarrival"] <= feats["detector_0_p95_interarrival"]
    # Fano factor near the Poisson baseline of 1 for sparse clicks
    assert 0.3 < feats["fano_factor"] < 1.7
    # per-pulse click indicators are ~independent -> lag-1 autocorr near 0
    assert abs(feats["autocorr_lag1"]) < 0.2
    # ground-truth afterpulse aggregates are non-negative or NaN
    if not np.isnan(feats["afterpulse_rate_ground_truth"]):
        assert 0.0 <= feats["afterpulse_rate_ground_truth"] <= 1.0
