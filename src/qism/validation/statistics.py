"""Statistical validation: empirical estimators and report generation.

Runs short simulations, compares empirical quantities against the analytical
model, and produces the validation figures (matplotlib only, no seaborn) plus
a Markdown/JSON report under ``reports/``.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless figure generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from qism import __version__ as SIMULATOR_VERSION
from qism.config import QKDConfig, apply_overrides
from qism.detector.detector import Detector
from qism.detector.probe import run_probe
from qism.random import derive_run_seed, make_rng
from qism.simulation.simulator import Simulator
from qism.validation.analytical import (
    channel_transmittance,
    gain_total,
    gain_vacuum,
    interarrival_mean_nonparalyzable,
    interarrival_std_nonparalyzable,
    misalignment_probability,
    observed_rate_nonparalyzable,
    tau_eff_rate_dependent,
)

log = logging.getLogger("qism.validation")

FIG_DIR_NAME = "figures"


# --------------------------------------------------------------------------- #
# Empirical estimators
# --------------------------------------------------------------------------- #
def run_events(
    config: QKDConfig,
    n_pulses: int,
    seed: int,
    chunk_size: int | None = None,
    run_id: str = "validation",
) -> pd.DataFrame:
    """Run a short simulation and return the event DataFrame (no persistence)."""
    sim = Simulator(config, run_id=run_id, seed=derive_run_seed(seed, run_id))
    cs = chunk_size or min(config.simulation.chunk_size, n_pulses)
    frames = []
    for start in range(0, n_pulses, cs):
        n = min(cs, n_pulses - start)
        chunk_index = start // cs
        frames.append(sim.simulate_chunk(chunk_index, n, start))
    return pd.concat(frames, ignore_index=True)


def empirical_frequencies(values: np.ndarray, n_labels: int = 2) -> np.ndarray:
    counts = np.bincount(values.astype(int), minlength=n_labels)
    return counts / counts.sum()


def empirical_gain(events: pd.DataFrame, cls: str) -> float:
    mask = events["intensity_class"].to_numpy() == cls
    n_sent = int(mask.sum())
    if n_sent == 0:
        return float("nan")
    n_click = int(
        (
            mask & (events["detector_0_click"].to_numpy() | events["detector_1_click"].to_numpy())
        ).sum()
    )
    return n_click / n_sent


def empirical_yields(events: pd.DataFrame, max_photon_number: int) -> dict[int, float]:
    clicked = events["detector_0_click"].to_numpy() | events["detector_1_click"].to_numpy()
    pn = events["photon_number"].to_numpy()
    yields_: dict[int, float] = {}
    for n in range(0, max_photon_number + 1):
        mask = pn == n
        n_sent = int(mask.sum())
        n_click = int(clicked[mask].sum())
        yields_[n] = n_click / n_sent if n_sent else float("nan")
    return yields_


def empirical_qber(events: pd.DataFrame, basis: int | None = None) -> float:
    sifted = events["alice_basis"].to_numpy() == events["bob_basis"].to_numpy()
    single = events["detector_0_click"].to_numpy() != events["detector_1_click"].to_numpy()
    be = events["bit_error"].to_numpy(dtype=np.float64)
    key = sifted & single
    if basis is not None:
        key = key & (events["alice_basis"].to_numpy() == basis)
    n_s = int(key.sum())
    if n_s == 0:
        return float("nan")
    return float(np.nansum(np.where(key, be, np.nan))) / n_s


# --------------------------------------------------------------------------- #
# Figure helpers
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, outdir: pathlib.Path, name: str) -> pathlib.Path:
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("figure written: %s", path)
    return path


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #
def generate_validation_report(
    config: QKDConfig,
    outdir: pathlib.Path,
    n_pulses: int = 200_000,
    seed: int = 20240101,
) -> dict[str, Any]:
    """Run the V0.1 analytical validation suite and write figures + report.

    Produces (in ``outdir/figures``):
        01_poisson_photon_number.png
        02_channel_transmittance_vs_distance.png
        03_gain_vs_detector_efficiency.png
        04_vacuum_gain_vs_dark_count_rate.png
        05_timing_distribution.png
        06_detector_imbalance_distribution.png
        07_qber_vs_misalignment.png
        08_signal_decoy_vacuum_gains.png
    """
    outdir = pathlib.Path(outdir)
    figdir = outdir / FIG_DIR_NAME
    figdir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "simulator_version": SIMULATOR_VERSION,
        "n_pulses_per_run": n_pulses,
        "seed": seed,
        "config_hash": config.config_hash(),
        "figures": [],
        "results": {},
    }

    # ---- 1. Poisson photon-number statistics (signal class, mu=0.5) ----
    ev = run_events(config, n_pulses, seed)
    sig = ev[ev["intensity_class"] == "signal"]
    counts, edges = np.histogram(sig["photon_number"].to_numpy(), bins=np.arange(-0.5, 9.5, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    p_theory = np.array([_pmf(k, 0.5) for k in range(len(counts))])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        centers,
        counts / counts.sum(),
        width=0.8,
        alpha=0.7,
        label="empirical P(n) (signal, mu=0.5)",
    )
    ax.plot(centers, p_theory, "r.-", label="theory Poisson(0.5)")
    ax.set_title("Poisson photon-number distribution (weak coherent pulse)")
    ax.set_xlabel("Photon number n")
    ax.set_ylabel("Probability P(n)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    report["figures"].append(str(_save(fig, figdir, "01_poisson_photon_number")))
    report["results"]["poisson_max_deviation"] = float(
        np.max(np.abs(counts / counts.sum() - p_theory))
    )

    # ---- 2. Channel transmittance vs distance ----
    distances = [0.0, 5.0, 10.0, 25.0, 50.0]
    alpha = config.channel.attenuation_db_per_km
    l_sys = config.channel.insertion_loss_db
    sim_means = []
    for d in distances:
        cfg = apply_overrides(config, {"channel.distance_km": d})
        evd = run_events(cfg, n_pulses // 4, seed)
        sim_means.append(float(np.mean(evd["channel_transmittance"])))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dd = np.linspace(0, 50, 200)
    ax.plot(
        dd,
        [channel_transmittance(alpha, x, l_sys) for x in dd],
        "b-",
        label="theory 10^(-(alpha*L+L_sys)/10)",
    )
    ax.plot(distances, sim_means, "ko", label="simulated (mean eta_ch)")
    ax.set_title("Channel transmittance vs distance")
    ax.set_xlabel("Distance L [km]")
    ax.set_ylabel(r"Channel transmittance $\eta_{ch}$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    report["figures"].append(str(_save(fig, figdir, "02_channel_transmittance_vs_distance")))

    # ---- 3. Gain vs detector efficiency ----
    etas = [0.05, 0.10, 0.15, 0.20, 0.30]
    mu_sig = config.source.mu_signal
    p_dark = 1 - np.exp(
        -config.detectors.detector_0.dark_count_rate_hz
        * config.receiver.effective_detection_interval_s
    )
    theory_gain = [_gain_vs_eta(mu_sig, config, e, p_dark) for e in etas]
    sim_gains = []
    for e in etas:
        cfg = apply_overrides(
            config, {"detectors.detector_0.efficiency": e, "detectors.detector_1.efficiency": e}
        )
        evd = run_events(cfg, n_pulses // 2, seed)
        sim_gains.append(empirical_gain(evd, "signal"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(etas, theory_gain, "b-", label="theory Q = 1 - exp(-mu eta_ch eta_d) (with dark)")
    ax.plot(etas, sim_gains, "ko", label="simulated signal gain")
    ax.set_title("Signal gain vs detector efficiency")
    ax.set_xlabel("Detector efficiency $\\eta_d$")
    ax.set_ylabel("Signal gain $Q_{signal}$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    report["figures"].append(str(_save(fig, figdir, "03_gain_vs_detector_efficiency")))

    # ---- 4. Vacuum gain vs dark count rate ----
    # At the default operating point (1 ns gate) vacuum gain is ~1e-6, which
    # needs tens of millions of pulses to resolve. This figure validates the
    # model relation P_dark = 1 - exp(-lambda * Delta_t) at a wider
    # measurement window (Delta_t = 100 ns), where the effect is measurable.
    meas_interval = 1e-7
    rates = [1e3, 2e3, 5e3, 1e4, 2e4]
    theory_vac = []
    sim_vac = []
    for r in rates:
        p0 = 1 - np.exp(-r * meas_interval)
        theory_vac.append(gain_vacuum(p0, p0))
        cfg = apply_overrides(
            config,
            {
                "receiver.detection_interval_s": meas_interval,
                "detectors.detector_0.dark_count_rate_hz": r,
                "detectors.detector_1.dark_count_rate_hz": r,
            },
        )
        evd = run_events(cfg, n_pulses * 3, seed)
        sim_vac.append(empirical_gain(evd, "vacuum"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rates, theory_vac, "b-", label="theory 1 - (1-p_dark)^2")
    ax.plot(rates, sim_vac, "ko", label="simulated vacuum gain")
    ax.set_title("Vacuum gain vs dark count rate (Delta_t = 100 ns)")
    ax.set_xlabel("Dark count rate $\\lambda_{dark}$ [Hz]")
    ax.set_ylabel("Vacuum gain $Q_{vacuum}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    report["figures"].append(str(_save(fig, figdir, "04_vacuum_gain_vs_dark_count_rate")))

    # ---- 5. Timing distribution (signal arrivals vs Gaussian jitter) ----
    offs = ev["timing_offset_s"].to_numpy()
    offs = offs[~np.isnan(offs)]
    sigma = config.detectors.detector_0.timing_jitter_s
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(offs * 1e12, bins=60, density=True, alpha=0.7, label="empirical arrival offsets")
    xs = np.linspace(-4 * sigma * 1e12, 4 * sigma * 1e12, 400)
    ax.plot(xs, norm.pdf(xs, 0, sigma * 1e12), "r-", label=f"theory N(0, {sigma * 1e12:g} ps)")
    ax.set_title("Timing distribution (arrival offset)")
    ax.set_xlabel("Arrival offset $t_{arr} - t_{exp}$ [ps]")
    ax.set_ylabel("Probability density [1/ps]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    report["figures"].append(str(_save(fig, figdir, "05_timing_distribution")))
    report["results"]["timing_empirical_std_s"] = float(np.std(offs, ddof=0))
    report["results"]["timing_configured_sigma_s"] = sigma

    # ---- 6. Detector imbalance distribution ----
    ws = config.telemetry.window_size
    n_windows = len(ev) // ws
    imbalances = []
    for w in range(n_windows):
        sub = ev.iloc[w * ws : (w + 1) * ws]
        c0 = int(sub["detector_0_click"].sum())
        c1 = int(sub["detector_1_click"].sum())
        if c0 + c1:
            imbalances.append((c0 - c1) / (c0 + c1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(imbalances, bins=30, alpha=0.7, label=f"per-window imbalance (N={len(imbalances)})")
    ax.axvline(0, color="r", ls="--", label="theory (symmetric detectors)")
    ax.set_title("Detector imbalance distribution")
    ax.set_xlabel("Detector imbalance $\\Delta_c = (C_0 - C_1)/(C_0 + C_1)$")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    report["figures"].append(str(_save(fig, figdir, "06_detector_imbalance_distribution")))
    report["results"]["imbalance_mean"] = float(np.mean(imbalances)) if imbalances else float("nan")

    # ---- 7. QBER vs polarization misalignment ----
    mis = [0.0, 0.5, 1.0, 2.0, 3.0]
    sim_qber = []
    for m in mis:
        cfg = apply_overrides(config, {"channel.misalignment_deg": m})
        evd = run_events(cfg, n_pulses, seed)
        sim_qber.append(empirical_qber(evd))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mm = np.linspace(0, 3, 100)
    ax.plot(
        mm, [misalignment_probability(x) for x in mm], "b-", label="theory p_mis = sin^2(theta)"
    )
    ax.plot(mis, sim_qber, "ko", label="simulated sifted QBER")
    ax.set_title("QBER vs polarization misalignment")
    ax.set_xlabel("Misalignment $\\theta$ [deg]")
    ax.set_ylabel("QBER")
    ax.grid(True, alpha=0.3)
    ax.legend()
    report["figures"].append(str(_save(fig, figdir, "07_qber_vs_misalignment")))

    # ---- 8. Signal / decoy / vacuum gains ----
    gains = {cls: empirical_gain(ev, cls) for cls in ("signal", "decoy", "vacuum")}
    mus = {
        "signal": config.source.mu_signal,
        "decoy": config.source.mu_decoy,
        "vacuum": config.source.mu_vacuum,
    }
    eta_ch = channel_transmittance(alpha, config.channel.distance_km, l_sys)
    theory8 = {
        cls: gain_total(mus[cls], eta_ch, config.detectors.detector_0.efficiency, p_dark, p_dark)
        for cls in gains
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(3)
    ax.bar(
        xs - 0.15, [gains[c] for c in ("signal", "decoy", "vacuum")], width=0.3, label="simulated"
    )
    ax.bar(
        xs + 0.15,
        [theory8[c] for c in ("signal", "decoy", "vacuum")],
        width=0.3,
        label="theory",
        alpha=0.8,
    )
    ax.set_xticks(xs, ["signal", "decoy", "vacuum"])
    ax.set_title("Gain by intensity class (default operating point)")
    ax.set_xlabel("Intensity class")
    ax.set_ylabel("Gain $Q_I$")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    report["figures"].append(str(_save(fig, figdir, "08_signal_decoy_vacuum_gains")))

    report["results"]["gains"] = gains

    # ---- Markdown report ----
    md = _render_markdown(report)
    (outdir / "validation_report.md").write_text(md, encoding="utf-8")
    (outdir / "validation_summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


def _pmf(k: int, mu: float) -> float:
    import math

    return math.exp(-mu) * mu**k / math.factorial(k)


def _gain_vs_eta(mu: float, config: QKDConfig, eta_d: float, p_dark: float) -> float:
    eta_ch = channel_transmittance(
        config.channel.attenuation_db_per_km,
        config.channel.distance_km,
        config.channel.insertion_loss_db,
    )
    return gain_total(mu, eta_ch, eta_d, p_dark, p_dark)


def _render_markdown(report: dict[str, Any]) -> str:
    res = report.get("results", {})
    lines = [
        "# QISM V0.2 Validation Report",
        "",
        f"- Simulator version: {report['simulator_version']}",
        f"- Pulses per run: {report['n_pulses_per_run']}",
        f"- Master seed: {report['seed']}",
        f"- Configuration hash: {report['config_hash']}",
        "",
        "## Results",
        "",
    ]
    for key, value in res.items():
        lines.append(f"- **{key}**: {value}")
    det = report.get("detector_validation", {})
    if det:
        lines += ["", "## Detector-dynamics validation (V0.2)", ""]
        for key, value in det.items():
            if key == "figures":
                continue
            lines.append(f"- **{key}**: {value}")
    lines += ["", "## Figures", ""]
    for f in report.get("figures", []):
        lines.append(f"- `{f}`")
    for f in det.get("figures", []):
        lines.append(f"- `{f}`")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# V0.2 detector-dynamics validation (free-running probe + QKD engine)
# --------------------------------------------------------------------------- #
def _probe_detector(
    *,
    dead_time_model: str = "fixed",
    tau_dead_s: float = 50e-9,
    tau_0_s: float = 50e-9,
    delta_tau_s: float = 50e-9,
    lambda_c_hz: float = 1e6,
    dead_time_enabled: bool = True,
    afterpulse_enabled: bool = False,
    p_afterpulse: float = 0.0,
    tau_afterpulse_s: float = 100e-9,
) -> Detector:
    """A standalone detector for free-running validation experiments."""
    from qism.config import AfterpulseConfig, DeadTimeConfig, DetectorConfig

    dcfg = DetectorConfig(
        label="probe", efficiency=1.0, dark_count_rate_hz=0.0, timing_jitter_s=0.0
    )
    dt = DeadTimeConfig(
        enabled=dead_time_enabled,
        model=dead_time_model,
        tau_dead_s=tau_dead_s,
        tau_0_s=tau_0_s,
        delta_tau_s=delta_tau_s,
        lambda_c_hz=lambda_c_hz,
    )
    ap = AfterpulseConfig(
        enabled=afterpulse_enabled,
        probability=p_afterpulse,
        tau_afterpulse_s=tau_afterpulse_s,
    )
    return Detector(dcfg, index=0, dead_time=dt, afterpulse=ap)


def generate_detector_validation_report(
    config: QKDConfig,
    outdir: pathlib.Path,
    seed: int = 20240101,
    n_qkd_pulses: int = 300_000,
) -> dict[str, Any]:
    """V0.2 detector-dynamics validation experiments.

    The detector state machine is validated in isolation with a free-running
    Poisson probe (sorted arrivals), where closed-form results exist:

      * observed rate     r_obs = lambda / (1 + lambda*tau)   (nonparalyzable)
      * inter-arrival     E[dt] = tau + 1/lambda, std(dt) = 1/lambda
      * afterpulse delay  mean(delay) = tau_AP (exponential)
      * rate-dependent    tau_eff(lambda) = tau_0 + delta_tau * lambda/(lambda+lambda_c)
      * busy fraction     B = N_clicks * tau / T_obs (disjoint intervals)

    Then the QKD pulse-train engine is run at the configured operating point
    to verify the emergent window features (dead-time loss, busy fraction,
    inter-arrival cutoff, effective dead time, Fano factor, autocorrelation).
    Produces figures 09-13 in ``outdir/figures``.
    """
    outdir = pathlib.Path(outdir)
    figdir = outdir / FIG_DIR_NAME
    figdir.mkdir(parents=True, exist_ok=True)
    rng = make_rng(seed)
    results: dict[str, Any] = {}
    figures: list[str] = []

    # ---- 9. dead-time rate suppression ------------------------------------
    tau = 50e-9
    rates = [2e4, 1e5, 4e5]
    obs_rates: list[float] = []
    for lam in rates:
        det = _probe_detector(dead_time_model="fixed", tau_dead_s=tau)
        run = run_probe(det, rng, rate_hz=lam, duration_s=1.0)
        obs_rates.append(run.observed_rate_hz)
    theory_rates = [observed_rate_nonparalyzable(lam, tau) for lam in rates]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lam_dense = np.logspace(4, 6.5, 200)
    ax.plot(
        lam_dense,
        [observed_rate_nonparalyzable(x, tau) for x in lam_dense],
        "b-",
        label="theory r = lambda/(1+lambda*tau)",
    )
    ax.plot(rates, obs_rates, "ko", label="simulated (probe)")
    ax.set_title(f"Nonparalyzable dead time: observed rate vs input rate (tau={tau * 1e9:g} ns)")
    ax.set_xlabel("Input Poisson rate $\\lambda$ [Hz]")
    ax.set_ylabel("Observed rate $r_{obs}$ [Hz]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figures.append(str(_save(fig, figdir, "09_dead_time_rate_suppression")))
    rel_dev = [abs(o - t) / t for o, t in zip(obs_rates, theory_rates)]
    results["dead_time_rate_max_rel_deviation"] = max(rel_dev)
    results["dead_time_observed_rates"] = dict(zip(rates, obs_rates))

    # ---- 10. inter-arrival distribution (dead-time cutoff) ----------------
    lam_ia = 1e5
    det = _probe_detector(dead_time_model="fixed", tau_dead_s=tau)
    run = run_probe(det, rng, rate_hz=lam_ia, duration_s=1.5)
    gaps = np.diff(run.click_times)
    gaps = gaps[gaps > 0]
    mean_ia = float(np.mean(gaps)) if gaps.size else float("nan")
    std_ia = float(np.std(gaps, ddof=1)) if gaps.size > 1 else float("nan")
    min_ia = float(np.min(gaps)) if gaps.size else float("nan")
    theory_mean = interarrival_mean_nonparalyzable(lam_ia, tau)
    theory_std = interarrival_std_nonparalyzable(lam_ia)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if gaps.size:
        ax.hist(gaps * 1e9, bins=60, density=True, alpha=0.7, label="observed gaps")
        ax.axvline(tau * 1e9, color="r", ls="--", label=f"dead-time cutoff tau={tau * 1e9:g} ns")
    ax.set_title("Inter-arrival times: nonparalyzable dead-time cutoff")
    ax.set_xlabel("Inter-arrival gap $\\Delta t$ [ns]")
    ax.set_ylabel("Density [1/ns]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figures.append(str(_save(fig, figdir, "10_interarrival_dead_time_cutoff")))
    results["interarrival_mean_s"] = mean_ia
    results["interarrival_mean_theory_s"] = theory_mean
    results["interarrival_std_s"] = std_ia
    results["interarrival_std_theory_s"] = theory_std
    results["interarrival_min_s"] = min_ia
    # the ideal model gives min gap >= tau (accepted clicks are spaced by >= tau)
    results["interarrival_min_above_tau"] = bool(not np.isnan(min_ia) and min_ia >= tau * 0.999)

    # ---- 11. afterpulse delay distribution --------------------------------
    tau_ap = 100e-9
    det = _probe_detector(
        dead_time_enabled=False,
        afterpulse_enabled=True,
        p_afterpulse=0.1,
        tau_afterpulse_s=tau_ap,
    )
    run = run_probe(det, rng, rate_hz=2e4, duration_s=3.0)
    delays = run.afterpulse_delays
    n_ap = run.n_afterpulse_clicks
    mean_delay = float(np.mean(delays)) if delays.size else float("nan")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if delays.size:
        ax.hist(delays * 1e9, bins=40, density=True, alpha=0.7, label=f"observed (N={n_ap})")
        xs = np.linspace(0, 5 * tau_ap * 1e9, 200)
        ax.plot(xs, np.exp(-xs / (tau_ap * 1e9)) / (tau_ap * 1e9), "r-", label="theory Exp(tau_AP)")
    ax.set_title("Afterpulse delay distribution (exponential model)")
    ax.set_xlabel("Afterpulse delay $\\Delta t_{AP}$ [ns]")
    ax.set_ylabel("Density [1/ns]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figures.append(str(_save(fig, figdir, "11_afterpulse_delay_distribution")))
    results["afterpulse_n"] = int(n_ap)
    results["afterpulse_mean_delay_s"] = mean_delay
    results["afterpulse_tau_s"] = tau_ap
    results["afterpulse_mean_delay_rel_dev"] = (
        abs(mean_delay - tau_ap) / tau_ap if not np.isnan(mean_delay) else float("nan")
    )

    # ---- 12. rate-dependent effective dead time ---------------------------
    # tau_eff is inferred from the per-window MINIMUM accepted inter-arrival
    # gap: after recovery the detector accepts the next arrival, so the
    # smallest gap in a window is tau_eff + O(1/(lambda*N)) (bias < 1% of tau
    # for the rates used here). The first rate window starts at tau_0; the
    # windows that follow use the tau_eff selected for the estimated rate.
    rd_rates = [1e5, 3e5, 1e6, 3e6]
    tau_0, delta_tau, lambda_c = 50e-9, 50e-9, 1e6
    rd_obs: list[float] = []
    rd_theory: list[float] = []
    rd_window_s = 0.25
    for lam in rd_rates:
        det = _probe_detector(
            dead_time_model="rate_dependent",
            tau_0_s=tau_0,
            delta_tau_s=delta_tau,
            lambda_c_hz=lambda_c,
        )
        run = run_probe(det, rng, rate_hz=lam, duration_s=1.0, rate_window_s=rd_window_s)
        ests: list[float] = []
        for w in range(1, 4):  # windows 1..3 (window 0 used tau_0)
            lo = w * rd_window_s
            hi = (w + 1) * rd_window_s
            ts = run.click_times[(run.click_times >= lo) & (run.click_times < hi)]
            if ts.size > 1:
                g = np.diff(ts)
                g = g[g > 0]
                if g.size:
                    ests.append(float(g.min()))
        tau_est = float(np.mean(ests)) if ests else float("nan")
        rd_obs.append(tau_est)
        rd_theory.append(tau_eff_rate_dependent(lam, tau_0, delta_tau, lambda_c))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lam_dense2 = np.logspace(3, 7, 200)
    ax.plot(
        lam_dense2,
        [tau_eff_rate_dependent(x, tau_0, delta_tau, lambda_c) * 1e9 for x in lam_dense2],
        "b-",
        label="theory tau_eff(lambda)",
    )
    ax.plot(rd_rates, [t * 1e9 for t in rd_obs], "ko", label="inferred (min inter-arrival gap)")
    ax.set_title("Rate-dependent effective dead time")
    ax.set_xlabel("Candidate rate $\\lambda$ [Hz]")
    ax.set_ylabel(r"$\tau_{eff}$ [ns]")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figures.append(str(_save(fig, figdir, "12_rate_dependent_dead_time")))
    rd_rel = [abs(o - t) / t for o, t in zip(rd_obs, rd_theory) if not np.isnan(o) and t > 0]
    results["rate_dependent_tau_eff_max_rel_deviation"] = max(rd_rel) if rd_rel else float("nan")
    results["rate_dependent_tau_eff_inferred"] = dict(zip(rd_rates, rd_obs))
    results["rate_dependent_tau_eff_theory"] = dict(zip(rd_rates, rd_theory))

    # ---- 13. QKD pulse-train engine: emergent window features -------------
    ev = run_events(config, n_qkd_pulses, seed)
    from qism.telemetry.windows import WindowAggregator

    agg = WindowAggregator(config)
    feat_rows = []
    for wid, sub in ev.groupby("window_id", sort=False):
        feat_rows.append(agg.aggregate(sub))
    feats = pd.DataFrame(feat_rows)
    dtl = feats["dead_time_loss_rate"].to_numpy()
    eff = feats["effective_dead_time"].to_numpy()
    busy0 = feats["detector_0_busy_fraction"].to_numpy()
    fano = feats["fano_factor"].to_numpy()
    acf1 = feats["autocorr_lag1"].to_numpy()

    # inter-arrival histogram of D0 clicks across the whole run
    arr0 = ev["detector_0_arrival_time_s"].dropna().to_numpy()
    gaps0 = np.diff(np.sort(arr0))
    gaps0 = gaps0[gaps0 > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(feats["window_id"], dtl, "o-", label="dead_time_loss_rate")
    ax.plot(feats["window_id"], busy0, "s-", label="detector_0_busy_fraction")
    ax.set_title("QKD engine: dead-time loss and busy fraction per window")
    ax.set_xlabel("Window id")
    ax.set_ylabel("Fraction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax = axes[1]
    if gaps0.size:
        ax.hist(gaps0 * 1e9, bins=60, alpha=0.7)
        cfg_dt = config.detectors.dead_time
        cutoff = cfg_dt.tau_0_s if cfg_dt.model == "rate_dependent" else cfg_dt.tau_dead_s
        ax.axvline(cutoff * 1e9, color="r", ls="--", label=f"tau_eff baseline {cutoff * 1e9:g} ns")
    ax.set_title("D0 inter-arrival gaps (pulse train)")
    ax.set_xlabel("Gap [ns]")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    figures.append(str(_save(fig, figdir, "13_qkd_engine_detector_dynamics")))

    results.update(
        {
            "qkd_dead_time_loss_rate_mean": float(np.nanmean(dtl)),
            "qkd_effective_dead_time_mean_s": float(np.nanmean(eff)),
            "qkd_busy_fraction_mean": float(np.nanmean(busy0)),
            "qkd_fano_factor_mean": float(np.nanmean(fano)),
            "qkd_autocorr_lag1_mean": float(np.nanmean(acf1)),
            "qkd_d0_interarrival_min_s": float(np.min(gaps0)) if gaps0.size else float("nan"),
            "qkd_d0_interarrival_mean_s": float(np.mean(gaps0)) if gaps0.size else float("nan"),
        }
    )

    return {"figures": figures, **results}
