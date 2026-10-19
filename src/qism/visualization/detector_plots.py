"""Single-Photon Avalanche Diode (SPAD) detector physics, dead time, and afterpulsing plots."""

from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qism.visualization.statistics import evaluate_dead_time_saturation
from qism.visualization.style import COLORS, FIGURE_SIZES, save_figure


def plot_detector_physics(
    out_dir: pathlib.Path,
    filename_stem: str = "04_detector_physics_and_dead_time",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Figure 04: SPAD detector saturation, rate-dependent dead time, and afterpulse decay."""
    fig, ((ax_sat, ax_tau), (ax_ap, ax_dark)) = plt.subplots(2, 2, figsize=FIGURE_SIZES["grid_2x2"])

    # 1. Nonparalyzable Dead Time Saturation
    incident_rates = np.logspace(4, 8, 300)  # 10 kHz to 100 MHz
    taus_ns = [50.0, 100.0]
    colors = [COLORS["alice"], COLORS["detector_1"]]

    ax_sat.plot(incident_rates, incident_rates, "k--", lw=1.5, alpha=0.6, label="Ideal Linear Detector (No Dead Time)")

    for tau_ns, c in zip(taus_ns, colors):
        tau_s = tau_ns * 1e-9
        obs = evaluate_dead_time_saturation(incident_rates, tau_dead_s=tau_s)
        sat_limit = 1.0 / tau_s
        ax_sat.plot(
            incident_rates, obs,
            lw=2.2, color=c,
            label=rf"SPAD $\tau_{{\mathrm{{dead}}}} = {tau_ns:.0f}\,\mathrm{{ns}}$ (Sat: {sat_limit/1e6:.1f} MHz)"
        )
        ax_sat.axhline(sat_limit, color=c, linestyle=":", alpha=0.5)

    ax_sat.set_xscale("log")
    ax_sat.set_yscale("log")
    ax_sat.set_xlabel(r"Incident Photon Arrival Rate $\lambda$ [Hz]", fontsize=10.5, fontweight="bold")
    ax_sat.set_ylabel(r"Observed Click Rate $r_{\mathrm{obs}}$ [counts/s]", fontsize=10.5, fontweight="bold")
    ax_sat.set_title(
        r"A: Nonparalyzable Saturation: $r_{\mathrm{obs}} = \frac{\lambda}{1 + \lambda \tau_{\mathrm{dead}}}$",
        fontsize=11.5, fontweight="bold", pad=8
    )
    ax_sat.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_sat.legend(loc="upper left", fontsize=8.5)

    # 2. Rate-Dependent Effective Dead Time
    rates_lambda = np.logspace(3, 7, 300)
    tau_0 = 50.0  # ns
    delta_tau = 50.0  # ns
    lambda_c = 1e6  # 1 MHz
    tau_eff_ns = tau_0 + delta_tau * (rates_lambda / (rates_lambda + lambda_c))

    ax_tau.plot(rates_lambda, tau_eff_ns, lw=2.5, color=COLORS["detector_0"])
    ax_tau.axhline(tau_0, color="gray", linestyle="--", label=rf"Base $\tau_0 = {tau_0:.0f}\,\mathrm{{ns}}$")
    ax_tau.axhline(tau_0 + delta_tau, color=COLORS["eve"], linestyle="--", label=rf"Max $\tau_0 + \Delta\tau = {tau_0+delta_tau:.0f}\,\mathrm{{ns}}$")
    ax_tau.axvline(lambda_c, color="green", linestyle=":", label=rf"Critical Rate $\lambda_c = {lambda_c/1e6:.1f}\,\mathrm{{MHz}}$")

    ax_tau.set_xscale("log")
    ax_tau.set_xlabel(r"Estimated Candidate Rate $\lambda$ [Hz]", fontsize=10.5, fontweight="bold")
    ax_tau.set_ylabel(r"Effective Dead Time $\tau_{\mathrm{eff}}(\lambda)$ [ns]", fontsize=10.5, fontweight="bold")
    ax_tau.set_title(
        r"B: Rate-Dependent Model: $\tau_{\mathrm{eff}} = \tau_0 + \Delta\tau \frac{\lambda}{\lambda + \lambda_c}$",
        fontsize=11.5, fontweight="bold", pad=8
    )
    ax_tau.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_tau.legend(loc="upper left", fontsize=8.5)

    # 3. Exponential Afterpulse Decay
    dt_ns = np.linspace(0, 500, 300)
    tau_ap_ns = 100.0
    p_ap = (1.0 / tau_ap_ns) * np.exp(-dt_ns / tau_ap_ns)

    ax_ap.plot(dt_ns, p_ap, lw=2.5, color=COLORS["afterpulse"], label=rf"Decay: $P(\Delta t) \propto e^{{-\Delta t / \tau_{{\mathrm{{ap}}}}}}$ ($\tau_{{\mathrm{{ap}}}} = {tau_ap_ns:.0f}\,\mathrm{{ns}}$)")
    ax_ap.axvline(50.0, color=COLORS["eve"], linestyle="--", label=r"Dead Time Window (Rejected if $\Delta t < \tau_{\mathrm{dead}}$)")
    ax_ap.fill_between(dt_ns[dt_ns < 50.0], p_ap[dt_ns < 50.0], color=COLORS["eve"], alpha=0.2, label="Dead-Time Suppressed")
    ax_ap.fill_between(dt_ns[dt_ns >= 50.0], p_ap[dt_ns >= 50.0], color=COLORS["afterpulse"], alpha=0.15, label="Observed Afterpulses")

    ax_ap.set_xlabel(r"Time Since Primary Avalanche $\Delta t$ [ns]", fontsize=10.5, fontweight="bold")
    ax_ap.set_ylabel("Probability Density [1/ns]", fontsize=10.5, fontweight="bold")
    ax_ap.set_title(
        r"C: Afterpulse Decay Queue & Dead Time Interception",
        fontsize=11.5, fontweight="bold", pad=8
    )
    ax_ap.grid(True, linestyle="--", alpha=0.5)
    ax_ap.legend(loc="upper right", fontsize=8.5)

    # 4. Dark Count Temperature Scaling
    temps_c = np.linspace(10, 40, 100)
    t_ref = 20.0
    lambda_0 = 1500.0  # Hz baseline dark count
    beta = 0.03  # +3% per deg C
    dark_rates = lambda_0 * (1.0 + beta * (temps_c - t_ref))

    ax_dark.plot(temps_c, dark_rates, lw=2.5, color=COLORS["dark_count"])
    ax_dark.scatter([t_ref], [lambda_0], color=COLORS["eve"], s=70, zorder=5, label=rf"Reference $T_{{\mathrm{{ref}}}} = {t_ref:.0f}^\circ\mathrm{{C}}$")

    ax_dark.set_xlabel(r"SPAD Substrate Temperature $T$ [$^\circ\mathrm{C}$]", fontsize=10.5, fontweight="bold")
    ax_dark.set_ylabel(r"Dark Count Rate $\lambda_{\mathrm{dark}}(T)$ [Hz]", fontsize=10.5, fontweight="bold")
    ax_dark.set_title(
        r"D: Dark Count Thermal Drift: $\lambda_{\mathrm{dark}}(T) = \lambda_0 [1 + \beta (T - T_{\mathrm{ref}})]$",
        fontsize=11.5, fontweight="bold", pad=8
    )
    ax_dark.grid(True, linestyle="--", alpha=0.5)
    ax_dark.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Bob SPAD Detector Model: Dead Time Kinetics, Afterpulsing Dynamics & Thermal Noise",
        fontsize=14, fontweight="bold", y=1.01
    )

    return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)
