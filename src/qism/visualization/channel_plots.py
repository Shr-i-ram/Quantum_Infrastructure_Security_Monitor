"""Quantum channel loss modeling, Beer-Lambert attenuation, and distance scaling plots."""

from __future__ import annotations

import pathlib
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qism.visualization.statistics import evaluate_theoretical_transmission
from qism.visualization.style import COLORS, FIGURE_SIZES, save_figure


def plot_channel_transmission(
    scenario_manifest: dict[str, Any] | None,
    telemetry_df: pd.DataFrame | None,
    out_dir: pathlib.Path,
    filename_stem: str = "03_channel_transmission_and_loss",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Figure 03: Quantum channel attenuation law, distance scaling, and dark-count rate floor."""
    fig, (ax_trans, ax_rate) = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_panel"])

    # 1. Beer-Lambert Transmittance vs Distance
    distances_km = np.linspace(0, 60, 200)
    alphas = [0.18, 0.20, 0.25]
    colors = ["#2563EB", "#059669", "#D97706"]

    for alpha, color in zip(alphas, colors):
        trans = evaluate_theoretical_transmission(distances_km, alpha_db_km=alpha, l_sys_db=1.0)
        ax_trans.plot(
            distances_km, trans,
            lw=2.2, color=color,
            label=rf"$\alpha = {alpha:.2f}\,\mathrm{{dB/km}}$ ($L_{{\mathrm{{sys}}}} = 1.0\,\mathrm{{dB}}$)"
        )

    # Overlay actual scenario instances if scenario_manifest is available
    if scenario_manifest and "scenarios" in scenario_manifest:
        scens = scenario_manifest["scenarios"]
        pts_d = []
        pts_t = []
        for s in scens:
            ch = s.get("config", {}).get("channel", {})
            d = ch.get("distance_km")
            a = ch.get("attenuation_db_per_km", 0.2)
            ins = ch.get("insertion_loss_db", 1.0)
            if d is not None:
                t = 10.0 ** (-(a * d + ins) / 10.0)
                pts_d.append(d)
                pts_t.append(t)
        if pts_d:
            ax_trans.scatter(
                pts_d, pts_t,
                color=COLORS["eve"], s=55, zorder=5, edgecolors="white",
                label="Simulated Scenario Instances"
            )

    ax_trans.set_yscale("log")
    ax_trans.set_xlabel("Fiber Distance L [km]", fontsize=11, fontweight="bold")
    ax_trans.set_ylabel("Channel Transmittance T(L) (log scale)", fontsize=11, fontweight="bold")
    ax_trans.set_title(
        r"A: Channel Loss: $T(L) = 10^{-(\alpha L + L_{\mathrm{sys}})/10}$",
        fontsize=12, fontweight="bold", pad=10
    )
    ax_trans.grid(True, which="both", linestyle="--", alpha=0.6)
    ax_trans.legend(loc="upper right", fontsize=9.5)

    # 2. Expected Detection Rate vs Distance (showing signal drop + dark count floor)
    f_rep = 1e6  # 1 MHz repetition rate
    mu = 0.5     # mean photon number
    eta_det = 0.15  # 15% detector efficiency
    r_dark = 2000.0  # 2 kHz dark count per detector (4 kHz total)

    t_loss = evaluate_theoretical_transmission(distances_km, alpha_db_km=0.20, l_sys_db=1.0)
    signal_rate = f_rep * mu * t_loss * eta_det
    total_rate = signal_rate + 2 * r_dark

    ax_rate.plot(distances_km, total_rate, lw=2.5, color=COLORS["alice"], label="Total Observed Count Rate")
    ax_rate.plot(distances_km, signal_rate, lw=1.8, linestyle="--", color=COLORS["bob"], label="Signal Detection Rate")
    ax_rate.axhline(2 * r_dark, lw=1.8, linestyle=":", color=COLORS["dark_count"], label=rf"Dark Count Noise Floor ($2 R_{{\mathrm{{dark}}}} = {int(2*r_dark)}\,\mathrm{{Hz}}$)")

    ax_rate.set_yscale("log")
    ax_rate.set_xlabel("Fiber Distance L [km]", fontsize=11, fontweight="bold")
    ax_rate.set_ylabel("Expected Detection Rate [Hz] (log scale)", fontsize=11, fontweight="bold")
    ax_rate.set_title(
        r"B: Expected Rate: $R_{\mathrm{det}}(L) = f_{\mathrm{rep}}\mu T(L)\eta_{\mathrm{det}} + 2R_{\mathrm{dark}}$",
        fontsize=12, fontweight="bold", pad=10
    )
    ax_rate.grid(True, which="both", linestyle="--", alpha=0.6)
    ax_rate.legend(loc="upper right", fontsize=9.5)

    fig.suptitle(
        "Quantum Channel Physics: Optical Attenuation and Dark-Count Noise Floor",
        fontsize=14, fontweight="bold", y=1.02
    )

    return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)
