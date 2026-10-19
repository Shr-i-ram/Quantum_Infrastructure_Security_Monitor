"""Physical attack lifecycle, signature profiles, and normal-vs-attack comparative plots."""

from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qism.visualization.style import COLORS, FIGURE_SIZES, save_figure


def plot_attack_lifecycle(
    telemetry_df: pd.DataFrame,
    out_dir: pathlib.Path,
    filename_stem: str = "08_attack_lifecycle_and_timeline",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Figure 08: Causal attack lifecycle timeline across Normal, Onset, Active, Termination, and Recovery."""
    fig, (ax_act, ax_qber, ax_rate) = plt.subplots(3, 1, figsize=FIGURE_SIZES["wide_timeline"], sharex=True)

    windows = telemetry_df.get("window_id", pd.Series(range(len(telemetry_df)))).to_numpy()

    # Activation / Severity
    severity = telemetry_df.get("ground_truth_mean_severity", telemetry_df.get("attack_strength", pd.Series(0.0, index=telemetry_df.index))).to_numpy()
    phases = telemetry_df.get("ground_truth_attack_phase", telemetry_df.get("attack_phase", pd.Series("normal", index=telemetry_df.index))).astype(str).to_numpy()

    # Colors for background shading of phases
    phase_colors = {
        "normal": "#FFFFFF",
        "onset": "#FEF3C7",      # Amber-100
        "active": "#FEE2E2",     # Red-100
        "termination": "#FFEDD5", # Orange-100
        "recovery": "#D1FAE5",   # Emerald-100
    }

    # Shade phase regions
    curr_phase = None
    start_w = 0
    for i, p_str in enumerate(phases):
        p_clean = p_str.lower()
        if p_clean != curr_phase:
            if curr_phase is not None:
                color = phase_colors.get(curr_phase, "#FFFFFF")
                if curr_phase != "normal":
                    for ax in (ax_act, ax_qber, ax_rate):
                        ax.axvspan(start_w - 0.5, i - 0.5, color=color, alpha=0.8, zorder=0)
            curr_phase = p_clean
            start_w = i
    if curr_phase is not None and curr_phase != "normal":
        color = phase_colors.get(curr_phase, "#FFFFFF")
        for ax in (ax_act, ax_qber, ax_rate):
            ax.axvspan(start_w - 0.5, len(phases) - 0.5, color=color, alpha=0.8, zorder=0)

    # Panel 1: Attack Severity & Activation Level
    ax_act.plot(windows, severity, color=COLORS["eve"], lw=2.4, marker="o", markersize=4, label="Attack Severity / Activation Level")
    ax_act.set_ylabel("Activation Factor [0-1]", fontsize=10.5, fontweight="bold")
    ax_act.set_ylim(-0.05, 1.05)
    ax_act.set_title("A: Physical Intervention Lifecycle: Smoothstep Onset -> Active -> Recovery", fontsize=11.5, fontweight="bold", pad=6)
    ax_act.legend(loc="upper right", fontsize=9)
    ax_act.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Quantum Bit Error Rate (QBER)
    qber_pct = telemetry_df.get("qber", pd.Series(0.0, index=telemetry_df.index)).to_numpy() * 100.0
    ax_qber.plot(windows, qber_pct, color=COLORS["normal"], lw=2.4, marker="s", markersize=4, label="Observed QBER (%)")
    ax_qber.axhline(11.0, color=COLORS["eve"], linestyle="--", lw=1.5, label="Shor-Preskill Security Bound (11.0%)")
    ax_qber.set_ylabel("QBER [%]", fontsize=10.5, fontweight="bold")
    ax_qber.set_title("B: Quantum Bit Error Rate (QBER) Temporal Response", fontsize=11.5, fontweight="bold", pad=6)
    ax_qber.legend(loc="upper right", fontsize=9)
    ax_qber.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Sifted Key Rate & Count Rate
    sifted_rate = telemetry_df.get("sifted_rate", telemetry_df.get("sifted_key_rate_hz", pd.Series(0.0, index=telemetry_df.index))).to_numpy()
    total_count = telemetry_df.get("total_count", telemetry_df.get("raw_count_rate_hz", pd.Series(0.0, index=telemetry_df.index))).to_numpy()

    ax_rate.plot(windows, sifted_rate, color=COLORS["bob"], lw=2.4, marker="^", markersize=4, label="Sifted Key Rate [Hz]")
    ax_rate.plot(windows, total_count, color=COLORS["channel"], lw=1.8, linestyle="--", label="Total Click Count Rate")
    ax_rate.set_ylabel("Rate [Hz]", fontsize=10.5, fontweight="bold")
    ax_rate.set_xlabel("Telemetry Window ID (Time Sequence)", fontsize=11, fontweight="bold")
    ax_rate.set_title("C: Sifted Key Generation Rate & Raw Detector Response", fontsize=11.5, fontweight="bold", pad=6)
    ax_rate.legend(loc="upper right", fontsize=9)
    ax_rate.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Attack Injection Mechanics: Causal Transition Dynamics & Telemetry Response",
        fontsize=14, fontweight="bold", y=0.99
    )

    return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)


def plot_multi_attack_signatures(
    out_dir: pathlib.Path,
    filename_stem: str = "09_multi_attack_signatures_comparison",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Figure 09: Distinct multi-dimensional physical footprints across all 6 implemented attacks."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_slide"])

    # 6 Implemented Attacks in QISM
    attacks = [
        "1. Intercept-Resend\n(Basis Measurement)",
        "2. Channel Loss\n(Extra Attenuation)",
        "3. Source Manip.\n(Intensity Offset Δμ)",
        "4. Detector Manip.\n(Blinding / Saturation)",
        "5. Timing Shift\n(Arrival Offset Jitter)",
        "6. PNS Attack\n(Multi-photon Splitting)",
    ]

    # Telemetry Footprints: [ΔQBER, ΔCountRate, ΔDecoyYield, ΔTimingJitter, ΔDeadTime]
    # Normalized relative shift: -1.0 to +1.0 (illustrating distinct signatures)
    footprints = {
        r"$\Delta$ QBER (Error Rate)": [0.95, 0.15, 0.05, -0.05, 0.35, 0.02],
        r"$\Delta$ Total Count Rate": [-0.50, -0.85, 0.60, -0.95, -0.70, -0.20],
        r"$\Delta$ Decoy Gain Residual": [0.10, -0.15, 0.80, -0.20, -0.10, 0.95],
        r"$\Delta$ Timing Offset / Jitter": [0.05, 0.02, 0.00, 0.10, 0.95, 0.05],
        r"$\Delta$ Detector Imbalance": [0.05, 0.00, 0.02, 0.90, 0.05, 0.00],
    }

    n_features = len(footprints)
    x = np.arange(len(attacks))
    width = 0.15
    feature_colors = ["#DC2626", "#2563EB", "#7C3AED", "#D97706", "#059669"]

    for idx, ((f_name, values), color) in enumerate(zip(footprints.items(), feature_colors)):
        offset = (idx - n_features / 2 + 0.5) * width
        rects = ax.bar(x + offset, values, width, label=f_name, color=color, edgecolor=COLORS["border"])

    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=10, fontweight="bold")
    ax.set_ylabel("Normalized Telemetry Deviation [Shift Relative to Baseline]", fontsize=11, fontweight="bold")
    ax.set_ylim(-1.15, 1.25)
    ax.set_title(
        "Multi-Attack Physical Footprints: Distinct Observational Signatures Across Telemetry Dimensions",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)


def plot_normal_vs_attack_comparison(
    normal_df: pd.DataFrame,
    attack_df: pd.DataFrame,
    out_dir: pathlib.Path,
    filename_stem: str = "10_normal_vs_attack_feature_comparison",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Figure 10: Side-by-side distribution comparison of key observables under normal vs attack conditions."""
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZES["grid_2x3"])
    axes = axes.flatten()

    features_to_plot = [
        ("qber", "Quantum Bit Error Rate (QBER)", "[%]", 100.0),
        ("total_count", "Total Detection Count Rate", "[counts/window]", 1.0),
        ("signal_gain", "Signal State Gain Q_μ", "[%]", 100.0),
        ("decoy_gain", "Decoy State Gain Q_ν", "[%]", 100.0),
        ("detector_imbalance", "Detector Click Imbalance", "[ratio]", 1.0),
        ("fano_factor", "Photon Click Fano Factor", "[variance/mean]", 1.0),
    ]

    for ax, (feat, title, unit, scale) in zip(axes, features_to_plot):
        n_vals = pd.to_numeric(normal_df.get(feat, pd.Series(dtype=float)), errors="coerce").dropna() * scale
        a_vals = pd.to_numeric(attack_df.get(feat, pd.Series(dtype=float)), errors="coerce").dropna() * scale

        if len(n_vals) == 0 and len(a_vals) == 0:
            ax.text(0.5, 0.5, f"{feat} not available", ha="center", va="center")
            continue

        data = [n_vals.to_numpy(), a_vals.to_numpy()]
        bplot = ax.boxplot(
            data,
            tick_labels=["Normal", "Attack"],
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            widths=0.5,
        )

        bplot["boxes"][0].set_facecolor("#DBEAFE")  # Blue-100
        bplot["boxes"][0].set_edgecolor(COLORS["normal"])
        bplot["boxes"][1].set_facecolor("#FEE2E2")  # Red-100
        bplot["boxes"][1].set_edgecolor(COLORS["eve"])

        ax.set_title(f"{title}", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(f"{unit}", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        r"Empirical Distribution Shift: Benign Baseline $X_{\mathrm{normal}}$ vs Attacked Telemetry $X_{\mathrm{attack}}$",
        fontsize=14, fontweight="bold", y=1.01
    )

    return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)
