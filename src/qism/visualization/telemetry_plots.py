"""Normal operating baseline telemetry distributions, temporal traces, and cross-correlation heatmaps."""

from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qism.visualization.statistics import compute_correlation_matrix
from qism.visualization.style import COLORS, FIGURE_SIZES, save_figure


def plot_normal_telemetry_distributions(
    normal_df: pd.DataFrame,
    out_dir: pathlib.Path,
    filename_stem: str = "06_normal_telemetry_distributions",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Figure 06: 6-panel distribution of key observables under benign normal operating conditions."""
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZES["grid_2x3"])
    axes = axes.flatten()

    items = [
        ("qber", "A: Quantum Bit Error Rate (QBER)", "[%]", 100.0, COLORS["normal"]),
        ("total_count", "B: Total Detector Counts", "[counts/window]", 1.0, COLORS["detector_0"]),
        ("signal_gain", "C: Signal Gain Q_μ", "[%]", 100.0, COLORS["signal"]),
        ("decoy_gain", "D: Decoy Gain Q_ν", "[%]", 100.0, COLORS["decoy"]),
        ("detector_imbalance", "E: Detector Imbalance (D0 vs D1)", "[ratio]", 1.0, COLORS["detector_1"]),
        ("sensor_temperature_mean", "F: Environmental Temperature", "[°C]", 1.0, COLORS["calibration"]),
    ]

    for ax, (feat, title, unit, scale, color) in zip(axes, items):
        vals = pd.to_numeric(normal_df.get(feat, pd.Series(dtype=float)), errors="coerce").dropna() * scale
        if len(vals) == 0:
            ax.text(0.5, 0.5, f"{feat} not available", ha="center", va="center")
            continue

        mean_v = float(vals.mean())
        std_v = float(vals.std()) if len(vals) > 1 else 0.0

        n_bins = min(25, max(8, len(vals) // 4))
        ax.hist(vals, bins=n_bins, color=color, alpha=0.75, edgecolor=COLORS["border"], density=True)
        ax.axvline(mean_v, color=COLORS["text_dark"], linestyle="--", lw=1.8, label=f"Mean: {mean_v:.2f}")

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(f"{unit}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Density", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Normal Operation Baseline: Multi-Dimensional Telemetry Distributions in Benign Regimes",
        fontsize=14, fontweight="bold", y=1.01
    )

    return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)


def plot_qber_temporal_evolution(
    normal_df: pd.DataFrame,
    attack_df: pd.DataFrame | None,
    out_dir: pathlib.Path,
    filename_stem: str = "07_qber_temporal_evolution",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Figure 07: Temporal evolution of QBER: stable benign regime vs attack excursion."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide_timeline"])

    n_qber = pd.to_numeric(normal_df.get("qber", pd.Series(dtype=float)), errors="coerce").dropna() * 100.0
    w_normal = np.arange(len(n_qber))

    ax.plot(w_normal, n_qber, color=COLORS["normal"], lw=2.2, marker="o", markersize=4, label="Normal Operation (Benign Drift)")

    if attack_df is not None and not attack_df.empty:
        a_qber = pd.to_numeric(attack_df.get("qber", pd.Series(dtype=float)), errors="coerce").dropna() * 100.0
        w_attack = np.arange(len(a_qber))
        ax.plot(w_attack, a_qber, color=COLORS["eve"], lw=2.2, marker="s", markersize=4, label="Attack Scenario (Injected Intervention)")

    ax.axhline(11.0, color="#DC2626", linestyle=":", lw=1.8, label="Shor-Preskill Threshold (11.0%)")
    ax.axhspan(0.0, 5.0, color="#EFF6FF", alpha=0.5, label="Expected Normal Band (0% - 5%)")

    ax.set_xlabel("Telemetry Window Index (Time)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Quantum Bit Error Rate (QBER) [%]", fontsize=11, fontweight="bold")
    ax.set_title(
        "Temporal Telemetry Dynamics: Benign Fluctuation Floor vs Attack Anomaly Deviation",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.6)

    return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)


def plot_telemetry_correlation_matrix(
    telemetry_df: pd.DataFrame,
    out_dir: pathlib.Path,
    filename_stem: str = "11_telemetry_cross_correlation",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Figure 11: Correlation heatmap demonstrating physical coupling across observable telemetry variables."""
    features = [
        "qber",
        "total_count",
        "signal_gain",
        "decoy_gain",
        "vacuum_gain",
        "detector_imbalance",
        "dark_count_rate",
        "dead_time_loss_rate",
        "sensor_temperature_mean",
        "fano_factor",
    ]

    labels = [
        "QBER",
        "Total Counts",
        "Signal Gain",
        "Decoy Gain",
        "Vacuum Gain",
        "Det Imbalance",
        "Dark Rate",
        "Dead Time Loss",
        "Sensor Temp",
        "Fano Factor",
    ]

    corr = compute_correlation_matrix(telemetry_df, features)
    if corr.empty:
        fig, ax = plt.subplots(figsize=FIGURE_SIZES["compact"])
        ax.text(0.5, 0.5, "Insufficient features for correlation matrix", ha="center", va="center")
        return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_slide"])
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    valid_labels = [labels[features.index(c)] if c in features else c for c in corr.columns]
    ax.set_xticklabels(valid_labels, rotation=45, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(valid_labels, fontsize=10, fontweight="bold")

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9.5, fontweight="bold")

    ax.set_title(
        "Telemetry Cross-Correlation Heatmap: Physical Coupling & Sensor Covariance",
        fontsize=13, fontweight="bold", pad=12
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation Coefficient r")

    return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)
