"""Publication-grade visualization suite for concept drift and safe adaptation (V0.8).

Generates:
1. plot_drift_telemetry_timeline: Multi-panel timeline of telemetry, drift score, anomaly score, and model version.
2. plot_distribution_comparison: Reference vs current telemetry distribution histograms and empirical CDFs.
3. plot_drift_heatmap: Feature-by-time drift magnitude heatmap.
4. plot_model_version_timeline: Operational timeline of version transitions, shadow validation, promotions, and rollbacks.
5. plot_fixed_vs_naive_vs_safe: Comparative trade-off analysis between Fixed, Naive, and Safe adaptation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling palette matching QISM standards
NAVY = "#0f172a"
BLUE = "#1e40af"
CYAN = "#06b6d4"
RED = "#b91c1c"
AMBER = "#d97706"
GREEN = "#059669"
PURPLE = "#7c3aed"
GRAY = "#64748b"
LIGHT_GRAY = "#f8fafc"


def plot_drift_telemetry_timeline(
    df_stream: pd.DataFrame,
    streaming_results: list[Any],
    output_path: str | Path,
    title: str = "QISM V0.8 Adaptive Operational Telemetry Monitoring",
    dpi: int = 300,
) -> None:
    """Generate comprehensive 4-panel timeline visualization."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    N = len(streaming_results)
    timestamps = np.array([r.timestamp_s if r.timestamp_s is not None else i * 0.002 for i, r in enumerate(streaming_results)])
    anomaly_scores = np.array([r.anomaly_score for r in streaming_results])
    anomaly_thresh = streaming_results[0].anomaly_threshold if streaming_results else 1.0
    drift_scores = np.array([r.drift_score for r in streaming_results])
    drift_thresh = streaming_results[0].drift_threshold if streaming_results else 0.5
    raw_alarms = np.array([r.raw_alarm for r in streaming_results])
    filtered_alarms = np.array([r.filtered_alarm for r in streaming_results])
    model_versions = [r.active_model_version for r in streaming_results]

    is_attack = df_stream["ground_truth_is_attack"].values[:N]
    is_drift = df_stream["ground_truth_is_drift"].values[:N]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0, 1.0, 0.8]})

    # --- Panel 1: Key Physical Telemetry Observables ---
    ax0 = axes[0]
    ax0.set_facecolor(LIGHT_GRAY)
    if "qber" in df_stream.columns:
        ax0.plot(timestamps, df_stream["qber"].values[:N] * 100.0, color=RED, label="QBER (%)", linewidth=1.8)
    if "signal_gain" in df_stream.columns:
        ax0.plot(timestamps, df_stream["signal_gain"].values[:N] * 100.0, color=BLUE, label="Signal Gain (x100)", linewidth=1.5, linestyle="--")
    if "sensor_temperature_mean" in df_stream.columns:
        ax0_twin = ax0.twinx()
        ax0_twin.plot(timestamps, df_stream["sensor_temperature_mean"].values[:N], color=AMBER, label="Temp (C)", linewidth=1.5, linestyle=":")
        ax0_twin.set_ylabel("Temp (C)", color=AMBER, fontsize=10, fontweight="bold")
        ax0_twin.tick_params(colors=AMBER)

    ax0.set_ylabel("Telemetry Rate (%)", fontsize=10, fontweight="bold")
    ax0.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=12)
    ax0.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax0.grid(True, linestyle="--", alpha=0.5)

    # Shade drift and attack intervals
    _shade_intervals(ax0, timestamps, is_drift, label="Ground Truth Drift", color=CYAN, alpha=0.15)
    _shade_intervals(ax0, timestamps, is_attack, label="Active Attack", color=RED, alpha=0.20)

    # --- Panel 2: Statistical Drift Score & Threshold ---
    ax1 = axes[1]
    ax1.set_facecolor(LIGHT_GRAY)
    ax1.plot(timestamps, drift_scores, color=CYAN, label="Multivariate Drift Score D_t", linewidth=2.0)
    ax1.axhline(drift_thresh, color=PURPLE, linestyle="--", linewidth=1.5, label=f"Drift Threshold tau_d={drift_thresh:.2f}")
    ax1.set_ylabel("Drift Distance", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    _shade_intervals(ax1, timestamps, is_drift, color=CYAN, alpha=0.15)
    _shade_intervals(ax1, timestamps, is_attack, color=RED, alpha=0.20)

    # --- Panel 3: Anomaly Score & Operational Alarm Threshold ---
    ax2 = axes[2]
    ax2.set_facecolor(LIGHT_GRAY)
    ax2.plot(timestamps, anomaly_scores, color=NAVY, label="Anomaly Score S(W_t)", linewidth=2.0)
    ax2.axhline(anomaly_thresh, color=RED, linestyle="--", linewidth=1.5, label=f"Alarm Threshold tau_a={anomaly_thresh:.2f}")

    # Mark filtered alarms
    alarm_indices = np.where(filtered_alarms)[0]
    if len(alarm_indices) > 0:
        ax2.scatter(timestamps[alarm_indices], anomaly_scores[alarm_indices], color=RED, s=25, zorder=5, label="Filtered Security Alarm")

    ax2.set_ylabel("Anomaly Score", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    _shade_intervals(ax2, timestamps, is_drift, color=CYAN, alpha=0.15)
    _shade_intervals(ax2, timestamps, is_attack, color=RED, alpha=0.20)

    # --- Panel 4: Active Model & Reference Version Lineage ---
    ax3 = axes[3]
    ax3.set_facecolor(LIGHT_GRAY)
    unique_vers = sorted(list(set(model_versions)))
    ver_map = {v: idx for idx, v in enumerate(unique_vers)}
    y_ver = [ver_map[v] for v in model_versions]

    ax3.step(timestamps, y_ver, where="post", color=GREEN, linewidth=2.5, label="Active Model Version")
    ax3.set_yticks(range(len(unique_vers)))
    ax3.set_yticklabels(unique_vers, fontsize=9, fontweight="bold")
    ax3.set_ylabel("Model State", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Simulated Time (seconds)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    _shade_intervals(ax3, timestamps, is_drift, color=CYAN, alpha=0.15)
    _shade_intervals(ax3, timestamps, is_attack, color=RED, alpha=0.20)

    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi)
    plt.close()


def plot_distribution_comparison(
    reference_data: np.ndarray,
    drifted_data: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    title: str = "Reference vs Drifted Empirical Telemetry Distributions",
    dpi: int = 300,
) -> None:
    """Generate empirical distribution comparisons (histograms & empirical CDFs)."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_feats = min(4, len(feature_names))
    fig, axes = plt.subplots(n_feats, 2, figsize=(12, 3 * n_feats))
    if n_feats == 1:
        axes = np.array([axes])

    for i in range(n_feats):
        feat = feature_names[i]
        ref_vals = reference_data[:, i]
        drift_vals = drifted_data[:, i]

        ax_hist = axes[i, 0]
        ax_cdf = axes[i, 1]

        # Histogram
        ax_hist.hist(ref_vals, bins=15, alpha=0.6, color=BLUE, label="Reference P_ref", density=True)
        ax_hist.hist(drift_vals, bins=15, alpha=0.6, color=CYAN, label="Current P_curr", density=True)
        ax_hist.set_title(f"{feat} — Density", fontsize=10, fontweight="bold")
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, linestyle="--", alpha=0.5)

        # Empirical CDF
        ref_sorted = np.sort(ref_vals)
        ref_cdf = np.linspace(0, 1, len(ref_sorted))
        drift_sorted = np.sort(drift_vals)
        drift_cdf = np.linspace(0, 1, len(drift_sorted))

        ax_cdf.step(ref_sorted, ref_cdf, color=BLUE, label="Reference CDF", linewidth=2.0)
        ax_cdf.step(drift_sorted, drift_cdf, color=CYAN, label="Current CDF", linewidth=2.0, linestyle="--")
        ax_cdf.set_title(f"{feat} — Empirical CDF", fontsize=10, fontweight="bold")
        ax_cdf.legend(fontsize=8)
        ax_cdf.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi)
    plt.close()


def plot_drift_heatmap(
    feature_drift_matrix: np.ndarray,
    feature_names: list[str],
    window_indices: list[int],
    output_path: str | Path,
    title: str = "Feature-Wise Statistical Drift Heatmap Over Time",
    dpi: int = 300,
) -> None:
    """Generate feature-by-time drift magnitude heatmap."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(feature_drift_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_xlabel("Monitoring Window Index", fontsize=10, fontweight="bold")
    ax.set_ylabel("Telemetry Observable", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Statistical Distance (W1)", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi)
    plt.close()


def plot_fixed_vs_naive_vs_safe(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Fixed vs Naive vs Safe Adaptation Performance & Security Trade-Off",
    dpi: int = 300,
) -> None:
    """Generate comparative grouped bar chart comparing the three adaptation policies."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    policies = comparison_df["policy"].tolist()
    far_vals = comparison_df["fpr"].values * 100.0
    rec_vals = comparison_df["recall"].values * 100.0

    colors = [BLUE, AMBER, GREEN]

    # Bar 1: False Positive Rate (FAR)
    bars1 = ax1.bar(policies, far_vals, color=colors, width=0.5, edgecolor=NAVY, linewidth=1.2)
    ax1.set_ylabel("False Positive Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Operational Burden (Lower is Better)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Bar 2: Attack Recall Retention
    bars2 = ax2.bar(policies, rec_vals, color=colors, width=0.5, edgecolor=NAVY, linewidth=1.2)
    ax2.set_ylabel("Attack Recall (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Security Retention (Higher is Better)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi)
    plt.close()


def _shade_intervals(ax: plt.Axes, timestamps: np.ndarray, mask: np.ndarray, label: str = "", color: str = CYAN, alpha: float = 0.20) -> None:
    """Helper to shade contiguous boolean masks as background spans."""
    in_span = False
    start_t = 0.0
    first = True
    for i in range(len(mask)):
        if mask[i] and not in_span:
            in_span = True
            start_t = timestamps[i]
        elif (not mask[i] or i == len(mask) - 1) and in_span:
            in_span = False
            end_t = timestamps[i]
            lbl = label if first and label else None
            ax.axvspan(start_t, end_t, color=color, alpha=alpha, label=lbl, zorder=0)
            first = False
