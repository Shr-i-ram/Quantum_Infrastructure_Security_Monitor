"""Temporal security monitoring visualization suite (V0.7).

Generates publication-quality figures:
1. Multi-panel timeline: observable telemetry, continuous anomaly score, raw vs filtered alarms.
2. Score distributions: normal vs attack separation with threshold overlay.
3. Feature reconstruction error decomposition: per-feature anomaly contributions E_j.
4. Window-length ablation: trade-off between detection recall, false alarms, and detection delay.
5. Detection delay distribution across attack families.
6. Point vs temporal detector ROC & PR comparison.
7. Event-level detection summary: episode timeline and detection status.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_temporal_monitoring_timeline(
    timestamps_s: np.ndarray | list[float],
    scores: np.ndarray,
    threshold: float,
    recovery_threshold: float,
    raw_alarms: np.ndarray,
    filtered_alarms: np.ndarray,
    attack_active: np.ndarray,
    telemetry_samples: pd.DataFrame | None = None,
    output_path: str | Path = "outputs/v07/plots/temporal_timeline.png",
) -> Path:
    """Generate high-resolution 3-panel operational monitoring timeline."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(timestamps_s)
    if len(t) == 0:
        t = np.arange(len(scores))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1.5, 2.0, 1.2]})

    # Panel 1: Key physical observables (e.g. QBER, counts/gain)
    ax0 = axes[0]
    if telemetry_samples is not None and not telemetry_samples.empty:
        if "qber" in telemetry_samples.columns:
            ax0.plot(t, telemetry_samples["qber"], color="#0066cc", lw=1.5, label="QBER")
            ax0.set_ylabel("QBER", color="#0066cc")
            ax0.tick_params(axis="y", labelcolor="#0066cc")
        if "signal_gain" in telemetry_samples.columns:
            ax0_twin = ax0.twinx()
            ax0_twin.plot(t, telemetry_samples["signal_gain"], color="#228833", lw=1.2, linestyle="--", label="Signal Gain")
            ax0_twin.set_ylabel("Gain", color="#228833")
            ax0_twin.tick_params(axis="y", labelcolor="#228833")
    else:
        ax0.text(0.5, 0.5, "Raw Telemetry Telemetry Inputs", ha="center", va="center", transform=ax0.transAxes)

    ax0.set_title("Operational QKD Telemetry & Temporal Anomaly Monitoring Timeline", fontsize=13, fontweight="bold")
    ax0.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Continuous anomaly score, thresholds, and attack intervals
    ax1 = axes[1]
    ax1.plot(t, scores, color="#332288", lw=1.8, label="Temporal Anomaly Score S(W)")
    ax1.axhline(threshold, color="#cc3311", linestyle="--", lw=1.6, label=f"Trigger Threshold (tau_high = {threshold:.2f})")
    if recovery_threshold < threshold:
        ax1.axhline(recovery_threshold, color="#ee7733", linestyle=":", lw=1.4, label=f"Recovery Threshold (tau_low = {recovery_threshold:.2f})")

    # Shade ground-truth attack regions
    in_atk = False
    start_t = t[0]
    for i, act in enumerate(attack_active):
        if act and not in_atk:
            in_atk = True
            start_t = t[i]
        elif not act and in_atk:
            in_atk = False
            ax1.axvspan(start_t, t[i-1], color="#ffcccc", alpha=0.45, label="Ground Truth Attack" if "Ground Truth Attack" not in ax1.get_legend_handles_labels()[1] else "")
            ax0.axvspan(start_t, t[i-1], color="#ffcccc", alpha=0.25)
    if in_atk:
        ax1.axvspan(start_t, t[-1], color="#ffcccc", alpha=0.45, label="Ground Truth Attack" if "Ground Truth Attack" not in ax1.get_legend_handles_labels()[1] else "")
        ax0.axvspan(start_t, t[-1], color="#ffcccc", alpha=0.25)

    ax1.set_ylabel("Anomaly Score", fontsize=11)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 3: Raw alarms vs filtered operational alarms
    ax2 = axes[2]
    ax2.step(t, raw_alarms, where="mid", color="#ee7733", lw=1.2, linestyle="--", label="Raw Alarm (S > tau)")
    ax2.step(t, filtered_alarms, where="mid", color="#cc3311", lw=2.2, label="Operational ALARM (Persistence Filtered)")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["NORMAL", "ALARM"])
    ax2.set_ylim([-0.2, 1.25])
    ax2.set_ylabel("Alarm State", fontsize=11)
    ax2.set_xlabel("Observation Time (s)", fontsize=11)
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_score_distribution_comparison(
    normal_scores: np.ndarray,
    attack_scores: np.ndarray,
    threshold: float,
    output_path: str | Path = "outputs/v07/plots/score_distributions.png",
) -> Path:
    """Plot anomaly score distributions comparing normal vs attack telemetry."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    if len(normal_scores) > 0:
        ax.hist(normal_scores, bins=30, alpha=0.6, color="#0077bb", label=f"Normal Telemetry (N={len(normal_scores)})", density=True)
    if len(attack_scores) > 0:
        ax.hist(attack_scores, bins=30, alpha=0.6, color="#cc3311", label=f"Attack Telemetry (N={len(attack_scores)})", density=True)

    ax.axvline(threshold, color="black", linestyle="--", lw=2.0, label=f"Calibrated Threshold (tau = {threshold:.2f})")
    ax.set_xlabel("Anomaly Score (higher = more anomalous)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title("Normal vs Attack Anomaly Score Separation", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_feature_decomposition(
    feature_names: list[str],
    contributions: np.ndarray,
    output_path: str | Path = "outputs/v07/plots/feature_decomposition.png",
    top_n: int = 12,
) -> Path:
    """Plot feature-level reconstruction error contributions (E_j) during anomalous intervals."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if contributions.ndim == 2:
        mean_contrib = np.mean(contributions, axis=0)
    else:
        mean_contrib = contributions

    # Select top N contributing features
    top_indices = np.argsort(mean_contrib)[::-1][:top_n]
    top_names = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in top_indices]
    top_vals = mean_contrib[top_indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals, color="#009988", alpha=0.85, edgecolor="#004433")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Reconstruction Error E_j", fontsize=11)
    ax.set_title("Top Physical Telemetry Dimensions Explaining Anomalies", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_window_length_study(
    lengths: list[int],
    recalls: list[float],
    fprs: list[float],
    median_delays_s: list[float],
    output_path: str | Path = "outputs/v07/plots/window_length_study.png",
) -> Path:
    """Plot trade-off curve between temporal window length, recall, FPR, and detection delay."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Subplot 1: Recall vs FPR
    ax1.plot(lengths, recalls, marker="o", color="#0077bb", lw=2, label="Recall")
    ax1.plot(lengths, fprs, marker="s", color="#cc3311", lw=2, linestyle="--", label="False Positive Rate")
    ax1.set_xlabel("Window Length L (steps)", fontsize=11)
    ax1.set_ylabel("Metric Rate", fontsize=11)
    ax1.set_title("Detection Performance vs Window Length", fontsize=12, fontweight="bold")
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend()

    # Subplot 2: Detection Delay
    valid_delays = [d if d is not None else 0.0 for d in median_delays_s]
    ax2.plot(lengths, valid_delays, marker="^", color="#ee7733", lw=2, label="Median Delay (s)")
    ax2.set_xlabel("Window Length L (steps)", fontsize=11)
    ax2.set_ylabel("Detection Latency (seconds)", fontsize=11)
    ax2.set_title("Detection Latency vs Window Length", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_roc_pr_comparison(
    y_true: np.ndarray,
    point_scores: np.ndarray,
    temporal_scores: np.ndarray,
    point_name: str = "V0.6 Point Baseline",
    temporal_name: str = "V0.7 Temporal Model",
    output_path: str | Path = "outputs/v07/plots/roc_pr_comparison.png",
) -> Path:
    """Plot side-by-side ROC and PR curves directly comparing V0.6 point baseline vs V0.7 temporal detector."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ROC
    fpr_pt, tpr_pt, _ = roc_curve(y_true, point_scores)
    auroc_pt = roc_auc_score(y_true, point_scores)
    fpr_tp, tpr_tp, _ = roc_curve(y_true, temporal_scores)
    auroc_tp = roc_auc_score(y_true, temporal_scores)

    ax1.plot(fpr_pt, tpr_pt, color="#888888", lw=1.8, linestyle="--", label=f"{point_name} (AUROC = {auroc_pt:.3f})")
    ax1.plot(fpr_tp, tpr_tp, color="#0055ff", lw=2.2, label=f"{temporal_name} (AUROC = {auroc_tp:.3f})")
    ax1.plot([0, 1], [0, 1], color="#cccccc", linestyle=":")
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # PR
    pr_pt, rec_pt, _ = precision_recall_curve(y_true, point_scores)
    auprc_pt = average_precision_score(y_true, point_scores)
    pr_tp, rec_tp, _ = precision_recall_curve(y_true, temporal_scores)
    auprc_tp = average_precision_score(y_true, temporal_scores)

    ax2.plot(rec_pt, pr_pt, color="#888888", lw=1.8, linestyle="--", label=f"{point_name} (AUPRC = {auprc_pt:.3f})")
    ax2.plot(rec_tp, pr_tp, color="#ff4400", lw=2.2, label=f"{temporal_name} (AUPRC = {auprc_tp:.3f})")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


class TemporalPlotter:
    """Convenience class providing access to all V0.7 temporal plotting functions."""

    plot_timeline = staticmethod(plot_temporal_monitoring_timeline)
    plot_distributions = staticmethod(plot_score_distribution_comparison)
    plot_decomposition = staticmethod(plot_feature_decomposition)
    plot_window_study = staticmethod(plot_window_length_study)
    plot_comparison = staticmethod(plot_roc_pr_comparison)

