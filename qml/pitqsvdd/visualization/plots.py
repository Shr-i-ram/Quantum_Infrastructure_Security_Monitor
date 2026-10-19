"""Publication-quality visualization generation for PI-TQSVDD.

Generates diagnostic, architectural, optimization, detection, and comparative figures.
All figures use high resolution (300 DPI), clean typography, descriptive legends,
and are saved to artifacts/pi_tqsvdd/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_architecture_diagram(out_path: str | Path) -> None:
    """Generate high-level architectural data flow diagram for PI-TQSVDD."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    # Draw boxes
    boxes = [
        ("Raw QKD Telemetry\n(48 Observable Features)", 0.05, 0.4, 0.18, 0.25, "#E1F5FE", "#0288D1"),
        ("Classical Feature Mapping\n(PCA / Learned Bottleneck /\nFeature Groups)", 0.28, 0.4, 0.20, 0.25, "#E8F5E9", "#388E3C"),
        ("Quantum Circuit\n(lightning.qubit, 2-8 Qubits)\nAngle / Data Reuploading\nEntangling CNOT Ring", 0.53, 0.4, 0.22, 0.25, "#EDE7F6", "#7B1FA2"),
        ("Latent Representations\nz_t in [-1, 1]^n\nPauli-Z Measurements", 0.80, 0.55, 0.18, 0.20, "#FFF3E0", "#F57C00"),
        ("Causal GRU Predictor\nhat{z}_{t+1} from history\nTemporal Residual Loss", 0.80, 0.20, 0.18, 0.20, "#FCE4EC", "#C2185B"),
    ]

    for title, x, y, w, h, bg, border in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor=border, linewidth=2, transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", transform=ax.transAxes, fontsize=9, weight="bold", color="#212121", zorder=3)

    # Connecting arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="#424242")
    ax.annotate("", xy=(0.28, 0.525), xytext=(0.23, 0.525), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.53, 0.525), xytext=(0.48, 0.525), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.80, 0.65), xytext=(0.75, 0.55), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.80, 0.30), xytext=(0.75, 0.50), xycoords="axes fraction", arrowprops=arrow_props)

    ax.set_title("Physics-Informed Temporal Quantum SVDD (PI-TQSVDD) Architecture", weight="bold", pad=15)
    plt.savefig(out_path)
    plt.close()


def plot_feature_inventory(X_df: pd.DataFrame, out_path: str | Path) -> None:
    """Plot correlation heatmap and variance distribution across the 4 physical families."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Feature counts by family
    from qism.ml.features.definitions import (
        DETECTOR_ONLY_FEATURES,
        ENVIRONMENTAL_ONLY_FEATURES,
        QKD_ONLY_FEATURES,
        TIMING_CHANNEL_FEATURES,
    )
    fam_counts = {
        "QBER & Yields": len(QKD_ONLY_FEATURES),
        "SPAD Dynamics": len(DETECTOR_ONLY_FEATURES),
        "Environmental Sensors": len(ENVIRONMENTAL_ONLY_FEATURES),
        "Timing & Loss": len(TIMING_CHANNEL_FEATURES),
    }
    axes[0].bar(fam_counts.keys(), fam_counts.values(), color=["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"], width=0.55)
    axes[0].set_ylabel("Number of Observable Features")
    axes[0].set_title("Full Telemetry Space: 48 Observable Channels", weight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for i, (k, v) in enumerate(fam_counts.items()):
        axes[0].text(i, v + 0.3, str(v), ha="center", weight="bold")

    # 2. PCA variance accumulation curve
    clean_cols = [c for c in X_df.columns if not X_df[c].isna().all()]
    sub_df = X_df[clean_cols].fillna(X_df[clean_cols].median())
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(sub_df)
    pca = PCA().fit(X_std)
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100.0

    qubit_marks = [2, 4, 6, 8]
    axes[1].plot(range(1, len(cum_var) + 1), cum_var, marker="o", markersize=3, color="#D32F2F", lw=2, label="Cumulative Explained Variance")
    for q in qubit_marks:
        if q <= len(cum_var):
            axes[1].axvline(q, color="#616161", linestyle=":", alpha=0.7)
            axes[1].scatter([q], [cum_var[q - 1]], color="#1976D2", s=50, zorder=5)
            axes[1].text(q + 0.5, cum_var[q - 1] - 4, f"{q}Q: {cum_var[q-1]:.1f}%", fontsize=9, weight="bold")

    axes[1].axhline(80.0, color="#388E3C", linestyle="--", alpha=0.7, label="80% Variance Threshold")
    axes[1].set_xlabel("Number of Principal Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("PCA Variance Retention vs Qubit Bandwidth", weight="bold")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_loss_curves(train_history: list[dict[str, Any]], out_path: str | Path) -> None:
    """Plot total loss, SVDD loss, and temporal loss curves over training epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = [rec["epoch"] for rec in train_history]
    train_total = [rec["train_loss"] for rec in train_history]
    val_total = [rec["val_loss"] for rec in train_history]
    train_svdd = [rec.get("train_loss_svdd", 0) for rec in train_history]
    train_temp = [rec.get("train_loss_temporal", 0) for rec in train_history]

    # Total loss
    axes[0].plot(epochs, train_total, label="Train Total Loss", color="#1976D2", lw=2)
    axes[0].plot(epochs, val_total, label="Val Total Loss", color="#D32F2F", lw=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Objective Loss")
    axes[0].set_title("Training and Validation Convergence", weight="bold")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Loss decomposition
    axes[1].plot(epochs, train_svdd, label="SVDD Normality Loss", color="#388E3C", lw=2)
    axes[1].plot(epochs, train_temp, label="Temporal Residual Loss", color="#F57C00", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss Component Magnitude")
    axes[1].set_title("Loss Component Decomposition", weight="bold")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_and_pr_curves(
    y_test: np.ndarray,
    scores_test: np.ndarray,
    out_path: str | Path,
    title_suffix: str = "PI-TQSVDD",
) -> None:
    """Plot ROC curve and Precision-Recall curve side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_test, scores_test)
    from sklearn.metrics import average_precision_score, roc_auc_score
    auroc = roc_auc_score(y_test, scores_test)
    auprc = average_precision_score(y_test, scores_test)

    # ROC
    axes[0].plot(fpr, tpr, color="#1976D2", lw=2.5, label=f"ROC (AUROC = {auroc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].set_title(f"ROC Curve — {title_suffix}", weight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # PR
    prec, rec, _ = precision_recall_curve(y_test, scores_test)
    axes[1].plot(rec, prec, color="#388E3C", lw=2.5, label=f"PR (AUPRC = {auprc:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve — {title_suffix}", weight="bold")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    out_path: str | Path,
) -> None:
    """Plot normal vs attack anomaly score distributions and decision threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))

    normal_scores = scores[labels == 0]
    attack_scores = scores[labels == 1]

    bins = np.linspace(min(scores), max(scores), 25)
    ax.hist(normal_scores, bins=bins, alpha=0.6, color="#1976D2", label="Normal Operation", density=True)
    ax.hist(attack_scores, bins=bins, alpha=0.6, color="#D32F2F", label="Attack Injected", density=True)

    ax.axvline(threshold, color="#212121", linestyle="--", lw=2.5, label=f"Calibrated Threshold (tau = {threshold:.3f})")
    ax.set_xlabel("Continuous Anomaly Score s_t")
    ax.set_ylabel("Empirical Density")
    ax.set_title("Anomaly Score Separation: Normal vs Attack", weight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_qubits_and_encodings_benchmark(registry_df: pd.DataFrame, out_path: str | Path) -> None:
    """Generate multi-panel comparison of qubit count, encodings, and runtime."""
    if registry_df.empty or "n_qubits" not in registry_df.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    completed = registry_df[registry_df["status"] == "completed"].copy()
    if completed.empty:
        return

    # Panel 1: Qubit count vs F1 score
    grouped_q = completed.groupby("n_qubits")["test_f1"].mean().reset_index()
    axes[0].plot(grouped_q["n_qubits"], grouped_q["test_f1"], marker="o", color="#1976D2", lw=2.5, markersize=8)
    axes[0].set_xlabel("Number of Qubits")
    axes[0].set_ylabel("Test F1 Score")
    axes[0].set_title("Detection Performance vs Qubit Count", weight="bold")
    axes[0].set_xticks([2, 4, 6, 8])
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Qubit count vs Wall Time
    grouped_t = completed.groupby("n_qubits")["wall_time_s"].mean().reset_index()
    axes[1].bar(grouped_t["n_qubits"].astype(str) + "Q", grouped_t["wall_time_s"], color="#7B1FA2", width=0.5)
    axes[1].set_xlabel("Qubit Configuration")
    axes[1].set_ylabel("Training Wall Time (s)")
    axes[1].set_title("Computational Cost vs Qubit Count", weight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Encoding comparison F1 score
    grouped_enc = completed.groupby("encoding_type")["test_f1"].mean().reset_index()
    axes[2].bar(grouped_enc["encoding_type"], grouped_enc["test_f1"], color="#388E3C", width=0.55)
    axes[2].set_xlabel("Quantum Encoding Strategy")
    axes[2].set_ylabel("Average Test F1 Score")
    axes[2].set_title("Encoding Strategy Comparison", weight="bold")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_model_comparison_bar(models_summary: dict[str, dict[str, Any]], out_path: str | Path) -> None:
    """Generate bar chart comparing PI-TQSVDD with classical and QML baselines."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(models_summary.keys())
    f1_scores = [models_summary[m].get("f1", 0.0) for m in names]
    aurocs = [models_summary[m].get("auroc", 0.0) if models_summary[m].get("auroc") is not None else 0.0 for m in names]

    x = np.arange(len(names))
    width = 0.35

    rects1 = ax.bar(x - width / 2, f1_scores, width, label="Test F1 Score", color="#1976D2")
    rects2 = ax.bar(x + width / 2, aurocs, width, label="Test AUROC", color="#388E3C")

    ax.set_ylabel("Metric Score")
    ax.set_title("Benchmarking: PI-TQSVDD vs Classical & Prior QML Baselines", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
