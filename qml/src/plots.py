"""Presentation-grade visualization generator for QML Review 2.

Generates Figures 1 through 11 as specified in the Review 2 audit guidelines:
- Fig 1: QML Architectures (VQC and QSVM pipeline diagram)
- Fig 2: Model Performance Comparison (F1, Precision, Recall, Balanced Accuracy)
- Fig 3: Confusion Matrices (VQC, QSVM, Classical RBF-SVM, Logistic Regression)
- Fig 4: ROC Curves (AUROC comparison)
- Fig 5: Precision-Recall Curves (AUPRC comparison)
- Fig 6: VQC Training History (Train vs. Val BCE loss)
- Fig 7: Quantum Kernel Gram Matrix Heatmap (Pairwise fidelity structure)
- Fig 8: Prediction Probability Distributions (Normal vs. Attack)
- Fig 9: Computational Runtime Comparison (Train time & Inference latency)
- Fig 10: Per-Attack Scenario Recall (Detection sensitivity by attack family)
- Fig 11: Feature Representation Space (PC1 vs. PC2 class separation)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


# Clean presentation style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})


def plot_qml_architectures(save_path: str | Path) -> None:
    """Generate Figure 1: Architectural diagram of VQC and Quantum Kernel SVM pipelines."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=300)

    # VQC diagram
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3)
    ax1.axis("off")
    ax1.set_title("A. Hybrid Variational Quantum Classifier (VQC)", loc="left", fontweight="bold", color="#1e1b4b")

    boxes_vqc = [
        ("QKD Telemetry\n(48 Raw Features)", 1.0, 1.5, "#dbeafe", "#1e40af"),
        ("Train-Only PCA\n(4 Components in [0, pi])", 3.0, 1.5, "#e0e7ff", "#3730a3"),
        ("Angle Embedding\nRY(x_i) on 4 Qubits", 5.0, 1.5, "#fae8ff", "#86198f"),
        ("Parameterized Unitary\nStrongly Entangling (L=2)", 7.0, 1.5, "#fce7f3", "#9d174d"),
        ("Pauli-Z Expvals\nLinear Head + Sigmoid\n-> p(Attack | x)", 9.0, 1.5, "#dcfce7", "#166534"),
    ]
    for text, cx, cy, fc, ec in boxes_vqc:
        ax1.text(cx, cy, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=1.5))
        if cx < 9.0:
            ax1.annotate("", xy=(cx + 0.85, cy), xytext=(cx + 1.15, cy),
                         arrowprops=dict(arrowstyle="->", lw=2, color="#4b5563"))

    # QSVM diagram
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 3)
    ax2.axis("off")
    ax2.set_title("B. Quantum Kernel Support Vector Machine (QSVM)", loc="left", fontweight="bold", color="#1e1b4b")

    boxes_qsvm = [
        ("QKD Telemetry\n(48 Raw Features)", 1.0, 1.5, "#dbeafe", "#1e40af"),
        ("Train-Only PCA\n(4 Components in [0, pi])", 3.0, 1.5, "#e0e7ff", "#3730a3"),
        ("Quantum Feature Map\nU_map(x) on 4 Qubits", 5.0, 1.5, "#fae8ff", "#86198f"),
        ("Fidelity Gram Matrix\nK(x, x') = |<phi(x)|phi(x')>|^2", 7.0, 1.5, "#ffedd5", "#c2410c"),
        ("Classical Dual SVM\nPrecomputed Kernel\n-> Decision Boundary", 9.0, 1.5, "#dcfce7", "#166534"),
    ]
    for text, cx, cy, fc, ec in boxes_qsvm:
        ax2.text(cx, cy, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=1.5))
        if cx < 9.0:
            ax2.annotate("", xy=(cx + 0.85, cy), xytext=(cx + 1.15, cy),
                         arrowprops=dict(arrowstyle="->", lw=2, color="#4b5563"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_performance_comparison(summary_df: pd.DataFrame, save_path: str | Path, split: str = "test") -> None:
    """Generate Figure 2: Grouped bar chart comparing metrics across models."""
    df_split = summary_df[summary_df["split"] == split].copy()
    if df_split.empty:
        return

    models = df_split["model_name"].tolist()
    metrics = ["f1_score", "precision", "recall", "balanced_accuracy"]
    metric_labels = ["F1 Score", "Precision", "Recall", "Balanced Acc"]

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed"]

    for idx, (m_col, m_label) in enumerate(zip(metrics, metric_labels)):
        vals = df_split[m_col].to_numpy()
        offset = (idx - 1.5) * width
        rects = ax.bar(x + offset, vals, width, label=m_label, color=colors[idx], alpha=0.9, edgecolor="black", lw=0.5)

    ax.set_ylabel("Metric Score [0.0 - 1.0]")
    ax.set_title(f"Model Performance Comparison ({split.upper()} Partition)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontweight="semibold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrices(preds_df: pd.DataFrame, save_path: str | Path, split: str = "test") -> None:
    """Generate Figure 3: Grid of confusion matrices for primary models."""
    df_split = preds_df[preds_df["split_name"] == split]
    models = ["Variational Quantum Classifier", "Quantum Kernel SVM", "Classical RBF-SVM", "Classical Logistic Regression"]
    available_models = [m for m in models if m in df_split["model_name"].unique()]
    if not available_models:
        available_models = df_split["model_name"].unique()[:4]

    n_models = len(available_models)
    fig, axes = plt.subplots(1, n_models, figsize=(3.8 * n_models, 3.8), dpi=300)
    if n_models == 1:
        axes = [axes]

    for ax, m_name in zip(axes, available_models):
        sub = df_split[df_split["model_name"] == m_name]
        cm = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14, fontweight="bold")

        ax.set_title(m_name, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Attack"])
        ax.set_yticklabels(["Normal", "Attack"])
        ax.set_xlabel("Predicted Label")
        if ax == axes[0]:
            ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(preds_df: pd.DataFrame, detailed_metrics: dict[str, Any], save_path: str | Path, split: str = "test") -> None:
    """Generate Figure 4: ROC Curves for all evaluated models."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    df_split = preds_df[preds_df["split_name"] == split]

    colors = ["#2563eb", "#7c3aed", "#059669", "#dc2626", "#d97706", "#4b5563"]
    color_idx = 0

    for m_name in df_split["model_name"].unique():
        sub = df_split[df_split["model_name"] == m_name]
        if len(sub["y_true"].unique()) > 1:
            fpr, tpr, _ = roc_curve(sub["y_true"], sub["y_prob"])
            key = f"{m_name}_{split}"
            auroc_val = detailed_metrics.get(key, {}).get("auroc", None)
            label_str = f"{m_name} (AUROC = {auroc_val:.3f})" if auroc_val is not None else m_name
            ax.plot(fpr, tpr, label=label_str, lw=2.2, color=colors[color_idx % len(colors)])
            color_idx += 1

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (0.500)", alpha=0.7)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title(f"Receiver Operating Characteristic (ROC) — {split.upper()} Set", fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curves(preds_df: pd.DataFrame, detailed_metrics: dict[str, Any], save_path: str | Path, split: str = "test") -> None:
    """Generate Figure 5: Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    df_split = preds_df[preds_df["split_name"] == split]

    colors = ["#2563eb", "#7c3aed", "#059669", "#dc2626", "#d97706", "#4b5563"]
    color_idx = 0

    # Baseline prevalence
    prev = float(df_split["y_true"].mean()) if not df_split.empty else 0.5

    for m_name in df_split["model_name"].unique():
        sub = df_split[df_split["model_name"] == m_name]
        if len(sub["y_true"].unique()) > 1:
            prec, rec, _ = precision_recall_curve(sub["y_true"], sub["y_prob"])
            key = f"{m_name}_{split}"
            auprc_val = detailed_metrics.get(key, {}).get("auprc", None)
            label_str = f"{m_name} (AUPRC = {auprc_val:.3f})" if auprc_val is not None else m_name
            ax.plot(rec, prec, label=label_str, lw=2.2, color=colors[color_idx % len(colors)])
            color_idx += 1

    ax.axhline(prev, color="k", linestyle="--", lw=1.2, label=f"Class Prevalence ({prev:.2f})", alpha=0.7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {split.upper()} Set", fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_vqc_training_history(train_record: dict[str, Any], save_path: str | Path) -> None:
    """Generate Figure 6: VQC training and validation loss curve across epochs."""
    train_losses = train_record.get("train_losses", [])
    val_losses = train_record.get("val_losses", [])
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)
    ax.plot(epochs, train_losses, label="Training Loss (BCE)", lw=2.2, color="#2563eb", marker="o", markersize=3)
    ax.plot(epochs, val_losses, label="Validation Loss (BCE)", lw=2.2, color="#dc2626", marker="s", markersize=3)

    best_epoch = np.argmin(val_losses) + 1
    best_val = val_losses[best_epoch - 1]
    ax.axvline(best_epoch, color="#059669", linestyle=":", lw=1.5, label=f"Best Checkpoint (Ep {best_epoch}: {best_val:.3f})")

    ax.set_xlabel("Optimization Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("VQC Optimization History (Adam, lr=0.02)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_quantum_kernel_matrix(K_train: np.ndarray, save_path: str | Path) -> None:
    """Generate Figure 7: Quantum fidelity Gram matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(K_train, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Quantum Fidelity |<phi(x)|phi(x')>|^2")

    ax.set_title("Training Quantum Kernel Gram Matrix (30 x 30)", fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_score_distributions(preds_df: pd.DataFrame, save_path: str | Path, split: str = "test") -> None:
    """Generate Figure 8: Normal vs Attack probability distributions with threshold."""
    df_split = preds_df[preds_df["split_name"] == split]
    models = ["Variational Quantum Classifier", "Quantum Kernel SVM"]
    available_models = [m for m in models if m in df_split["model_name"].unique()]
    if not available_models:
        available_models = df_split["model_name"].unique()[:2]

    fig, axes = plt.subplots(1, len(available_models), figsize=(5.5 * len(available_models), 4.2), dpi=300)
    if len(available_models) == 1:
        axes = [axes]

    for ax, m_name in zip(axes, available_models):
        sub = df_split[df_split["model_name"] == m_name]
        norm_scores = sub[sub["y_true"] == 0]["y_prob"]
        atk_scores = sub[sub["y_true"] == 1]["y_prob"]

        ax.hist(norm_scores, bins=12, alpha=0.6, color="#059669", label="Normal (y=0)", density=True, edgecolor="black")
        ax.hist(atk_scores, bins=12, alpha=0.6, color="#dc2626", label="Attack (y=1)", density=True, edgecolor="black")

        # Threshold line
        th = sub["y_pred"].iloc[0] if not sub.empty else 0.5  # placeholder
        ax.axvline(0.50, color="black", linestyle="--", lw=1.8, label="Decision Threshold (0.50)")

        ax.set_title(m_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Attack Probability p(y=1|x)")
        ax.set_ylabel("Density")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper center", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_runtime_comparison(summary_df: pd.DataFrame, runtime_records: dict[str, Any], save_path: str | Path) -> None:
    """Generate Figure 9: Training duration and per-sample inference latency comparison."""
    df_test = summary_df[summary_df["split"] == "test"].copy()
    if df_test.empty:
        return

    models = df_test["model_name"].tolist()
    train_times = [runtime_records.get(m, {}).get("train_time_s", 0.1) for m in models]
    latencies = df_test["latency_ms_per_sample"].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Train time
    y_pos = np.arange(len(models))
    ax1.barh(y_pos, train_times, color="#2563eb", alpha=0.85, edgecolor="black", lw=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontweight="semibold")
    ax1.set_xlabel("Training Duration [seconds]")
    ax1.set_title("A. Training Duration", loc="left", fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    # Inference latency
    ax2.barh(y_pos, latencies, color="#059669", alpha=0.85, edgecolor="black", lw=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Inference Latency [ms / sample]")
    ax2.set_title("B. Inference Latency", loc="left", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_per_attack_recall(detailed_metrics: dict[str, Any], save_path: str | Path, split: str = "test") -> None:
    """Generate Figure 10: Detection recall grouped by attack scenario."""
    attack_data: dict[str, dict[str, float]] = {}
    models = ["Variational Quantum Classifier", "Quantum Kernel SVM", "Classical RBF-SVM", "Classical Logistic Regression"]

    for m in models:
        key = f"{m}_{split}"
        per_atk = detailed_metrics.get(key, {}).get("per_attack_recall", {})
        for atk_name, val in per_atk.items():
            attack_data.setdefault(atk_name, {})[m] = val.get("recall", 0.0)

    if not attack_data:
        return

    atks = list(attack_data.keys())
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    x = np.arange(len(atks))
    width = 0.2

    colors = ["#2563eb", "#7c3aed", "#059669", "#dc2626"]
    for idx, m in enumerate(models):
        recalls = [attack_data[a].get(m, 0.0) for a in atks]
        ax.bar(x + (idx - 1.5) * width, recalls, width, label=m, color=colors[idx], alpha=0.85, edgecolor="black", lw=0.5)

    ax.set_ylabel("Detection Recall")
    ax.set_title(f"Attack Sensitivity by Threat Family ({split.upper()} Set)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", " ").title() for a in atks], fontweight="semibold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_space(X_train: np.ndarray, y_train: np.ndarray, save_path: str | Path) -> None:
    """Generate Figure 11: 2D feature projection showing normal vs. attack distribution."""
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    norm_mask = (y_train == 0)
    atk_mask = (y_train == 1)

    ax.scatter(X_train[norm_mask, 0], X_train[norm_mask, 1], c="#059669", label="Normal (y=0)", s=50, alpha=0.8, edgecolors="k", lw=0.5)
    ax.scatter(X_train[atk_mask, 0], X_train[atk_mask, 1], c="#dc2626", label="Attack (y=1)", s=60, alpha=0.8, marker="^", edgecolors="k", lw=0.5)

    ax.set_xlabel("Component 1 [scaled to 0 - pi]")
    ax.set_ylabel("Component 2 [scaled to 0 - pi]")
    ax.set_title("QKD Feature Representation Space (Training Partition)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
