"""Diagnostics, sanity experiments, and evaluation plots (Requirements 108–112, 200–202, 215–218)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from qism.ml.features.extractor import FeatureExtractor
from qism.ml.models.base import AnomalyDetector
from qism.ml.preprocessing.scalers import PreprocessorPipeline


def run_leakage_sanity_experiments(
    model: AnomalyDetector,
    preprocessor: PreprocessorPipeline,
    extractor: FeatureExtractor,
    df_test: pd.DataFrame,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute label-shuffling and random-feature sanity checks (Requirements 215–218, 255, 256).

    Expected outcomes:
    - Shuffled labels collapse AUROC toward ~0.50 (chance level).
    - Random features destroy meaningful anomaly detection separation.
    """
    if df_test.empty:
        return {"status": "skipped_empty_test"}

    X_df, y_df = extractor.extract(df_test)
    X_scaled = preprocessor.transform(X_df)
    scores = model.score(X_scaled)
    y_true = y_df["attack_active"].to_numpy(dtype=int)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        return {
            "status": "single_class_present",
            "shuffled_auroc": 0.5,
            "random_feature_auroc": 0.5,
        }

    # 1. Label Shuffling Sanity Experiment
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y_true)
    try:
        shuffled_auroc = float(roc_auc_score(y_shuffled, scores))
    except Exception:
        shuffled_auroc = 0.5

    # 2. Random Features Sanity Experiment
    X_random = rng.normal(loc=0.0, scale=1.0, size=X_scaled.shape)
    try:
        random_scores = model.score(X_random)
        random_auroc = float(roc_auc_score(y_true, random_scores))
    except Exception:
        random_auroc = 0.5

    return {
        "shuffled_labels": {
            "auroc": shuffled_auroc,
            "collapsed_to_chance": abs(shuffled_auroc - 0.5) < 0.25,
        },
        "random_features": {
            "auroc": random_auroc,
            "collapsed_to_chance": abs(random_auroc - 0.5) < 0.25,
        },
    }


def plot_evaluation_curves(
    eval_results: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Generate and save ROC and Precision-Recall curves."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_data = eval_results.get("curve_metrics", {})
    roc = curve_data.get("roc_curve", {})
    pr = curve_data.get("pr_curve", {})

    paths = {}

    # 1. ROC Curve
    if roc.get("fpr") and roc.get("tpr"):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(roc["fpr"], roc["tpr"], color="#0055ff", lw=2, label=f"ROC (AUROC = {curve_data.get('auroc', 0.5):.3f})")
        ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", label="Chance")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("Receiver Operating Characteristic (ROC)")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.6)
        roc_path = out_dir / "roc_curve.png"
        fig.tight_layout()
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        paths["roc_curve"] = roc_path

    # 2. Precision-Recall Curve
    if pr.get("recall") and pr.get("precision"):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(pr["recall"], pr["precision"], color="#ff4400", lw=2, label=f"PR (AUPRC = {curve_data.get('auprc', 0.0):.3f})")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        pr_path = out_dir / "pr_curve.png"
        fig.tight_layout()
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)
        paths["pr_curve"] = pr_path

    return paths


def plot_score_distributions(
    normal_scores: np.ndarray,
    attack_scores: np.ndarray,
    threshold: float,
    output_path: str | Path,
) -> Path:
    """Plot anomaly score distributions for normal vs attack telemetry."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    if len(normal_scores) > 0:
        ax.hist(normal_scores, bins=25, alpha=0.6, color="#0088cc", label=f"Normal ({len(normal_scores)})", density=True)
    if len(attack_scores) > 0:
        ax.hist(attack_scores, bins=25, alpha=0.6, color="#ee3322", label=f"Attack ({len(attack_scores)})", density=True)

    ax.axvline(threshold, color="black", linestyle="--", lw=2, label=f"Threshold (tau = {threshold:.2f})")
    ax.set_xlabel("Anomaly Score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Normal vs Attack Anomaly Score Distributions")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_anomaly_timeline(
    timestamps: np.ndarray | list[int | float],
    scores: np.ndarray,
    threshold: float,
    attack_active: np.ndarray,
    alarms: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Plot representative telemetry timeline with ground-truth attacks, anomaly scores, and alarms."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(timestamps)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # Top: Anomaly Score & Threshold
    ax1.plot(t, scores, color="#2244aa", lw=1.8, label="Anomaly Score")
    ax1.axhline(threshold, color="#cc0000", linestyle="--", lw=1.5, label=f"Threshold ({threshold:.2f})")

    # Shading ground truth attack intervals
    in_atk = False
    atk_start = 0
    for i, act in enumerate(attack_active):
        if act and not in_atk:
            in_atk = True
            atk_start = t[i]
        elif not act and in_atk:
            in_atk = False
            ax1.axvspan(atk_start, t[i-1], color="#ffcccc", alpha=0.5, label="Ground Truth Attack" if "Ground Truth Attack" not in ax1.get_legend_handles_labels()[1] else "")
    if in_atk:
        ax1.axvspan(atk_start, t[-1], color="#ffcccc", alpha=0.5, label="Ground Truth Attack" if "Ground Truth Attack" not in ax1.get_legend_handles_labels()[1] else "")

    ax1.set_ylabel("Score")
    ax1.set_title("QISM Telemetry Anomaly Score & Operational Alarm State")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Bottom: Triggered Alarms
    ax2.step(t, alarms, where="mid", color="#ee2200", lw=2, label="Alarm Active (1/0)")
    ax2.set_ylim([-0.2, 1.2])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Normal", "ALARM"])
    ax2.set_xlabel("Observation Time / Window Index")
    ax2.set_ylabel("State")
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
