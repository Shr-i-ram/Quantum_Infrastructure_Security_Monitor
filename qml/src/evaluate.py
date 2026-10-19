"""Unified evaluation pipeline for QML and Classical models (Review 2).

Evaluates models on IID test and OOD generalization partitions, calculates all standard
and ranking metrics, measures per-sample inference latency, and saves structured prediction
records and metric summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from qml.src.metrics import compute_binary_metrics


def evaluate_model(
    model_name: str,
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metadata_df: pd.DataFrame,
    split_name: str = "test",
    precomputed_K: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate a single model on a dataset partition, measuring metrics and inference latency."""
    # Measure inference time
    t0 = time.time()
    if hasattr(model, "predict_proba"):
        if precomputed_K is not None:
            probs = model.predict_proba(X, K=precomputed_K)
        else:
            probs = model.predict_proba(X)
    elif hasattr(model, "score_samples"):
        probs = model.score_samples(X)
    else:
        probs = np.zeros(len(X))

    if hasattr(model, "predict"):
        if precomputed_K is not None:
            preds = model.predict(X, K=precomputed_K)
        else:
            preds = model.predict(X)
    else:
        preds = (probs >= 0.50).astype(int)

    inference_duration_s = time.time() - t0
    latency_per_sample_ms = (inference_duration_s / max(1, len(X))) * 1000.0

    # Extract attack types from metadata if available
    attack_types = None
    if "attack_type" in metadata_df.columns:
        attack_types = metadata_df["attack_type"].to_numpy()
    elif "ground_truth_attack_type" in metadata_df.columns:
        attack_types = metadata_df["ground_truth_attack_type"].to_numpy()

    metrics = compute_binary_metrics(
        y_true=y,
        y_pred=preds,
        y_prob=probs,
        attack_types=attack_types,
    )

    metrics["model_name"] = model_name
    metrics["split_name"] = split_name
    metrics["inference_duration_s"] = inference_duration_s
    metrics["latency_per_sample_ms"] = latency_per_sample_ms
    metrics["threshold"] = getattr(model, "threshold", 0.50)

    # Build predictions dataframe
    pred_df = metadata_df.copy()
    pred_df["model_name"] = model_name
    pred_df["split_name"] = split_name
    pred_df["y_true"] = y
    pred_df["y_prob"] = probs
    pred_df["y_pred"] = preds

    return metrics, pred_df


def evaluate_all_models(
    models: dict[str, Any],
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    metadata: dict[str, pd.DataFrame],
    kernel_matrices: dict[str, dict[str, np.ndarray]] | None = None,
    output_dir: str | Path = "artifacts/qml_review2",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate all models across test and generalization splits, saving metrics and predictions."""
    out_dir = Path(output_dir)
    pred_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    all_preds_list: list[pd.DataFrame] = []

    for split in ["test", "gen"]:
        X, y = data[split]
        meta_df = metadata[split]

        for model_name, model in models.items():
            pre_K = None
            if kernel_matrices and model_name == "Quantum Kernel SVM":
                pre_K = kernel_matrices.get(split)

            m, p_df = evaluate_model(
                model_name=model_name,
                model=model,
                X=X,
                y=y,
                metadata_df=meta_df,
                split_name=split,
                precomputed_K=pre_K,
            )

            key = f"{model_name}_{split}"
            all_metrics[key] = m
            all_preds_list.append(p_df)

            summary_rows.append({
                "model_name": model_name,
                "split": split,
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "specificity": m["specificity"],
                "fpr": m["false_positive_rate"],
                "fnr": m["false_negative_rate"],
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "latency_ms_per_sample": m["latency_per_sample_ms"],
                "threshold": m["threshold"],
            })

    # Save summary dataframe
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(metrics_dir / "benchmark_summary.csv", index=False)

    # Save JSON metrics
    with open(metrics_dir / "detailed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Save concatenated predictions
    if all_preds_list:
        combined_preds = pd.concat(all_preds_list, ignore_index=True)
        combined_preds.to_csv(pred_dir / "all_model_predictions.csv", index=False)

    return all_metrics, summary_df
