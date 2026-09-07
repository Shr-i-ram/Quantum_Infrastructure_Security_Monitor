"""Automated execution script for V0.6 Experiments A through L and consolidated benchmarking (Requirements 266–270)."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from qism.ml.data.loader import MLDatasetLoader
from qism.ml.evaluation.runner import EvaluationRunner
from qism.ml.experiments.config import ExperimentConfig
from qism.ml.experiments.diagnostics import run_leakage_sanity_experiments
from qism.ml.experiments.runner import ExperimentRunner
from qism.ml.features.extractor import FeatureExtractor
from qism.ml.models.registry import build_model
from qism.ml.preprocessing.scalers import PreprocessorPipeline
from qism.ml.thresholding.calibration import ThresholdCalibrator


def run_all_v06_experiments(dataset_root: str = "data/v0.5_small/v05_small_dataset") -> dict[str, Any]:
    report: dict[str, Any] = {}
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    exp_output_root = Path("experiments")
    exp_output_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"STARTING QISM V0.6 EXPERIMENT MATRIX (Experiments A–L) — Dataset: {dataset_root}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: Model Benchmark (Experiments A–E: Statistical, IF, SVM, LOF, AE, Temporal AE)
    # -------------------------------------------------------------------------
    models_to_test = [
        ("statistical_mahalanobis", "mahalanobis", {}),
        ("robust_covariance", "robust_covariance", {}),
        ("isolation_forest", "isolation_forest", {"n_estimators": 50}),
        ("one_class_svm", "one_class_svm", {"nu": 0.05}),
        ("lof", "lof", {"n_neighbors": 15}),
        ("autoencoder", "autoencoder", {"epochs": 20, "lr": 1e-3, "hidden_dim": 32, "latent_dim": 8}),
        ("temporal_autoencoder", "temporal_autoencoder", {"epochs": 20, "seq_len": 3, "hidden_dim": 24, "latent_dim": 12}),
    ]

    model_benchmark_results = []
    for exp_name, m_type, hparams in models_to_test:
        print(f"\n>> Running Experiment: {exp_name} ({m_type})...")
        cfg = ExperimentConfig(
            experiment_id=f"exp_{exp_name}",
            dataset_root=dataset_root,
            model_type=m_type,
            hyperparameters=hparams,
            feature_set="full",
            add_temporal_features=(m_type != "temporal_autoencoder"),
            output_dir=str(exp_output_root),
        )
        runner = ExperimentRunner(cfg)
        res = runner.run()

        iid_wm = res["iid_test"]["window_metrics"]
        ood_wm = res["ood_generalization"]["window_metrics"]
        iid_cm = res["iid_test"]["curve_metrics"]
        im = res["iid_test"]["instance_metrics"]
        delay = im.get("detection_delay_summary", {}).get("median_delay_s")

        rec = {
            "experiment": exp_name,
            "model_type": m_type,
            "iid_precision": iid_wm["precision"],
            "iid_recall": iid_wm["recall"],
            "iid_f1": iid_wm["f1"],
            "iid_fpr": iid_wm["fpr"],
            "iid_auroc": iid_cm["auroc"],
            "iid_auprc": iid_cm["auprc"],
            "ood_recall": ood_wm["recall"],
            "ood_fpr": ood_wm["fpr"],
            "recall_gap": res["generalization_gap"]["recall_gap"],
            "fpr_gap": res["generalization_gap"]["fpr_gap"],
            "median_delay_s": delay,
            "false_alarms_per_hour": im.get("false_alarms_per_hour", 0.0),
        }
        model_benchmark_results.append(rec)
        print(f"   IID F1: {iid_wm['f1']:.3f} | IID AUROC: {iid_cm['auroc']:.3f} | OOD Rec: {ood_wm['recall']:.3f} | OOD FPR: {ood_wm['fpr']:.3f}")

    report["model_benchmark"] = model_benchmark_results

    # -------------------------------------------------------------------------
    # PART 2: Experiment F — Feature Ablation (QKD-only, detector-only, env-only, full)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">> Running Experiment F: Feature Ablation Study...")
    ablation_sets = ["qkd_only", "detector_only", "environmental_only", "infrastructure_only", "full"]
    ablation_results = []
    for fset in ablation_sets:
        cfg = ExperimentConfig(
            experiment_id=f"exp_ablation_{fset}",
            dataset_root=dataset_root,
            model_type="isolation_forest",
            hyperparameters={"n_estimators": 50},
            feature_set=fset,
            output_dir=str(exp_output_root),
        )
        runner = ExperimentRunner(cfg)
        res = runner.run()
        iid_wm = res["iid_test"]["window_metrics"]
        iid_cm = res["iid_test"]["curve_metrics"]
        ablation_results.append({
            "feature_set": fset,
            "precision": iid_wm["precision"],
            "recall": iid_wm["recall"],
            "f1": iid_wm["f1"],
            "fpr": iid_wm["fpr"],
            "auroc": iid_cm["auroc"],
            "auprc": iid_cm["auprc"],
        })
        print(f"   Feature Set: {fset:20s} | F1: {iid_wm['f1']:.3f} | AUROC: {iid_cm['auroc']:.3f} | FPR: {iid_wm['fpr']:.3f}")

    report["feature_ablation"] = ablation_results

    # -------------------------------------------------------------------------
    # PART 3: Experiments G & H — Device and Environmental Generalization Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">> Evaluating Experiments G & H: Generalization Gap across Models...")
    gen_results = []
    for m_res in model_benchmark_results:
        gen_results.append({
            "model": m_res["model_type"],
            "iid_recall": m_res["iid_recall"],
            "iid_fpr": m_res["iid_fpr"],
            "ood_recall": m_res["ood_recall"],
            "ood_fpr": m_res["ood_fpr"],
            "recall_gap": m_res["recall_gap"],
            "fpr_gap": m_res["fpr_gap"],
        })
    report["generalization_analysis"] = gen_results

    # -------------------------------------------------------------------------
    # PART 4: Experiment I — Attack Type and Severity Breakdown
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">> Evaluating Experiment I: Performance by Attack Type & Severity...")
    loader = MLDatasetLoader(dataset_root, master_seed=42)
    df_test = loader.load_split_dataframe("test")
    # Load isolation_forest results
    if_res_file = exp_output_root / "exp_isolation_forest" / "results.json"
    with open(if_res_file, "r", encoding="utf-8") as f:
        if_res = json.load(f)
    report["attack_breakdown"] = if_res.get("iid_test", {}).get("attack_type_breakdown", {})

    # -------------------------------------------------------------------------
    # PART 5: Experiment K — False Alarm Stability (Extended Normal-Only Run)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">> Running Experiment K: False-Alarm Stability on Normal Telemetry...")
    df_train = loader.load_split_dataframe("train")
    df_val = loader.load_split_dataframe("val")
    df_normal = pd.concat([df_train, df_val], ignore_index=True)
    prep = PreprocessorPipeline.load(exp_output_root / "exp_isolation_forest" / "preprocessor.json")
    ext = FeatureExtractor(feature_set="full", add_temporal_features=True)
    X_norm, _ = ext.extract(df_normal)
    X_norm_sc = prep.transform(X_norm)
    model_if = build_model("isolation_forest").load(exp_output_root / "exp_isolation_forest" / "model.bin")
    scores_norm = model_if.score(X_norm_sc)
    th = if_res["calibration"]["threshold"]
    normal_alarms = int(np.sum(scores_norm > th))
    report["false_alarm_stability"] = {
        "total_normal_windows": len(scores_norm),
        "false_alarm_windows": normal_alarms,
        "empirical_normal_fpr": float(normal_alarms / max(1, len(scores_norm))),
    }
    print(f"   Normal Windows: {len(scores_norm)} | False Alarms: {normal_alarms} | Empirical FPR: {report['false_alarm_stability']['empirical_normal_fpr']:.4f}")

    # -------------------------------------------------------------------------
    # PART 6: Experiment L — Leakage Sanity Checks Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">> Checking Experiment L: Sanity Experiments...")
    report["sanity_checks"] = if_res.get("sanity_experiments", {})
    print(f"   Label Shuffle AUROC   : {report['sanity_checks'].get('shuffled_labels', {}).get('auroc', 0.5):.3f}")
    print(f"   Random Feature AUROC  : {report['sanity_checks'].get('random_features', {}).get('auroc', 0.5):.3f}")

    # -------------------------------------------------------------------------
    # Save Final Consolidated JSON and Markdown Reports
    # -------------------------------------------------------------------------
    with open(reports_dir / "v06_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Markdown table generation
    md_lines = [
        "# QISM V0.6 Benchmark Summary Report",
        "",
        "## 1. Consolidated Baseline Model Benchmark",
        "| Model | IID FPR | IID Recall | IID F1 | IID AUROC | IID AUPRC | OOD Recall | OOD FPR | Recall Gap | Median Delay (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in model_benchmark_results:
        d_str = f"{r['median_delay_s']:.4f}" if r['median_delay_s'] is not None else "N/A"
        md_lines.append(
            f"| {r['model_type']} | {r['iid_fpr']:.3f} | {r['iid_recall']:.3f} | {r['iid_f1']:.3f} | {r['iid_auroc']:.3f} | {r['iid_auprc']:.3f} | {r['ood_recall']:.3f} | {r['ood_fpr']:.3f} | {r['recall_gap']:+.3f} | {d_str} |"
        )

    md_lines.extend([
        "",
        "## 2. Feature Ablation Study",
        "| Feature Set | Precision | Recall | F1 | FPR | AUROC | AUPRC |",
        "|---|---|---|---|---|---|---|",
    ])
    for a in ablation_results:
        md_lines.append(
            f"| {a['feature_set']} | {a['precision']:.3f} | {a['recall']:.3f} | {a['f1']:.3f} | {a['fpr']:.3f} | {a['auroc']:.3f} | {a['auprc']:.3f} |"
        )

    with open(reports_dir / "v06_benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n" + "=" * 80)
    print("ALL V0.6 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Summary reports saved to {reports_dir / 'v06_benchmark_summary.json'} and {reports_dir / 'v06_benchmark_summary.md'}")
    print("=" * 80)
    return report


if __name__ == "__main__":
    run_all_v06_experiments()
