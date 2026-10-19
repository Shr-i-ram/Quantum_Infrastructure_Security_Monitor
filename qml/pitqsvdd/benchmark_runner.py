"""Automated Benchmark Suite for PI-TQSVDD (Mode B Pilot Investigations).

Runs controlled, budget-limited experiments covering:
1. Qubit configurations: 2, 4, 6, and 8 qubits.
2. Encodings: Angle, Data Reuploading, Amplitude, IQP/ZZ, Feature-Group, Learned Projection.
3. Architectures: Static QSVDD vs. Temporal PI-TQSVDD vs. Context-Aware PI-TQSVDD.
4. Classical Baselines: Classical Deep SVDD, Isolation Forest, One-Class SVM.
5. Generates the comprehensive publication figure collection in artifacts/pi_tqsvdd/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from qml.pitqsvdd.config import CircuitConfig, DataConfig, EvaluationConfig, ModelConfig, PITQSVDDConfig, TrainingConfig
from qml.pitqsvdd.data import PITQSVDDDatasetManager
from qml.pitqsvdd.evaluation.comparisons import (
    train_and_eval_classical_deep_svdd,
    train_and_eval_isolation_forest,
    train_and_eval_one_class_svm,
)
from qml.pitqsvdd.training.experiment_runner import PITQSVDDExperimentRunner
from qml.pitqsvdd.visualization.plots import (
    plot_architecture_diagram,
    plot_feature_inventory,
    plot_loss_curves,
    plot_model_comparison_bar,
    plot_qubits_and_encodings_benchmark,
    plot_roc_and_pr_curves,
    plot_score_distribution,
)


def run_pilot_benchmark_suite(
    artifacts_dir: str | Path = "artifacts/pi_tqsvdd",
    dataset_root: str = "data/v0.5_small/v05_small_dataset",
    pilot_epochs: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute complete pilot experiment suite and generate publication artifacts."""
    art_path = Path(artifacts_dir)
    fig_dir = art_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    runner = PITQSVDDExperimentRunner(artifacts_dir=art_path)

    print("\n========================================================================")
    print("  QISM PI-TQSVDD: PILOT EXPERIMENT BENCHMARK SUITE (MODE B)")
    print("========================================================================")

    # 1. Architecture diagram
    print("[1/5] Generating architectural schematics...")
    plot_architecture_diagram(fig_dir / "01_pitqsvdd_architecture_overview.png")

    # 2. Feature inventory and PCA curve
    dm_init = PITQSVDDDatasetManager(DataConfig(dataset_root=dataset_root, n_qubits=4))
    train_scens = dm_init.scenarios_transformed["train"]
    train_arr = np.concatenate(train_scens, axis=0)
    train_df = pd.DataFrame(train_arr, columns=[f"feat_{i}" for i in range(train_arr.shape[1])])
    plot_feature_inventory(train_df, fig_dir / "02_telemetry_features_and_pca_variance.png")

    # 3. Controlled Qubit Count and Encoding Experiments
    # Matrix of planned experiments:
    experiments_plan = [
        # Flagship Temporal PI-TQSVDD across all 4 required qubit counts
        {"id": "pilot_2q_angle", "n_q": 2, "enc": "angle", "rep": "pca", "model": "temporal_pitqsvdd"},
        {"id": "pilot_4q_angle", "n_q": 4, "enc": "angle", "rep": "pca", "model": "temporal_pitqsvdd"},
        {"id": "pilot_6q_angle", "n_q": 6, "enc": "angle", "rep": "pca", "model": "temporal_pitqsvdd"},
        {"id": "pilot_8q_angle", "n_q": 8, "enc": "angle", "rep": "pca", "model": "temporal_pitqsvdd"},

        # Encoding variations on 4 qubits
        {"id": "pilot_4q_reupload", "n_q": 4, "enc": "data_reuploading", "rep": "highdim_pca", "model": "temporal_pitqsvdd"},
        {"id": "pilot_4q_amplitude", "n_q": 4, "enc": "amplitude", "rep": "highdim_pca", "model": "temporal_pitqsvdd"},
        {"id": "pilot_4q_iqp_zz", "n_q": 4, "enc": "iqp_zz", "rep": "pca", "model": "temporal_pitqsvdd"},
        {"id": "pilot_4q_feature_group", "n_q": 4, "enc": "feature_group", "rep": "feature_group", "model": "temporal_pitqsvdd"},
        {"id": "pilot_4q_learned_proj", "n_q": 4, "enc": "learned_projection", "rep": "learned_bottleneck", "model": "temporal_pitqsvdd"},

        # Data Reuploading on 8 qubits
        {"id": "pilot_8q_reupload", "n_q": 8, "enc": "data_reuploading", "rep": "highdim_pca", "model": "temporal_pitqsvdd"},

        # Architectural baselines on 4 qubits
        {"id": "pilot_4q_static_qsvdd", "n_q": 4, "enc": "angle", "rep": "pca", "model": "static_qsvdd"},
        {"id": "pilot_4q_context_aware", "n_q": 4, "enc": "angle", "rep": "pca", "model": "context_aware_pitqsvdd"},
    ]

    print(f"\n[2/5] Executing {len(experiments_plan)} Quantum Model Pilot Experiments...")
    executed_results = {}
    for plan in experiments_plan:
        cfg = PITQSVDDConfig(
            data=DataConfig(
                dataset_root=dataset_root,
                representation=plan["rep"],
                n_qubits=plan["n_q"],
                sequence_length=3,
                master_seed=seed,
            ),
            circuit=CircuitConfig(
                n_qubits=plan["n_q"],
                n_layers=2,
                encoding_type=plan["enc"],
                backend="lightning.qubit",
                diff_method="adjoint",
                reupload_blocks=2,
            ),
            model=ModelConfig(
                model_type=plan["model"],
                lambda_temporal=0.5,
                lambda_reg=1e-4,
                temporal_hidden_dim=16,
            ),
            training=TrainingConfig(
                lr=0.01,
                batch_size=8,
                max_epochs=pilot_epochs,
                early_stopping_patience=6,
                save_dir=str(art_path / "checkpoints"),
                run_id=plan["id"],
                resume_mode="auto",
            ),
            evaluation=EvaluationConfig(
                threshold_percentile=95.0,
                alpha_svdd=0.5,
                beta_temporal=0.5,
                output_dir=str(art_path),
            ),
        )

        res = runner.run_experiment(cfg)
        executed_results[plan["id"]] = res

    # 4. Classical Baselines Execution
    print("\n[3/5] Evaluating Classical ML Baselines on the exact same partitions...")
    # Extract raw 48-feature arrays from dm_init
    X_train_raw = np.concatenate([dm_init.scenarios_transformed["train"][i] for i in range(len(dm_init.scenarios_transformed["train"]))], axis=0)
    y_train_raw = np.zeros(len(X_train_raw))

    X_val_raw = np.concatenate([dm_init.scenarios_transformed["val"][i] for i in range(len(dm_init.scenarios_transformed["val"]))], axis=0)
    y_val_raw = np.concatenate([dm_init.scenarios_metadata["val"][i]["attack_active"].to_numpy(dtype=int) for i in range(len(dm_init.scenarios_transformed["val"]))], axis=0)

    X_test_raw = np.concatenate([dm_init.scenarios_transformed["test"][i] for i in range(len(dm_init.scenarios_transformed["test"]))], axis=0)
    y_test_raw = np.concatenate([dm_init.scenarios_metadata["test"][i]["attack_active"].to_numpy(dtype=int) for i in range(len(dm_init.scenarios_transformed["test"]))], axis=0)

    deep_svdd_res = train_and_eval_classical_deep_svdd(X_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw, epochs=25, seed=seed)
    iso_res = train_and_eval_isolation_forest(X_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw, seed=seed)
    ocsvm_res = train_and_eval_one_class_svm(X_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw)

    classical_summary = {
        "Classical Deep SVDD": deep_svdd_res["test_metrics"],
        "Isolation Forest": iso_res["test_metrics"],
        "One-Class SVM": ocsvm_res["test_metrics"],
    }
    with open(art_path / "classical_baselines_summary.json", "w", encoding="utf-8") as f:
        json.dump(classical_summary, f, indent=2)

    # 5. Generate Figures
    print("\n[4/5] Generating Diagnostic & Comparative Figures...")
    registry_df = runner.registry_df

    # Multi-panel benchmark: qubits, encodings, runtime
    plot_qubits_and_encodings_benchmark(registry_df, fig_dir / "03_qubits_and_encodings_benchmark.png")

    # Flagship model diagnostics (using pilot_4q_reupload or pilot_4q_angle)
    flagship_res = executed_results.get("pilot_4q_reupload", executed_results.get("pilot_4q_angle"))
    if flagship_res:
        hist = flagship_res.get("train_history", [])
        if hist:
            plot_loss_curves(hist, fig_dir / "04_flagship_loss_convergence.png")

        eval_data = flagship_res.get("evaluation", {}).get("test", {})
        if eval_data and "scores_combined" in eval_data:
            s_test = np.array(eval_data["scores_combined"])
            y_test = np.array(eval_data["labels"])
            th = flagship_res["evaluation"]["calibrated_threshold"]
            plot_score_distribution(s_test, y_test, th, fig_dir / "05_anomaly_score_distribution.png")
            plot_roc_and_pr_curves(y_test, s_test, fig_dir / "06_roc_pr_curves_flagship.png", title_suffix="PI-TQSVDD (4Q Reupload)")

    # Model comparison bar chart: PI-TQSVDD vs Static QSVDD vs Classical Deep SVDD vs Isolation Forest vs OCSVM vs VQC Review 2
    comp_models = {
        "PI-TQSVDD (4Q Reupload)": executed_results.get("pilot_4q_reupload", {}).get("summary", {}),
        "Static QSVDD (4Q)": executed_results.get("pilot_4q_static_qsvdd", {}).get("summary", {}),
        "Classical Deep SVDD": deep_svdd_res["test_metrics"],
        "Isolation Forest": iso_res["test_metrics"],
        "One-Class SVM": ocsvm_res["test_metrics"],
        "Review 2 VQC (Supervised)": {"f1": 0.8889, "auroc": 0.8800},
    }
    plot_model_comparison_bar(comp_models, fig_dir / "07_model_family_comparison.png")

    print("\n[5/5] Pilot Benchmark Complete!")
    print(f"Figures saved in: {fig_dir}")
    print(f"Registry saved in: {runner.registry_csv}")

    return {
        "executed_results": executed_results,
        "classical_results": classical_summary,
        "registry_records": len(runner.registry_df),
    }


if __name__ == "__main__":
    run_pilot_benchmark_suite()
