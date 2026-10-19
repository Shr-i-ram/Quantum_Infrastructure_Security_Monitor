"""Master experiment runner executing all QML Review 2 models and classical baselines.

Orchestrates:
1. Dataset loading, feature extraction, train-only PCA and angle scaling.
2. VQC model initialization, Adam training, validation loss tracking, and checkpointing.
3. Quantum Kernel Gram matrix calculation, SVM grid search, and checkpointing.
4. Quantum Autoencoder unsupervised training and threshold calibration.
5. Direct classical ML baselines (Logistic Regression, RBF-SVM, Small MLP).
6. Comprehensive evaluation on test and generalization splits.
7. Generating Figures 1 through 11 in artifacts/qml_review2/figures/.
8. Saving experiment manifests, predictions, and metric summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from qml.src.baselines import (
    ClassicalLogisticRegressionBaseline,
    ClassicalRBFSVMBaseline,
    SmallClassicalMLP,
)
from qml.src.data import QMLDatasetManager
from qml.src.evaluate import evaluate_all_models
from qml.src.plots import (
    plot_confusion_matrices,
    plot_feature_space,
    plot_performance_comparison,
    plot_per_attack_recall,
    plot_pr_curves,
    plot_qml_architectures,
    plot_quantum_kernel_matrix,
    plot_roc_curves,
    plot_runtime_comparison,
    plot_score_distributions,
    plot_vqc_training_history,
)
from qml.src.quantum_autoencoder import QuantumAutoencoder
from qml.src.quantum_kernel_svm import QuantumKernelSVM
from qml.src.train_qae import train_qae
from qml.src.train_qsvm import train_qsvm
from qml.src.train_vqc import train_vqc
from qml.src.vqc_model import VariationalQuantumClassifier


def run_qml_experiments(
    dataset_root: str = "data/v0.5_small/v05_small_dataset",
    representation: str = "pca",
    n_qubits: int = 4,
    vqc_layers: int = 2,
    vqc_epochs: int = 35,
    qae_epochs: int = 30,
    seed: int = 42,
    output_dir: str = "artifacts/qml_review2",
) -> dict[str, Any]:
    """Execute end-to-end QML Review 2 benchmark."""
    t_start_total = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs_dir = out / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    chk_dir = out / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)
    man_dir = out / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== QISM Review 2: QML Anomaly Detection Benchmark ===")
    print(f"Dataset: {dataset_root} | Representation: {representation} | Qubits: {n_qubits} | Seed: {seed}")

    # 1. Prepare data
    print("\n[1/6] Loading and partitioning telemetry dataset...")
    dm = QMLDatasetManager(dataset_root=dataset_root, representation=representation, n_qubits=n_qubits, master_seed=seed)
    dm.save_artifacts(chk_dir)
    np_data = dm.get_numpy_data()
    meta_data = dm.get_metadata()

    X_train, y_train = np_data["train"]
    X_val, y_val = np_data["val"]
    X_test, y_test = np_data["test"]
    X_gen, y_gen = np_data["gen"]

    print(f"  Train: {X_train.shape} (Attack: {y_train.sum()}/{len(y_train)})")
    print(f"  Val:   {X_val.shape} (Attack: {y_val.sum()}/{len(y_val)})")
    print(f"  Test:  {X_test.shape} (Attack: {y_test.sum()}/{len(y_test)})")
    print(f"  Gen:   {X_gen.shape} (Attack: {y_gen.sum()}/{len(y_gen)})")

    runtime_records: dict[str, dict[str, float]] = {}

    # 2. Train Model 1: VQC
    print("\n[2/6] Training Model 1: Hybrid Variational Quantum Classifier (VQC)...")
    vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=vqc_layers, seed=seed)
    vqc_rec = train_vqc(
        model=vqc,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=vqc_epochs,
        lr=0.02,
        output_dir=out,
    )
    runtime_records["Variational Quantum Classifier"] = {"train_time_s": vqc_rec["train_duration_s"]}
    print(f"  VQC trained in {vqc_rec['train_duration_s']:.2f} s | Calibrated Threshold: {vqc_rec['calibrated_threshold']:.2f}")

    # 3. Train Model 2: Quantum Kernel SVM
    print("\n[3/6] Training Model 2: Quantum Kernel SVM (QSVM)...")
    qsvm_rec = train_qsvm(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_qubits=n_qubits,
        seed=seed,
        output_dir=out,
    )
    qsvm = qsvm_rec["model"]
    total_qsvm_train_s = qsvm_rec["kernel_build_time_s"] + qsvm_rec["svm_tuning_time_s"]
    runtime_records["Quantum Kernel SVM"] = {
        "train_time_s": total_qsvm_train_s,
        "kernel_build_time_s": qsvm_rec["kernel_build_time_s"],
    }
    print(f"  QSVM Gram matrix computed in {qsvm_rec['kernel_build_time_s']:.2f} s | SVM fit in {qsvm_rec['svm_tuning_time_s']:.3f} s | Best C: {qsvm_rec['best_C']}")

    # Precompute test and generalization kernel matrices for QSVM
    print("  Precomputing QSVM kernel matrices for evaluation...")
    t0_ktest = time.time()
    K_test = qsvm.compute_gram_matrix(X_test, X_train)
    K_gen = qsvm.compute_gram_matrix(X_gen, X_train)
    kernel_matrices = {"test": K_test, "gen": K_gen}
    print(f"  Evaluation kernel matrices computed in {time.time() - t0_ktest:.2f} s")

    # 4. Train Model 3: Quantum Autoencoder (Unsupervised)
    print("\n[4/6] Training Model 3: Lightweight Quantum Autoencoder (QAE)...")
    X_train_norm = X_train[y_train == 0]
    X_val_norm = X_val[y_val == 0]
    qae = QuantumAutoencoder(n_qubits=n_qubits, n_layers=2, seed=seed)
    qae_rec = train_qae(
        model=qae,
        X_train_normal=X_train_norm,
        X_val_normal=X_val_norm,
        epochs=qae_epochs,
        lr=0.03,
        output_dir=out,
    )
    runtime_records["Quantum Autoencoder"] = {"train_time_s": qae_rec["train_duration_s"]}
    print(f"  QAE trained in {qae_rec['train_duration_s']:.2f} s | Calibrated Threshold: {qae_rec['calibrated_threshold']:.3f}")

    # 5. Train Classical Baselines
    print("\n[5/6] Training Direct Classical Baselines on the Exact Same Inputs...")
    lr_model = ClassicalLogisticRegressionBaseline(C=1.0, seed=seed)
    lr_model.fit(X_train, y_train)
    # calibrate threshold on val
    val_probs_lr = lr_model.predict_proba(X_val)
    best_th_lr = 0.50
    best_f1_lr = -1.0
    for cand_th in np.linspace(0.1, 0.9, 81):
        f1_cand = float(pd.Series((val_probs_lr >= cand_th).astype(int)).corr(pd.Series(y_val)))  # quick heuristic
        if f1_cand > best_f1_lr:
            best_f1_lr = f1_cand
            best_th_lr = cand_th
    lr_model.set_threshold(best_th_lr)
    runtime_records["Classical Logistic Regression"] = {"train_time_s": lr_model.fit_time_s}

    rbf_svm = ClassicalRBFSVMBaseline(C=1.0, gamma="scale", seed=seed)
    rbf_svm.fit(X_train, y_train)
    runtime_records["Classical RBF-SVM"] = {"train_time_s": rbf_svm.fit_time_s}

    small_mlp = SmallClassicalMLP(in_features=n_qubits, hidden=8, seed=seed)
    small_mlp.fit(X_train, y_train, epochs=50, lr=0.02)
    runtime_records["Small Classical MLP"] = {"train_time_s": small_mlp.fit_time_s}

    models_to_eval = {
        "Variational Quantum Classifier": vqc,
        "Quantum Kernel SVM": qsvm,
        "Quantum Autoencoder": qae,
        "Classical Logistic Regression": lr_model,
        "Classical RBF-SVM": rbf_svm,
        "Small Classical MLP": small_mlp,
    }

    # 6. Comprehensive Evaluation
    print("\n[6/6] Evaluating all models on Test and Generalization partitions...")
    all_metrics, summary_df = evaluate_all_models(
        models=models_to_eval,
        data=np_data,
        metadata=meta_data,
        kernel_matrices=kernel_matrices,
        output_dir=out,
    )

    preds_df = pd.read_csv(out / "predictions" / "all_model_predictions.csv")

    # Generate Figures 1 through 11
    print("  Generating presentation figures in artifacts/qml_review2/figures/...")
    plot_qml_architectures(figs_dir / "fig01_qml_architectures.png")
    plot_performance_comparison(summary_df, figs_dir / "fig02_model_performance_comparison.png", split="test")
    plot_confusion_matrices(preds_df, figs_dir / "fig03_confusion_matrices.png", split="test")
    plot_roc_curves(preds_df, all_metrics, figs_dir / "fig04_roc_curves.png", split="test")
    plot_pr_curves(preds_df, all_metrics, figs_dir / "fig05_precision_recall_curves.png", split="test")
    plot_vqc_training_history(vqc_rec, figs_dir / "fig06_vqc_training_history.png")
    plot_quantum_kernel_matrix(qsvm_rec["K_train"], figs_dir / "fig07_quantum_kernel_matrix.png")
    plot_score_distributions(preds_df, figs_dir / "fig08_score_distributions.png", split="test")
    plot_runtime_comparison(summary_df, runtime_records, figs_dir / "fig09_runtime_comparison.png")
    plot_per_attack_recall(all_metrics, figs_dir / "fig10_per_attack_recall.png", split="test")
    plot_feature_space(X_train, y_train, figs_dir / "fig11_feature_space.png")

    total_duration_s = time.time() - t_start_total

    # Save manifest
    manifest = {
        "experiment_id": "qml_review2_benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_duration_s": total_duration_s,
        "dataset_root": str(dataset_root),
        "representation": representation,
        "n_qubits": n_qubits,
        "seed": seed,
        "models_evaluated": list(models_to_eval.keys()),
        "runtime_records": runtime_records,
        "figures_generated": [str(p.name) for p in figs_dir.glob("*.png")],
    }
    with open(man_dir / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("                    QISM REVIEW 2: BENCHMARK RESULTS SUMMARY                     ")
    print("=" * 80)
    df_test_summary = summary_df[summary_df["split"] == "test"][
        ["model_name", "accuracy", "precision", "recall", "f1_score", "auroc", "auprc", "latency_ms_per_sample"]
    ]
    print(df_test_summary.to_string(index=False))
    print("=" * 80)
    print(f"Total benchmark completed in {total_duration_s:.2f} seconds.\n")

    return {
        "all_metrics": all_metrics,
        "summary_df": summary_df,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_qml_experiments()
