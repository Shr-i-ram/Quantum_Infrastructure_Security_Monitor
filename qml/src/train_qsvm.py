"""Training and hyperparameter selection for Quantum Kernel SVM (Review 2).

Includes:
- Pairwise quantum fidelity Gram matrix computation.
- Hyperparameter tuning for SVM penalty C on validation data.
- Validation probability threshold tuning.
- Artifact persistence (cached kernel matrices, fitted SVM, metadata).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from qml.src.metrics import compute_binary_metrics
from qml.src.quantum_kernel_svm import QuantumKernelSVM


def train_qsvm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_qubits: int = 4,
    candidate_C: list[float] | None = None,
    seed: int = 42,
    output_dir: str | Path = "artifacts/qml_review2",
) -> dict[str, Any]:
    """Train Quantum Kernel SVM, optimize C on validation data, and save artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if candidate_C is None:
        candidate_C = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

    # Initialize model to compute Gram matrices
    base_model = QuantumKernelSVM(n_qubits=n_qubits, C=1.0, seed=seed)

    # 1. Compute training Gram matrix
    t0_k = time.time()
    K_train = base_model.compute_gram_matrix(X_train)
    kernel_build_time_s = time.time() - t0_k

    # 2. Compute validation kernel matrix K_val (N_val x N_train)
    K_val = base_model.compute_gram_matrix(X_val, X_train)

    # 3. Grid search C on validation performance
    best_C = 1.0
    best_f1 = -1.0
    best_th = 0.50
    best_model: QuantumKernelSVM | None = None

    t0_fit = time.time()
    for c_val in candidate_C:
        model = QuantumKernelSVM(n_qubits=n_qubits, C=c_val, seed=seed)
        model.fit(X_train, y_train, K_train=K_train)
        val_probs = model.predict_proba(X_val, K=K_val)

        # Threshold search for this C
        for cand_th in np.linspace(0.1, 0.9, 81):
            cand_preds = (val_probs >= cand_th).astype(int)
            m = compute_binary_metrics(y_val, cand_preds, val_probs)
            if m["f1_score"] > best_f1:
                best_f1 = m["f1_score"]
                best_C = c_val
                best_th = float(cand_th)
                best_model = model

    svm_tuning_time_s = time.time() - t0_fit

    if best_model is None:
        best_model = QuantumKernelSVM(n_qubits=n_qubits, C=best_C, seed=seed)
        best_model.fit(X_train, y_train, K_train=K_train)

    best_model.set_threshold(best_th)

    # Save model and artifacts
    chk_path = out_dir / "checkpoints" / "qsvm_model.joblib"
    best_model.save_artifacts(chk_path, meta={
        "best_C": best_C,
        "calibrated_threshold": best_th,
        "val_f1_at_best_th": best_f1,
        "kernel_build_time_s": kernel_build_time_s,
        "svm_tuning_time_s": svm_tuning_time_s,
    })

    # Save K_train as numpy file for figure plotting
    np.save(out_dir / "checkpoints" / "K_train.npy", K_train)

    record = {
        "best_C": best_C,
        "calibrated_threshold": best_th,
        "val_f1_at_calibrated_th": best_f1,
        "kernel_build_time_s": kernel_build_time_s,
        "svm_tuning_time_s": svm_tuning_time_s,
        "checkpoint_path": str(chk_path),
        "model": best_model,
        "K_train": K_train,
        "K_val": K_val,
    }

    return record
