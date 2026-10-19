r"""Quantum Kernel Support Vector Machine (QSVM) using PennyLane and Scikit-learn.

Mathematical Formulation:
1. Quantum Feature Map:
   |phi(x)> = U_map(x)|0>^n
   where U_map(x) consists of AngleEmbedding(rotation='Y') followed by a CNOT entangling ring.
2. Fidelity Quantum Kernel:
   K(x, x') = |<phi(x)|phi(x')>|^2 = |<0000| U_map^\dagger(x') U_map(x) |0000>|^2
   Evaluated as the measurement probability of state |0000>.
3. Properties:
   - K(x, x) = 1.0 (Unit self-fidelity)
   - K(x, x') = K(x', x) (Symmetry)
   - K is positive semi-definite (Gram matrix)
4. Decision Boundary:
   Trained via classical dual SVM optimization using precomputed kernel matrix K.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pennylane as qml
from sklearn.svm import SVC


class QuantumKernelSVM:
    """Quantum Kernel Support Vector Machine with PennyLane fidelity kernel and Scikit-learn SVC."""

    def __init__(
        self,
        n_qubits: int = 4,
        C: float = 1.0,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.C = C
        self.seed = seed
        self.threshold = 0.50

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        def _feature_map(x):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        @qml.qnode(self.dev)
        def _kernel_circuit(x1, x2):
            _feature_map(x1)
            qml.adjoint(_feature_map)(x2)
            return qml.probs(wires=range(n_qubits))

        self.kernel_qnode = _kernel_circuit

        # Classical SVM with precomputed kernel
        self.svc = SVC(
            kernel="precomputed",
            C=C,
            probability=True,
            random_state=seed,
        )

        self.X_train_cached: np.ndarray | None = None
        self.K_train: np.ndarray | None = None
        self.kernel_build_time_s: float = 0.0
        self.svm_fit_time_s: float = 0.0

    def compute_kernel_value(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute scalar fidelity kernel K(x1, x2)."""
        probs = self.kernel_qnode(x1, x2)
        return float(probs[0])

    def compute_gram_matrix(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        """Compute pairwise kernel Gram matrix K(X1, X2).

        If X2 is None, computes symmetric square matrix K(X1, X1) in O(N^2 / 2).
        """
        n1 = len(X1)
        if X2 is None:
            # Symmetric matrix
            K = np.zeros((n1, n1), dtype=np.float64)
            for i in range(n1):
                K[i, i] = 1.0
                for j in range(i + 1, n1):
                    val = self.compute_kernel_value(X1[i], X1[j])
                    K[i, j] = val
                    K[j, i] = val
            return K
        else:
            # Rectangular matrix (n1 x n2)
            n2 = len(X2)
            K = np.zeros((n1, n2), dtype=np.float64)
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.compute_kernel_value(X1[i], X2[j])
            return K

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, K_train: np.ndarray | None = None) -> QuantumKernelSVM:
        """Fit Quantum Kernel SVM on training data."""
        self.X_train_cached = np.array(X_train, dtype=np.float64)

        # 1. Compute or reuse training Gram matrix
        t0 = time.time()
        if K_train is not None:
            self.K_train = K_train
            self.kernel_build_time_s = 0.0
        else:
            self.K_train = self.compute_gram_matrix(self.X_train_cached)
            self.kernel_build_time_s = time.time() - t0

        # Verify symmetry and finite values
        if not np.all(np.isfinite(self.K_train)):
            raise ValueError("Training Gram matrix contains non-finite values.")
        if not np.allclose(self.K_train, self.K_train.T, atol=1e-5):
            raise ValueError("Training Gram matrix is not symmetric.")

        # 2. Fit classical SVM
        t1 = time.time()
        self.svc.fit(self.K_train, y_train)
        self.svm_fit_time_s = time.time() - t1

        return self

    def predict_proba(self, X: np.ndarray, K: np.ndarray | None = None) -> np.ndarray:
        """Predict class probabilities p(y=1|X)."""
        if self.X_train_cached is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        if K is None:
            K = self.compute_gram_matrix(X, self.X_train_cached)
        probs = self.svc.predict_proba(K)[:, 1]
        return probs

    def predict(self, X: np.ndarray, K: np.ndarray | None = None, threshold: float | None = None) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X, K=K)
        return (probs >= th).astype(int)

    def set_threshold(self, threshold: float) -> None:
        """Set decision threshold calibrated on validation set."""
        self.threshold = float(threshold)

    def save_artifacts(self, path: str | Path, meta: dict[str, Any] | None = None) -> None:
        """Persist trained SVM, kernel matrix, and experiment metadata."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "svc": self.svc,
            "C": self.C,
            "n_qubits": self.n_qubits,
            "seed": self.seed,
            "threshold": self.threshold,
            "kernel_build_time_s": self.kernel_build_time_s,
            "svm_fit_time_s": self.svm_fit_time_s,
            "K_train": self.K_train,
            "X_train_cached": self.X_train_cached,
            "meta": meta or {},
        }
        joblib.dump(payload, p)

    @classmethod
    def load_artifacts(cls, path: str | Path) -> QuantumKernelSVM:
        """Load trained QSVM from file."""
        payload = joblib.load(path)
        model = cls(n_qubits=payload["n_qubits"], C=payload["C"], seed=payload["seed"])
        model.svc = payload["svc"]
        model.K_train = payload["K_train"]
        model.X_train_cached = payload.get("X_train_cached")
        model.threshold = payload.get("threshold", 0.50)
        model.kernel_build_time_s = payload.get("kernel_build_time_s", 0.0)
        model.svm_fit_time_s = payload.get("svm_fit_time_s", 0.0)
        return model
