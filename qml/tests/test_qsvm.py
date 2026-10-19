"""Unit tests for Quantum Kernel Support Vector Machine (QSVM)."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from qml.src.quantum_kernel_svm import QuantumKernelSVM


def test_quantum_kernel_properties():
    n_qubits = 4
    model = QuantumKernelSVM(n_qubits=n_qubits, C=1.0, seed=42)

    x1 = np.array([0.2, 0.8, 1.4, 2.0])
    x2 = np.array([0.5, 1.1, 0.3, 2.5])

    # 1. Unit self-fidelity
    k11 = model.compute_kernel_value(x1, x1)
    assert np.isclose(k11, 1.0, atol=1e-5), f"Self-fidelity is {k11}, expected 1.0"

    # 2. Symmetry
    k12 = model.compute_kernel_value(x1, x2)
    k21 = model.compute_kernel_value(x2, x1)
    assert np.isclose(k12, k21, atol=1e-5), f"Kernel not symmetric: K(1,2)={k12}, K(2,1)={k21}"

    # 3. Gram matrix properties
    X = np.random.uniform(0.0, np.pi, size=(6, n_qubits))
    K = model.compute_gram_matrix(X)

    assert K.shape == (6, 6)
    assert np.allclose(K, K.T, atol=1e-5), "Gram matrix not symmetric"
    np.testing.assert_allclose(np.diag(K), np.ones(6), atol=1e-5)

    # 4. Positive semi-definiteness: all eigenvalues >= -1e-5
    eigvals = np.linalg.eigvalsh(K)
    assert (eigvals >= -1e-5).all(), f"Negative eigenvalues found: {eigvals}"


def test_qsvm_fit_predict_save():
    model = QuantumKernelSVM(n_qubits=4, C=1.0, seed=42)
    X_train = np.random.uniform(0.0, np.pi, size=(8, 4))
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    model.fit(X_train, y_train)

    X_test = np.random.uniform(0.0, np.pi, size=(4, 4))
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)

    assert len(probs) == 4
    assert len(preds) == 4
    assert set(preds).issubset({0, 1})

    with tempfile.TemporaryDirectory() as tmpdir:
        chk_path = Path(tmpdir) / "qsvm.joblib"
        model.save_artifacts(chk_path)
        loaded = QuantumKernelSVM.load_artifacts(chk_path)
        loaded_preds = loaded.predict(X_test)
        np.testing.assert_array_equal(preds, loaded_preds)
