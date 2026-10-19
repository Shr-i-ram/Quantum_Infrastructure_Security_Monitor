"""Unit tests for Variational Quantum Classifier (VQC)."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from qml.src.vqc_model import VariationalQuantumClassifier


def test_vqc_forward_and_gradients():
    n_qubits = 4
    model = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=2, seed=42)

    # Batch of 4 samples
    x = torch.rand(4, n_qubits) * np.pi
    probs = model(x)

    assert probs.shape == (4,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Test gradient calculation
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = torch.nn.BCELoss()(probs, target)
    loss.backward()

    # Weights must have non-zero gradients
    assert model.q_weights.grad is not None
    assert model.q_weights.grad.norm().item() > 0.0
    assert model.head.weight.grad is not None
    assert model.head.weight.grad.norm().item() > 0.0


def test_vqc_predict_and_checkpoint():
    model = VariationalQuantumClassifier(n_qubits=4, n_layers=1, seed=42)
    model.set_threshold(0.60)

    X = np.random.uniform(0.0, np.pi, size=(5, 4))
    probs = model.predict_proba(X)
    preds = model.predict(X)

    assert len(probs) == 5
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})

    # Test save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        chk_path = Path(tmpdir) / "vqc.pt"
        model.save_checkpoint(chk_path)
        loaded = VariationalQuantumClassifier.load_checkpoint(chk_path)

        assert loaded.threshold == 0.60
        loaded_probs = loaded.predict_proba(X)
        np.testing.assert_allclose(probs, loaded_probs, atol=1e-5)
