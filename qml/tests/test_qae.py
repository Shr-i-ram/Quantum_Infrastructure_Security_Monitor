"""Unit tests for Quantum Autoencoder (QAE)."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from qml.src.quantum_autoencoder import QuantumAutoencoder


def test_qae_compression_and_gradients():
    n_qubits = 4
    model = QuantumAutoencoder(n_qubits=n_qubits, n_layers=2, seed=42)

    x = torch.rand(4, n_qubits) * np.pi
    scores = model(x)

    assert scores.shape == (4,)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    # Test gradient
    loss = scores.mean()
    loss.backward()

    assert model.q_weights.grad is not None
    assert model.q_weights.grad.norm().item() > 0.0


def test_qae_predict_and_checkpoint():
    model = QuantumAutoencoder(n_qubits=4, n_layers=1, seed=42)
    model.set_threshold(0.40)

    X = np.random.uniform(0.0, np.pi, size=(5, 4))
    scores = model.score_samples(X)
    preds = model.predict(X)

    assert len(scores) == 5
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})

    with tempfile.TemporaryDirectory() as tmpdir:
        chk_path = Path(tmpdir) / "qae.pt"
        model.save_checkpoint(chk_path)
        loaded = QuantumAutoencoder.load_checkpoint(chk_path)

        assert loaded.threshold == 0.40
        loaded_scores = loaded.score_samples(X)
        np.testing.assert_allclose(scores, loaded_scores, atol=1e-5)
