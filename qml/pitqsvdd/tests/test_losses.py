"""Tests for SVDD, temporal, and multi-centre loss formulations."""

import pytest
import torch

from qml.pitqsvdd.circuits.variational import VariationalQuantumCircuit
from qml.pitqsvdd.config import CircuitConfig, ModelConfig
from qml.pitqsvdd.encodings import get_encoding
from qml.pitqsvdd.models import build_model


@pytest.mark.parametrize("model_type", ["static_qsvdd", "temporal_pitqsvdd", "context_aware_pitqsvdd"])
def test_model_loss_computation_and_gradients(model_type: str):
    """Verify that all three model formulations compute finite loss and backpropagate gradients."""
    n_q = 4
    c_cfg = CircuitConfig(n_qubits=n_q, n_layers=1, backend="lightning.qubit")
    enc = get_encoding("angle", n_qubits=n_q)
    circuit = VariationalQuantumCircuit(c_cfg, enc)

    m_cfg = ModelConfig(model_type=model_type, lambda_temporal=0.5, lambda_reg=1e-4)
    model = build_model(model_type, circuit, m_cfg)

    # Fake input batch: (B=2, L=3, D=4)
    x_windows = torch.randn(2, 3, n_q)
    targets = torch.randn(2, n_q)
    has_target = torch.tensor([True, False])

    # Initialize centres
    if hasattr(model, "init_centre"):
        model.init_centre(x_windows)
    elif hasattr(model, "init_centres"):
        model.init_centres(x_windows)

    out = model.compute_loss(x_windows, targets=targets, has_target=has_target)
    loss = out["loss"]

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    loss.backward()
    assert circuit.q_weights.grad is not None
    assert torch.isfinite(circuit.q_weights.grad).all()
