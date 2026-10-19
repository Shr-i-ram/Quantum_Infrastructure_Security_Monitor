"""Tests for quantum circuit construction and execution on lightning.qubit."""

import pytest
import torch

from qml.pitqsvdd.circuits.variational import VariationalQuantumCircuit
from qml.pitqsvdd.config import CircuitConfig
from qml.pitqsvdd.encodings import get_encoding


@pytest.mark.parametrize("n_q", [2, 4, 6, 8])
def test_circuit_execution_and_gradients_on_lightning(n_q: int):
    """Verify forward and backward passes on lightning.qubit for required qubit configurations."""
    cfg = CircuitConfig(n_qubits=n_q, n_layers=1, backend="lightning.qubit", diff_method="adjoint")
    enc = get_encoding("angle", n_qubits=n_q)
    circuit = VariationalQuantumCircuit(cfg, enc)

    x = torch.randn(2, n_q)
    z = circuit(x)

    # 1. Output shape must match (B, n_qubits)
    assert z.shape == (2, n_q)

    # 2. Expectation values of Pauli-Z must lie strictly in [-1, 1]
    assert torch.all(z >= -1.0001) and torch.all(z <= 1.0001)

    # 3. Adjoint gradient must be non-zero and finite
    loss = z.sum()
    loss.backward()
    assert circuit.q_weights.grad is not None
    assert torch.isfinite(circuit.q_weights.grad).all()
    assert circuit.q_weights.grad.norm().item() > 0.0
