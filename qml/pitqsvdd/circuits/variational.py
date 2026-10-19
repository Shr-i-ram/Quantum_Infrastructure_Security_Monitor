"""Parameterized Quantum Circuits for PI-TQSVDD with PennyLane lightning.qubit."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from qml.pitqsvdd.config import CircuitConfig
from qml.pitqsvdd.encodings.base import BaseEncoding
from qml.pitqsvdd.encodings.data_reuploading import DataReuploadingEncoding


class VariationalQuantumCircuit(nn.Module):
    """Parameterized Quantum Circuit executing on lightning.qubit with adjoint differentiation."""

    def __init__(
        self,
        config: CircuitConfig,
        encoder: BaseEncoding,
        seed: int = 42,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.seed = seed

        # Initialize PennyLane device
        # Verification: lightning.qubit is strictly required
        try:
            self.dev = qml.device(config.backend, wires=self.n_qubits)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize required quantum backend '{config.backend}': {e}"
            )

        # Build QNode with PyTorch adjoint backprop
        self.qnode = self._build_qnode()

        # Initialize trainable variational weights: shape (n_layers, n_qubits, 3)
        torch.manual_seed(seed)
        # Random initialisation away from zero to avoid SVDD representation collapse
        self.q_weights = nn.Parameter(
            0.2 * torch.randn(self.n_layers, self.n_qubits, 3, dtype=torch.float32) + 0.1
        )

    def _build_qnode(self):
        """Construct the PennyLane QNode."""
        n_q = self.n_qubits
        n_l = self.n_layers
        wires = list(range(n_q))
        entangle_type = self.config.entanglement
        encoder = self.encoder
        is_reuploading = isinstance(encoder, DataReuploadingEncoding)

        @qml.qnode(self.dev, interface="torch", diff_method=self.config.diff_method)
        def _circuit(inputs, weights):
            if not is_reuploading:
                # Apply initial state encoding
                encoder.apply(inputs, wires=wires)

            for l in range(n_l):
                if is_reuploading:
                    encoder.apply(inputs, wires=wires, layer_idx=l)

                # Parameterized rotation layer: RX, RY, RZ per qubit
                for i in range(n_q):
                    qml.RX(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                    qml.RZ(weights[l, i, 2], wires=i)

                # Entanglement pattern
                if n_q > 1:
                    if entangle_type == "ring":
                        for i in range(n_q):
                            qml.CNOT(wires=[i, (i + 1) % n_q])
                    elif entangle_type == "linear":
                        for i in range(n_q - 1):
                            qml.CNOT(wires=[i, i + 1])
                    elif entangle_type == "all-to-all":
                        for i in range(n_q):
                            for j in range(i + 1, n_q):
                                qml.CNOT(wires=[i, j])

            return [qml.expval(qml.PauliZ(w)) for w in wires]

        return _circuit

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate circuit for a single feature vector. Returns shape (n_qubits,)."""
        res = self.qnode(x, self.q_weights)
        return torch.stack(res)

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Batch evaluation of quantum circuit.

        Args:
            x_batch: Tensor of shape (B, D) or (B, L, D).

        Returns:
            Tensor of latent representations: shape (B, n_qubits) or (B, L, n_qubits).
        """
        orig_shape = x_batch.shape
        if x_batch.ndim == 3:
            B, L, D = orig_shape
            x_flat = x_batch.view(B * L, D)
            z_flat = torch.stack([self.forward_single(x_flat[i]) for i in range(B * L)])
            return z_flat.view(B, L, self.n_qubits)
        elif x_batch.ndim == 2:
            B, D = orig_shape
            return torch.stack([self.forward_single(x_batch[i]) for i in range(B)])
        else:
            return self.forward_single(x_batch)

    def get_circuit_metrics(self) -> dict[str, Any]:
        """Compute gate counts, parameter counts, and depth estimates."""
        n_q = self.n_qubits
        n_l = self.n_layers
        rotations_per_layer = n_q * 3
        if self.config.entanglement == "ring":
            cnots_per_layer = n_q
        elif self.config.entanglement == "linear":
            cnots_per_layer = max(0, n_q - 1)
        elif self.config.entanglement == "all-to-all":
            cnots_per_layer = (n_q * (n_q - 1)) // 2
        else:
            cnots_per_layer = n_q

        total_cnots = cnots_per_layer * n_l
        total_rotations = rotations_per_layer * n_l
        trainable_params = self.q_weights.numel()

        return {
            "backend": self.config.backend,
            "diff_method": self.config.diff_method,
            "n_qubits": n_q,
            "n_layers": n_l,
            "entanglement": self.config.entanglement,
            "trainable_quantum_params": trainable_params,
            "cnot_count": total_cnots,
            "rotation_gate_count": total_rotations,
            "estimated_depth": n_l * (3 + (2 if n_q > 1 else 0)),
        }
