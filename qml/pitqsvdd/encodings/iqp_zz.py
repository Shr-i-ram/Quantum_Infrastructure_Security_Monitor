"""IQP-style and ZZ Feature Map Encoding for PI-TQSVDD."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pennylane as qml
import torch

from qml.pitqsvdd.encodings.base import BaseEncoding


class IQPZZEncoding(BaseEncoding):
    """IQP / ZZ feature map embedding continuous features with 2-qubit interactions.

    Mathematical formulation:
        U_ZZ(x) = exp( i sum_j x_j Z_j + i sum_{j<k} (pi - x_j)(pi - x_k) Z_j Z_k ) H^{\otimes n}
    """

    def __init__(self, n_qubits: int, reps: int = 1):
        super().__init__(n_qubits=n_qubits, name="iqp_zz_encoding")
        self.reps = reps

    def apply(self, inputs: torch.Tensor, wires: Sequence[int], **kwargs: Any) -> None:
        """Apply IQP/ZZ feature map across specified wires."""
        n_w = len(wires)
        feats = inputs[:n_w] if inputs.shape[0] >= n_w else inputs

        for _ in range(self.reps):
            # 1. Hadamard on all wires
            for wire in wires:
                qml.Hadamard(wires=wire)

            # 2. Single-qubit RZ rotations
            for idx, wire in enumerate(wires):
                val = feats[idx] if idx < feats.shape[0] else torch.tensor(0.0)
                qml.RZ(2.0 * val, wires=wire)

            # 3. Two-qubit ZZ entangling gates
            for i in range(n_w):
                j = (i + 1) % n_w
                if n_w == 2 and i == 1:
                    continue  # Avoid duplicate pair for 2 qubits
                w_i = wires[i]
                w_j = wires[j]
                val_i = feats[i] if i < feats.shape[0] else torch.tensor(0.0)
                val_j = feats[j] if j < feats.shape[0] else torch.tensor(0.0)
                # Interaction phase
                phi = 2.0 * (np.pi - val_i) * (np.pi - val_j)

                qml.CNOT(wires=[w_i, w_j])
                qml.RZ(phi, wires=w_j)
                qml.CNOT(wires=[w_i, w_j])

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["reps"] = self.reps
        return info
