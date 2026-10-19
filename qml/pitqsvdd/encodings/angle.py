"""Basic Angle Encoding for PI-TQSVDD."""

from __future__ import annotations

from typing import Any, Sequence
import pennylane as qml
import torch

from qml.pitqsvdd.encodings.base import BaseEncoding


class AngleEncoding(BaseEncoding):
    """Angle embedding rotating each qubit by theta_j = a_j * x_j + b_j."""

    def __init__(self, n_qubits: int, rotation: str = "Y"):
        super().__init__(n_qubits=n_qubits, name="angle_encoding")
        self.rotation = rotation.upper()

    def apply(self, inputs: torch.Tensor, wires: Sequence[int], **kwargs: Any) -> None:
        """Apply single-qubit rotations corresponding to input features."""
        # Truncate or use available inputs
        n_w = len(wires)
        features = inputs[:n_w] if inputs.shape[0] >= n_w else inputs

        for idx, wire in enumerate(wires):
            val = features[idx] if idx < features.shape[0] else torch.tensor(0.0)
            if self.rotation == "Y":
                qml.RY(val, wires=wire)
            elif self.rotation == "X":
                qml.RX(val, wires=wire)
            elif self.rotation == "Z":
                qml.RZ(val, wires=wire)
            else:
                qml.RY(val, wires=wire)

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["rotation"] = self.rotation
        return info
