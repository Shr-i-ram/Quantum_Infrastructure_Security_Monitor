"""Feature-Group Domain-Aware Quantum Encoding for PI-TQSVDD."""

from __future__ import annotations

from typing import Any, Sequence
import pennylane as qml
import torch

from qml.pitqsvdd.encodings.base import BaseEncoding


class FeatureGroupEncoding(BaseEncoding):
    """Encodes physical telemetry families into dedicated qubit registers."""

    def __init__(self, n_qubits: int, num_groups: int = 4, rotation: str = "Y"):
        super().__init__(n_qubits=n_qubits, name="feature_group_encoding")
        self.num_groups = num_groups
        self.rotation = rotation.upper()

    def apply(self, inputs: torch.Tensor, wires: Sequence[int], **kwargs: Any) -> None:
        """Apply group-specific rotations to designated wire partitions."""
        n_w = len(wires)
        feats = inputs.flatten()
        dim = feats.shape[0]

        # Allocate wires per group
        wires_per_grp = max(1, n_w // self.num_groups)

        for grp_idx in range(self.num_groups):
            w_start = grp_idx * wires_per_grp
            w_end = min(w_start + wires_per_grp, n_w)
            grp_wires = wires[w_start:w_end]

            # Features belonging to this group
            f_start = grp_idx * wires_per_grp
            for local_idx, wire in enumerate(grp_wires):
                f_idx = (f_start + local_idx) % dim
                val = feats[f_idx]
                if self.rotation == "Y":
                    qml.RY(val, wires=wire)
                elif self.rotation == "X":
                    qml.RX(val, wires=wire)
                elif self.rotation == "Z":
                    qml.RZ(val, wires=wire)
                else:
                    qml.RY(val, wires=wire)

        # Entangle across group boundaries
        for i in range(n_w):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n_w]])

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["num_groups"] = self.num_groups
        info["rotation"] = self.rotation
        return info
