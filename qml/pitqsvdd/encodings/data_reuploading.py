"""Multi-layer Data Reuploading Encoding for PI-TQSVDD."""

from __future__ import annotations

from typing import Any, Sequence
import pennylane as qml
import torch

from qml.pitqsvdd.encodings.base import BaseEncoding


class DataReuploadingEncoding(BaseEncoding):
    """Multi-layer Data Reuploading scheme interleaving feature injections with rotations.

    Mathematical formulation:
        |psi(x; theta)> = U_L(x; theta_L) ... U_1(x; theta_1) |0>^n
    where each layer U_l injects a slice of features x followed by trainable rotations.
    """

    def __init__(self, n_qubits: int, reupload_blocks: int = 2, rotation: str = "Y"):
        super().__init__(n_qubits=n_qubits, name="data_reuploading")
        self.reupload_blocks = reupload_blocks
        self.rotation = rotation.upper()

    def apply(self, inputs: torch.Tensor, wires: Sequence[int], layer_idx: int = 0, **kwargs: Any) -> None:
        """Apply a single reuploading block for the specified layer index.

        Distributes features across layers if input dimension > wire count.
        """
        n_w = len(wires)
        total_dim = inputs.shape[0]

        # Determine feature slice for this layer
        start_idx = (layer_idx * n_w) % total_dim
        for i, wire in enumerate(wires):
            feat_idx = (start_idx + i) % total_dim
            val = inputs[feat_idx]
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
        info["reupload_blocks"] = self.reupload_blocks
        info["rotation"] = self.rotation
        return info
