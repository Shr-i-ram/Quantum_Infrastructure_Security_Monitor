"""Amplitude Encoding for PI-TQSVDD."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pennylane as qml
import torch

from qml.pitqsvdd.encodings.base import BaseEncoding


class AmplitudeEncoding(BaseEncoding):
    """Amplitude embedding representing normalized input in the state vector amplitudes.

    Mathematical formulation:
        |psi(x)> = sum_{j=0}^{2^n - 1} c_j |j>
        where c_j = x_j / ||x||_2
    """

    def __init__(self, n_qubits: int):
        super().__init__(n_qubits=n_qubits, name="amplitude_encoding")
        self.state_dim = 2 ** n_qubits

    def apply(self, inputs: torch.Tensor, wires: Sequence[int], **kwargs: Any) -> None:
        """Apply amplitude embedding, padding or truncating to 2^n amplitudes."""
        n_w = len(wires)
        required_dim = 2 ** n_w

        # Ensure 1D and proper size
        feats = inputs.flatten()
        if feats.shape[0] < required_dim:
            pad = torch.zeros(required_dim - feats.shape[0], dtype=feats.dtype, device=feats.device)
            feats = torch.cat([feats, pad])
        elif feats.shape[0] > required_dim:
            feats = feats[:required_dim]

        # Avoid all-zero division
        norm = torch.norm(feats, p=2)
        if norm < 1e-8:
            feats = torch.zeros_like(feats)
            feats[0] = 1.0
        else:
            feats = feats / norm

        qml.AmplitudeEmbedding(feats, wires=wires, pad_with=0.0, normalize=True)

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["state_dim"] = self.state_dim
        return info
