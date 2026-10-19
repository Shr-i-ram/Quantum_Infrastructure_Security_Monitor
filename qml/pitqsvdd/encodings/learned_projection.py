"""Learned Classical Projection + Quantum Encoding for PI-TQSVDD."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from qml.pitqsvdd.encodings.base import BaseEncoding


class LearnedProjectionModule(nn.Module):
    """Trainable classical bottleneck module mapping inputs to angle rotations."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map from [-1, 1] to [0, pi]
        out = self.net(x)
        return (out + 1.0) * (np.pi / 2.0)


class LearnedProjectionEncoding(BaseEncoding):
    """Combines a trainable classical neural bottleneck with variational quantum encoding."""

    def __init__(self, n_qubits: int, in_dim: int = 48, hidden_dim: int = 24, seed: int = 42):
        super().__init__(n_qubits=n_qubits, name="learned_projection_encoding")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.proj = LearnedProjectionModule(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=n_qubits, seed=seed)

    def project(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply classical projection to get quantum rotation angles."""
        return self.proj(inputs)

    def apply(self, inputs: torch.Tensor, wires: Sequence[int], **kwargs: Any) -> None:
        """Apply quantum rotations using the projected features."""
        # If inputs are not yet projected, project them
        if inputs.shape[0] != len(wires):
            angles = self.project(inputs)
        else:
            angles = inputs

        for idx, wire in enumerate(wires):
            val = angles[idx] if idx < angles.shape[0] else torch.tensor(0.0)
            qml.RY(val, wires=wire)

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["in_dim"] = self.in_dim
        info["hidden_dim"] = self.hidden_dim
        return info
