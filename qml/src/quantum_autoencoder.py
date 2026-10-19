r"""Lightweight Quantum Autoencoder (QAE) for Unsupervised Anomaly Detection.

Mathematical Formulation:
1. State Preparation:
   |psi(x)> = U_enc(x)|0>^4  on 4 qubits.
2. Compression Unitary:
   |psi(x, theta)> = U_var(theta) |psi(x)>
   Partition wires into Latent (wires 0, 1) and Trash (wires 2, 3).
3. Compression Target:
   Train U_var(theta) such that for normal telemetry x, the trash subsystem
   is mapped to the reference state |00>:
   F(theta, x) = <psi(x, theta)| (I_4 \otimes |00><00|) |psi(x, theta)>
               = P(trash wires = |00>)
4. Training Loss on Normal Data:
   L_QAE(theta) = 1.0 - 1/N \sum_{i=1}^N P_i(trash = |00>)
5. Unsupervised Anomaly Score:
   s(x) = 1.0 - P(trash = |00>) \in [0, 1]
   - Normal telemetry: s(x) \approx 0 (high compression fidelity).
   - Attack anomalies: s(x) -> 1 (low compression fidelity; trash excitation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


class QuantumAutoencoder(nn.Module):
    """Lightweight Quantum Autoencoder compressing 4 qubits into 2 latent qubits."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.latent_wires = [0, 1]
        self.trash_wires = [2, 3]
        self.n_layers = n_layers
        self.seed = seed
        self.threshold = 0.50

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=self.trash_wires)

        self.qnode = _circuit

        # Trainable unitary weights
        torch.manual_seed(seed)
        weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.q_weights = nn.Parameter(0.1 * torch.randn(weights_shape, dtype=torch.float32))

    def forward_trash_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Compute probability distribution on trash wires. Shape: (B, 4)."""
        return self.qnode(x, self.q_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score s(x) = 1.0 - P(trash = |00>). Returns shape: (B,)."""
        probs = self.forward_trash_probs(x)
        # Index 0 corresponds to |00> state on trash wires
        if probs.ndim == 1:
            fidelity = probs[0]
        else:
            fidelity = probs[:, 0]
        return 1.0 - fidelity

    def score_samples(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Inference returning anomaly scores in [0, 1]."""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.tensor(X, dtype=torch.float32)
            else:
                X_t = X
            scores = self.forward(X_t).cpu().numpy()
        return scores

    def predict(self, X: np.ndarray | torch.Tensor, threshold: float | None = None) -> np.ndarray:
        """Inference returning binary anomaly predictions (1 for anomaly, 0 for normal)."""
        th = threshold if threshold is not None else self.threshold
        scores = self.score_samples(X)
        return (scores >= th).astype(int)

    def set_threshold(self, threshold: float) -> None:
        """Set decision threshold calibrated on validation set."""
        self.threshold = float(threshold)

    def save_checkpoint(self, path: str | Path, meta: dict[str, Any] | None = None) -> None:
        """Persist state dict, circuit architecture, and calibration metadata."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "n_qubits": self.n_qubits,
            "latent_wires": self.latent_wires,
            "trash_wires": self.trash_wires,
            "n_layers": self.n_layers,
            "seed": self.seed,
            "threshold": self.threshold,
            "meta": meta or {},
        }
        torch.save(checkpoint, p)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> QuantumAutoencoder:
        """Load model from saved checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            n_qubits=checkpoint["n_qubits"],
            n_layers=checkpoint["n_layers"],
            seed=checkpoint.get("seed", 42),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.set_threshold(checkpoint.get("threshold", 0.50))
        return model
