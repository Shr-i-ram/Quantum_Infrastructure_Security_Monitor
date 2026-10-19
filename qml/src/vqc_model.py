r"""Hybrid Variational Quantum Classifier (VQC) using PennyLane and PyTorch.

Mathematical Formulation:
1. State Preparation:
   |psi(x)> = U_enc(x)|0>^n = \bigotimes_{i=0}^{n-1} R_Y(x_i) |0>^n
2. Variational Unitary:
   |psi(x, theta)> = U_var(theta) |psi(x)>
   where U_var(theta) comprises L layers of parameterized rotations (R_X, R_Y, R_Z)
   and periodic CNOT entanglement.
3. Observables Measurement:
   z_i(x, theta) = <psi(x, theta)| Z_i |psi(x, theta)> \in [-1, 1]  for i = 0..n-1
4. Classical Classification Head:
   logit = w^T z + b,  p(y=1|x) = \sigma(logit)
5. Training Objective:
   L_BCE = -1/N \sum_j [y_j \log(p_j) + (1-y_j) \log(1-p_j)]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


class VariationalQuantumClassifier(nn.Module):
    """Hybrid Variational Quantum Classifier with PennyLane quantum circuit and PyTorch classical head."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed
        self.threshold = 0.50

        # PennyLane device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Build QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = _circuit

        # Parameterized quantum weights: (n_layers, n_qubits, 3)
        torch.manual_seed(seed)
        weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.q_weights = nn.Parameter(0.1 * torch.randn(weights_shape, dtype=torch.float32))

        # Classical classification head: Linear (n_qubits -> 1)
        self.head = nn.Linear(n_qubits, 1)

    def forward_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate quantum circuit on input batch x. Returns expectation values tensor (B, n_qubits)."""
        expvals = self.qnode(x, self.q_weights)
        return torch.stack(expvals, dim=1).to(dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing predicted probabilities p(y=1|x)."""
        z = self.forward_quantum(x)
        logits = self.head(z).squeeze(-1)
        return torch.sigmoid(logits)

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Inference returning class probabilities p(y=1|x) as NumPy array."""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.tensor(X, dtype=torch.float32)
            else:
                X_t = X
            probs = self.forward(X_t).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray | torch.Tensor, threshold: float | None = None) -> np.ndarray:
        """Inference returning binary class predictions (0 or 1)."""
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(int)

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
            "n_layers": self.n_layers,
            "seed": self.seed,
            "threshold": self.threshold,
            "meta": meta or {},
        }
        torch.save(checkpoint, p)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> VariationalQuantumClassifier:
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
