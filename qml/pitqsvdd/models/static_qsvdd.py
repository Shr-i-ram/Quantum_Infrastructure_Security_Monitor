"""Static Quantum Support Vector Data Description (QSVDD) Model."""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn

from qml.pitqsvdd.circuits.variational import VariationalQuantumCircuit
from qml.pitqsvdd.config import ModelConfig


class StaticQSVDD(nn.Module):
    """Static Quantum SVDD baseline mapping normal telemetry to a compact hypersphere."""

    def __init__(
        self,
        circuit: VariationalQuantumCircuit,
        config: ModelConfig,
    ):
        super().__init__()
        self.circuit = circuit
        self.config = config
        self.n_qubits = circuit.n_qubits
        self.lambda_reg = config.lambda_reg

        # Hypersphere reference centre c registered as a persistent buffer (fixed, not trained)
        self.register_buffer("centre", torch.zeros(self.n_qubits, dtype=torch.float32))
        self.register_buffer("centre_initialized", torch.tensor(False, dtype=torch.bool))

    def init_centre(self, x_normal_batch: torch.Tensor) -> None:
        """Initialize reference centre c as the empirical mean of normal latent states."""
        with torch.no_grad():
            if x_normal_batch.ndim == 3:
                x_eval = x_normal_batch[:, -1, :]
            else:
                x_eval = x_normal_batch
            z = self.circuit(x_eval)
            mean_z = torch.mean(z, dim=0)
            # Ensure centre does not collapse to zero vector
            eps = 0.1
            mean_z = torch.where(torch.abs(mean_z) < 1e-4, torch.full_like(mean_z, eps), mean_z)
            self.centre.copy_(mean_z)
            self.centre_initialized.copy_(torch.tensor(True))

    def forward_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantum latent state z = Q_theta(x). Shape: (B, n_qubits)."""
        if x.ndim == 3:
            # Use current step in window
            x_eval = x[:, -1, :]
        else:
            x_eval = x
        return self.circuit(x_eval)

    def compute_loss(self, x_batch: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute one-class SVDD loss on normal data.

        Objective:
            L_SVDD = 1/N sum_i ||z_i - c||_2^2 + lambda_reg ||theta||_2^2
        """
        z = self.forward_latent(x_batch)
        dist_sq = torch.sum((z - self.centre) ** 2, dim=-1)
        loss_svdd = torch.mean(dist_sq)

        # L2 parameter regularization
        l2_reg = self.circuit.q_weights.pow(2).sum()
        total_loss = loss_svdd + self.lambda_reg * l2_reg

        return {
            "loss": total_loss,
            "loss_svdd": loss_svdd,
            "loss_temporal": torch.tensor(0.0, device=x_batch.device),
            "loss_reg": self.lambda_reg * l2_reg,
        }

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score s_normal(x) = ||z - c||_2^2."""
        with torch.no_grad():
            z = self.forward_latent(x)
            dist_sq = torch.sum((z - self.centre) ** 2, dim=-1)
            return dist_sq
