"""Flagship Architecture: Physics-Informed Temporal Quantum SVDD (PI-TQSVDD).

Mathematical Formulation:
1. Quantum Representation:
   Given causal window X_t = [x_{t-L+1}, ..., x_t], evaluate quantum circuit:
   z_tau = Q_theta(v_tau)  for tau in [t-L+1, ..., t]
   yielding latent sequence Z_t = [z_{t-L+1}, ..., z_t] in R^{L x n_qubits}.
2. One-Class SVDD Normality Objective:
   L_SVDD = 1/N sum_i ||z_{i, t} - c||_2^2
   where c is a fixed reference centre estimated from normal training representations.
3. Causal Temporal Prediction Objective:
   hat{z}_{t+1} = g_psi(Z_t)
   L_temporal = 1/N_valid sum_{i in valid} ||z_{i, t+1} - hat{z}_{i, t+1}||_2^2
4. Total Training Objective:
   L = L_SVDD + lambda_temporal * L_temporal + lambda_reg * (||theta||_2^2 + ||psi||_2^2)
5. Combined Continuous Anomaly Score:
   s_t = alpha * s_normal(t) + beta * s_temporal(t)
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn

from qml.pitqsvdd.circuits.variational import VariationalQuantumCircuit
from qml.pitqsvdd.config import ModelConfig
from qml.pitqsvdd.models.temporal_predictor import LatentTemporalPredictor


class TemporalPITQSVDD(nn.Module):
    """Flagship PI-TQSVDD architecture combining quantum SVDD with temporal latent dynamics."""

    def __init__(
        self,
        circuit: VariationalQuantumCircuit,
        config: ModelConfig,
    ):
        super().__init__()
        self.circuit = circuit
        self.config = config
        self.n_qubits = circuit.n_qubits
        self.lambda_temporal = config.lambda_temporal
        self.lambda_reg = config.lambda_reg

        # Hypersphere reference centre c registered as persistent buffer
        self.register_buffer("centre", torch.zeros(self.n_qubits, dtype=torch.float32))
        self.register_buffer("centre_initialized", torch.tensor(False, dtype=torch.bool))

        # Causal temporal predictor
        self.temporal_predictor = LatentTemporalPredictor(
            latent_dim=self.n_qubits,
            hidden_dim=config.temporal_hidden_dim,
            seed=42,
        )

    def init_centre(self, x_normal_batch: torch.Tensor) -> None:
        """Initialize reference centre c from normal training forward pass."""
        with torch.no_grad():
            if x_normal_batch.ndim == 3:
                x_eval = x_normal_batch[:, -1, :]
            else:
                x_eval = x_normal_batch
            z = self.circuit(x_eval)
            mean_z = torch.mean(z, dim=0)
            eps = 0.1
            mean_z = torch.where(torch.abs(mean_z) < 1e-4, torch.full_like(mean_z, eps), mean_z)
            self.centre.copy_(mean_z)
            self.centre_initialized.copy_(torch.tensor(True))

    def forward_latent_sequence(self, x_windows: torch.Tensor) -> torch.Tensor:
        """Compute quantum latent sequence for windows of shape (B, L, D) -> (B, L, n_qubits)."""
        return self.circuit(x_windows)

    def compute_loss(
        self,
        x_windows: torch.Tensor,
        targets: torch.Tensor | None = None,
        has_target: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute combined PI-TQSVDD loss."""
        # 1. Quantum latent representations for the window
        z_seq = self.forward_latent_sequence(x_windows)  # (B, L, n_qubits)
        z_current = z_seq[:, -1, :]  # (B, n_qubits)

        # 2. SVDD normality loss on current step
        dist_sq = torch.sum((z_current - self.centre) ** 2, dim=-1)
        loss_svdd = torch.mean(dist_sq)

        # 3. Temporal prediction loss
        loss_temporal = torch.tensor(0.0, device=x_windows.device)
        if targets is not None and has_target is not None and has_target.any():
            valid_mask = has_target.bool()
            valid_z_seq = z_seq[valid_mask]
            valid_targets = targets[valid_mask]

            # Predict next latent state
            hat_z_next = self.temporal_predictor(valid_z_seq)  # (B_valid, n_qubits)
            # True next quantum state
            z_next_true = self.circuit(valid_targets)  # (B_valid, n_qubits)

            temporal_sq = torch.sum((z_next_true - hat_z_next) ** 2, dim=-1)
            loss_temporal = torch.mean(temporal_sq)

        # 4. L2 regularization on quantum and classical weights
        l2_q = self.circuit.q_weights.pow(2).sum()
        l2_c = sum(p.pow(2).sum() for p in self.temporal_predictor.parameters())
        loss_reg = self.lambda_reg * (l2_q + l2_c)

        total_loss = loss_svdd + self.lambda_temporal * loss_temporal + loss_reg

        return {
            "loss": total_loss,
            "loss_svdd": loss_svdd,
            "loss_temporal": loss_temporal,
            "loss_reg": loss_reg,
        }

    def score(
        self,
        x_windows: torch.Tensor,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute continuous anomaly scores.

        Returns:
            (combined_score, normal_score, temporal_score)
        """
        with torch.no_grad():
            z_seq = self.forward_latent_sequence(x_windows)
            z_current = z_seq[:, -1, :]

            # Normal distance to hypersphere centre
            s_normal = torch.sum((z_current - self.centre) ** 2, dim=-1)

            # Temporal prediction error: predict z_t from past steps [z_{t-L+1}, ..., z_{t-1}]
            if z_seq.shape[1] > 1:
                past_seq = z_seq[:, :-1, :]  # (B, L-1, n_qubits)
                hat_z_current = self.temporal_predictor(past_seq)
                s_temporal = torch.sum((z_current - hat_z_current) ** 2, dim=-1)
            else:
                s_temporal = torch.zeros_like(s_normal)

            s_combined = alpha * s_normal + beta * s_temporal
            return s_combined, s_normal, s_temporal
