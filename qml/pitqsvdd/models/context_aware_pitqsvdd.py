"""Context-Aware & Multi-Centre Variant of PI-TQSVDD."""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn

from qml.pitqsvdd.circuits.variational import VariationalQuantumCircuit
from qml.pitqsvdd.config import ModelConfig
from qml.pitqsvdd.models.temporal_predictor import LatentTemporalPredictor


class ContextAwarePITQSVDD(nn.Module):
    """Context-aware multi-centre quantum SVDD accommodating legitimate benign operating regimes."""

    def __init__(
        self,
        circuit: VariationalQuantumCircuit,
        config: ModelConfig,
    ):
        super().__init__()
        self.circuit = circuit
        self.config = config
        self.n_qubits = circuit.n_qubits
        self.num_regimes = config.num_regimes
        self.lambda_temporal = config.lambda_temporal
        self.lambda_reg = config.lambda_reg

        # K centres registered as persistent buffer
        self.register_buffer("centres", torch.zeros(self.num_regimes, self.n_qubits, dtype=torch.float32))
        self.register_buffer("centres_initialized", torch.tensor(False, dtype=torch.bool))

        # Temporal predictor
        self.temporal_predictor = LatentTemporalPredictor(
            latent_dim=self.n_qubits,
            hidden_dim=config.temporal_hidden_dim,
            seed=42,
        )

    def init_centres(self, x_normal_batch: torch.Tensor) -> None:
        """Initialize K centres across normal data regimes using deterministic offsets or k-means."""
        with torch.no_grad():
            if x_normal_batch.ndim == 3:
                x_eval = x_normal_batch[:, -1, :]
            else:
                x_eval = x_normal_batch
            z = self.circuit(x_eval)
            mean_z = torch.mean(z, dim=0)

            # Assign spread centres around mean
            for k in range(self.num_regimes):
                offset = (k - (self.num_regimes - 1) / 2.0) * 0.2
                c_k = mean_z + offset
                eps = 0.05
                c_k = torch.where(torch.abs(c_k) < 1e-4, torch.full_like(c_k, eps), c_k)
                self.centres[k].copy_(c_k)

            self.centres_initialized.copy_(torch.tensor(True))

    def compute_loss(
        self,
        x_windows: torch.Tensor,
        targets: torch.Tensor | None = None,
        has_target: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-centre SVDD loss combined with temporal loss."""
        z_seq = self.circuit(x_windows)
        z_current = z_seq[:, -1, :]  # (B, n_qubits)

        # Multi-centre distance: min_k ||z - c_k||^2
        # z_current: (B, 1, n_qubits), centres: (1, K, n_qubits)
        diff = z_current.unsqueeze(1) - self.centres.unsqueeze(0)  # (B, K, n_qubits)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (B, K)
        min_dist_sq, _ = torch.min(dist_sq, dim=-1)  # (B,)
        loss_svdd = torch.mean(min_dist_sq)

        # Temporal loss
        loss_temporal = torch.tensor(0.0, device=x_windows.device)
        if targets is not None and has_target is not None and has_target.any():
            valid_mask = has_target.bool()
            valid_z_seq = z_seq[valid_mask]
            valid_targets = targets[valid_mask]

            hat_z_next = self.temporal_predictor(valid_z_seq)
            z_next_true = self.circuit(valid_targets)
            temporal_sq = torch.sum((z_next_true - hat_z_next) ** 2, dim=-1)
            loss_temporal = torch.mean(temporal_sq)

        # L2 regularizer
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
        """Compute anomaly score relative to closest operating regime."""
        with torch.no_grad():
            z_seq = self.circuit(x_windows)
            z_current = z_seq[:, -1, :]

            diff = z_current.unsqueeze(1) - self.centres.unsqueeze(0)
            dist_sq = torch.sum(diff ** 2, dim=-1)
            s_normal, _ = torch.min(dist_sq, dim=-1)

            if z_seq.shape[1] > 1:
                past_seq = z_seq[:, :-1, :]
                hat_z_current = self.temporal_predictor(past_seq)
                s_temporal = torch.sum((z_current - hat_z_current) ** 2, dim=-1)
            else:
                s_temporal = torch.zeros_like(s_normal)

            s_combined = alpha * s_normal + beta * s_temporal
            return s_combined, s_normal, s_temporal
