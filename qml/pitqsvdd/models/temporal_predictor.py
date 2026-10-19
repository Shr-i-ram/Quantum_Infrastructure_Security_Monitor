"""Lightweight Causal Temporal Predictor for PI-TQSVDD."""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentTemporalPredictor(nn.Module):
    """Causal neural network predicting next quantum latent state z_{t+1} from history."""

    def __init__(self, latent_dim: int = 4, hidden_dim: int = 16, num_layers: int = 1, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Causal GRU network
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),  # Latent expectations are in [-1, 1]
        )

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Predict next latent representation hat{z}_{t+1}.

        Args:
            z_seq: Tensor of shape (B, L, latent_dim).

        Returns:
            hat{z}: Tensor of shape (B, latent_dim).
        """
        out, _ = self.gru(z_seq)
        last_hidden = out[:, -1, :]  # (B, hidden_dim)
        return self.head(last_hidden)
