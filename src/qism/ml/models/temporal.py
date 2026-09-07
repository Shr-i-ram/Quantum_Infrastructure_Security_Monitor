"""PyTorch LSTM temporal sequence autoencoder anomaly detector (Requirements 11, 226, 264)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from qism.ml.models.base import AnomalyDetector


class _LSTMSeq2SeqAutoencoder(nn.Module):
    """LSTM encoder-decoder for multivariate sequence reconstruction."""

    def __init__(self, input_dim: int, hidden_dim: int = 24, latent_dim: int = 12, seq_len: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)

        self.latent_to_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.encoder(x)
        # h_n: (1, batch, hidden_dim)
        latent = self.enc_to_latent(h_n.squeeze(0))  # (batch, latent_dim)

        dec_init = self.latent_to_dec(latent).unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)
        dec_out, _ = self.decoder(dec_init)
        reconstruction = self.output_layer(dec_out)  # (batch, seq_len, input_dim)
        return reconstruction


class TemporalAutoencoderDetector(AnomalyDetector):
    """Temporal sequence autoencoder for multivariate sequential telemetry."""

    def __init__(
        self,
        seq_len: int = 3,
        hidden_dim: int = 24,
        latent_dim: int = 12,
        lr: float = 1e-3,
        epochs: int = 25,
        batch_size: int = 16,
        random_state: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            **kwargs,
        )
        self.seq_len = max(2, seq_len)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: _LSTMSeq2SeqAutoencoder | None = None

    def _create_sequences(self, X_2d: np.ndarray) -> np.ndarray:
        """Convert 2D array of (n_samples, n_features) into 3D causal sliding sequences (n_samples, seq_len, n_features)."""
        n_samples, n_features = X_2d.shape
        L = self.seq_len
        sequences = np.zeros((n_samples, L, n_features), dtype=np.float32)
        for i in range(n_samples):
            start_idx = max(0, i - L + 1)
            sub = X_2d[start_idx : i + 1]
            if len(sub) < L:
                # Causal padding with first observation of window
                pad_len = L - len(sub)
                pad = np.repeat(sub[:1], pad_len, axis=0)
                sequences[i] = np.vstack([pad, sub])
            else:
                sequences[i] = sub
        return sequences

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> TemporalAutoencoderDetector:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 2:
            X_seq = self._create_sequences(X_arr)
        else:
            X_seq = X_arr

        n_samples, seq_len, input_dim = X_seq.shape
        self.net = _LSTMSeq2SeqAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            seq_len=seq_len,
        ).to(self.device)

        dataset = TensorDataset(torch.from_numpy(X_seq))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, n_samples), shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.net.train()
        for _ in range(self.epochs):
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.net(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.net is None:
            raise RuntimeError("Model must be fit before score.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 2:
            X_seq = self._create_sequences(X_arr)
        else:
            X_seq = X_arr

        self.net.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_seq).to(self.device)
            recon = self.net(tensor_x)
            # Anomaly score: mean squared error across time and feature dimensions
            mse = torch.mean((tensor_x - recon) ** 2, dim=(1, 2))
            scores = mse.cpu().numpy().astype(np.float64)
        return scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.net is None:
            raise RuntimeError("Cannot save unfitted model.")
        payload = {
            "hyperparameters": self.hyperparameters,
            "threshold": self.threshold,
            "state_dict": self.net.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> TemporalAutoencoderDetector:
        path = Path(path)
        payload = torch.load(path, map_location="cpu")
        det = cls(**payload["hyperparameters"])
        det.threshold = payload["threshold"]

        w_enc = payload["state_dict"]["encoder.weight_ih_l0"]
        hidden_4, input_dim = w_enc.shape
        hidden_dim = hidden_4 // 4
        latent_dim = payload["state_dict"]["enc_to_latent.weight"].shape[0]
        seq_len = payload["hyperparameters"].get("seq_len", 3)

        det.net = _LSTMSeq2SeqAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
        )
        det.net.load_state_dict(payload["state_dict"])
        det.net.to(det.device)
        det.is_fitted = True
        return det
