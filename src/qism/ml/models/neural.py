"""PyTorch feedforward autoencoder anomaly detector (Requirements 9, 10, 57–59, 61–63, 125–128)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from qism.ml.models.base import AnomalyDetector


class _AutoencoderNetwork(nn.Module):
    """Compact symmetric MLP autoencoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderDetector(AnomalyDetector):
    """Neural autoencoder anomaly detector trained on normal telemetry."""

    def __init__(
        self,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 16,
        early_stopping_patience: int = 5,
        random_state: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = early_stopping_patience
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: _AutoencoderNetwork | None = None
        self.training_history: list[dict[str, float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> AutoencoderDetector:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        n_samples, input_dim = X_arr.shape

        effective_hidden = min(self.hidden_dim, max(4, input_dim * 2))
        effective_latent = min(self.latent_dim, max(2, effective_hidden // 2))

        self.net = _AutoencoderNetwork(
            input_dim=input_dim,
            hidden_dim=effective_hidden,
            latent_dim=effective_latent,
        ).to(self.device)

        dataset = TensorDataset(torch.from_numpy(X_arr))
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, n_samples),
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.net.train()
        best_loss = float("inf")
        patience_counter = 0

        self.training_history = []
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.net(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_x)

            epoch_loss /= n_samples
            self.training_history.append({"epoch": epoch, "loss": epoch_loss})

            if epoch_loss < best_loss - 1e-5:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.net is None:
            raise RuntimeError("Model must be fit before score.")
        X_arr = np.asarray(X, dtype=np.float32)
        self.net.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_arr).to(self.device)
            recon = self.net(tensor_x)
            # Row-wise MSE: mean squared error per feature
            mse = torch.mean((tensor_x - recon) ** 2, dim=1)
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
            "training_history": self.training_history,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> AutoencoderDetector:
        path = Path(path)
        payload = torch.load(path, map_location="cpu")
        det = cls(**payload["hyperparameters"])
        det.threshold = payload["threshold"]
        det.training_history = payload.get("training_history", [])

        # Recover network architecture from state dict
        weight_keys = [k for k in payload["state_dict"].keys() if "encoder.0.weight" in k]
        if weight_keys:
            w0 = payload["state_dict"]["encoder.0.weight"]
            hidden_dim, input_dim = w0.shape
            w2 = payload["state_dict"]["encoder.2.weight"]
            latent_dim, _ = w2.shape
            det.net = _AutoencoderNetwork(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
            det.net.load_state_dict(payload["state_dict"])
            det.net.to(det.device)
            det.is_fitted = True
        return det
