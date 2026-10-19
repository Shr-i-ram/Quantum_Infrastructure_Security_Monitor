"""Temporal anomaly detection models and baselines (V0.7).

Provides:
1. RollingZScoreDetector: Lightweight, causal, interpretable rolling statistics baseline.
2. EWMAAnomalyDetector: Causal exponentially weighted moving average deviation detector.
3. SequenceAutoencoderDetector: GRU/LSTM sequence-to-sequence autoencoder with per-feature
   reconstruction error decomposition E_j = (1/L) sum_t (x_{t,j} - x_hat_{t,j})^2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from qism.ml.models.base import AnomalyDetector

BaseTemporalDetector = AnomalyDetector



# =============================================================================
# 1. Temporal Statistical Baselines
# =============================================================================

class RollingZScoreDetector(AnomalyDetector):
    """Causal rolling Z-score multivariate anomaly detector.

    Maintains a causal rolling buffer of past observations. Anomaly score is
    the normalized Euclidean deviation of the current observation from the rolling mean:
        S_t = sqrt( sum_j ((x_{t,j} - mu_{t,j}) / (sigma_{t,j} + eps))^2 )
    """

    def __init__(self, window_length: int = 5, eps: float = 1e-4, **kwargs: Any):
        super().__init__(window_length=window_length, eps=eps, **kwargs)
        self.window_length = max(2, window_length)
        self.eps = eps
        self.baseline_mean_: np.ndarray | None = None
        self.baseline_std_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> RollingZScoreDetector:
        """Estimate global normal baseline statistics from training telemetry."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 3:
            # Flatten window dimension if 3D tensor is passed
            X_arr = X_arr[:, -1, :]
        self.baseline_mean_ = np.mean(X_arr, axis=0)
        self.baseline_std_ = np.std(X_arr, axis=0)
        self.baseline_std_[self.baseline_std_ < self.eps] = 1.0
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute causal rolling Z-score anomaly score."""
        if not self.is_fitted or self.baseline_mean_ is None or self.baseline_std_ is None:
            raise RuntimeError("Model must be fit before score.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 3:
            # X_arr is (n_samples, L, d)
            n_samples, L, d = X_arr.shape
            scores = np.zeros(n_samples, dtype=np.float64)
            for i in range(n_samples):
                seq = X_arr[i]  # (L, d)
                curr = seq[-1]  # current observation
                past = seq[:-1] if L > 1 else seq
                mu = np.mean(past, axis=0)
                std = np.std(past, axis=0)
                std[std < self.eps] = self.baseline_std_[std < self.eps]
                z = (curr - mu) / (std + self.eps)
                scores[i] = float(np.sqrt(np.mean(z ** 2)))
            return scores
        else:
            # 2D array: compute causal rolling statistics
            n_samples, d = X_arr.shape
            scores = np.zeros(n_samples, dtype=np.float64)
            L = self.window_length
            for i in range(n_samples):
                start = max(0, i - L)
                past = X_arr[start : i]
                curr = X_arr[i]
                if len(past) > 1:
                    mu = np.mean(past, axis=0)
                    std = np.std(past, axis=0)
                    std[std < self.eps] = self.baseline_std_[std < self.eps]
                else:
                    mu = self.baseline_mean_
                    std = self.baseline_std_
                z = (curr - mu) / (std + self.eps)
                scores[i] = float(np.sqrt(np.mean(z ** 2)))
            return scores


    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "hyperparameters": self.hyperparameters,
            "threshold": self.threshold,
            "baseline_mean": self.baseline_mean_.tolist() if self.baseline_mean_ is not None else None,
            "baseline_std": self.baseline_std_.tolist() if self.baseline_std_ is not None else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> RollingZScoreDetector:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        det = cls(**data["hyperparameters"])
        det.threshold = data["threshold"]
        det.baseline_mean_ = np.array(data["baseline_mean"], dtype=np.float64) if data["baseline_mean"] else None
        det.baseline_std_ = np.array(data["baseline_std"], dtype=np.float64) if data["baseline_std"] else None
        det.is_fitted = True
        return det


class EWMAAnomalyDetector(AnomalyDetector):
    """Causal Exponentially Weighted Moving Average (EWMA) anomaly detector."""

    def __init__(self, alpha: float = 0.3, **kwargs: Any):
        super().__init__(alpha=alpha, **kwargs)
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.baseline_mean_: np.ndarray | None = None
        self.baseline_cov_inv_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> EWMAAnomalyDetector:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 3:
            X_arr = X_arr[:, -1, :]
        self.baseline_mean_ = np.mean(X_arr, axis=0)
        cov = np.cov(X_arr, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        elif cov.ndim == 1:
            cov = np.diag(cov)
        cov_reg = cov + 1e-4 * np.eye(cov.shape[0])
        self.baseline_cov_inv_ = np.linalg.pinv(cov_reg)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.baseline_mean_ is None or self.baseline_cov_inv_ is None:
            raise RuntimeError("Model must be fit before score.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 3:
            # 3D: evaluate EWMA across sequence steps for each window
            n_samples, L, d = X_arr.shape
            scores = np.zeros(n_samples, dtype=np.float64)
            for i in range(n_samples):
                seq = X_arr[i]
                ewma = seq[0].copy()
                for t in range(1, L):
                    ewma = self.alpha * seq[t] + (1.0 - self.alpha) * ewma
                diff = seq[-1] - ewma
                scores[i] = float(np.sqrt(np.maximum(0.0, diff @ self.baseline_cov_inv_ @ diff)))
            return scores
        else:
            n_samples, d = X_arr.shape
            scores = np.zeros(n_samples, dtype=np.float64)
            ewma = self.baseline_mean_.copy()
            for i in range(n_samples):
                curr = X_arr[i]
                ewma = self.alpha * curr + (1.0 - self.alpha) * ewma
                diff = curr - ewma
                scores[i] = float(np.sqrt(np.maximum(0.0, diff @ self.baseline_cov_inv_ @ diff)))
            return scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "hyperparameters": self.hyperparameters,
            "threshold": self.threshold,
            "baseline_mean": self.baseline_mean_.tolist() if self.baseline_mean_ is not None else None,
            "baseline_cov_inv": self.baseline_cov_inv_.tolist() if self.baseline_cov_inv_ is not None else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> EWMAAnomalyDetector:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        det = cls(**data["hyperparameters"])
        det.threshold = data["threshold"]
        det.baseline_mean_ = np.array(data["baseline_mean"], dtype=np.float64) if data["baseline_mean"] else None
        det.baseline_cov_inv_ = np.array(data["baseline_cov_inv"], dtype=np.float64) if data["baseline_cov_inv"] else None
        det.is_fitted = True
        return det


# =============================================================================
# 2. Sequence Reconstruction Autoencoder (GRU / LSTM)
# =============================================================================

class _RecurrentSeq2SeqNetwork(nn.Module):
    """Sequence autoencoder network using GRU or LSTM cells."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        seq_len: int = 5,
        cell_type: Literal["gru", "lstm"] = "gru",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.cell_type = cell_type

        # Encoder
        if cell_type == "lstm":
            self.encoder_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.decoder_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.decoder_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.enc_dense = nn.Linear(hidden_dim, latent_dim)
        self.dec_dense_in = nn.Linear(latent_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        if self.cell_type == "lstm":
            _, (h_n, _) = self.encoder_rnn(x)
            hidden = h_n[-1]  # (batch_size, hidden_dim)
        else:
            _, h_n = self.encoder_rnn(x)
            hidden = h_n[-1]

        latent = torch.relu(self.enc_dense(hidden))  # (batch_size, latent_dim)
        dec_init = torch.relu(self.dec_dense_in(latent)).unsqueeze(1).repeat(1, self.seq_len, 1)

        dec_out, _ = self.decoder_rnn(dec_init)
        reconstruction = self.output_layer(dec_out)  # (batch_size, seq_len, input_dim)
        return reconstruction


class SequenceAutoencoderDetector(AnomalyDetector):
    """Causal Recurrent Sequence Autoencoder Anomaly Detector (GRU / LSTM).

    Reconstructs multivariate telemetry sequences W = [x_{t-L+1}, ..., x_t].
    Anomaly score is the mean squared reconstruction error:
        S(W) = (1 / (L * d)) * sum_t sum_j (x_{t,j} - x_hat_{t,j})^2

    Supports feature-level decomposition:
        E_j = (1 / L) * sum_t (x_{t,j} - x_hat_{t,j})^2
    """

    def __init__(
        self,
        seq_len: int = 5,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        cell_type: Literal["gru", "lstm"] = "gru",
        lr: float = 1e-3,
        epochs: int = 25,
        batch_size: int = 16,
        patience: int = 5,
        random_state: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            cell_type=cell_type,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            random_state=random_state,
            **kwargs,
        )
        self.seq_len = max(2, seq_len)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cell_type = cell_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: _RecurrentSeq2SeqNetwork | None = None
        self.training_loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SequenceAutoencoderDetector:
        """Fit sequence autoencoder on normal training sequences."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 2:
            # Fallback causal sequence packaging if 2D array passed
            from qism.temporal.windowing import TemporalWindowBuilder, TemporalWindowConfig
            builder = TemporalWindowBuilder(TemporalWindowConfig(window_length=self.seq_len))
            X_arr, _, _ = builder.build_windows(pd.DataFrame(X_arr))

        n_samples, seq_len, input_dim = X_arr.shape
        self.seq_len = seq_len

        self.net = _RecurrentSeq2SeqNetwork(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            seq_len=seq_len,
            cell_type=self.cell_type,
        ).to(self.device)

        dataset = TensorDataset(torch.from_numpy(X_arr))
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, n_samples)),
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.net.train()
        best_loss = float("inf")
        patience_count = 0
        self.training_loss_history.clear()

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

            epoch_loss /= max(1, n_samples)
            self.training_loss_history.append(epoch_loss)

            if epoch_loss < best_loss - 1e-5:
                best_loss = epoch_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    break

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly score (mean squared reconstruction error)."""
        if not self.is_fitted or self.net is None:
            raise RuntimeError("Model must be fit before score.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 2:
            from qism.temporal.windowing import TemporalWindowBuilder, TemporalWindowConfig
            builder = TemporalWindowBuilder(TemporalWindowConfig(window_length=self.seq_len))
            X_arr, _, _ = builder.build_windows(pd.DataFrame(X_arr))

        self.net.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_arr).to(self.device)
            recon = self.net(tensor_x)
            # S(W) = mean over time (dim 1) and features (dim 2)
            mse = torch.mean((tensor_x - recon) ** 2, dim=(1, 2))
            return mse.cpu().numpy().astype(np.float64)

    def score_decomposition(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute total score AND per-feature reconstruction error contributions.

        Returns:
            total_scores: 1D array of shape (n_windows,).
            feature_contributions: 2D array of shape (n_windows, n_features).
                E_j = (1/L) * sum_t (x_{t,j} - x_hat_{t,j})^2
        """
        if not self.is_fitted or self.net is None:
            raise RuntimeError("Model must be fit before score.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 2:
            from qism.temporal.windowing import TemporalWindowBuilder, TemporalWindowConfig
            builder = TemporalWindowBuilder(TemporalWindowConfig(window_length=self.seq_len))
            X_arr, _, _ = builder.build_windows(pd.DataFrame(X_arr))

        self.net.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_arr).to(self.device)
            recon = self.net(tensor_x)
            # Per-feature error: mean over time (dim 1)
            feat_mse = torch.mean((tensor_x - recon) ** 2, dim=1)  # (batch, n_features)
            total_mse = torch.mean(feat_mse, dim=1)               # (batch,)

            return (
                total_mse.cpu().numpy().astype(np.float64),
                feat_mse.cpu().numpy().astype(np.float64),
            )

    def feature_error_contributions(self, X: np.ndarray) -> np.ndarray:
        """Compute per-feature reconstruction error contributions E_j."""
        _, feat_mse = self.score_decomposition(X)
        return feat_mse

    def save(self, path: str | Path) -> None:

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.net is None:
            raise RuntimeError("Cannot save unfitted model.")
        payload = {
            "hyperparameters": self.hyperparameters,
            "threshold": self.threshold,
            "state_dict": self.net.state_dict(),
            "training_history": self.training_loss_history,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> SequenceAutoencoderDetector:
        payload = torch.load(path, map_location="cpu")
        det = cls(**payload["hyperparameters"])
        det.threshold = payload["threshold"]
        det.training_loss_history = payload.get("training_history", [])

        state = payload["state_dict"]
        # Recover dimensions from state dict
        w_enc = state["encoder_rnn.weight_ih_l0"]
        hidden_mult, input_dim = w_enc.shape
        cell_type = payload["hyperparameters"].get("cell_type", "gru")
        hidden_dim = hidden_mult // (4 if cell_type == "lstm" else 3)
        latent_dim = state["enc_dense.weight"].shape[0]
        seq_len = payload["hyperparameters"].get("seq_len", 5)

        det.net = _RecurrentSeq2SeqNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            cell_type=cell_type,
        )
        det.net.load_state_dict(state)
        det.net.to(det.device)
        det.is_fitted = True
        return det
