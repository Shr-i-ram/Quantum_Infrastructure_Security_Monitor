"""Classical and prior ML baselines evaluated under identical data conditions."""

from __future__ import annotations

import time
from typing import Any
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
import torch.optim as optim

from qml.pitqsvdd.evaluation.metrics import compute_point_metrics


class ClassicalDeepSVDD(nn.Module):
    """Classical Deep SVDD baseline (Ruff et al., 2018) mapping telemetry to a hypersphere."""

    def __init__(self, in_features: int = 48, hidden_dim: int = 24, rep_dim: int = 4, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.rep_dim = rep_dim
        # Network without bias to prevent representation collapse
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, rep_dim, bias=False),
        )
        self.register_buffer("centre", torch.zeros(rep_dim))
        self.register_buffer("centre_initialized", torch.tensor(False))

    def init_centre(self, X_train: torch.Tensor) -> None:
        with torch.no_grad():
            reps = self.net(X_train)
            c = torch.mean(reps, dim=0)
            eps = 0.1
            c = torch.where(torch.abs(c) < 1e-4, torch.full_like(c, eps), c)
            self.centre.copy_(c)
            self.centre_initialized.copy_(torch.tensor(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reps = self.forward(x)
            return torch.sum((reps - self.centre) ** 2, dim=-1)


class ClassicalTemporalAutoencoder(nn.Module):
    """Classical GRU Temporal Autoencoder baseline for sequence reconstruction."""

    def __init__(self, in_features: int = 48, hidden_dim: int = 16, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.GRU(in_features, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, in_features, batch_first=True)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, L, D)
        _, h = self.encoder(x_seq)
        L = x_seq.shape[1]
        # Repeat context vector across sequence
        rep = h.permute(1, 0, 2).repeat(1, L, 1)
        recon, _ = self.decoder(rep)
        return recon

    def score(self, x_seq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.forward(x_seq)
            mse = torch.mean((x_seq - recon) ** 2, dim=[1, 2])
            return mse


def train_and_eval_classical_deep_svdd(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30,
    lr: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate Classical Deep SVDD baseline."""
    in_dim = X_train.shape[1]
    model = ClassicalDeepSVDD(in_features=in_dim, seed=seed)

    t_train = torch.tensor(X_train, dtype=torch.float32)
    model.init_centre(t_train)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        reps = model(t_train)
        loss = torch.mean(torch.sum((reps - model.centre) ** 2, dim=-1))
        loss.backward()
        optimizer.step()
    fit_time = time.time() - t0

    # Score validation normal to calibrate threshold
    model.eval()
    t_val = torch.tensor(X_val, dtype=torch.float32)
    val_scores = model.score(t_val).numpy()
    val_normal_scores = val_scores[y_val == 0] if (y_val == 0).any() else val_scores
    threshold = float(np.percentile(val_normal_scores, 95.0))

    # Score test
    t_test = torch.tensor(X_test, dtype=torch.float32)
    test_scores = model.score(t_test).numpy()
    test_preds = (test_scores >= threshold).astype(int)

    metrics = compute_point_metrics(y_test, test_preds, test_scores)
    return {
        "model_name": "Classical Deep SVDD",
        "fit_time_s": fit_time,
        "calibrated_threshold": threshold,
        "test_metrics": metrics,
    }


def train_and_eval_isolation_forest(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate classical Isolation Forest."""
    iso = IsolationForest(contamination=0.05, random_state=seed)
    t0 = time.time()
    iso.fit(X_train)
    fit_time = time.time() - t0

    # In IsolationForest, lower score = more anomalous. Invert for standard score
    val_scores = -iso.score_samples(X_val)
    val_normal = val_scores[y_val == 0] if (y_val == 0).any() else val_scores
    threshold = float(np.percentile(val_normal, 95.0))

    test_scores = -iso.score_samples(X_test)
    test_preds = (test_scores >= threshold).astype(int)
    metrics = compute_point_metrics(y_test, test_preds, test_scores)

    return {
        "model_name": "Classical Isolation Forest",
        "fit_time_s": fit_time,
        "calibrated_threshold": threshold,
        "test_metrics": metrics,
    }


def train_and_eval_one_class_svm(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Train and evaluate classical One-Class SVM."""
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    t0 = time.time()
    ocsvm.fit(X_train)
    fit_time = time.time() - t0

    val_scores = -ocsvm.score_samples(X_val)
    val_normal = val_scores[y_val == 0] if (y_val == 0).any() else val_scores
    threshold = float(np.percentile(val_normal, 95.0))

    test_scores = -ocsvm.score_samples(X_test)
    test_preds = (test_scores >= threshold).astype(int)
    metrics = compute_point_metrics(y_test, test_preds, test_scores)

    return {
        "model_name": "Classical One-Class SVM",
        "fit_time_s": fit_time,
        "calibrated_threshold": threshold,
        "test_metrics": metrics,
    }
