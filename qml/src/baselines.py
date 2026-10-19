"""Direct Classical ML baselines evaluated under identical data and feature conditions (Review 2).

Provides fair, apples-to-apples classical reference comparisons for:
- Variational Quantum Classifier (VQC) vs. Classical Logistic Regression & Small Classical MLP
- Quantum Kernel SVM vs. Classical RBF Support Vector Classifier
- Quantum Autoencoder vs. Classical Dense Autoencoder
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import torch.nn as nn


class ClassicalLogisticRegressionBaseline:
    """Logistic Regression baseline with probability calibration."""

    def __init__(self, C: float = 1.0, seed: int = 42):
        self.C = C
        self.seed = seed
        self.model = LogisticRegression(C=C, random_state=seed, max_iter=500)
        self.threshold = 0.50
        self.fit_time_s = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> ClassicalLogisticRegressionBaseline:
        t0 = time.time()
        self.model.fit(X, y)
        self.fit_time_s = time.time() - t0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(int)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)


class ClassicalRBFSVMBaseline:
    """Support Vector Machine with Radial Basis Function (RBF) kernel."""

    def __init__(self, C: float = 1.0, gamma: str = "scale", seed: int = 42):
        self.C = C
        self.gamma = gamma
        self.seed = seed
        self.model = SVC(C=C, gamma=gamma, kernel="rbf", probability=True, random_state=seed)
        self.threshold = 0.50
        self.fit_time_s = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> ClassicalRBFSVMBaseline:
        t0 = time.time()
        self.model.fit(X, y)
        self.fit_time_s = time.time() - t0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(int)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)


class SmallClassicalMLP(nn.Module):
    """Small classical MLP with comparable parameter count to VQC (4 -> 8 -> 1)."""

    def __init__(self, in_features: int = 4, hidden: int = 8, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.threshold = 0.50
        self.fit_time_s = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        lr: float = 0.02,
    ) -> SmallClassicalMLP:
        t0 = time.time()
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self(X_t)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()

        self.fit_time_s = time.time() - t0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            probs = self(X_t).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(int)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)
