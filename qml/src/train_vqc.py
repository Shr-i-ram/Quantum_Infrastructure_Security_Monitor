"""Training pipeline for Hybrid Variational Quantum Classifier (Review 2).

Includes:
- Batch training loop with Adam optimizer and Binary Cross-Entropy loss.
- Epoch-by-epoch training and validation loss tracking.
- Non-parametric validation threshold tuning (maximizing validation F1).
- Complete artifact persistence (model weights, training history, predictions).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from qml.src.metrics import compute_binary_metrics
from qml.src.vqc_model import VariationalQuantumClassifier


def train_vqc(
    model: VariationalQuantumClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    lr: float = 0.02,
    batch_size: int = 10,
    output_dir: str | Path = "artifacts/qml_review2",
) -> dict[str, Any]:
    """Train VQC and return comprehensive training record and calibrated model."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val_loss = float("inf")
    best_weights: dict[str, Any] | None = None

    t0 = time.time()
    n_samples = len(X_train_t)

    for epoch in range(1, epochs + 1):
        model.train()
        # Shuffle indices
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i : i + batch_size]
            bx, by = X_train_t[batch_idx], y_train_t[batch_idx]

            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(1, n_batches)
        train_losses.append(avg_train_loss)

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_duration_s = time.time() - t0

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Threshold calibration on validation set
    val_probs = model.predict_proba(X_val)
    best_th = 0.50
    best_f1 = -1.0
    for candidate_th in np.linspace(0.1, 0.9, 81):
        cand_preds = (val_probs >= candidate_th).astype(int)
        cand_metrics = compute_binary_metrics(y_val, cand_preds, val_probs)
        if cand_metrics["f1_score"] > best_f1:
            best_f1 = cand_metrics["f1_score"]
            best_th = float(candidate_th)

    model.set_threshold(best_th)

    # Save checkpoint
    chk_path = out_dir / "checkpoints" / "vqc_model.pt"
    model.save_checkpoint(chk_path, meta={
        "train_duration_s": train_duration_s,
        "best_val_loss": best_val_loss,
        "calibrated_threshold": best_th,
        "epochs": epochs,
        "lr": lr,
    })

    record = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_duration_s": train_duration_s,
        "best_val_loss": best_val_loss,
        "calibrated_threshold": best_th,
        "val_f1_at_calibrated_th": best_f1,
        "checkpoint_path": str(chk_path),
    }

    return record
