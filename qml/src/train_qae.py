"""Training pipeline for Quantum Autoencoder (Review 2).

Includes:
- Training strictly on normal QKD telemetry (y == 0).
- Minimizing trash qubit excitation (maximizing compression fidelity with |00>).
- Calibrating unsupervised anomaly score threshold on validation normal data.
- Checkpoint persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from qml.src.quantum_autoencoder import QuantumAutoencoder


def train_qae(
    model: QuantumAutoencoder,
    X_train_normal: np.ndarray,
    X_val_normal: np.ndarray,
    epochs: int = 35,
    lr: float = 0.03,
    output_dir: str | Path = "artifacts/qml_review2",
) -> dict[str, Any]:
    """Train Quantum Autoencoder on normal telemetry and calibrate anomaly threshold."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train_t = torch.tensor(X_train_normal, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_normal, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: list[float] = []
    val_losses: list[float] = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        scores = model(X_train_t)
        loss = scores.mean()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_scores = model(X_val_t)
            val_loss = val_scores.mean().item()
            val_losses.append(val_loss)

    train_duration_s = time.time() - t0

    # Calibrate threshold on validation normal scores (95th percentile)
    val_scores_np = model.score_samples(X_val_normal)
    threshold = float(np.percentile(val_scores_np, 95.0))
    model.set_threshold(threshold)

    # Save checkpoint
    chk_path = out_dir / "checkpoints" / "qae_model.pt"
    model.save_checkpoint(chk_path, meta={
        "train_duration_s": train_duration_s,
        "calibrated_threshold": threshold,
        "epochs": epochs,
        "lr": lr,
    })

    record = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_duration_s": train_duration_s,
        "calibrated_threshold": threshold,
        "checkpoint_path": str(chk_path),
    }

    return record
