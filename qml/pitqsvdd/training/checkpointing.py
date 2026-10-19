"""Crash-resistant atomic checkpointing for PI-TQSVDD.

Guarantees:
- Atomic writes: Temporary file write followed by atomic replacement prevents corrupt states.
- Complete state preservation: Model parameters, optimizer states, epoch, step,
  loss history, random states (Torch, NumPy, Python), and backend metadata.
- Tri-tier checkpoint maintenance:
  - checkpoint_latest.pt (always the most recent completed step)
  - checkpoint_best.pt (model with lowest validation loss)
  - checkpoint_epoch_{epoch}.pt (periodic archival checkpoints)
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointManager:
    """Manages crash-resilient atomic serialization and restoration."""

    REQUIRED_KEYS = (
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "global_step",
        "best_val_loss",
        "config",
        "rng_states",
    )

    def __init__(self, save_dir: str | Path, run_id: str):
        self.run_dir = Path(save_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.run_dir / "checkpoint_latest.pt"
        self.best_path = self.run_dir / "checkpoint_best.pt"

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        global_step: int,
        val_loss: float,
        best_val_loss: float,
        config_dict: dict[str, Any],
        metrics_history: list[dict[str, Any]],
        scheduler: Any = None,
        is_best: bool = False,
        is_periodic: bool = False,
    ) -> Path:
        """Atomically persist a checkpoint dictionary to disk."""
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }

        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": float(val_loss),
            "best_val_loss": float(best_val_loss),
            "config": config_dict,
            "metrics_history": metrics_history,
            "rng_states": rng_states,
            "backend_meta": {
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
            },
        }

        # Step 1: Write to temporary file in the same filesystem directory
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=self.run_dir, prefix="ckpt_", suffix=".tmp")
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)

        try:
            torch.save(payload, tmp_path)
            # Step 2: Atomic rename/replace to checkpoint_latest.pt
            tmp_path.replace(self.latest_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Atomic checkpoint save failed: {e}")

        # Step 3: Maintain best checkpoint
        if is_best:
            shutil.copy2(self.latest_path, self.best_path)

        # Step 4: Maintain periodic checkpoint
        if is_periodic:
            epoch_path = self.run_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            shutil.copy2(self.latest_path, epoch_path)

        return self.latest_path

    def has_checkpoint(self) -> bool:
        """Check if a valid latest checkpoint exists."""
        return self.latest_path.exists()

    def load_checkpoint(self, path: Path | str | None = None) -> dict[str, Any]:
        """Load and validate checkpoint payload."""
        target_path = Path(path) if path is not None else self.latest_path
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {target_path}")

        payload = torch.load(target_path, map_location="cpu", weights_only=False)

        # Validate integrity
        for key in self.REQUIRED_KEYS:
            if key not in payload:
                raise ValueError(f"Corrupt checkpoint! Missing required key '{key}' in {target_path}")

        return payload

    def restore_state(
        self,
        checkpoint: dict[str, Any],
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any = None,
    ) -> tuple[int, int, float, list[dict[str, Any]]]:
        """Restore model, optimizer, RNG, and training progress from checkpoint.

        Returns:
            (epoch, global_step, best_val_loss, metrics_history)
        """
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore RNG states
        rng = checkpoint.get("rng_states", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch_cpu" in rng:
            torch.set_rng_state(rng["torch_cpu"])

        epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        metrics_history = checkpoint.get("metrics_history", [])

        return epoch, global_step, best_val_loss, metrics_history
