"""Resumable Trainer implementation for PI-TQSVDD."""

from __future__ import annotations

import time
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim

from qml.pitqsvdd.config import EvaluationConfig, PITQSVDDConfig, TrainingConfig
from qml.pitqsvdd.data import PITQSVDDDatasetManager
from qml.pitqsvdd.training.checkpointing import CheckpointManager
from qml.pitqsvdd.training.resume import resolve_resume_target


class PITQSVDDTrainer:
    """Resumable, crash-resistant trainer for PI-TQSVDD models."""

    def __init__(
        self,
        model: nn.Module,
        dataset_manager: PITQSVDDDatasetManager,
        config: PITQSVDDConfig,
    ):
        self.model = model
        self.dm = dataset_manager
        self.config = config
        self.train_cfg = config.training
        self.eval_cfg = config.evaluation

        self.checkpoint_mgr = CheckpointManager(
            save_dir=self.train_cfg.save_dir,
            run_id=self.train_cfg.run_id,
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )

        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.metrics_history: list[dict[str, Any]] = []

    def setup_or_resume(self) -> None:
        """Resolve resume state or initialize fresh model centres."""
        resume_ckpt_path = resolve_resume_target(
            checkpoint_mgr=self.checkpoint_mgr,
            resume_mode=self.train_cfg.resume_mode,
            resume_path=self.train_cfg.resume_path,
            overwrite=self.train_cfg.overwrite,
        )

        if resume_ckpt_path is not None:
            ckpt = self.checkpoint_mgr.load_checkpoint(resume_ckpt_path)
            ep, step, best_loss, hist = self.checkpoint_mgr.restore_state(
                ckpt, self.model, self.optimizer
            )
            self.start_epoch = ep + 1
            self.global_step = step
            self.best_val_loss = best_loss
            self.metrics_history = hist
            print(f"[Trainer] Resumed run '{self.train_cfg.run_id}' from epoch {ep} (step {step}).")
        else:
            # Initialize reference centre c using first batch of normal training data
            train_loader = self.dm.get_dataloader("train", batch_size=self.train_cfg.batch_size, shuffle=False)
            first_batch = next(iter(train_loader))
            windows, _, _, _ = first_batch
            if hasattr(self.model, "init_centre"):
                self.model.init_centre(windows)
            elif hasattr(self.model, "init_centres"):
                self.model.init_centres(windows)
            print(f"[Trainer] Initialized fresh run '{self.train_cfg.run_id}' with reference centre.")

    def evaluate_validation(self) -> dict[str, float]:
        """Compute average loss components over validation partition."""
        val_loader = self.dm.get_dataloader("val", batch_size=self.train_cfg.batch_size, shuffle=False)
        self.model.eval()

        total_loss = 0.0
        total_svdd = 0.0
        total_temp = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                windows, targets, has_target, _ = batch
                out = self.model.compute_loss(windows, targets=targets, has_target=has_target)
                total_loss += out["loss"].item()
                total_svdd += out["loss_svdd"].item()
                total_temp += out["loss_temporal"].item()
                n_batches += 1

        if n_batches == 0:
            return {"loss": 0.0, "loss_svdd": 0.0, "loss_temporal": 0.0}

        return {
            "loss": total_loss / n_batches,
            "loss_svdd": total_svdd / n_batches,
            "loss_temporal": total_temp / n_batches,
        }

    def train(self) -> dict[str, Any]:
        """Execute or continue training loop."""
        self.setup_or_resume()

        train_loader = self.dm.get_dataloader(
            "train", batch_size=self.train_cfg.batch_size, shuffle=True
        )

        epochs_without_improvement = 0
        t0_train = time.time()

        final_epoch = max(0, self.start_epoch - 1)
        if self.start_epoch >= self.train_cfg.max_epochs:
            print(
                f"[Trainer] Run '{self.train_cfg.run_id}' already completed "
                f"{self.start_epoch} epochs (max_epochs={self.train_cfg.max_epochs}). Nothing to train."
            )

        for epoch in range(self.start_epoch, self.train_cfg.max_epochs):
            self.model.train()
            ep_loss = 0.0
            ep_svdd = 0.0
            ep_temp = 0.0
            n_batches = 0

            for batch in train_loader:
                windows, targets, has_target, _ = batch

                self.optimizer.zero_grad()
                out = self.model.compute_loss(windows, targets=targets, has_target=has_target)
                loss = out["loss"]
                loss.backward()

                # Gradient clipping to ensure numerical stability
                if self.train_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.train_cfg.grad_clip
                    )

                self.optimizer.step()

                ep_loss += loss.item()
                ep_svdd += out["loss_svdd"].item()
                ep_temp += out["loss_temporal"].item()
                n_batches += 1
                self.global_step += 1

            final_epoch = epoch
            train_metrics = {
                "train_loss": ep_loss / max(1, n_batches),
                "train_loss_svdd": ep_svdd / max(1, n_batches),
                "train_loss_temporal": ep_temp / max(1, n_batches),
            }

            # Evaluate on validation
            val_metrics = self.evaluate_validation()
            val_loss = val_metrics["loss"]

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            record = {
                "epoch": epoch,
                "global_step": self.global_step,
                **train_metrics,
                "val_loss": val_loss,
                "val_loss_svdd": val_metrics["loss_svdd"],
                "val_loss_temporal": val_metrics["loss_temporal"],
                "is_best": is_best,
            }
            self.metrics_history.append(record)

            # Atomic checkpoint save
            is_periodic = (epoch + 1) % self.train_cfg.checkpoint_freq == 0
            self.checkpoint_mgr.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                global_step=self.global_step,
                val_loss=val_loss,
                best_val_loss=self.best_val_loss,
                config_dict=self.config.to_dict(),
                metrics_history=self.metrics_history,
                is_best=is_best,
                is_periodic=is_periodic,
            )

            # Early stopping check
            if epochs_without_improvement >= self.train_cfg.early_stopping_patience:
                print(f"[Trainer] Early stopping triggered at epoch {epoch}.")
                break

        total_time = time.time() - t0_train
        return {
            "total_time_s": total_time,
            "final_epoch": final_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "metrics_history": self.metrics_history,
        }
