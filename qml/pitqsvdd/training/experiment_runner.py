"""Experiment runner, manifest generator, and machine-readable registry manager."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
import torch

from qml.pitqsvdd.circuits.variational import VariationalQuantumCircuit
from qml.pitqsvdd.config import PITQSVDDConfig
from qml.pitqsvdd.data import PITQSVDDDatasetManager
from qml.pitqsvdd.encodings import get_encoding
from qml.pitqsvdd.evaluation.scoring import evaluate_model
from qml.pitqsvdd.evaluation.thresholding import ThresholdCalibrator
from qml.pitqsvdd.models import build_model
from qml.pitqsvdd.training.trainer import PITQSVDDTrainer


class PITQSVDDExperimentRunner:
    """Manages execution of single experiments or benchmark sweeps with crash recovery."""

    def __init__(self, artifacts_dir: str | Path = "artifacts/pi_tqsvdd"):
        self.artifacts_dir = Path(artifacts_dir)
        self.registry_csv = self.artifacts_dir / "experiment_registry.csv"
        self.registry_json = self.artifacts_dir / "experiment_registry.json"
        self.manifests_dir = self.artifacts_dir / "manifests"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.checkpoints_dir = self.artifacts_dir / "checkpoints"

        for p in [self.manifests_dir, self.metrics_dir, self.checkpoints_dir]:
            p.mkdir(parents=True, exist_ok=True)

        self.registry_df = self._load_registry()

    def _load_registry(self) -> pd.DataFrame:
        """Load existing experiment registry or initialize empty DataFrame."""
        if self.registry_csv.exists():
            try:
                return pd.read_csv(self.registry_csv)
            except Exception:
                pass
        return pd.DataFrame()

    def _save_registry(self) -> None:
        """Save registry to both CSV and JSON formats atomically."""
        self.registry_df.to_csv(self.registry_csv, index=False)
        self.registry_df.to_json(self.registry_json, orient="records", indent=2)

    def is_experiment_completed(self, experiment_id: str) -> bool:
        """Check if an experiment is recorded as completed with existing artifacts."""
        if self.registry_df.empty or "experiment_id" not in self.registry_df.columns:
            return False
        row = self.registry_df[self.registry_df["experiment_id"] == experiment_id]
        if not row.empty and row.iloc[0].get("status") == "completed":
            metric_file = self.metrics_dir / f"{experiment_id}.json"
            if metric_file.exists():
                return True
        return False

    def run_experiment(self, config: PITQSVDDConfig) -> dict[str, Any]:
        """Execute a single PI-TQSVDD experiment end-to-end."""
        exp_id = config.training.run_id
        timestamp_str = datetime.now().isoformat()

        # Check if already completed and resume_mode != "none"
        if self.is_experiment_completed(exp_id) and config.training.resume_mode != "none" and not config.training.overwrite:
            print(f"[Runner] Experiment '{exp_id}' is already completed. Skipping.")
            metric_path = self.metrics_dir / f"{exp_id}.json"
            with open(metric_path, "r", encoding="utf-8") as f:
                return json.load(f)

        print(f"\n========================================================")
        print(f"  STARTING EXPERIMENT: {exp_id}")
        print(f"  Qubits: {config.circuit.n_qubits} | Encoding: {config.circuit.encoding_type} | Model: {config.model.model_type}")
        print(f"  Representation: {config.data.representation} | Backend: {config.circuit.backend}")
        print(f"========================================================")

        t0_start = time.time()
        status = "running"
        error_msg = None
        circuit_metrics: dict[str, Any] = {}
        train_res: dict[str, Any] = {}
        eval_res: dict[str, Any] = {}

        try:
            # 1. Dataset loading and partitioning
            dm = PITQSVDDDatasetManager(config.data)

            # 2. Encoding and Circuit instantiation
            encoder = get_encoding(
                config.circuit.encoding_type,
                n_qubits=config.circuit.n_qubits,
                reupload_blocks=config.circuit.reupload_blocks,
                rotation=config.circuit.rotation,
            )
            circuit = VariationalQuantumCircuit(
                config=config.circuit,
                encoder=encoder,
                seed=config.data.master_seed,
            )
            circuit_metrics = circuit.get_circuit_metrics()

            # 3. Model construction
            model = build_model(
                model_type=config.model.model_type,
                circuit=circuit,
                config=config.model,
            )

            # 4. Training
            trainer = PITQSVDDTrainer(
                model=model,
                dataset_manager=dm,
                config=config,
            )
            train_res = trainer.train()

            # 5. Evaluation
            calibrator = ThresholdCalibrator(percentile=config.evaluation.threshold_percentile)
            eval_res = evaluate_model(
                model=model,
                dm=dm,
                calibrator=calibrator,
                alpha=config.evaluation.alpha_svdd,
                beta=config.evaluation.beta_temporal,
            )

            status = "completed"

        except Exception as e:
            status = "failed"
            error_msg = str(e)
            print(f"[Runner] ERROR in experiment '{exp_id}': {e}")
            raise e

        finally:
            wall_time = time.time() - t0_start
            test_m = eval_res.get("test", {}).get("metrics", {}) if status == "completed" else {}
            gen_m = eval_res.get("generalization", {}).get("metrics", {}) if status == "completed" else {}

            record = {
                "experiment_id": exp_id,
                "status": status,
                "timestamp": timestamp_str,
                "encoding_type": config.circuit.encoding_type,
                "n_qubits": config.circuit.n_qubits,
                "n_layers": config.circuit.n_layers,
                "representation": config.data.representation,
                "model_type": config.model.model_type,
                "backend": config.circuit.backend,
                "diff_method": config.circuit.diff_method,
                "wall_time_s": round(wall_time, 2),
                "trainable_quantum_params": circuit_metrics.get("trainable_quantum_params", 0) if status == "completed" else 0,
                "cnot_count": circuit_metrics.get("cnot_count", 0) if status == "completed" else 0,
                "rotation_gate_count": circuit_metrics.get("rotation_gate_count", 0) if status == "completed" else 0,
                "estimated_depth": circuit_metrics.get("estimated_depth", 0) if status == "completed" else 0,
                "best_val_loss": round(train_res.get("best_val_loss", float("nan")), 4) if status == "completed" else None,
                "calibrated_threshold": round(eval_res.get("calibrated_threshold", float("nan")), 4) if status == "completed" else None,
                "test_precision": round(test_m.get("precision", float("nan")), 4) if status == "completed" else None,
                "test_recall": round(test_m.get("recall", float("nan")), 4) if status == "completed" else None,
                "test_f1": round(test_m.get("f1", float("nan")), 4) if status == "completed" else None,
                "test_auroc": round(test_m.get("auroc", float("nan")), 4) if (status == "completed" and test_m.get("auroc") is not None) else None,
                "test_auprc": round(test_m.get("auprc", float("nan")), 4) if (status == "completed" and test_m.get("auprc") is not None) else None,
                "test_specificity": round(test_m.get("specificity", float("nan")), 4) if status == "completed" else None,
                "gen_f1": round(gen_m.get("f1", float("nan")), 4) if status == "completed" else None,
                "gen_recall": round(gen_m.get("recall", float("nan")), 4) if status == "completed" else None,
                "error_msg": error_msg,
            }

            # Update registry
            if self.registry_df.empty:
                self.registry_df = pd.DataFrame([record])
            else:
                # Remove existing row with same ID if updating
                self.registry_df = self.registry_df[self.registry_df["experiment_id"] != exp_id]
                self.registry_df = pd.concat([self.registry_df, pd.DataFrame([record])], ignore_index=True)
            self._save_registry()

            # Save detailed experiment manifest and metrics
            if status == "completed":
                manifest_payload = {
                    "experiment_id": exp_id,
                    "timestamp": timestamp_str,
                    "config": config.to_dict(),
                    "circuit_metrics": circuit_metrics,
                    "train_summary": {
                        "total_time_s": train_res.get("total_time_s"),
                        "final_epoch": train_res.get("final_epoch"),
                        "global_step": train_res.get("global_step"),
                        "best_val_loss": train_res.get("best_val_loss"),
                    },
                    "summary_metrics": record,
                }
                with open(self.manifests_dir / f"{exp_id}.json", "w", encoding="utf-8") as f:
                    json.dump(manifest_payload, f, indent=2)

                full_metrics_payload = {
                    "experiment_id": exp_id,
                    "summary": record,
                    "circuit_metrics": circuit_metrics,
                    "train_history": train_res.get("metrics_history", []),
                    "evaluation": eval_res,
                }
                with open(self.metrics_dir / f"{exp_id}.json", "w", encoding="utf-8") as f:
                    json.dump(full_metrics_payload, f, indent=2)

        return full_metrics_payload
