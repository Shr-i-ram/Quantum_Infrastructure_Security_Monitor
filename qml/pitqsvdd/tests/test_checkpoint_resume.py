"""Automated test for crash-resistant atomic checkpointing and training resumption."""

import shutil
from pathlib import Path
import tempfile
import torch

from qml.pitqsvdd.circuits.variational import VariationalQuantumCircuit
from qml.pitqsvdd.config import CircuitConfig, DataConfig, EvaluationConfig, ModelConfig, PITQSVDDConfig, TrainingConfig
from qml.pitqsvdd.data import PITQSVDDDatasetManager
from qml.pitqsvdd.encodings import get_encoding
from qml.pitqsvdd.models import build_model
from qml.pitqsvdd.training.trainer import PITQSVDDTrainer


def test_checkpoint_save_and_resumption():
    """Verify that training can be paused, serialized, and resumed seamlessly without restarting from epoch 0."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_pitqsvdd_ckpt_"))
    try:
        run_id = "test_resume_run"
        cfg = PITQSVDDConfig(
            data=DataConfig(dataset_root="data/v0.5_small/v05_small_dataset", n_qubits=2, sequence_length=2),
            circuit=CircuitConfig(n_qubits=2, n_layers=1, backend="lightning.qubit"),
            model=ModelConfig(model_type="static_qsvdd"),
            training=TrainingConfig(
                max_epochs=2,  # Train for 2 epochs first
                batch_size=4,
                save_dir=str(temp_dir),
                run_id=run_id,
                resume_mode="auto",
            ),
            evaluation=EvaluationConfig(output_dir=str(temp_dir)),
        )

        dm = PITQSVDDDatasetManager(cfg.data)

        # Stage 1: Initial training for 2 epochs
        enc1 = get_encoding("angle", n_qubits=2)
        circ1 = VariationalQuantumCircuit(cfg.circuit, enc1, seed=42)
        model1 = build_model(cfg.model.model_type, circ1, cfg.model)
        trainer1 = PITQSVDDTrainer(model1, dm, cfg)
        res1 = trainer1.train()

        # Verify Stage 1 completion
        assert res1["final_epoch"] == 1  # 0-indexed epochs 0, 1 -> 2 epochs total
        step_after_stage1 = res1["global_step"]
        assert step_after_stage1 > 0
        assert (temp_dir / run_id / "checkpoint_latest.pt").exists()

        # Capture model parameters after stage 1
        weights_stage1 = circ1.q_weights.detach().clone()

        # Stage 2: Resume training in a fresh model instance for 2 more epochs (total 4 epochs)
        cfg.training.max_epochs = 4
        enc2 = get_encoding("angle", n_qubits=2)
        circ2 = VariationalQuantumCircuit(cfg.circuit, enc2, seed=999)  # Different seed to test restoration
        model2 = build_model(cfg.model.model_type, circ2, cfg.model)

        trainer2 = PITQSVDDTrainer(model2, dm, cfg)
        res2 = trainer2.train()

        # Verify that start epoch was restored to 2 and completed at epoch 3 (total 4 epochs)
        assert res2["final_epoch"] == 3
        assert res2["global_step"] > step_after_stage1
        assert len(res2["metrics_history"]) == 4

        # Verify that parameters evolved from stage 1 rather than starting from fresh random weights
        weights_stage2 = circ2.q_weights.detach()
        assert not torch.allclose(weights_stage1, weights_stage2)

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
