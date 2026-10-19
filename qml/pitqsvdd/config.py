"""Configuration schemas and YAML loaders for PI-TQSVDD."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import yaml


@dataclass
class DataConfig:
    """Telemetry data loading and feature configuration."""
    dataset_root: str = "data/v0.5_small/v05_small_dataset"
    representation: Literal["pca", "learned_bottleneck", "feature_group", "highdim_pca", "random_proj"] = "pca"
    n_qubits: int = 4
    sequence_length: int = 3
    temporal_stride: int = 1
    master_seed: int = 42
    normal_only_training: bool = True
    train_val_ratio: float = 0.70


@dataclass
class CircuitConfig:
    """Quantum circuit architecture and backend configuration."""
    n_qubits: int = 4
    n_layers: int = 2
    encoding_type: Literal[
        "angle",
        "data_reuploading",
        "amplitude",
        "iqp_zz",
        "feature_group",
        "learned_projection",
    ] = "angle"
    entanglement: Literal["ring", "all-to-all", "linear"] = "ring"
    rotation: Literal["X", "Y", "Z", "rot"] = "Y"
    backend: str = "lightning.qubit"
    diff_method: str = "adjoint"
    reupload_blocks: int = 2


@dataclass
class ModelConfig:
    """PI-TQSVDD model formulation and objective hyperparameters."""
    model_type: Literal["static_qsvdd", "temporal_pitqsvdd", "context_aware_pitqsvdd"] = "temporal_pitqsvdd"
    lambda_temporal: float = 0.5
    lambda_reg: float = 1e-4
    temporal_hidden_dim: int = 16
    num_regimes: int = 2
    centre_init: Literal["first_batch_mean", "train_mean"] = "train_mean"
    fix_centre: bool = True
    bottleneck_hidden_dim: int = 24


@dataclass
class TrainingConfig:
    """Training optimization, budget, and checkpointing configuration."""
    lr: float = 0.01
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 25
    early_stopping_patience: int = 8
    grad_clip: float = 1.0
    checkpoint_freq: int = 5
    save_dir: str = "artifacts/pi_tqsvdd/checkpoints"
    run_id: str = "pitqsvdd_experiment"
    resume_mode: Literal["auto", "none", "explicit"] = "auto"
    resume_path: str | None = None
    overwrite: bool = False


@dataclass
class EvaluationConfig:
    """Anomaly scoring and threshold calibration parameters."""
    threshold_percentile: float = 95.0
    alpha_svdd: float = 0.5
    beta_temporal: float = 0.5
    persistence_window: int = 1
    output_dir: str = "artifacts/pi_tqsvdd"


@dataclass
class PITQSVDDConfig:
    """Comprehensive master configuration for a PI-TQSVDD run."""
    data: DataConfig = field(default_factory=DataConfig)
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        """Persist configuration to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, indent=2)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PITQSVDDConfig:
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return cls(
            data=DataConfig(**raw.get("data", {})),
            circuit=CircuitConfig(**raw.get("circuit", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        )
