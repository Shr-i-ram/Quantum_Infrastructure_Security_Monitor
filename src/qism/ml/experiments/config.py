"""Configuration objects for ML experiments and benchmarks (Requirements 54–56, 148)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import yaml


class ExperimentConfig(BaseModel):
    """Configuration for a complete anomaly detection experiment."""

    experiment_id: str
    dataset_root: str
    version: str = "0.6.0"
    master_seed: int = 42

    # Feature configuration
    feature_set: str = "full"  # "qkd_only", "detector_only", "environmental_only", "infrastructure_only", "minimal", "full"
    add_temporal_features: bool = False
    rolling_window_length: int = 3

    # Preprocessing
    scaler: str = "standard"  # "standard", "robust", "none"
    impute_strategy: str = "median"
    clip_sigma: float = 10.0

    # Model configuration
    model_type: str = "isolation_forest"  # "mahalanobis", "isolation_forest", "one_class_svm", "lof", "autoencoder", "temporal_autoencoder"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    # Thresholding & operational alarms
    target_fpr: float = 0.01
    consecutive_k: int = 2

    # Splits & Output
    train_val_split_ratio: float = 0.70
    output_dir: str = "experiments"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load experiment configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
