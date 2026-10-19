"""Configuration dataclass and defaults for the QISM visualization suite."""

from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import Sequence


@dataclass
class VisualizationConfig:
    """Configures paths, styling, formats, and sampling for visualizations."""

    dataset_path: pathlib.Path = field(
        default_factory=lambda: pathlib.Path("data/v0.5_small/v05_small_dataset")
    )
    output_root: pathlib.Path = field(
        default_factory=lambda: pathlib.Path("outputs/visualizations")
    )
    formats: list[str] = field(default_factory=lambda: ["png"])
    dpi: int = 300
    seed: int = 20260502
    max_event_samples: int = 20_000
    clean_output: bool = False

    @property
    def presentation_dir(self) -> pathlib.Path:
        return self.output_root / "presentation"

    @property
    def exploratory_dir(self) -> pathlib.Path:
        return self.output_root / "exploratory"

    @property
    def metadata_dir(self) -> pathlib.Path:
        return self.output_root / "metadata"

    @property
    def report_dir(self) -> pathlib.Path:
        return self.output_root / "report"

    def ensure_directories(self) -> None:
        """Create all required output directories."""
        self.presentation_dir.mkdir(parents=True, exist_ok=True)
        self.exploratory_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
