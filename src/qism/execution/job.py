"""Simulation job definitions for scalable execution.

In accordance with Requirements 28, 30:
Self-contained picklable job and result objects tracking execution state,
timing, throughput, and error metadata.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

from qism.config import QKDConfig
from qism.types import JobStatus


class SimulationJob:
    """A self-contained simulation execution task."""

    scenario_id: str = ""
    run_id: str = ""
    seed: int = 0
    config: QKDConfig = field(default_factory=QKDConfig)
    output_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("data"))
    checkpoint_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("checkpoints"))
    shard_id: str = "shard_000"
    max_retries: int = 2
    retry_count: int = 0
    timeout_s: float = 600.0

    def __init__(
        self,
        scenario_id: str = "",
        run_id: str = "",
        seed: int = 0,
        config: QKDConfig | None = None,
        output_dir: str | pathlib.Path = "data",
        checkpoint_dir: str | pathlib.Path = "checkpoints",
        shard_id: str = "shard_000",
        max_retries: int = 2,
        retry_count: int = 0,
        timeout_s: float = 600.0,
        scenario: Any | None = None,
    ):
        if scenario is not None:
            self.scenario_id = getattr(scenario, "scenario_id", scenario_id)
            self.config = getattr(scenario, "config", config or QKDConfig())
        else:
            self.scenario_id = scenario_id
            self.config = config if config is not None else QKDConfig()

        self.run_id = run_id
        self.seed = seed
        self.output_dir = pathlib.Path(output_dir)
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.shard_id = shard_id
        self.max_retries = max_retries
        self.retry_count = retry_count
        self.timeout_s = timeout_s

    @property
    def job_id(self) -> str:
        return f"{self.scenario_id}__{self.run_id}"


@dataclass
class JobResult:
    """Outcome of executing a SimulationJob."""

    scenario_id: str
    run_id: str
    seed: int
    status: JobStatus
    summary: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    stack_trace: str | None = None
    wall_time_s: float = 0.0
    events_generated: int = 0
    events_per_sec: float = 0.0

    @property
    def job_id(self) -> str:
        return f"{self.scenario_id}__{self.run_id}"

    @property
    def is_success(self) -> bool:
        return self.status == JobStatus.COMPLETED

    @property
    def metrics(self) -> dict[str, Any]:
        return self.summary or {}
