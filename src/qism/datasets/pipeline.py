"""End-to-end dataset generation pipeline (Requirement 116).

Coordinates:
Master Config -> Scenario Generator -> Job Scheduler -> Parallel Workers ->
Shard Writer -> Manifests -> Diagnostics.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pathlib
import time
from typing import Any

from qism.config import Scenario, SplitConfig, V05DatasetConfig
from qism.datasets.provenance import build_provenance_record
from qism.datasets.sharding import DatasetShardWriter
from qism.execution.job import JobResult, SimulationJob
from qism.execution.scheduler import JobScheduler
from qism.population.diagnostics import (
    extract_scenario_dataframe,
    generate_coverage_report,
    plot_population_coverage,
    validate_population_distributions,
)
from qism.population.generator import ScenarioGenerator
from qism.random import derive_dataset_seed, derive_run_seed

log = logging.getLogger("qism.datasets.pipeline")


class DatasetPipeline:
    """Orchestrates large-scale dataset generation across scenarios and parallel workers."""

    def __init__(self, config: V05DatasetConfig):
        self.config = config
        self.output_root = pathlib.Path(config.output_root) / config.dataset_id
        self.checkpoint_root = pathlib.Path(config.checkpoint_root) / config.dataset_id
        self.scenarios_dir = self.output_root / "scenarios"
        self.diagnostics_dir = self.output_root / "diagnostics"

    def estimate_workload(self) -> dict[str, Any]:
        """Provide an advance estimate of scenarios, runs, pulses, and storage (Requirement 110)."""
        tot_scenarios = sum(s.n_scenarios for s in self.config.splits.values())
        tot_runs = sum(s.n_scenarios * s.runs_per_scenario for s in self.config.splits.values())
        tot_pulses = sum(s.n_scenarios * s.runs_per_scenario * s.pulses_per_run for s in self.config.splits.values())
        win_size = (
            self.config.base.telemetry.window_size
            if (self.config.base and self.config.base.telemetry)
            else 2000
        )
        tot_windows = sum(
            s.n_scenarios * s.runs_per_scenario * (s.pulses_per_run // max(1, win_size))
            for s in self.config.splits.values()
        )
        # Empirical storage: ~120 bytes per pulse for full events + telemetry
        est_mb = (tot_pulses * 120) / (1024 * 1024)

        return {
            "dataset_id": self.config.dataset_id,
            "total_splits": len(self.config.splits),
            "total_scenarios": tot_scenarios,
            "total_runs": tot_runs,
            "total_pulses": tot_pulses,
            "total_estimated_windows": tot_windows,
            "estimated_storage_mb": round(est_mb, 1),
            "sampling_strategy": self.config.sampling_strategy,
            "worker_count": self.config.worker_count,
        }

    def dry_run(self) -> dict[str, Any]:
        """Generate, validate, and preview scenarios without executing simulations (Requirement 111)."""
        scenarios = self._generate_all_scenarios()
        df = extract_scenario_dataframe(scenarios)
        workload = self.estimate_workload()
        return {
            "status": "VALIDATED",
            "workload": workload,
            "scenarios_generated": len(scenarios),
            "parameters": list(df.columns),
        }

    def _generate_all_scenarios(self) -> list[Scenario]:
        """Generate Scenario instances for all configured splits."""
        generator = ScenarioGenerator(
            self.config.population,
            base_template=self.config.base,
            master_seed=self.config.master_seed,
            sampling_strategy=self.config.sampling_strategy,
        )
        dataset_seed = derive_dataset_seed(self.config.master_seed, self.config.dataset_id)
        all_scenarios: list[Scenario] = []

        for split_name, split_cfg in self.config.splits.items():
            split_seed = derive_dataset_seed(dataset_seed, split_name)
            scenarios = generator.generate_split_scenarios(split_name, split_cfg, split_seed)
            all_scenarios.extend(scenarios)

        return all_scenarios

    def run(self, resume: bool = False, fresh: bool = False) -> dict[str, Any]:
        """Execute the end-to-end dataset generation pipeline."""
        t0 = time.perf_counter()
        if fresh and self.output_root.exists():
            import shutil
            log.warning("Fresh run requested: wiping %s", self.output_root)
            shutil.rmtree(self.output_root, ignore_errors=True)
            shutil.rmtree(self.checkpoint_root, ignore_errors=True)

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate & validate scenarios
        log.info("Generating population scenarios across splits...")
        all_scenarios = self._generate_all_scenarios()
        log.info("Generated and validated %d scenarios", len(all_scenarios))

        # Save scenario manifest
        scen_manifest = {
            "dataset_id": self.config.dataset_id,
            "total_scenarios": len(all_scenarios),
            "scenarios": [s.model_dump() for s in all_scenarios],
        }
        with open(self.scenarios_dir / "scenario_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(scen_manifest, fh, indent=2, default=str)

        # 2. Build simulation jobs partitioned by shard
        jobs: list[SimulationJob] = []
        shard_writers: dict[str, DatasetShardWriter] = {}

        for split_name, split_cfg in self.config.splits.items():
            split_dir = self.output_root / split_name
            shard_writer = DatasetShardWriter(split_dir, shard_size=self.config.shard_size)
            shard_writers[split_name] = shard_writer

            split_scenarios = [s for s in all_scenarios if s.dataset_split == split_name]
            run_counter = 0

            for sc in split_scenarios:
                for r_idx in range(split_cfg.runs_per_scenario):
                    run_id = f"run_{run_counter:04d}"
                    shard_dir = shard_writer.get_shard_dir(run_counter)
                    run_seed = derive_run_seed(sc.seed, run_id)

                    job = SimulationJob(
                        scenario_id=sc.scenario_id,
                        run_id=run_id,
                        seed=run_seed,
                        config=copy.deepcopy(sc.config),
                        output_dir=shard_dir,
                        checkpoint_dir=self.checkpoint_root / split_name / shard_dir.name,
                        shard_id=shard_dir.name,
                        max_retries=self.config.max_retries,
                    )
                    jobs.append(job)
                    run_counter += 1

        # 3. Schedule and execute jobs
        state_file = self.checkpoint_root / "job_manifest.json"
        scheduler = JobScheduler(
            jobs,
            state_file=state_file,
            worker_count=self.config.worker_count,
            parallel=(self.config.worker_count > 1),
        )
        results = scheduler.run()

        # 4. Finalize manifests & markers
        n_completed = sum(1 for r in results if r.is_success)
        n_failed = sum(1 for r in results if not r.is_success)
        wall_time = time.perf_counter() - t0

        for split_name in self.config.splits.keys():
            marker = self.output_root / split_name / "dataset_complete.marker"
            with open(marker, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "dataset_id": self.config.dataset_id,
                        "split": split_name,
                        "status": "COMPLETED",
                        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    fh,
                    indent=2,
                )

        # Top-level manifest
        provenance = build_provenance_record(
            self.config.dataset_id,
            self.config.master_seed,
            self.config.model_dump(),
        )
        status = "COMPLETED" if n_failed == 0 else "PARTIAL"
        top_manifest = {
            **provenance,
            "status": status,
            "total_scenarios": len(all_scenarios),
            "total_jobs": len(jobs),
            "completed_jobs": n_completed,
            "failed_jobs": n_failed,
            "wall_time_s": wall_time,
            "splits": list(self.config.splits.keys()),
        }
        with open(self.output_root / "dataset_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(top_manifest, fh, indent=2, default=str)

        # 5. Run population diagnostics and generate coverage plots
        log.info("Generating population coverage reports and plots...")
        df_scen = extract_scenario_dataframe(all_scenarios)
        generate_coverage_report(df_scen, self.diagnostics_dir / "parameter_coverage.csv")
        plot_population_coverage(df_scen, self.diagnostics_dir)

        log.info("Dataset generation completed in %.2fs (%d completed, %d failed)", wall_time, n_completed, n_failed)

        return {
            "dataset_id": self.config.dataset_id,
            "status": status,
            "output_dir": str(self.output_root),
            "total_scenarios": len(all_scenarios),
            "completed_runs": n_completed,
            "failed_runs": n_failed,
            "wall_time_s": wall_time,
        }
