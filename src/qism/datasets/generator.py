"""Dataset generation: calibration sweeps and normal operational runs.

CALIBRATION — controlled runs where exactly one physical parameter is varied
while all others stay fixed (distance, attenuation, mu, detector efficiency,
dark-count rate, misalignment, timing jitter). Calibration data validates the
simulator against its analytical model; it must NOT be mixed into ML training.

NORMAL — legitimate QKD operational data at the default operating point, with
multiple independent deterministic seeds. This is the V0.1 normal dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import pathlib

from qism.config import (
    AttackDatasetConfig,
    CalibrationConfig,
    GeneralizationDatasetConfig,
    NormalConfig,
    apply_overrides,
)
from qism.datasets.manifest import update_manifest_field, write_manifest_atomic
from qism.simulation.runner import RunManager, RunSummary

log = logging.getLogger("qism.generator")


def _format_value(value: float) -> str:
    return f"{value:g}"


def _run_config(overrides: dict, pulses_per_run: int | None, family: str) -> dict:
    cfg_dict = dict(overrides)
    cfg_dict["dataset.family"] = family
    if pulses_per_run is not None:
        cfg_dict["protocol.n_pulses"] = int(pulses_per_run)
    return cfg_dict


class CalibrationGenerator:
    """Generates the calibration dataset (one parameter varied per run)."""

    def __init__(
        self,
        config: CalibrationConfig,
        *,
        output_root: str | pathlib.Path | None = None,
        checkpoint_root: str | pathlib.Path | None = None,
    ):
        self.config = config
        self.output_root = str(output_root) if output_root is not None else None
        self.checkpoint_root = str(checkpoint_root) if checkpoint_root is not None else None

    def generate(
        self,
        *,
        fresh: bool = False,
        resume: bool = False,
        pulses_per_run: int | None = None,
        only: list[str] | None = None,
    ) -> list[RunSummary]:
        """Generate every calibration run. Returns run summaries.

        ``only`` restricts the sweep labels (e.g. ["distance_km", "mu_signal"]).
        """
        pulses = pulses_per_run or self.config.pulses_per_run
        results: list[RunSummary] = []
        for label, paths, values in self.config.sweeps.items():
            if only and label not in only:
                continue
            for value in values:
                overrides = {path: value for path in paths}
                overrides["simulation.scenario_id"] = f"calib_{label}_{_format_value(value)}"
                cfg_dict = _run_config(overrides, pulses, "calibration")
                cfg = apply_overrides(self.config.base, cfg_dict)
                run_id = f"calib_{label}_{_format_value(value)}"
                log.info("calibration run %s: %s = %s", run_id, label, value)
                manager = RunManager(
                    cfg,
                    run_id,
                    output_root=self.output_root,
                    checkpoint_root=self.checkpoint_root,
                )
                summary = manager.run(resume=resume, fresh=fresh)
                update_manifest_field(
                    manager.run_dir / "manifest.json",
                    varied_parameter=label,
                    varied_value=float(value),
                )
                results.append(summary)
        return results


class NormalGenerator:
    """Generates the normal operational dataset (N runs, deterministic seeds)."""

    def __init__(
        self,
        config: NormalConfig,
        *,
        output_root: str | pathlib.Path | None = None,
        checkpoint_root: str | pathlib.Path | None = None,
    ):
        self.config = config
        self.output_root = str(output_root) if output_root is not None else None
        self.checkpoint_root = str(checkpoint_root) if checkpoint_root is not None else None

    def generate(
        self,
        *,
        runs: int | None = None,
        pulses_per_run: int | None = None,
        fresh: bool = False,
        resume: bool = False,
        start_index: int = 0,
    ) -> list[RunSummary]:
        """Generate the normal dataset. Each run uses a unique deterministic
        seed derived from (master_seed, run_id); no global randomness."""
        n_runs = runs or self.config.runs
        pulses = pulses_per_run or self.config.pulses_per_run or self.config.base.protocol.n_pulses
        results: list[RunSummary] = []
        for i in range(start_index, start_index + n_runs):
            run_id = f"run_{i:04d}"
            cfg_dict = _run_config({}, pulses, "normal")
            cfg_dict["simulation.scenario_id"] = "normal"
            cfg = apply_overrides(self.config.base, cfg_dict)
            log.info("normal run %s: %d pulses, family=normal", run_id, pulses)
            manager = RunManager(
                cfg,
                run_id,
                output_root=self.output_root,
                checkpoint_root=self.checkpoint_root,
            )
            summary = manager.run(resume=resume, fresh=fresh)
            marker_file = manager.run_dir / "dataset_complete.marker"
            write_manifest_atomic(
                {
                    "run_id": run_id,
                    "status": "COMPLETED",
                    "total_pulses": pulses,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                marker_file,
            )
            results.append(summary)
        return results


class AttackDatasetGenerator:
    """Generates the attack dataset (multiple attack scenarios, parameter variations)."""

    def __init__(
        self,
        config: AttackDatasetConfig,
        *,
        output_root: str | pathlib.Path | None = None,
        checkpoint_root: str | pathlib.Path | None = None,
    ):
        self.config = config
        self.output_root = str(output_root) if output_root is not None else None
        self.checkpoint_root = str(checkpoint_root) if checkpoint_root is not None else None

    def generate(
        self,
        *,
        fresh: bool = False,
        resume: bool = False,
        pulses_per_run: int | None = None,
        scenarios: list[str] | None = None,
    ) -> list[RunSummary]:
        """Generate attack dataset runs for each scenario.

        ``scenarios`` optionally restricts to specific scenario IDs or attack types.
        """
        pulses = pulses_per_run or self.config.pulses_per_run or self.config.base.protocol.n_pulses
        results: list[RunSummary] = []
        last_dir: pathlib.Path | None = None

        for sc in self.config.scenarios:
            if scenarios and (sc.scenario_id not in scenarios and sc.type not in scenarios):
                continue
            run_id = f"attack_{sc.scenario_id}"
            overrides = {
                "attack.enabled": True,
                "attack.scenarios": [sc.model_dump()],
                "simulation.scenario_id": sc.scenario_id,
            }
            cfg_dict = _run_config(overrides, pulses, "attacks")
            cfg = apply_overrides(self.config.base, cfg_dict)
            log.info("attack run %s: type=%s, severity=%.2f, pulses=%d", run_id, sc.type, sc.severity, pulses)
            manager = RunManager(
                cfg,
                run_id,
                output_root=self.output_root,
                checkpoint_root=self.checkpoint_root,
            )
            summary = manager.run(resume=resume, fresh=fresh)
            update_manifest_field(
                manager.run_dir / "manifest.json",
                attack_type=sc.type,
                attack_strength=float(sc.severity),
                attack_start=int(sc.start_pulse),
                attack_end=int(sc.end_pulse) if sc.end_pulse is not None else None,
                attack_scenario=sc.scenario_id,
                onset_profile=sc.onset_profile,
                recovery_profile=sc.recovery_profile,
            )
            # Write individual run completion marker
            marker_file = manager.run_dir / "dataset_complete.marker"
            write_manifest_atomic(
                {
                    "run_id": run_id,
                    "status": "COMPLETED",
                    "attack_type": sc.type,
                    "severity": sc.severity,
                    "total_pulses": pulses,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                marker_file,
            )
            results.append(summary)
            last_dir = manager.run_dir.parent

        if last_dir is not None and results:
            write_manifest_atomic(
                {
                    "dataset_family": "attacks",
                    "status": "COMPLETED",
                    "total_runs": len(results),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                last_dir / "dataset_complete.marker",
            )
        return results


class GeneralizationDatasetGenerator:
    """Generates the generalization dataset (held-out attack types and severities)."""

    def __init__(
        self,
        config: GeneralizationDatasetConfig,
        *,
        output_root: str | pathlib.Path | None = None,
        checkpoint_root: str | pathlib.Path | None = None,
    ):
        self.config = config
        self.output_root = str(output_root) if output_root is not None else None
        self.checkpoint_root = str(checkpoint_root) if checkpoint_root is not None else None

    def generate(
        self,
        *,
        fresh: bool = False,
        resume: bool = False,
        pulses_per_run: int | None = None,
    ) -> list[RunSummary]:
        """Generate generalization dataset runs."""
        pulses = pulses_per_run or self.config.pulses_per_run or self.config.base.protocol.n_pulses
        results: list[RunSummary] = []
        last_dir: pathlib.Path | None = None

        for sc in self.config.scenarios:
            run_id = f"gen_{sc.scenario_id}"
            overrides = {
                "attack.enabled": True,
                "attack.scenarios": [sc.model_dump()],
                "simulation.scenario_id": sc.scenario_id,
            }
            cfg_dict = _run_config(overrides, pulses, "generalization")
            cfg = apply_overrides(self.config.base, cfg_dict)
            log.info("generalization run %s: type=%s, severity=%.2f, pulses=%d", run_id, sc.type, sc.severity, pulses)
            manager = RunManager(
                cfg,
                run_id,
                output_root=self.output_root,
                checkpoint_root=self.checkpoint_root,
            )
            summary = manager.run(resume=resume, fresh=fresh)
            update_manifest_field(
                manager.run_dir / "manifest.json",
                attack_type=sc.type,
                attack_strength=float(sc.severity),
                attack_start=int(sc.start_pulse),
                attack_end=int(sc.end_pulse) if sc.end_pulse is not None else None,
                attack_scenario=sc.scenario_id,
                is_generalization=True,
            )
            marker_file = manager.run_dir / "dataset_complete.marker"
            write_manifest_atomic(
                {
                    "run_id": run_id,
                    "status": "COMPLETED",
                    "attack_type": sc.type,
                    "severity": sc.severity,
                    "total_pulses": pulses,
                    "is_generalization": True,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                marker_file,
            )
            results.append(summary)
            last_dir = manager.run_dir.parent

        if last_dir is not None and results:
            write_manifest_atomic(
                {
                    "dataset_family": "generalization",
                    "status": "COMPLETED",
                    "total_runs": len(results),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                last_dir / "dataset_complete.marker",
            )
        return results
