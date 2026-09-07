"""Worker execution function for running self-contained simulation jobs.

In accordance with Requirements 25, 28, 29:
- Workers have completely independent RNG states initialized from job.seed.
- Single worker crashes are captured, recorded with tracebacks, and never crash the scheduler.
"""

from __future__ import annotations

import logging
import time
import traceback

import numpy as np

from qism.execution.job import JobResult, SimulationJob
from qism.simulation.runner import RunManager
from qism.types import DatasetStatus, JobStatus

log = logging.getLogger("qism.execution.worker")


def execute_simulation_job(job: SimulationJob) -> JobResult:
    """Execute a single SimulationJob in a worker process."""
    t0 = time.perf_counter()
    try:
        # Override config master_seed with the job's deterministic seed
        cfg = job.config
        cfg.simulation.master_seed = job.seed
        cfg.simulation.scenario_id = job.scenario_id

        # Physical validation guard before running (Requirements 20, 70)
        from qism.population.validator import validate_scenario_config
        val_errors = validate_scenario_config(cfg)
        if val_errors:
            raise ValueError(f"Scenario {job.scenario_id} failed physical validation: {'; '.join(val_errors)}")

        # Instantiate RunManager using the job's shard output root with scenario isolation
        ckpt_dir = job.checkpoint_dir / job.scenario_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        manager = RunManager(
            cfg,
            job.run_id,
            output_root=str(job.output_dir),
            checkpoint_root=str(ckpt_dir),
        )

        # Run simulation with crash recovery enabled
        summary = manager.run(resume=True, fresh=False)
        wall_time = time.perf_counter() - t0

        if summary.status == DatasetStatus.COMPLETED and not summary.issues:
            status = JobStatus.COMPLETED
            err_type = None
            err_msg = None
            stack = None
        else:
            status = JobStatus.FAILED
            err_type = "DataQualityError" if summary.issues else "IncompleteRunError"
            err_msg = "; ".join(summary.issues) if summary.issues else f"Status was {summary.status}"
            stack = None

        n_events = summary.pulses_completed
        rate = n_events / wall_time if wall_time > 0 else 0.0

        sum_dict = summary.to_dict()
        sum_dict["sifted_qber"] = summary.qber
        try:
            df_win = manager.writer.read_final_telemetry()
            if df_win is not None and not df_win.empty:
                for det_col in ("n_detection", "total_clicks", "raw_detections"):
                    if det_col in df_win.columns:
                        sum_dict["raw_detections"] = int(df_win[det_col].sum())
                        break
                if "n_dark_0" in df_win.columns and "n_dark_1" in df_win.columns:
                    sum_dict["dark_counts"] = int(df_win["n_dark_0"].sum() + df_win["n_dark_1"].sum())
                else:
                    for dark_col in ("n_dark", "dark_counts"):
                        if dark_col in df_win.columns:
                            sum_dict["dark_counts"] = int(df_win[dark_col].sum())
                            break
                if "dark_counts" not in sum_dict and "dark_count_rate" in df_win.columns:
                    sum_dict["dark_counts"] = int(round(float(df_win["dark_count_rate"].mean()) * summary.pulses_completed))
        except Exception:
            pass
        if "raw_detections" not in sum_dict:
            gain = summary.signal_gain if (summary.signal_gain is not None and not np.isnan(summary.signal_gain)) else 0.0
            sum_dict["raw_detections"] = int(gain * summary.pulses_completed)
        if "dark_counts" not in sum_dict:
            sum_dict["dark_counts"] = 0

        return JobResult(
            scenario_id=job.scenario_id,
            run_id=job.run_id,
            seed=job.seed,
            status=status,
            summary=sum_dict,
            error_type=err_type,
            error_message=err_msg,
            stack_trace=stack,
            wall_time_s=wall_time,
            events_generated=n_events,
            events_per_sec=rate,
        )

    except Exception as exc:
        wall_time = time.perf_counter() - t0
        err_type = type(exc).__name__
        err_msg = str(exc)
        stack = traceback.format_exc()
        log.error("Worker error in job %s: %s", job.job_id, exc)

        return JobResult(
            scenario_id=job.scenario_id,
            run_id=job.run_id,
            seed=job.seed,
            status=JobStatus.FAILED,
            summary=None,
            error_type=err_type,
            error_message=err_msg,
            stack_trace=stack,
            wall_time_s=wall_time,
            events_generated=0,
            events_per_sec=0.0,
        )
