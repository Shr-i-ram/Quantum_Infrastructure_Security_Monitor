"""Job scheduler for parallel and serial batch execution.

In accordance with Requirements 26–32, 72–75:
- Manages ProcessPoolExecutor or serial execution.
- Maintains job_manifest.json for atomic checkpointing, resume, and failure logging.
- Retries transient worker failures up to max_retries.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import pathlib
import time
from typing import Callable

from tqdm import tqdm

from qism.execution.job import JobResult, SimulationJob
from qism.execution.worker import execute_simulation_job
from qism.types import JobStatus

log = logging.getLogger("qism.execution.scheduler")


class JobScheduler:
    """Schedules and coordinates execution of simulation jobs with persistent state."""

    def __init__(
        self,
        jobs: list[SimulationJob],
        state_file: pathlib.Path | str | None = None,
        worker_count: int = 4,
        parallel: bool = True,
        max_retries: int = 2,
    ):
        self.jobs = jobs
        self.state_file = pathlib.Path(state_file or "checkpoints/job_manifest.json")
        self.worker_count = max(1, worker_count)
        self.parallel = parallel and (self.worker_count > 1)
        self.max_retries = max_retries
        self.state: dict[str, dict] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load existing execution state from state_file if resuming."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        if "jobs" in data and isinstance(data["jobs"], dict):
                            self.state = data["jobs"]
                        else:
                            self.state = {k: v for k, v in data.items() if isinstance(v, dict) and "status" in v}
                    log.info("Loaded execution state for %d jobs from %s", len(self.state), self.state_file)
            except Exception as exc:
                log.warning("Could not load state from %s: %s; starting fresh", self.state_file, exc)
                self.state = {}

    def _save_state(self) -> None:
        """Atomically persist execution state to state_file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        completed = sum(1 for j in self.state.values() if isinstance(j, dict) and j.get("status") == JobStatus.COMPLETED.value)
        failed = sum(1 for j in self.state.values() if isinstance(j, dict) and j.get("status") == JobStatus.FAILED.value)

        payload = {
            "total_jobs": len(self.jobs),
            "completed": completed,
            "failed": failed,
            "jobs": self.state,
            **self.state,
        }
        tmp_file = self.state_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        tmp_file.replace(self.state_file)

    def summary(self) -> dict[str, int]:
        """Return execution progress summary counts."""
        completed = sum(1 for j in self.state.values() if isinstance(j, dict) and j.get("status") == JobStatus.COMPLETED.value)
        failed = sum(1 for j in self.state.values() if isinstance(j, dict) and j.get("status") == JobStatus.FAILED.value)
        total = len(self.jobs)
        pending = max(0, total - completed - failed)
        return {"total": total, "completed": completed, "failed": failed, "pending": pending}

    def run(
        self,
        progress_callback: Callable[[JobResult], None] | None = None,
        mode: str | None = None,
    ) -> list[JobResult]:
        """Execute all pending jobs using configured execution mode."""
        if mode == "serial":
            self.parallel = False
        elif mode == "parallel":
            self.parallel = True and (self.worker_count > 1)

        # 1. Filter jobs: skip already completed runs
        pending_jobs: list[SimulationJob] = []
        results: list[JobResult] = []

        for job in self.jobs:
            jid = job.job_id
            st = self.state.get(jid, {})
            status = st.get("status")
            if status == JobStatus.COMPLETED.value:
                # Job previously finished
                results.append(
                    JobResult(
                        scenario_id=job.scenario_id,
                        run_id=job.run_id,
                        seed=job.seed,
                        status=JobStatus.COMPLETED,
                        summary=st.get("summary"),
                        wall_time_s=st.get("wall_time_s", 0.0),
                        events_generated=st.get("events_generated", 0),
                    )
                )
            else:
                pending_jobs.append(job)

        log.info(
            "Scheduler starting: %d total jobs (%d completed, %d pending, mode=%s, workers=%d)",
            len(self.jobs),
            len(results),
            len(pending_jobs),
            "parallel" if self.parallel else "serial",
            self.worker_count if self.parallel else 1,
        )

        if not pending_jobs:
            return results

        pbar = tqdm(total=len(self.jobs), initial=len(results), desc="Simulations", unit="run")

        # 2. Execution loop
        if not self.parallel:
            # Serial execution
            for job in pending_jobs:
                res = self._execute_with_retry(job)
                self._record_result(res)
                results.append(res)
                pbar.update(1)
                if progress_callback:
                    progress_callback(res)
        else:
            # Parallel execution via ProcessPoolExecutor
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.worker_count) as executor:
                active_futures = {executor.submit(execute_simulation_job, j): j for j in pending_jobs}
                while active_futures:
                    done, _ = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        job = active_futures.pop(future)
                        try:
                            res = future.result()
                        except Exception as exc:
                            res = JobResult(
                                scenario_id=job.scenario_id,
                                run_id=job.run_id,
                                seed=job.seed,
                                status=JobStatus.FAILED,
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                            )

                        # Handle retry if failed
                        # Requirement 73: Deterministic failure detection
                        is_deterministic_err = res.error_type in ("ValueError", "DataQualityError", "ValidationError")
                        if not res.is_success and not is_deterministic_err and job.retry_count < self.max_retries:
                            job.retry_count += 1
                            log.warning("Retrying failed job %s (attempt %d/%d)", job.job_id, job.retry_count, self.max_retries)
                            fut = executor.submit(execute_simulation_job, job)
                            active_futures[fut] = job
                        else:
                            self._record_result(res)
                            results.append(res)
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(res)

        pbar.close()
        self._save_state()
        return results

    def _execute_with_retry(self, job: SimulationJob) -> JobResult:
        """Execute a job serially with retry handling."""
        while True:
            res = execute_simulation_job(job)
            is_deterministic_err = res.error_type in ("ValueError", "DataQualityError", "ValidationError")
            if res.is_success or is_deterministic_err or job.retry_count >= self.max_retries:
                return res
            job.retry_count += 1
            log.warning("Retrying failed job %s (attempt %d/%d)", job.job_id, job.retry_count, self.max_retries)

    def _record_result(self, res: JobResult) -> None:
        """Record result in state memory and checkpoint manifest."""
        self.state[res.job_id] = {
            "scenario_id": res.scenario_id,
            "run_id": res.run_id,
            "seed": res.seed,
            "status": res.status.value,
            "summary": res.summary,
            "error_type": res.error_type,
            "error_message": res.error_message,
            "wall_time_s": res.wall_time_s,
            "events_generated": res.events_generated,
        }
        self._save_state()
