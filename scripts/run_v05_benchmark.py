#!/usr/bin/env python3
"""Scalability throughput benchmark comparing serial vs parallel execution (Requirements 76, 77, 198)."""

import argparse
import json
import os
import pathlib
import sys
import time

from _common import setup

setup("run_v05_benchmark")

from qism.config import load_v05_dataset_config
from qism.datasets.pipeline import DatasetPipeline


def main():
    parser = argparse.ArgumentParser(description="QISM V0.5 Scalability Benchmark")
    parser.add_argument("config", help="Path to V0.5 dataset YAML configuration")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker process count")
    args = parser.parse_args()

    cfg = load_v05_dataset_config(args.config)
    print("\n" + "=" * 70)
    print(f"QISM V0.5 Scalability Benchmark — '{cfg.dataset_id}'")
    print("=" * 70)

    # 1. Serial run (1 worker)
    cfg_s = load_v05_dataset_config(args.config)
    cfg_s.dataset_id = f"{cfg.dataset_id}_bench_serial"
    cfg_s.worker_count = 1
    p_s = DatasetPipeline(cfg_s)
    t0 = time.perf_counter()
    res_s = p_s.run(fresh=True)
    time_s = time.perf_counter() - t0

    # 2. Parallel run (N workers)
    cfg_p = load_v05_dataset_config(args.config)
    cfg_p.dataset_id = f"{cfg.dataset_id}_bench_parallel"
    cfg_p.worker_count = args.workers
    p_p = DatasetPipeline(cfg_p)
    t0 = time.perf_counter()
    res_p = p_p.run(fresh=True)
    time_p = time.perf_counter() - t0

    speedup = time_s / time_p if time_p > 0 else 1.0
    runs_completed = res_s["completed_runs"]
    runs_per_sec_s = runs_completed / time_s if time_s > 0 else 0.0
    runs_per_sec_p = runs_completed / time_p if time_p > 0 else 0.0

    print(f"Serial Execution (1 worker)        : {time_s:.2f} s ({runs_per_sec_s:.2f} runs/s)")
    print(f"Parallel Execution ({args.workers} workers)    : {time_p:.2f} s ({runs_per_sec_p:.2f} runs/s)")
    print(f"Measured Speedup                   : {speedup:.2f}x")
    print("=" * 70)

    out_dir = pathlib.Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "v05_benchmark.json"
    with open(report_file, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset_id": cfg.dataset_id,
                "total_runs": runs_completed,
                "serial_time_s": time_s,
                "serial_runs_per_sec": runs_per_sec_s,
                "parallel_workers": args.workers,
                "parallel_time_s": time_p,
                "parallel_runs_per_sec": runs_per_sec_p,
                "speedup": speedup,
            },
            fh,
            indent=2,
        )
    print(f"Benchmark report saved to {report_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
