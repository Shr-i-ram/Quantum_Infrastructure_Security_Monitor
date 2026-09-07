#!/usr/bin/env python3
"""Run large-scale V0.5 dataset generation across population scenarios."""

import argparse
import sys
from _common import setup

setup("generate_v05_dataset")

from qism.config import load_v05_dataset_config
from qism.datasets.pipeline import DatasetPipeline


def main():
    parser = argparse.ArgumentParser(description="QISM V0.5 Dataset Pipeline Generator")
    parser.add_argument("config", help="Path to V0.5 dataset YAML configuration")
    parser.add_argument("--workers", type=int, default=None, help="Override parallel worker count")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete dataset generation")
    parser.add_argument("--fresh", action="store_true", help="Wipe existing dataset and restart")
    args = parser.parse_args()

    cfg = load_v05_dataset_config(args.config)
    if args.workers is not None:
        cfg.worker_count = int(args.workers)

    pipeline = DatasetPipeline(cfg)
    summary = pipeline.run(resume=args.resume, fresh=args.fresh)

    print("\n" + "=" * 70)
    print(f"Dataset Pipeline Run Complete — '{summary['dataset_id']}'")
    print("=" * 70)
    print(f"Output Directory : {summary['output_dir']}")
    print(f"Total Scenarios  : {summary['total_scenarios']}")
    print(f"Completed Runs   : {summary['completed_runs']}")
    print(f"Failed Runs      : {summary['failed_runs']}")
    print(f"Execution Time   : {summary['wall_time_s']:.2f} s")
    print("=" * 70)
    return 0 if summary["failed_runs"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
