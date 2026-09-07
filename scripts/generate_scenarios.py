#!/usr/bin/env python3
"""Generate, validate, and preview population scenarios without simulation."""

import argparse
import sys
from _common import setup

setup("generate_scenarios")

from qism.config import load_v05_dataset_config
from qism.datasets.pipeline import DatasetPipeline


def main():
    parser = argparse.ArgumentParser(description="QISM V0.5 Scenario Generator")
    parser.add_argument("config", help="Path to V0.5 dataset YAML configuration")
    args = parser.parse_args()

    cfg = load_v05_dataset_config(args.config)
    pipeline = DatasetPipeline(cfg)
    res = pipeline.dry_run()

    print("\n" + "=" * 70)
    print(f"QISM V0.5 Scenarios Generated & Validated — '{cfg.dataset_id}'")
    print("=" * 70)
    print(f"Total Scenarios  : {res['scenarios_generated']}")
    print(f"Total Runs       : {res['workload']['total_runs']}")
    print(f"Total Pulses     : {res['workload']['total_pulses']:,}")
    print(f"Sampling Method  : {res['workload']['sampling_strategy']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
