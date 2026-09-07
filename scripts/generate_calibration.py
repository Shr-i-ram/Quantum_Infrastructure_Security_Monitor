#!/usr/bin/env python3
"""Generate the QISM V0.2 calibration dataset (parameter sweeps).

Each run varies exactly one physical parameter; every run carries a manifest
with varied_parameter / varied_value and the full base configuration.

Example:
    python scripts/generate_calibration.py --config configs/calibration.yaml
    python scripts/generate_calibration.py --pulses 20000 --only distance_km
"""

from __future__ import annotations

import argparse
import sys

from _common import default_config_path, setup


def main() -> int:
    setup("generate_calibration")

    parser = argparse.ArgumentParser(description="QISM V0.2 calibration dataset generator")
    parser.add_argument(
        "--config", default=str(default_config_path()), help="Calibration YAML configuration"
    )
    parser.add_argument(
        "--pulses", type=int, default=None, help="Override pulses per calibration run"
    )
    parser.add_argument("--seed", type=int, default=None, help="Override simulation.master_seed")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict to sweep labels, e.g. --only distance_km mu_signal",
    )
    parser.add_argument("--output-dir", default=None, help="Override simulation.output_root")
    parser.add_argument(
        "--checkpoint-dir", default=None, help="Override simulation.checkpoint_root"
    )
    parser.add_argument("--resume", action="store_true", help="Resume interrupted runs")
    parser.add_argument("--fresh", action="store_true", help="Discard existing runs and restart")
    args = parser.parse_args()

    from qism.config import apply_overrides, load_calibration_config
    from qism.datasets.generator import CalibrationGenerator

    calib = load_calibration_config(args.config)
    overrides = {}
    if args.seed is not None:
        overrides["simulation.master_seed"] = args.seed
    calib.base = apply_overrides(calib.base, overrides)

    generator = CalibrationGenerator(
        calib,
        output_root=args.output_dir,
        checkpoint_root=args.checkpoint_dir,
    )
    results = generator.generate(
        fresh=args.fresh,
        resume=args.resume,
        pulses_per_run=args.pulses,
        only=args.only,
    )

    print(f"\ncalibration dataset complete: {len(results)} runs")
    n_failed = sum(1 for r in results if r.issues)
    print(f"  completed: {len(results) - n_failed}, failed quality checks: {n_failed}")
    for r in results:
        print(f"  {r.run_id:<40s} status={r.status.value:<10s} pulses={r.pulses_completed}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
