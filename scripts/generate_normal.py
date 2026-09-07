#!/usr/bin/env python3
"""Generate the QISM V0.2 normal operational dataset.

Multiple independent runs; each run uses a deterministic seed derived from
(master_seed, run_id). No global randomness is used.

Examples:
    python scripts/generate_normal.py --config configs/normal.yaml
    python scripts/generate_normal.py --runs 5 --pulses 10000
"""

from __future__ import annotations

import argparse
import sys

from _common import default_config_path, setup


def main() -> int:
    setup("generate_normal")

    parser = argparse.ArgumentParser(description="QISM V0.2 normal dataset generator")
    parser.add_argument(
        "--config", default=str(default_config_path()), help="Normal YAML configuration"
    )
    parser.add_argument("--runs", type=int, default=None, help="Override number of runs")
    parser.add_argument("--pulses", type=int, default=None, help="Override pulses per run")
    parser.add_argument("--seed", type=int, default=None, help="Override simulation.master_seed")
    parser.add_argument("--start-index", type=int, default=0, help="First run index (default 0)")
    parser.add_argument("--output-dir", default=None, help="Override simulation.output_root")
    parser.add_argument(
        "--checkpoint-dir", default=None, help="Override simulation.checkpoint_root"
    )
    parser.add_argument("--resume", action="store_true", help="Resume interrupted runs")
    parser.add_argument("--fresh", action="store_true", help="Discard existing runs and restart")
    args = parser.parse_args()

    from qism.config import apply_overrides, load_normal_config
    from qism.datasets.generator import NormalGenerator

    normal = load_normal_config(args.config)
    overrides = {}
    if args.seed is not None:
        overrides["simulation.master_seed"] = args.seed
    normal.base = apply_overrides(normal.base, overrides)

    generator = NormalGenerator(
        normal,
        output_root=args.output_dir,
        checkpoint_root=args.checkpoint_dir,
    )
    results = generator.generate(
        runs=args.runs,
        pulses_per_run=args.pulses,
        fresh=args.fresh,
        resume=args.resume,
        start_index=args.start_index,
    )

    print(f"\nnormal dataset: {len(results)} runs")
    n_failed = sum(1 for r in results if r.issues)
    print(f"  completed: {len(results) - n_failed}, failed quality checks: {n_failed}")
    for r in results:
        print(
            f"  {r.run_id:<12s} status={r.status.value:<10s} pulses={r.pulses_completed} "
            f"QBER={r.qber:.6f}"
        )
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
