#!/usr/bin/env python3
"""Run a single QISM V0.2 simulation (with checkpointing / resume support).

Examples:
    python scripts/run_simulation.py --config configs/default.yaml
    python scripts/run_simulation.py --config configs/default.yaml --pulses 10000
    python scripts/run_simulation.py --config configs/default.yaml --resume
    python scripts/run_simulation.py --config configs/default.yaml --fresh
    python scripts/run_simulation.py --runs 5 --pulses 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from _common import default_config_path, setup

log = logging.getLogger("qism.cli")


def main() -> int:
    setup("run_simulation")

    parser = argparse.ArgumentParser(description="QISM V0.2 simulation runner")
    parser.add_argument(
        "--config", default=str(default_config_path()), help="YAML configuration file"
    )
    parser.add_argument("--pulses", type=int, default=None, help="Override protocol.n_pulses")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (indexed run_0000..)")
    parser.add_argument("--seed", type=int, default=None, help="Override simulation.master_seed")
    parser.add_argument("--run-id", default=None, help="Run identifier (default: scenario-based)")
    parser.add_argument("--scenario", default=None, help="Override simulation.scenario_id")
    parser.add_argument("--output-dir", default=None, help="Override simulation.output_root")
    parser.add_argument(
        "--checkpoint-dir", default=None, help="Override simulation.checkpoint_root"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Discard old checkpoints and restart")
    parser.add_argument("--summary", default=None, help="Write the run summary JSON to this path")
    args = parser.parse_args()

    from qism.config import apply_overrides, load_config
    from qism.simulation.runner import RunManager

    overrides = {}
    if args.pulses is not None:
        overrides["protocol.n_pulses"] = args.pulses
    if args.seed is not None:
        overrides["simulation.master_seed"] = args.seed
    if args.scenario is not None:
        overrides["simulation.scenario_id"] = args.scenario
    config = apply_overrides(load_config(args.config), overrides)

    default_run_id = args.run_id or f"{config.simulation.scenario_id}_run"
    run_ids = [f"run_{i:04d}" for i in range(args.runs)] if args.runs > 1 else [default_run_id]

    for rid in run_ids:
        manager = RunManager(
            config,
            rid,
            output_root=args.output_dir,
            checkpoint_root=args.checkpoint_dir,
        )
        summary = manager.run(resume=args.resume, fresh=args.fresh)

        if summary.status.value == "INTERRUPTED":
            print(f"run interrupted at pulse {summary.pulses_completed}; checkpoint preserved.")
            print(f"resume with: python scripts/run_simulation.py --config {args.config} --resume")
            return 130

        print(f"\nrun {summary.run_id}: status={summary.status.value}")
        print(f"  pulses completed  : {summary.pulses_completed}")
        print(f"  windows           : {summary.n_windows}")
        print(
            f"  QBER              : {summary.qber:.6f}  (Z: {summary.qber_z:.6f}, X: {summary.qber_x:.6f})"
        )
        print(
            f"  gains             : signal={summary.signal_gain:.6f} "
            f"decoy={summary.decoy_gain:.6f} vacuum={summary.vacuum_gain:.6g}"
        )
        print(f"  detector imbalance: {summary.detector_imbalance:+.6f}")
        print(f"  double-click rate : {summary.double_click_rate:.6g}")
        print(f"  output            : {summary.run_dir}")

        if args.summary:
            with open(args.summary, "w", encoding="utf-8") as fh:
                json.dump(summary.to_dict(), fh, indent=2, default=str)
            log.info("run summary written to %s", args.summary)

        if summary.issues:
            print(f"WARNING: {len(summary.issues)} data-quality issue(s); dataset marked FAILED")
            for issue in summary.issues:
                print(f"  - {issue}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
