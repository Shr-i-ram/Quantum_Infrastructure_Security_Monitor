#!/usr/bin/env python3
"""Generate the QISM V0.3 attack-aware dataset.

Simulates attack scenarios across all 6 causal attack types and severity sweeps.
Writes per-run and dataset-level completion markers.

Examples:
    python scripts/generate_attacks.py --config configs/v0.3_attack.yaml
    python scripts/generate_attacks.py --config configs/v0.3_smoke.yaml
    python scripts/generate_attacks.py --scenarios ir_sev_06 loss_sev_05 --pulses 20000
"""

from __future__ import annotations

import argparse
import sys

from _common import PROJECT_ROOT, setup


def default_attack_config_path():
    return PROJECT_ROOT / "configs" / "v0.3_attack.yaml"


def main() -> int:
    setup("generate_attacks")

    parser = argparse.ArgumentParser(description="QISM V0.3 attack dataset generator")
    parser.add_argument(
        "--config", default=str(default_attack_config_path()), help="Attack YAML configuration"
    )
    parser.add_argument("--pulses", type=int, default=None, help="Override pulses per run")
    parser.add_argument("--seed", type=int, default=None, help="Override simulation.master_seed")
    parser.add_argument(
        "--scenarios", nargs="+", default=None, help="Specific scenario IDs or types to run"
    )
    parser.add_argument("--output-dir", default=None, help="Override simulation.output_root")
    parser.add_argument(
        "--checkpoint-dir", default=None, help="Override simulation.checkpoint_root"
    )
    parser.add_argument("--resume", action="store_true", help="Resume interrupted runs")
    parser.add_argument("--fresh", action="store_true", help="Discard existing runs and restart")
    args = parser.parse_args()

    from qism.config import apply_overrides, load_attack_config
    from qism.datasets.generator import AttackDatasetGenerator

    atk_cfg = load_attack_config(args.config)
    overrides = {}
    if args.seed is not None:
        overrides["simulation.master_seed"] = args.seed
    if overrides:
        atk_cfg.base = apply_overrides(atk_cfg.base, overrides)

    generator = AttackDatasetGenerator(
        atk_cfg,
        output_root=args.output_dir,
        checkpoint_root=args.checkpoint_dir,
    )
    results = generator.generate(
        pulses_per_run=args.pulses,
        scenarios=args.scenarios,
        fresh=args.fresh,
        resume=args.resume,
    )

    print(f"\nattack dataset: {len(results)} runs")
    n_failed = sum(1 for r in results if r.issues)
    print(f"  completed: {len(results) - n_failed}, failed quality checks: {n_failed}")
    for r in results:
        print(
            f"  {r.run_id:<24s} status={r.status.value:<10s} pulses={r.pulses_completed} "
            f"QBER={r.qber:.6f}"
        )
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
