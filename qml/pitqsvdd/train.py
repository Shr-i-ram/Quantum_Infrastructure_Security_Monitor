"""Command-line entry point for training and resuming PI-TQSVDD models.

Usage examples:
    python -m qml.pitqsvdd.train --config qml/pitqsvdd/configs/pilot.yaml --run-id pilot_run --resume auto
    python -m qml.pitqsvdd.train --config qml/pitqsvdd/configs/pitqsvdd_final.yaml --run-id pitqsvdd_final --resume auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qml.pitqsvdd.config import PITQSVDDConfig
from qml.pitqsvdd.training.experiment_runner import PITQSVDDExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or resume PI-TQSVDD models for QKD anomaly detection."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="qml/pitqsvdd/configs/pilot.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Experiment identifier (overrides config run_id if specified).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        choices=["auto", "none", "explicit"],
        default=None,
        help="Resume mode: 'auto' (latest checkpoint), 'none' (start fresh), 'explicit' (use --resume-path).",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Explicit path to checkpoint file when --resume explicit is specified.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing experiment outputs if starting fresh.",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        choices=[2, 4, 6, 8],
        default=None,
        help="Override qubit count in configuration (2, 4, 6, 8).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="Override quantum encoding method.",
    )

    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Error: Configuration file not found at {cfg_path}", file=sys.stderr)
        sys.exit(1)

    config = PITQSVDDConfig.from_yaml(cfg_path)

    # CLI Overrides
    if args.run_id:
        config.training.run_id = args.run_id
    if args.resume:
        config.training.resume_mode = args.resume
    if args.resume_path:
        config.training.resume_path = args.resume_path
    if args.overwrite:
        config.training.overwrite = True
    if args.qubits:
        config.circuit.n_qubits = args.qubits
        config.data.n_qubits = args.qubits
    if args.encoding:
        config.circuit.encoding_type = args.encoding

    runner = PITQSVDDExperimentRunner(artifacts_dir=config.evaluation.output_dir)
    res = runner.run_experiment(config)

    print("\n[Done] Experiment finished successfully.")
    print("Summary:")
    for k, v in res.get("summary", {}).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
