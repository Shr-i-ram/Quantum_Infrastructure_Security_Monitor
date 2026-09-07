#!/usr/bin/env python3
"""Run the QISM validation suite (V0.1 analytical + V0.2 detector dynamics).

Generates validation figures and a report under reports/:

    reports/figures/01_poisson_photon_number.png ... 08_signal_decoy_vacuum_gains.png
    reports/figures/09_dead_time_rate_suppression.png
    reports/figures/10_interarrival_dead_time_cutoff.png
    reports/figures/11_afterpulse_delay_distribution.png
    reports/figures/12_rate_dependent_dead_time.png
    reports/figures/13_qkd_engine_detector_dynamics.png
    reports/validation_report.md
    reports/validation_summary.json

Example:
    python scripts/run_validation.py --config configs/default.yaml --pulses 200000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from _common import default_config_path, setup


def main() -> int:
    setup("run_validation")

    parser = argparse.ArgumentParser(
        description="QISM validation (V0.1 analytical + V0.2 detector)"
    )
    parser.add_argument(
        "--config", default=str(default_config_path()), help="Base YAML configuration"
    )
    parser.add_argument("--pulses", type=int, default=200_000, help="Pulses per validation run")
    parser.add_argument("--seed", type=int, default=20240101, help="Validation master seed")
    parser.add_argument("--outdir", default="reports", help="Output directory for figures/report")
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip figure generation (stats only)"
    )
    args = parser.parse_args()

    from qism.config import load_config
    from qism.validation.statistics import (
        generate_detector_validation_report,
        generate_validation_report,
    )

    config = load_config(args.config)
    outdir = pathlib.Path(args.outdir)

    report = generate_validation_report(
        config,
        outdir=outdir,
        n_pulses=args.pulses,
        seed=args.seed,
    )
    det_report = generate_detector_validation_report(
        config,
        outdir=outdir,
        seed=args.seed,
        n_qkd_pulses=max(args.pulses, 300_000),
    )
    report["detector_validation"] = det_report

    # re-render the markdown with the detector section merged in
    from qism.validation.statistics import _render_markdown

    md = _render_markdown(report)
    (outdir / "validation_report.md").write_text(md, encoding="utf-8")
    (outdir / "validation_summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    n_figs = len(report["figures"]) + len(det_report["figures"])
    print(
        f"\nvalidation complete: {n_figs} figures, "
        f"{len(report['results']) + len(det_report) - 1} result entries"
    )
    print(f"  figures  : {outdir / 'figures'}")
    print(f"  report   : {outdir / 'validation_report.md'}")
    print("\nV0.1 results:")
    for key, value in report["results"].items():
        print(f"  {key:36s}: {value}")
    print("\nV0.2 detector-dynamics results:")
    for key, value in det_report.items():
        if key != "figures":
            print(f"  {key:36s}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
