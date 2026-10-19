#!/usr/bin/env python3
"""Automated presentation-quality visualization generator for QISM V0.1–V0.5.

Generates:
    - 15 curated presentation figures under outputs/visualizations/presentation/
    - Exploratory diagnostic figures under outputs/visualizations/exploratory/
    - Provenance metadata under outputs/visualizations/metadata/figure_metadata.json
    - Technical Markdown report under outputs/visualizations/report/visualization_report.md

Usage:
    python scripts/generate_visualizations.py
    python scripts/generate_visualizations.py --dataset data/v0.5_small/v05_small_dataset --outdir outputs/visualizations
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

# Ensure src is on sys.path
_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qism import verify_python_version
from qism.visualization import VisualizationConfig, generate_all_visualizations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("qism.generate_visualizations")


def main() -> int:
    verify_python_version()

    parser = argparse.ArgumentParser(
        description="Generate presentation-quality visualizations for QISM V0.1–V0.5 simulator and dataset."
    )
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=pathlib.Path("data/v0.5_small/v05_small_dataset"),
        help="Path to input dataset root directory containing partitions (calibration, normal, attack, generalization).",
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/visualizations"),
        help="Root output directory for generated figures and report.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png",
        help="Comma-separated image formats to generate (e.g. png,svg,pdf).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Target DPI for rasterized figures (default: 300).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing output directory before generating figures.",
    )

    args = parser.parse_args()

    format_list = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    config = VisualizationConfig(
        dataset_path=args.dataset,
        output_root=args.outdir,
        formats=format_list,
        dpi=args.dpi,
        clean_output=args.clean,
    )

    if not config.dataset_path.exists():
        log.error("Dataset path '%s' does not exist.", config.dataset_path)
        return 1

    print("\n" + "=" * 78)
    print("QISM V0.1–V0.5 Technical Presentation Visualization Generator")
    print("=" * 78)
    print(f"Dataset Path : {config.dataset_path}")
    print(f"Output Root  : {config.output_root}")
    print(f"Formats      : {config.formats}")
    print(f"Target DPI   : {config.dpi}")
    print("=" * 78 + "\n")

    t0 = time.perf_counter()
    summary = generate_all_visualizations(config)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 78)
    print("Generation Summary")
    print("=" * 78)
    print(f"Status                  : {summary['status']}")
    print(f"Total Presentation Figs : {summary['total_presentation_figures']}")
    print(f"Presentation Directory  : {summary['presentation_dir']}")
    print(f"Exploratory Directory   : {summary['exploratory_dir']}")
    print(f"Metadata File           : {summary['metadata_file']}")
    print(f"Presentation Report     : {summary['report_file']}")
    print(f"Execution Time          : {elapsed:.2f} s")
    print("=" * 78 + "\n")

    print("Curated Presentation Figures Generated:")
    for fig in summary["figures"]:
        print(f"  [{fig['id']}] {fig['filename']:<42} : {fig['title']}")
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
