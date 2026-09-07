#!/usr/bin/env python3
"""Inspect a QISM run dataset directory.

Prints the manifest summary, file inventory, event schema summary, and key
telemetry statistics.

Example:
    python scripts/inspect_dataset.py --path data/normal/run_0000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pyarrow.parquet as pq

from _common import setup


def main() -> int:
    setup("inspect_dataset")

    parser = argparse.ArgumentParser(description="Inspect a QISM run dataset")
    parser.add_argument("--path", required=True, help="Run dataset directory")
    parser.add_argument("--events-head", type=int, default=5, help="Rows of events to preview")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.path)
    if not run_dir.is_dir():
        print(f"error: {run_dir} is not a directory", file=sys.stderr)
        return 1

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        print("=== manifest ===")
        for key in (
            "run_id",
            "scenario",
            "family",
            "status",
            "pulses_completed",
            "number_of_pulses",
            "window_size",
            "configuration_hash",
            "simulator_version",
            "seed",
            "master_seed",
            "varied_parameter",
            "varied_value",
            "git_commit_if_available",
            "python_version",
        ):
            if key in manifest:
                print(f"  {key:24s}: {manifest[key]}")
    else:
        print("=== manifest: MISSING ===")

    print("\n=== files ===")
    for f in sorted(run_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:24s} {f.stat().st_size:>12,} bytes")

    ev_path = run_dir / "events.parquet"
    if ev_path.exists():
        table = pq.read_table(ev_path)
        print("\n=== events.parquet ===")
        print(f"  rows      : {table.num_rows:,}")
        print(f"  columns   : {table.num_columns}")
        print("  schema    :")
        for name, typ in zip(table.schema.names, table.schema.types):
            print(f"    {name:32s} {typ}")
        if args.events_head:
            print(f"\n  first {args.events_head} rows:")
            df = pq.read_table(ev_path).to_pandas()
            cols = [
                "pulse_id",
                "alice_bit",
                "alice_basis",
                "intensity_class",
                "mu",
                "photon_number",
                "bob_basis",
                "detector_0_click",
                "detector_1_click",
                "click_type",
                "sifted",
                "bit_error",
            ]
            print(df[cols].head(args.events_head).to_string(index=False))

    tel_path = run_dir / "telemetry.parquet"
    if tel_path.exists():
        df = pq.read_table(tel_path).to_pandas()
        print(f"\n=== telemetry.parquet ({len(df)} windows) ===")
        cols = [
            "window_id",
            "qber",
            "signal_gain",
            "decoy_gain",
            "vacuum_gain",
            "detector_imbalance",
            "double_click_rate",
            "sifted_count",
        ]
        print(df[cols].head(10).to_string(index=False))
        print("\n  window means:")
        means = [
            "qber",
            "signal_gain",
            "decoy_gain",
            "vacuum_gain",
            "detector_imbalance",
            "double_click_rate",
            "mean_arrival_offset",
            "std_arrival_offset",
            "timing_skewness",
            "timing_kurtosis",
        ]
        print(df[means].mean().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
