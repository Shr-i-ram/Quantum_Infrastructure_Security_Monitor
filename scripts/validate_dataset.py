#!/usr/bin/env python3
"""Validate generated QISM datasets for integrity, schema adherence, and zero label leakage.

Checks:
  1. Completion marker existence (dataset_complete.marker).
  2. Manifest integrity and status == 'completed'.
  3. Parquet files schema adherence.
  4. Zero label leakage: verifies observable telemetry features contain no ground truth columns.
  5. Scenario-level splitting validation between datasets.

Examples:
    python scripts/validate_dataset.py data/attacks
    python scripts/validate_dataset.py data/attacks data/generalization --check-split
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from _common import setup


def validate_single_dataset(dataset_dir: pathlib.Path) -> list[str]:
    """Validate a dataset folder."""
    issues: list[str] = []
    if not dataset_dir.exists():
        return [f"Directory does not exist: {dataset_dir}"]

    # 1. Completion marker check
    marker_path = dataset_dir / "dataset_complete.marker"
    if not marker_path.exists():
        # Check subdirectories for run markers
        sub_markers = list(dataset_dir.glob("*/dataset_complete.marker"))
        if not sub_markers:
            issues.append(f"No dataset_complete.marker found in {dataset_dir} or its run subdirectories")
    else:
        try:
            with marker_path.open("r", encoding="utf-8") as fh:
                marker_data = json.load(fh)
                if marker_data.get("status") != "COMPLETED":
                    issues.append(f"Marker status in {marker_path} is {marker_data.get('status')}, expected COMPLETED")
        except Exception as exc:
            issues.append(f"Failed to read completion marker {marker_path}: {exc}")

    # 2. Find all run directories (having manifest.json or parquet files)
    run_dirs = [p.parent for p in dataset_dir.glob("**/manifest.json")]
    if not run_dirs:
        # Check if dataset_dir itself is a run dir
        if (dataset_dir / "telemetry.parquet").exists() or (dataset_dir / "events.parquet").exists():
            run_dirs = [dataset_dir]

    if not run_dirs:
        issues.append(f"No runs found in {dataset_dir}")
        return issues

    import pandas as pd
    from qism.telemetry.features import (
        GROUND_TRUTH_WINDOW_COLUMNS,
        get_ground_truth_feature_names,
        get_observable_feature_names,
    )

    observable_names = set(get_observable_feature_names())
    gt_names = set(get_ground_truth_feature_names()) | set(GROUND_TRUTH_WINDOW_COLUMNS)

    # Check zero label leakage in definition
    overlap = observable_names.intersection(gt_names)
    if overlap:
        issues.append(f"CRITICAL: Observable feature set leaks ground truth columns: {overlap}")

    for rdir in run_dirs:
        m_file = rdir / "manifest.json"
        if m_file.exists():
            try:
                with m_file.open("r", encoding="utf-8") as fh:
                    m_data = json.load(fh)
                    if str(m_data.get("status", "")).lower() not in ("completed", "checkpointed"):
                        issues.append(f"Run {rdir.name} manifest status is {m_data.get('status')}")
            except Exception as exc:
                issues.append(f"Run {rdir.name} manifest read error: {exc}")

        telem_file = rdir / "telemetry.parquet"
        if telem_file.exists():
            try:
                df = pd.read_parquet(telem_file)
                cols = set(df.columns)
                # Verify that observable columns exist
                missing_obs = [col for col in observable_names if col not in cols]
                if missing_obs:
                    issues.append(f"Run {rdir.name} telemetry missing observable features: {missing_obs[:5]}")

                # Verify observable feature separation
                # (any column that starts with ground_truth_ must be in gt_names, never treated as observable)
                for col in cols:
                    if col.startswith("ground_truth_") and col in observable_names:
                        issues.append(f"Run {rdir.name} observable feature leaks ground truth: {col}")
            except Exception as exc:
                issues.append(f"Failed to read {telem_file}: {exc}")

    return issues


def validate_scenario_split(train_dir: pathlib.Path, test_dir: pathlib.Path) -> list[str]:
    """Verify scenario-level splitting: no scenario ID in test_dir appears in train_dir."""
    issues: list[str] = []
    import json

    train_scenarios = set()
    for m in train_dir.glob("**/manifest.json"):
        try:
            with m.open("r", encoding="utf-8") as fh:
                d = json.load(fh)
                sc = d.get("attack_scenario") or d.get("scenario")
                if sc and sc != "normal":
                    train_scenarios.add(sc)
        except Exception:
            pass

    test_scenarios = set()
    for m in test_dir.glob("**/manifest.json"):
        try:
            with m.open("r", encoding="utf-8") as fh:
                d = json.load(fh)
                sc = d.get("attack_scenario") or d.get("scenario")
                if sc and sc != "normal":
                    test_scenarios.add(sc)
        except Exception:
            pass

    leak = train_scenarios.intersection(test_scenarios)
    if leak:
        issues.append(f"Scenario leakage detected! Scenarios present in both train and test: {leak}")

    return issues


def main() -> int:
    setup("validate_dataset")

    parser = argparse.ArgumentParser(description="QISM V0.3 dataset validator")
    parser.add_argument("directories", nargs="+", help="Dataset directory/directories to validate")
    parser.add_argument(
        "--check-split",
        action="store_true",
        help="If two directories provided, check scenario-level splitting integrity",
    )
    args = parser.parse_args()

    all_issues: list[str] = []
    dirs = [pathlib.Path(d) for d in args.directories]

    for d in dirs:
        print(f"Validating dataset directory: {d} ...")
        issues = validate_single_dataset(d)
        if issues:
            print(f"  FAILED with {len(issues)} issues:")
            for issue in issues:
                print(f"    - {issue}")
            all_issues.extend(issues)
        else:
            print("  PASSED (schema verified, zero label leakage, completion marker OK)")

    if args.check_split and len(dirs) >= 2:
        print(f"\nChecking scenario-level split between {dirs[0]} and {dirs[1]} ...")
        split_issues = validate_scenario_split(dirs[0], dirs[1])
        if split_issues:
            print(f"  SPLIT VIOLATION with {len(split_issues)} issues:")
            for issue in split_issues:
                print(f"    - {issue}")
            all_issues.extend(split_issues)
        else:
            print("  PASSED (Zero scenario leakage between training and generalization)")

    return 1 if all_issues else 0


if __name__ == "__main__":
    sys.exit(main())
