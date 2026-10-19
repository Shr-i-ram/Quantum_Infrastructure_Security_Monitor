"""Unit and integration tests for the QISM visualization subsystem (V0.1–V0.5)."""

from __future__ import annotations

import pathlib
import pytest
import numpy as np
import pandas as pd

from qism.visualization import (
    VisualizationConfig,
    generate_all_visualizations,
)
from qism.visualization.loaders import (
    discover_runs,
    load_dataset_manifest,
    load_scenario_manifest,
    load_split_telemetry,
    validate_required_columns,
)
from qism.visualization.statistics import (
    compute_correlation_matrix,
    compute_distribution_summary,
    evaluate_dead_time_saturation,
    evaluate_theoretical_transmission,
)
from qism.visualization.style import save_figure, apply_presentation_style
from qism.visualization.qkd_plots import plot_simulation_pipeline, plot_basis_sifting_funnel
from qism.visualization.channel_plots import plot_channel_transmission
from qism.visualization.detector_plots import plot_detector_physics
from qism.visualization.attack_plots import plot_multi_attack_signatures
from qism.visualization.dataset_plots import plot_dataset_composition


@pytest.fixture
def dataset_path() -> pathlib.Path:
    p = pathlib.Path("data/v0.5_small/v05_small_dataset")
    if not p.exists():
        pytest.skip(f"Test dataset not found at {p}")
    return p


def test_visualization_config(tmp_path: pathlib.Path):
    cfg = VisualizationConfig(output_root=tmp_path / "vis_test")
    cfg.ensure_directories()

    assert cfg.presentation_dir.exists()
    assert cfg.exploratory_dir.exists()
    assert cfg.metadata_dir.exists()
    assert cfg.report_dir.exists()


def test_loaders_manifest_and_runs(dataset_path: pathlib.Path):
    d_manifest = load_dataset_manifest(dataset_path)
    assert d_manifest.get("dataset_id") == "v05_small_dataset"
    assert "splits" in d_manifest

    s_manifest = load_scenario_manifest(dataset_path)
    assert "scenarios" in s_manifest
    assert len(s_manifest["scenarios"]) == 20

    runs = discover_runs(dataset_path)
    assert len(runs) == 20
    splits = {r["split"] for r in runs}
    assert {"calibration", "normal", "attack", "generalization"}.issubset(splits)


def test_loader_split_telemetry(dataset_path: pathlib.Path):
    df_normal = load_split_telemetry(dataset_path, "normal", columns=["qber", "total_count"])
    assert not df_normal.empty
    assert "qber" in df_normal.columns
    assert "total_count" in df_normal.columns
    assert (df_normal["qber"] >= 0.0).all()


def test_validate_required_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    missing = validate_required_columns(df, ["a", "b", "c"], context="test")
    assert missing == ["c"]


def test_statistics_calculations():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    summary = compute_distribution_summary(s)
    assert summary["mean"] == 3.0
    assert summary["count"] == 5

    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [2.0, 4.0, 6.0, 8.0],
    })
    corr = compute_correlation_matrix(df, ["x", "y"])
    assert np.isclose(corr.loc["x", "y"], 1.0)

    # Physics formulas
    trans = evaluate_theoretical_transmission(np.array([0.0, 10.0]), alpha_db_km=0.2, l_sys_db=0.0)
    assert np.isclose(trans[0], 1.0)
    assert trans[1] < 1.0

    sat = evaluate_dead_time_saturation(np.array([1e6]), tau_dead_s=50e-9)
    assert sat[0] < 1e6


def test_individual_plot_generation(tmp_path: pathlib.Path, dataset_path: pathlib.Path):
    apply_presentation_style()
    out_dir = tmp_path / "plots"

    # Pipeline
    res1 = plot_simulation_pipeline(out_dir, "test_pipeline", formats=["png"])
    assert res1["png"].exists()
    assert res1["png"].stat().st_size > 1000

    # Channel
    s_manifest = load_scenario_manifest(dataset_path)
    res2 = plot_channel_transmission(s_manifest, None, out_dir, "test_channel", formats=["png"])
    assert res2["png"].exists()

    # Detector
    res3 = plot_detector_physics(out_dir, "test_detector", formats=["png"])
    assert res3["png"].exists()

    # Attacks
    res4 = plot_multi_attack_signatures(out_dir, "test_attacks", formats=["png"])
    assert res4["png"].exists()


def test_end_to_end_visualization_generation(tmp_path: pathlib.Path, dataset_path: pathlib.Path):
    cfg = VisualizationConfig(
        dataset_path=dataset_path,
        output_root=tmp_path / "e2e_vis",
        formats=["png"],
        dpi=150,  # fast test DPI
        max_event_samples=500,
    )

    summary = generate_all_visualizations(cfg)
    assert summary["status"] == "COMPLETED"
    assert summary["total_presentation_figures"] >= 12

    # Verify key presentation figures exist
    expected_files = [
        "01_simulation_pipeline.png",
        "02_event_timeline.png",
        "03_channel_transmission_and_loss.png",
        "04_detector_physics_and_dead_time.png",
        "05_basis_selection_and_sifting_funnel.png",
        "06_normal_telemetry_distributions.png",
        "07_qber_temporal_evolution.png",
        "08_attack_lifecycle_and_timeline.png",
        "09_multi_attack_signatures_comparison.png",
        "10_normal_vs_attack_feature_comparison.png",
        "11_telemetry_cross_correlation.png",
        "12_dataset_partition_composition.png",
        "13_population_parameter_distributions.png",
        "14_generalization_distribution_shift.png",
        "15_stochastic_consistency_and_reproducibility.png",
    ]

    for fname in expected_files:
        fig_path = cfg.presentation_dir / fname
        assert fig_path.exists(), f"Expected figure {fname} was not created!"
        assert fig_path.stat().st_size > 5000, f"Figure {fname} is unexpectedly small!"

    # Verify metadata and report
    assert pathlib.Path(summary["metadata_file"]).exists()
    assert pathlib.Path(summary["report_file"]).exists()
