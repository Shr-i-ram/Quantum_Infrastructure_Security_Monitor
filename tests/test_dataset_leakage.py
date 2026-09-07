"""Tests for zero label leakage between observable telemetry features and ground truth labels."""

import pytest

from qism.config import AttackScenarioConfig, default_config
from qism.simulation.runner import RunManager
from qism.telemetry.features import (
    GROUND_TRUTH_WINDOW_COLUMNS,
    IDENTIFIER_WINDOW_COLUMNS,
    get_ground_truth_feature_names,
    get_observable_feature_names,
)


def test_feature_list_isolation():
    """Verify that observable feature names and ground truth columns are strictly disjoint."""
    observable = set(get_observable_feature_names())
    ground_truth = set(get_ground_truth_feature_names()) | set(GROUND_TRUTH_WINDOW_COLUMNS)
    identifiers = set(IDENTIFIER_WINDOW_COLUMNS)

    # 1. Zero overlap between observable and ground truth
    leakage = observable.intersection(ground_truth)
    assert len(leakage) == 0, f"Observable features leak ground truth columns: {leakage}"

    # 2. Zero overlap between observable and identifiers
    id_leakage = observable.intersection(identifiers)
    assert len(id_leakage) == 0, f"Observable features leak identifier columns: {id_leakage}"

    # 3. No observable column contains suspicious ground truth keywords
    for col in observable:
        lower = col.lower()
        assert "attack" not in lower, f"Observable column {col} contains 'attack'"
        assert "ground_truth" not in lower, f"Observable column {col} contains 'ground_truth'"
        assert "eve_" not in lower and "eavesdrop" not in lower, f"Observable column {col} contains 'eve'"
        assert "secret" not in lower, f"Observable column {col} contains 'secret'"


def test_telemetry_table_leakage_isolation(tmp_path):
    """Run an attack simulation and verify that observable feature slicing contains zero labels."""
    sc = AttackScenarioConfig(
        scenario_id="leakage_test",
        type="intercept_resend",
        severity=1.0,
        start_pulse=2000,
        end_pulse=6000,
        onset_pulses=500,
        termination_pulses=500,
        params={"fraction": 1.0},
    )
    cfg = default_config()
    cfg.protocol.n_pulses = 8000
    cfg.telemetry.window_size = 2000
    cfg.simulation.chunk_size = 2000
    cfg.simulation.checkpoint_interval = 2000
    cfg.attack.enabled = True
    cfg.attack.scenarios = [sc]

    mgr = RunManager(cfg, "leakage_run", output_root=tmp_path / "data", checkpoint_root=tmp_path / "ckpt")
    mgr.run(fresh=True)
    df = mgr.writer.read_final_telemetry()

    obs_names = get_observable_feature_names()
    gt_names = get_ground_truth_feature_names()

    # All observable and GT columns must be present in the parquet file
    for col in obs_names:
        assert col in df.columns, f"Missing observable column: {col}"
    for col in gt_names:
        assert col in df.columns, f"Missing ground truth column: {col}"

    # Verify that slice X = df[obs_names] has shape (n_windows, len(obs_names)) and zero GT columns
    X = df[obs_names]
    assert X.shape[1] == len(obs_names)
    for col in X.columns:
        assert not col.startswith("ground_truth_")
        assert not col.startswith("attack_")

    # Ground truth labels exist in df[gt_names]
    y = df[gt_names]
    assert y["ground_truth_attack_active"].any()
    assert (y["ground_truth_attack_type"] == "intercept_resend").any()
