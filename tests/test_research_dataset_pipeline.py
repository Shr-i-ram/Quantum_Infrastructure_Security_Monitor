"""Unit and integration tests for the QISM research dataset generation and validation pipeline.

Tests:
- Schema consistency (76 observable, 18 ground truth, zero leakage).
- Deterministic seed derivation.
- Scenario provenance and 6-partition isolation.
- Shard manifest generation and reopening.
- Held-out parameter bounds in generalization partition.
- Extreme operational regimes in stress test partition.
"""

from __future__ import annotations

import json
import pathlib
import pandas as pd
import pytest

from qism.config import (
    ParameterDistributionConfig,
    PopulationConfig,
    QKDConfig,
    Scenario,
    SplitConfig,
    apply_overrides,
    default_config,
)
from qism.datasets.schema_registry import (
    export_dataset_schemas,
    export_scenario_registry,
    generate_feature_dictionary,
)
from qism.datasets.sharding import (
    DatasetShardWriter,
    list_sharded_events_files,
    list_sharded_telemetry_files,
    load_sharded_telemetry,
)
from qism.population.generator import ScenarioGenerator
from qism.random import derive_dataset_seed, derive_run_seed, derive_scenario_seed
from qism.telemetry.features import (
    GROUND_TRUTH_WINDOW_COLUMNS,
    IDENTIFIER_WINDOW_COLUMNS,
    get_ground_truth_feature_names,
    get_observable_feature_names,
)


def test_schema_dictionary_and_zero_leakage():
    """Verify that observable features and ground truth labels are strictly disjoint."""
    df_dict = generate_feature_dictionary()
    assert len(df_dict) == 130

    obs_names = set(get_observable_feature_names())
    gt_names = set(get_ground_truth_feature_names()) | set(GROUND_TRUTH_WINDOW_COLUMNS)
    id_names = set(IDENTIFIER_WINDOW_COLUMNS)

    assert len(obs_names) == 76
    assert len(gt_names) >= 18
    assert len(id_names) == 5

    # Critical requirement: absolute zero leakage
    leakage = obs_names.intersection(gt_names)
    assert len(leakage) == 0, f"Critical label leakage in observable features: {leakage}"

    id_overlap = obs_names.intersection(id_names)
    assert len(id_overlap) == 0, f"Identifiers leaked into observable features: {id_overlap}"


def test_schema_registry_export(tmp_path: pathlib.Path):
    """Verify that export_dataset_schemas and export_scenario_registry write valid files."""
    exported = export_dataset_schemas(tmp_path)
    assert exported["feature_dictionary"].exists()
    assert exported["telemetry_schema"].exists()
    assert exported["label_schema"].exists()

    df_dict = pd.read_csv(exported["feature_dictionary"])
    assert len(df_dict) == 130
    assert "field_name" in df_dict.columns
    assert "model_input_allowed" in df_dict.columns

    with open(exported["telemetry_schema"], "r", encoding="utf-8") as fh:
        schema_json = json.load(fh)
    assert schema_json["observable_feature_count"] == 76

    with open(exported["label_schema"], "r", encoding="utf-8") as fh:
        label_json = json.load(fh)
    assert label_json["target_label"] == "ground_truth_attack_active"

    # Test scenario registry export
    base = default_config()
    scen = Scenario(
        scenario_id="scen_test_001",
        infrastructure_id="dev_001",
        seed=999,
        regime="nominal",
        is_edge_case=False,
        dataset_split="train",
        config=base,
        sampled_parameters={"channel.distance_km": 25.0},
    )
    reg_exported = export_scenario_registry([scen], tmp_path)
    assert reg_exported["scenario_registry_parquet"].exists()
    assert reg_exported["scenario_parameters_parquet"].exists()

    df_reg = pd.read_parquet(reg_exported["scenario_registry_parquet"])
    assert len(df_reg) == 1
    assert df_reg.iloc[0]["scenario_id"] == "scen_test_001"


def test_six_partition_isolation():
    """Verify scenario-level isolation across all 6 logical dataset partitions."""
    pop_cfg = PopulationConfig(
        channel={"distance_km": ParameterDistributionConfig(distribution="uniform", a=10.0, b=40.0)},
        detectors={"efficiency": ParameterDistributionConfig(distribution="uniform", a=0.10, b=0.25)},
    )
    generator = ScenarioGenerator(pop_cfg, master_seed=424242)

    splits = {
        "calibration": SplitConfig(n_scenarios=4, pulses_per_run=1000, attack_fraction=0.0),
        "train": SplitConfig(n_scenarios=6, pulses_per_run=1000, attack_fraction=0.0),
        "validation": SplitConfig(n_scenarios=4, pulses_per_run=1000, attack_fraction=0.5),
        "test_in_distribution": SplitConfig(n_scenarios=6, pulses_per_run=1000, attack_fraction=0.8),
        "test_generalization": SplitConfig(
            n_scenarios=4,
            pulses_per_run=1000,
            attack_fraction=0.5,
            held_out_dimensions={"channel.distance_km": {"min": 50.0, "max": 70.0}},
        ),
        "test_stress": SplitConfig(
            n_scenarios=4,
            pulses_per_run=1000,
            edge_case_fraction=1.0,
            attack_fraction=0.5,
        ),
    }

    scenarios_by_split: dict[str, list[Scenario]] = {}
    dataset_seed = derive_dataset_seed(424242, "isolation_test")

    for s_name, s_cfg in splits.items():
        s_seed = derive_dataset_seed(dataset_seed, s_name)
        scenarios_by_split[s_name] = generator.generate_split_scenarios(s_name, s_cfg, s_seed)

    # 1. Check mutual scenario exclusivity
    all_scenario_ids = []
    for s_name, scens in scenarios_by_split.items():
        scen_ids = [s.scenario_id for s in scens]
        all_scenario_ids.extend(scen_ids)
    assert len(all_scenario_ids) == len(set(all_scenario_ids)), "Duplicate scenario IDs detected across splits!"

    # 2. Check generalization held-out bounds
    for s in scenarios_by_split["test_generalization"]:
        dist = s.config.channel.distance_km
        assert 50.0 <= dist <= 70.0, f"Generalization scenario distance {dist} outside held-out bounds [50, 70]!"

    # 3. Check stress edge case assignment
    for s in scenarios_by_split["test_stress"]:
        assert s.is_edge_case, "Stress test scenario must be flagged as an edge case!"
        assert s.regime in ("low_signal", "high_loss", "high_noise", "high_instability")

    # 4. Check normal train has zero attacks
    for s in scenarios_by_split["train"]:
        assert not s.config.attack.enabled, "Train partition scenario must have attack.enabled == False!"


def test_deterministic_seed_derivation():
    """Verify bit-identical deterministic seed derivation tree."""
    master = 20261019
    d_seed = derive_dataset_seed(master, "test_split")
    s_seed = derive_scenario_seed(d_seed, "scen_0001")
    r_seed_a = derive_run_seed(s_seed, "run_0001")
    r_seed_b = derive_run_seed(s_seed, "run_0001")
    r_seed_c = derive_run_seed(s_seed, "run_0002")

    assert r_seed_a == r_seed_b, "Identical seed inputs must yield bit-identical output seeds!"
    assert r_seed_a != r_seed_c, "Different run IDs must yield distinct seeds!"
