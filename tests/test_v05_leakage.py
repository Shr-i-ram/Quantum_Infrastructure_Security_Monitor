"""Tests for QISM V0.5 split integrity, generalization holdout, and zero feature leakage (Requirements 53–56, 129, 130, 151, 194–196)."""

import pandas as pd
import pytest

from qism.config import (
    ParameterDistributionConfig,
    PopulationConfig,
    SplitConfig,
    V05DatasetConfig,
    default_config,
)
from qism.datasets.pipeline import DatasetPipeline
from qism.population.generator import ScenarioGenerator


def test_scenario_level_split_isolation():
    """Verify scenarios never leak across splits (Requirements 54, 55, 151, 194)."""
    pop_cfg = PopulationConfig(
        channel={"distance_km": ParameterDistributionConfig(distribution="uniform", a=10.0, b=30.0)}
    )
    generator = ScenarioGenerator(pop_cfg, master_seed=12345)

    split_calib = SplitConfig(n_scenarios=3, runs_per_scenario=2, pulses_per_run=2000)
    split_norm = SplitConfig(n_scenarios=4, runs_per_scenario=2, pulses_per_run=2000)
    split_atk = SplitConfig(n_scenarios=4, runs_per_scenario=2, pulses_per_run=2000)
    split_gen = SplitConfig(n_scenarios=3, runs_per_scenario=2, pulses_per_run=2000)

    scens_calib = generator.generate_split_scenarios("calibration", split_calib, split_seed=101)
    scens_norm = generator.generate_split_scenarios("normal", split_norm, split_seed=102)
    scens_atk = generator.generate_split_scenarios("attack", split_atk, split_seed=103)
    scens_gen = generator.generate_split_scenarios("generalization", split_gen, split_seed=104)

    ids_calib = {s.scenario_id for s in scens_calib}
    ids_norm = {s.scenario_id for s in scens_norm}
    ids_atk = {s.scenario_id for s in scens_atk}
    ids_gen = {s.scenario_id for s in scens_gen}

    # Pairwise disjoint
    assert len(ids_calib.intersection(ids_norm)) == 0
    assert len(ids_calib.intersection(ids_atk)) == 0
    assert len(ids_norm.intersection(ids_atk)) == 0
    assert len(ids_norm.intersection(ids_gen)) == 0
    assert len(ids_atk.intersection(ids_gen)) == 0


def test_generalization_parameter_holdout():
    """Verify generalization scenarios respect configured held-out dimension boundaries (Requirements 53, 56, 133)."""
    pop_cfg = PopulationConfig(
        channel={"distance_km": ParameterDistributionConfig(distribution="uniform", a=10.0, b=30.0)}
    )
    generator = ScenarioGenerator(pop_cfg, master_seed=54321)

    norm_cfg = SplitConfig(n_scenarios=8, pulses_per_run=2000)
    gen_cfg = SplitConfig(
        n_scenarios=6,
        pulses_per_run=2000,
        held_out_dimensions={"channel.distance_km": {"min": 45.0, "max": 65.0}},
    )

    scens_norm = generator.generate_split_scenarios("normal", norm_cfg, split_seed=201)
    scens_gen = generator.generate_split_scenarios("generalization", gen_cfg, split_seed=202)

    norm_dists = [s.config.channel.distance_km for s in scens_norm]
    gen_dists = [s.config.channel.distance_km for s in scens_gen]

    # Training distributions should be <= 35 km
    assert max(norm_dists) <= 35.0
    # Generalization distribution must be in strictly held out range [45, 65] km
    assert min(gen_dists) >= 45.0
    assert max(gen_dists) <= 65.0
    # Complete absence of overlap
    assert max(norm_dists) < min(gen_dists)


def test_zero_feature_leakage_in_observable_telemetry(tmp_path):
    """Observable telemetry must never contain attack ground-truth columns (Requirements 129, 130, 196)."""
    cfg = V05DatasetConfig(
        dataset_id="test_leakage_ds",
        master_seed=777,
        output_root=str(tmp_path / "data"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        worker_count=1,
        splits={
            "attack": SplitConfig(
                n_scenarios=2,
                runs_per_scenario=1,
                pulses_per_run=4000,
                attack_fraction=1.0,
                attack_types=["intercept_resend"],
            )
        },
    )

    pipeline = DatasetPipeline(cfg)
    result = pipeline.run(fresh=True)
    assert result["status"] == "COMPLETED"

    # Inspect written telemetry files
    from qism.datasets.sharding import load_sharded_telemetry
    df_atk = load_sharded_telemetry(pipeline.output_root / "attack")
    assert df_atk is not None and not df_atk.empty

    forbidden_leakage_columns = [
        "attack_present",
        "attack_type",
        "attack_severity",
        "true_state",
        "ground_truth_eavesdropper",
    ]

    for col in forbidden_leakage_columns:
        assert col not in df_atk.columns, f"Information leakage detected: forbidden column {col} in observable telemetry"
