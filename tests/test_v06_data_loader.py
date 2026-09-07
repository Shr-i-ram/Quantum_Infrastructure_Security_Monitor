"""Unit tests for ML dataset loading and scenario-level split isolation (Requirements 16–20, 141–144)."""

import pathlib
import pytest

from qism.ml.data.loader import MLDatasetLoader
from qism.ml.data.splits import SplitManager


def test_split_manager_scenario_isolation():
    """Scenario-level separation guarantees zero overlap between train and test/generalization."""
    scenario_manifest = {
        "scenarios": [
            {"scenario_id": f"norm_{i:02d}", "infrastructure_id": f"dev_{i:02d}", "dataset_split": "normal"}
            for i in range(10)
        ] + [
            {"scenario_id": f"atk_{i:02d}", "infrastructure_id": f"dev_atk_{i:02d}", "dataset_split": "attack"}
            for i in range(6)
        ] + [
            {"scenario_id": f"gen_{i:02d}", "infrastructure_id": f"dev_gen_{i:02d}", "dataset_split": "generalization"}
            for i in range(4)
        ]
    }

    sm = SplitManager(master_seed=12345, train_val_split_ratio=0.70)
    partitions = sm.partition_scenarios(scenario_manifest)

    train_ids = {s["scenario_id"] for s in partitions["train"]}
    val_ids = {s["scenario_id"] for s in partitions["val"]}
    test_atk_ids = {s["scenario_id"] for s in partitions["test"] if s["dataset_split"] == "attack"}
    gen_ids = {s["scenario_id"] for s in partitions["generalization"]}

    # Train and test attack must never intersect
    assert len(train_ids.intersection(test_atk_ids)) == 0
    # Train and generalization must never intersect
    assert len(train_ids.intersection(gen_ids)) == 0
    # Val and test attack must never intersect
    assert len(val_ids.intersection(test_atk_ids)) == 0
    # Train and val must not intersect
    assert len(train_ids.intersection(val_ids)) == 0

    assert len(train_ids) == 7
    assert len(val_ids) == 3


def test_ml_dataset_loader_reads_smoke_dataset():
    """MLDatasetLoader loads real parquet shards and partitions data correctly."""
    dataset_path = pathlib.Path("data/v0.5_smoke/v05_smoke_dataset")
    if not dataset_path.exists():
        pytest.skip("v05_smoke_dataset not found on disk")

    loader = MLDatasetLoader(dataset_path, master_seed=42)
    summary = loader.summary()
    assert summary["dataset_id"] == "v05_smoke_dataset"
    assert "train" in summary["split_scenario_counts"]

    # Load train and test dataframes
    df_train = loader.load_split_dataframe("train")
    df_test = loader.load_split_dataframe("test")

    assert not df_train.empty
    assert not df_test.empty
    assert "qber" in df_train.columns
    assert "run_id" in df_train.columns
