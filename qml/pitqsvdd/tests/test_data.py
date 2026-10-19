"""Tests for dataset loading, zero-leakage partitions, and temporal windowing."""

from pathlib import Path
import numpy as np
import pytest
import torch

from qml.pitqsvdd.config import DataConfig
from qml.pitqsvdd.data import PITQSVDDDatasetManager
from qml.pitqsvdd.temporal_windows import TemporalWindowBuilder


def test_dataset_manager_partitions_and_zero_leakage():
    """Verify that training set strictly contains normal scenarios with zero attack labels."""
    cfg = DataConfig(
        dataset_root="data/v0.5_small/v05_small_dataset",
        n_qubits=4,
        sequence_length=3,
        master_seed=42,
    )
    dm = PITQSVDDDatasetManager(cfg)

    # 1. Check training set labels: MUST be strictly 0
    train_labels = dm.temporal_data["train"]["labels"]
    assert len(train_labels) > 0, "Training set is empty!"
    assert torch.all(train_labels == 0), f"Leakage detected! Training set contains attacks: {train_labels.sum()}"

    # 2. Check test set labels: contains both normal and attack
    test_labels = dm.temporal_data["test"]["labels"]
    assert len(test_labels) > 0
    assert torch.any(test_labels == 1), "Test set must contain attack samples for evaluation!"
    assert torch.any(test_labels == 0), "Test set must contain normal samples for evaluation!"

    # 3. Check shapes
    train_windows = dm.temporal_data["train"]["windows"]
    assert train_windows.ndim == 3
    assert train_windows.shape[1] == 3  # Sequence length L=3
    assert train_windows.shape[2] == 4  # Dimension = n_qubits = 4


def test_temporal_window_builder_boundary_isolation():
    """Verify sliding windows do not cross scenario boundaries."""
    builder = TemporalWindowBuilder(sequence_length=3, stride=1)
    T, D = 5, 4
    X_scen1 = np.ones((T, D)) * 1.0
    y_scen1 = np.zeros((T,))
    df1 = pytest.importorskip("pandas").DataFrame({"attack_active": y_scen1})

    res = builder.build_scenario_windows(X_scen1, df1)
    windows = res["windows"]
    # For T=5, L=3, stride=1: windows start at t=2, 3, 4 -> exactly 3 windows
    assert len(windows) == 3
    assert windows.shape == (3, 3, 4)
    # Check causal target: targets are available for t=2 (next t=3) and t=3 (next t=4), but not t=4
    assert res["has_target"].tolist() == [True, True, False]
