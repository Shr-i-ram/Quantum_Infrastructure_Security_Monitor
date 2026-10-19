"""Unit tests for QML dataset pipeline and anti-leakage constraints."""

import numpy as np
import pytest

from qml.src.data import QMLDatasetManager


def test_qml_dataset_loading_pca():
    dm = QMLDatasetManager(representation="pca", n_qubits=4, master_seed=42)
    np_data = dm.get_numpy_data()

    X_train, y_train = np_data["train"]
    X_val, y_val = np_data["val"]
    X_test, y_test = np_data["test"]
    X_gen, y_gen = np_data["gen"]

    # Check shapes
    assert X_train.shape == (30, 4)
    assert X_val.shape == (10, 4)
    assert X_test.shape == (20, 4)
    assert X_gen.shape == (20, 4)

    # Check finite values and no NaNs
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test), ("gen", X_gen)]:
        assert np.all(np.isfinite(X)), f"Non-finite values in {name}"
        assert not np.any(np.isnan(X)), f"NaN values in {name}"
        assert X.min() >= -1e-5, f"Values below 0 in {name}"
        assert X.max() <= np.pi + 1e-5, f"Values above pi in {name}"

    # Check class presence
    assert y_train.sum() > 0, "No positive samples in train"
    assert (len(y_train) - y_train.sum()) > 0, "No negative samples in train"
    assert y_test.sum() > 0, "No positive samples in test"


def test_qml_dataset_loading_physical():
    dm = QMLDatasetManager(representation="physical", n_qubits=4, master_seed=42)
    np_data = dm.get_numpy_data()

    X_train, y_train = np_data["train"]
    assert X_train.shape == (30, 4)
    assert np.all(np.isfinite(X_train))
    assert X_train.min() >= -1e-5
    assert X_train.max() <= np.pi + 1e-5


def test_zero_scenario_leakage():
    dm = QMLDatasetManager(representation="pca")
    meta = dm.get_metadata()

    train_scens = set(meta["train"]["scenario_id"])
    val_scens = set(meta["val"]["scenario_id"])
    test_scens = set(meta["test"]["scenario_id"])
    gen_scens = set(meta["gen"]["scenario_id"])

    # Verify zero scenario intersection
    assert len(train_scens.intersection(val_scens)) == 0, "Train and Val scenarios overlap!"
    assert len(train_scens.intersection(test_scens)) == 0, "Train and Test scenarios overlap!"
    assert len(train_scens.intersection(gen_scens)) == 0, "Train and Gen scenarios overlap!"
    assert len(val_scens.intersection(test_scens)) == 0, "Val and Test scenarios overlap!"
