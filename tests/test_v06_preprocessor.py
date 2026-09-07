"""Unit tests for preprocessor isolation, serialization, and robust scaling (Requirements 32–35, 140, 245)."""

import numpy as np
import pandas as pd
import pytest

from qism.ml.preprocessing.scalers import PreprocessorPipeline


def test_preprocessor_fit_on_train_only():
    """Preprocessor computes statistics strictly from X_train and applies them to X_test."""
    X_train = np.array([
        [10.0, 1.0],
        [20.0, 2.0],
        [30.0, 3.0],
    ])
    # Mean: [20.0, 2.0], Std: [8.1649658, 0.81649658]

    X_test = np.array([
        [20.0, 2.0],  # Centered at training mean -> should transform to [0.0, 0.0]
        [200.0, 20.0],
    ])

    pipe = PreprocessorPipeline(scaler="standard")
    pipe.fit(X_train)

    X_test_trans = pipe.transform(X_test)
    assert np.allclose(X_test_trans[0], [0.0, 0.0], atol=1e-5)


def test_preprocessor_handles_nans_and_zero_variance():
    """Preprocessor safely handles missing values and constant zero-variance features."""
    X_train = np.array([
        [5.0, np.nan, 42.0],
        [5.0, 10.0,   42.0],
        [5.0, 20.0,   42.0],
    ])  # Feature 0 and 2 are constant (zero variance), Feature 1 has NaN

    pipe = PreprocessorPipeline(scaler="standard", impute_strategy="median")
    X_trans = pipe.fit_transform(X_train)

    assert not np.isnan(X_trans).any()
    assert not np.isinf(X_trans).any()


def test_preprocessor_serialization(tmp_path):
    """Fitted preprocessor saves to JSON and reloads identically."""
    X = np.random.randn(20, 5)
    pipe = PreprocessorPipeline(scaler="robust", clip_sigma=5.0)
    X_trans = pipe.fit_transform(X)

    save_path = tmp_path / "pipe.json"
    pipe.save(save_path)

    loaded_pipe = PreprocessorPipeline.load(save_path)
    X_trans_loaded = loaded_pipe.transform(X)

    assert np.allclose(X_trans, X_trans_loaded)
