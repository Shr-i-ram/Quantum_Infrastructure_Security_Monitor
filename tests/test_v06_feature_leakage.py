"""Unit tests guaranteeing zero ground-truth and metadata leakage in feature extraction (Requirements 28–31, 112, 243)."""

import numpy as np
import pandas as pd
import pytest

from qism.ml.features.definitions import GROUND_TRUTH_FIELDS, METADATA_FIELDS, is_ground_truth_field
from qism.ml.features.extractor import FeatureExtractor


def test_ground_truth_identification():
    """Ground truth helper accurately identifies forbidden fields."""
    assert is_ground_truth_field("attack_active")
    assert is_ground_truth_field("attack_type")
    assert is_ground_truth_field("ground_truth_mean_severity")
    assert is_ground_truth_field("true_temperature_c")
    assert is_ground_truth_field("yield_0")
    assert is_ground_truth_field("mean_photon_number")

    # Legitimate observable features must NOT be flagged as ground truth
    assert not is_ground_truth_field("qber")
    assert not is_ground_truth_field("signal_gain")
    assert not is_ground_truth_field("sensor_temperature_mean")
    assert not is_ground_truth_field("detector_imbalance")


def test_feature_extractor_forbids_ground_truth_and_metadata():
    """FeatureExtractor rejects any feature set containing ground-truth or metadata fields."""
    # Instantiating a valid feature extractor
    ext = FeatureExtractor(feature_set="full")
    names = ext.get_feature_names()

    for name in names:
        assert not is_ground_truth_field(name), f"Leakage: '{name}' is in observable feature set!"
        assert name not in METADATA_FIELDS, f"Metadata leakage: '{name}' is in feature set!"


def test_feature_extraction_zero_leakage_on_data():
    """Extracting features separates observable X from ground truth y."""
    raw_df = pd.DataFrame({
        "qber": [0.01, 0.02, 0.15],
        "signal_gain": [0.05, 0.04, 0.02],
        "sensor_temperature_mean": [21.0, 21.2, 21.5],
        # Ground truth columns present in raw telemetry
        "attack_active": [False, False, True],
        "attack_type": ["none", "none", "intercept_resend"],
        "attack_strength": [0.0, 0.0, 0.8],
        "true_temperature_c": [20.9, 21.1, 21.4],
        "mean_photon_number": [0.5, 0.5, 0.5],
        # Provenance columns
        "scenario_id": ["scen_01", "scen_01", "scen_01"],
        "run_id": ["run_01", "run_01", "run_01"],
        "window_id": [0, 1, 2],
    })

    ext = FeatureExtractor(feature_set="minimal")
    X, y = ext.extract(raw_df)

    # Observable feature matrix X must contain NO ground truth columns
    for col in X.columns:
        assert not is_ground_truth_field(col)
        assert col not in METADATA_FIELDS

    # y sidecar must isolate ground truth and metadata
    assert "attack_active" in y.columns
    assert "attack_type" in y.columns
    assert "scenario_id" in y.columns
    assert y["attack_active"].tolist() == [False, False, True]
