"""Tests for QISM V0.9 telemetry ingestion, schema validation, and security leakage guards."""

import pytest
import pandas as pd
import numpy as np

from qism.monitor.ingestion import (
    TelemetryIngestor,
    IngestionConfig,
    TelemetryValidationError,
    FORBIDDEN_GROUND_TRUTH_KEYS,
)
from qism.monitor.types import TelemetryRecord


def test_valid_telemetry_ingestion():
    ingestor = TelemetryIngestor()
    rec = TelemetryRecord(
        timestamp_s=0.002,
        sequence_id=1,
        subsystem="qkd_composite",
        features={"qber": 0.015, "signal_gain": 0.045, "dark_count_rate": 25.0},
    )
    validated = ingestor.validate_and_sanitize(rec)
    assert validated.timestamp_s == 0.002
    assert validated.sequence_id == 1
    assert "qber" in validated.features
    assert validated.features["qber"] == 0.015


def test_non_monotonic_timestamp_rejection():
    ingestor = TelemetryIngestor()
    rec1 = TelemetryRecord(
        timestamp_s=0.010,
        sequence_id=1,
        subsystem="qkd_composite",
        features={"qber": 0.015},
    )
    ingestor.validate_and_sanitize(rec1)

    rec2 = TelemetryRecord(
        timestamp_s=0.005,  # Backwards timestamp
        sequence_id=2,
        subsystem="qkd_composite",
        features={"qber": 0.015},
    )
    with pytest.raises(TelemetryValidationError, match="Non-monotonic timestamp"):
        ingestor.validate_and_sanitize(rec2)


def test_duplicate_sequence_id_rejection():
    ingestor = TelemetryIngestor()
    rec1 = TelemetryRecord(
        timestamp_s=0.010,
        sequence_id=10,
        subsystem="qkd_composite",
        features={"qber": 0.015},
    )
    ingestor.validate_and_sanitize(rec1)

    rec2 = TelemetryRecord(
        timestamp_s=0.012,
        sequence_id=10,  # Duplicate sequence id
        subsystem="qkd_composite",
        features={"qber": 0.015},
    )
    with pytest.raises(TelemetryValidationError, match="Duplicate sequence_id"):
        ingestor.validate_and_sanitize(rec2)


def test_non_finite_feature_rejection():
    ingestor = TelemetryIngestor()
    rec = TelemetryRecord(
        timestamp_s=0.010,
        sequence_id=1,
        subsystem="qkd_composite",
        features={"qber": float("nan")},
    )
    with pytest.raises(TelemetryValidationError, match="non-finite"):
        ingestor.validate_and_sanitize(rec)


def test_physical_range_bounds_rejection():
    ingestor = TelemetryIngestor()
    rec = TelemetryRecord(
        timestamp_s=0.010,
        sequence_id=1,
        subsystem="qkd_composite",
        features={"qber": 1.50},  # QBER cannot exceed 1.0
    )
    with pytest.raises(TelemetryValidationError, match="violates physical bounds"):
        ingestor.validate_and_sanitize(rec)


def test_zero_ground_truth_leakage():
    """Verify that ground truth sidecars are strictly stripped from operational features."""
    ingestor = TelemetryIngestor()
    raw_dict = {
        "timestamp_s": 0.020,
        "sequence_id": 5,
        "qber": 0.018,
        "signal_gain": 0.042,
        # Malicious or evaluation sidecars
        "attack_type": "intercept_resend",
        "attack_active": True,
        "is_attack": True,
        "ground_truth_label": 1,
    }
    validated = ingestor.validate_and_sanitize(raw_dict)
    # Ensure zero ground truth keys leaked into operational features
    for forbidden in FORBIDDEN_GROUND_TRUTH_KEYS:
        assert forbidden not in validated.features
    assert "qber" in validated.features
    assert "signal_gain" in validated.features
