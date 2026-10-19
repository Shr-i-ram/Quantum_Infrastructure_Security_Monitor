"""Unit tests for V0.7 temporal anomaly detectors."""

import numpy as np
import pytest

from qism.temporal.models import (
    EWMAAnomalyDetector,
    RollingZScoreDetector,
    SequenceAutoencoderDetector,
)


def test_rolling_zscore_detector():
    """Verify RollingZScoreDetector fit and score causal behavior."""
    X_train = np.random.RandomState(42).normal(loc=0.0, scale=1.0, size=(50, 4))
    detector = RollingZScoreDetector(window_length=5)
    detector.fit(X_train)

    assert detector.is_fitted
    # Test scoring on 2D array
    scores = detector.score(X_train)
    assert len(scores) == 50
    assert np.all(scores >= 0.0)

    # Injected massive outlier should produce large score
    X_outlier = np.copy(X_train[:5])
    X_outlier[-1] += 50.0
    outlier_scores = detector.score(X_outlier)
    assert outlier_scores[-1] > scores[-1] * 5.0


def test_ewma_anomaly_detector():
    """Verify EWMA detector smoothing and anomaly scoring."""
    X_train = np.ones((40, 3))
    detector = EWMAAnomalyDetector(alpha=0.3)
    detector.fit(X_train)

    assert detector.is_fitted
    # Constant data should have near zero score
    scores = detector.score(X_train)
    assert np.all(scores < 0.01)

    # Shift at step 10
    X_shift = np.ones((20, 3))
    X_shift[10:] = 10.0
    shift_scores = detector.score(X_shift)
    assert shift_scores[10] > 5.0


def test_sequence_autoencoder():
    """Verify SequenceAutoencoderDetector training and feature error decomposition."""
    n_samples, seq_len, d = 30, 4, 3
    rng = np.random.RandomState(42)
    X_seq = rng.normal(loc=0.0, scale=0.5, size=(n_samples, seq_len, d)).astype(np.float32)

    detector = SequenceAutoencoderDetector(
        sequence_length=seq_len,
        feature_dim=d,
        rnn_type="GRU",
        hidden_dim=16,
        latent_dim=8,
        epochs=3,
        batch_size=8,
        device="cpu",
        seed=42,
    )
    detector.fit(X_seq)
    assert detector.is_fitted

    scores = detector.score(X_seq)
    assert len(scores) == n_samples
    assert np.all(scores >= 0.0)

    # Feature error decomposition E_j
    e_j = detector.feature_error_contributions(X_seq)
    assert e_j.shape == (n_samples, d)
    assert np.all(e_j >= 0.0)
