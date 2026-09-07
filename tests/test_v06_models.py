"""Unit tests for model baselines, score orientation, and serialization (Requirements 8–14, 46, 248–250)."""

import numpy as np
import pytest

from qism.ml.models import (
    AutoencoderDetector,
    IsolationForestDetector,
    LOFDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
    RobustCovarianceDetector,
    TemporalAutoencoderDetector,
    build_model,
)

MODELS = [
    "mahalanobis",
    "robust_covariance",
    "isolation_forest",
    "one_class_svm",
    "lof",
    "autoencoder",
    "temporal_autoencoder",
]


@pytest.mark.parametrize("model_name", MODELS)
def test_model_interface_and_score_orientation(model_name, tmp_path):
    """Every model produces higher scores for anomalous inputs and serializes/reloads cleanly."""
    np.random.seed(42)
    # Normal training distribution centered around 0
    X_train = np.random.normal(loc=0.0, scale=1.0, size=(50, 4))
    # Test distribution: 10 normal samples, 5 extreme anomalies (+15.0)
    X_test_normal = np.random.normal(loc=0.0, scale=1.0, size=(10, 4))
    X_test_anomalous = np.random.normal(loc=15.0, scale=1.0, size=(5, 4))
    X_test = np.vstack([X_test_normal, X_test_anomalous])

    kwargs = {}
    if model_name == "autoencoder":
        kwargs = {"epochs": 10, "lr": 1e-3, "hidden_dim": 16, "latent_dim": 4}
    elif model_name == "temporal_autoencoder":
        kwargs = {"epochs": 10, "seq_len": 3, "hidden_dim": 16, "latent_dim": 8}
    elif model_name == "isolation_forest":
        kwargs = {"n_estimators": 30}

    model = build_model(model_name, **kwargs)
    model.fit(X_train)
    assert model.is_fitted

    scores = model.score(X_test)
    assert len(scores) == 15

    # Contract: HIGHER score must strictly mean MORE ANOMALOUS
    mean_normal_score = float(np.mean(scores[:10]))
    mean_anomalous_score = float(np.mean(scores[10:]))
    assert mean_anomalous_score > mean_normal_score, (
        f"Score orientation failure in {model_name}! "
        f"Normal mean: {mean_normal_score:.3f}, Anomalous mean: {mean_anomalous_score:.3f}"
    )

    # Test predict with threshold
    th = float(np.percentile(scores[:10], 90))
    model.set_threshold(th)
    preds = model.predict(X_test)
    assert preds.shape == (15,)
    # Anomalous samples should mostly be predicted as 1
    assert np.sum(preds[10:]) >= 3

    # Test serialization and reloading
    save_path = tmp_path / f"{model_name}.bin"
    model.save(save_path)

    loaded_model = build_model(model_name).load(save_path)
    loaded_scores = loaded_model.score(X_test)
    assert np.allclose(scores, loaded_scores, atol=1e-5)
