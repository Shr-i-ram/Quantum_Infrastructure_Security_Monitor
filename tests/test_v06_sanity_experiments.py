"""Unit tests for leakage sanity experiments: label shuffling and random features (Requirements 215–218, 255, 256)."""

import numpy as np
import pandas as pd
import pytest

from qism.ml.experiments.diagnostics import run_leakage_sanity_experiments
from qism.ml.features.extractor import FeatureExtractor
from qism.ml.models.statistical import MahalanobisDetector
from qism.ml.preprocessing.scalers import PreprocessorPipeline


def test_leakage_sanity_experiments():
    """Sanity checks: label shuffling collapses AUROC to chance level."""
    np.random.seed(42)
    # Synthetic normal and attack features
    n_samples = 40
    raw_df = pd.DataFrame({
        "qber": np.concatenate([np.random.normal(0.02, 0.005, 20), np.random.normal(0.20, 0.02, 20)]),
        "signal_gain": np.concatenate([np.random.normal(0.05, 0.005, 20), np.random.normal(0.01, 0.002, 20)]),
        "attack_active": [False]*20 + [True]*20,
        "attack_type": ["none"]*20 + ["intercept_resend"]*20,
    })

    ext = FeatureExtractor(feature_set="minimal")
    X, y = ext.extract(raw_df)

    prep = PreprocessorPipeline(scaler="standard")
    # Fit only on normal portion
    X_train = X.iloc[:20]
    prep.fit(X_train)

    model = MahalanobisDetector()
    model.fit(prep.transform(X_train))

    # Evaluate sanity experiments on test set
    sanity_res = run_leakage_sanity_experiments(
        model=model,
        preprocessor=prep,
        extractor=ext,
        df_test=raw_df,
        seed=123,
    )

    assert "shuffled_labels" in sanity_res
    assert "random_features" in sanity_res
    assert sanity_res["shuffled_labels"]["collapsed_to_chance"]
