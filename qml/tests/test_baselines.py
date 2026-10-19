"""Unit tests for classical reference baselines."""

import numpy as np
import pytest

from qml.src.baselines import (
    ClassicalLogisticRegressionBaseline,
    ClassicalRBFSVMBaseline,
    SmallClassicalMLP,
)


def test_classical_baselines():
    X_train = np.random.uniform(0.0, np.pi, size=(12, 4))
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X_test = np.random.uniform(0.0, np.pi, size=(6, 4))

    # Logistic Regression
    lr = ClassicalLogisticRegressionBaseline()
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_test)
    assert p_lr.shape == (6,)
    assert (p_lr >= 0.0).all() and (p_lr <= 1.0).all()

    # RBF SVM
    svm = ClassicalRBFSVMBaseline()
    svm.fit(X_train, y_train)
    p_svm = svm.predict_proba(X_test)
    assert p_svm.shape == (6,)
    assert (p_svm >= 0.0).all() and (p_svm <= 1.0).all()

    # Small MLP
    mlp = SmallClassicalMLP()
    mlp.fit(X_train, y_train, epochs=10)
    p_mlp = mlp.predict_proba(X_test)
    assert p_mlp.shape == (6,)
    assert (p_mlp >= 0.0).all() and (p_mlp <= 1.0).all()
