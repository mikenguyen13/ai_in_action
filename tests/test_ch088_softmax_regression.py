"""Tests for aiinaction.ch088_softmax_regression and the shared fixtures.

The fixtures here are the single source of truth: the Julia and Rust test
suites assert against the same numbers, which is what keeps the three
implementations at parity (1e-9 tolerance).
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction.ch088_softmax_regression import (
    SoftmaxRegression,
    cross_entropy,
    softmax,
)

# --- Shared fixtures, mirrored in julia/AIInAction/test and rust tests. ---

# softmax of two rows of logits.
SOFTMAX_Z = [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]
SOFTMAX_EXPECTED = [
    [0.09003057317038046, 0.24472847105479764, 0.6652409557748218],
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
]

# cross-entropy of a small probability matrix against integer labels.
CE_PROBS = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]
CE_Y = [0, 1]
CE_EXPECTED = 0.2899092476264711

# A tiny 3-class training problem (same data + hyperparameters in all langs).
TRAIN_X = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [2.0, 0.0],
]
TRAIN_Y = [0, 1, 2, 1, 2, 1]
TRAIN_LR = 0.5
TRAIN_ITERS = 200
TRAIN_PRED = [0, 1, 2, 1, 2, 1]
# predict_proba on the first training row [0,0] after fitting.
TRAIN_PROBA_ROW0 = [0.8222160377229213, 0.16537455654418495, 0.012409405732893796]

TOL = 1e-9


def test_softmax_matches_fixture():
    out = softmax(SOFTMAX_Z)
    assert out == pytest.approx(np.array(SOFTMAX_EXPECTED), abs=TOL)


def test_softmax_rows_sum_to_one():
    out = softmax(SOFTMAX_Z)
    assert out.sum(axis=1) == pytest.approx(np.ones(2), abs=TOL)


def test_cross_entropy_matches_fixture():
    assert cross_entropy(CE_PROBS, CE_Y) == pytest.approx(CE_EXPECTED, abs=TOL)


def test_fit_predicts_training_labels():
    model = SoftmaxRegression(learning_rate=TRAIN_LR, n_iter=TRAIN_ITERS).fit(
        TRAIN_X, TRAIN_Y
    )
    assert list(model.predict(TRAIN_X)) == TRAIN_PRED


def test_fit_proba_matches_fixture():
    model = SoftmaxRegression(learning_rate=TRAIN_LR, n_iter=TRAIN_ITERS).fit(
        TRAIN_X, TRAIN_Y
    )
    proba = model.predict_proba([[0.0, 0.0]])[0]
    assert list(proba) == pytest.approx(TRAIN_PROBA_ROW0, abs=TOL)
    assert sum(proba) == pytest.approx(1.0, abs=TOL)


def test_n_classes_inferred():
    model = SoftmaxRegression(n_iter=10).fit(TRAIN_X, TRAIN_Y)
    assert model.n_classes == 3
    assert model.W.shape == (2, 3)


# --- Edge cases / validation ---


def test_softmax_requires_2d():
    with pytest.raises(ValueError, match="2-dimensional"):
        softmax([1.0, 2.0, 3.0])


def test_softmax_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        softmax([[1.0, float("inf")]])


def test_cross_entropy_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        cross_entropy([[0.5, 0.5]], [0, 1])


def test_cross_entropy_label_out_of_range():
    with pytest.raises(ValueError, match="labels must lie"):
        cross_entropy([[0.5, 0.5]], [2])


def test_fit_rejects_bad_hyperparameters():
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        SoftmaxRegression(learning_rate=0.0)
    with pytest.raises(ValueError, match="n_iter must be positive"):
        SoftmaxRegression(n_iter=0)
    with pytest.raises(ValueError, match="l2 must be non-negative"):
        SoftmaxRegression(l2=-1.0)


def test_fit_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        SoftmaxRegression().fit([[0.0, 0.0], [1.0, 1.0]], [0])


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        SoftmaxRegression().predict([[0.0, 0.0]])


def test_predict_feature_mismatch():
    model = SoftmaxRegression(n_iter=10).fit(TRAIN_X, TRAIN_Y)
    with pytest.raises(ValueError, match="features"):
        model.predict([[0.0, 0.0, 0.0]])
