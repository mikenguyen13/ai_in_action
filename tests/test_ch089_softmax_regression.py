"""Tests for aiinaction.ch089_softmax_regression and the shared parity fixtures.

The numbers in this file are the single source of truth: the Julia and Rust
test suites assert against the identical fixtures, which is what keeps the three
implementations at parity. Tolerances are 1e-9.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aiinaction.ch089_softmax_regression import (
    SoftmaxRegression,
    cross_entropy_from_logits,
    log_sum_exp,
    softmax,
)

# --- Shared scalar fixtures (mirrored in Julia and Rust) ---------------------
SOFTMAX_Z = [1.0, 2.0, 3.0]
SOFTMAX_EXPECTED = [
    0.09003057317038046,
    0.24472847105479764,
    0.6652409557748218,
]
SOFTMAX_T2_EXPECTED = [
    0.1863237232258476,
    0.3071958857184984,
    0.506480391055654,
]
LSE_EXPECTED = 0.6931471805599453        # log_sum_exp([0, 0]) == log 2
CE_Z = [2.0, 1.0, 0.0]
CE_EXPECTED = 0.4076059644443806         # cross_entropy_from_logits(CE_Z, 0)

# --- Shared classifier fixture: 3 points, 3 classes, 2 gradient steps ---------
FIT_X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
FIT_Y = [0, 1, 2]
FIT_LR = 1.0
FIT_N_ITER = 2
FIT_W = [
    -0.20927700627356688, -0.20927700627356688,
    0.41855401254713376, -0.20927700627356688,
    -0.20927700627356688, 0.41855401254713376,
]
FIT_B = [0.0258904318973107, -0.012945215948655314, -0.012945215948655314]
FIT_LOSS = 0.8486364761645943
FIT_PROBA = [
    0.3420186020672228, 0.3289906989663886, 0.3289906989663886,
    0.265668759951062, 0.4787821251716869, 0.25554911487725124,
    0.265668759951062, 0.25554911487725124, 0.4787821251716869,
]

TOL = 1e-9


def test_softmax_matches_fixture():
    out = softmax(SOFTMAX_Z)
    assert out == pytest.approx(SOFTMAX_EXPECTED, abs=TOL)
    assert sum(out) == pytest.approx(1.0, abs=TOL)


def test_softmax_temperature():
    assert softmax(SOFTMAX_Z, temperature=2.0) == pytest.approx(SOFTMAX_T2_EXPECTED, abs=TOL)


def test_softmax_shift_invariance():
    base = softmax(SOFTMAX_Z)
    shifted = softmax([v + 100.0 for v in SOFTMAX_Z])
    assert shifted == pytest.approx(base, abs=TOL)


def test_softmax_no_overflow():
    out = softmax([1000.0, 1000.0, 1000.0])
    assert out == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=TOL)


def test_log_sum_exp_matches_fixture():
    assert log_sum_exp([0.0, 0.0]) == pytest.approx(LSE_EXPECTED, abs=TOL)


def test_log_sum_exp_stable_large():
    # log_sum_exp([m, m]) == m + log 2 even for large m, without overflow.
    assert log_sum_exp([800.0, 800.0]) == pytest.approx(800.0 + math.log(2.0), abs=1e-6)


def test_cross_entropy_matches_fixture():
    assert cross_entropy_from_logits(CE_Z, 0) == pytest.approx(CE_EXPECTED, abs=TOL)


def test_cross_entropy_equals_neg_log_softmax():
    z = [0.3, -1.2, 2.1, 0.0]
    for c in range(len(z)):
        expected = -math.log(softmax(z)[c])
        assert cross_entropy_from_logits(z, c) == pytest.approx(expected, abs=TOL)


def test_classifier_parameters_match_fixture():
    model = SoftmaxRegression(learning_rate=FIT_LR, n_iter=FIT_N_ITER, l2=0.0).fit(FIT_X, FIT_Y)
    assert model.W.ravel().tolist() == pytest.approx(FIT_W, abs=TOL)
    assert model.b.tolist() == pytest.approx(FIT_B, abs=TOL)


def test_classifier_loss_matches_fixture():
    model = SoftmaxRegression(learning_rate=FIT_LR, n_iter=FIT_N_ITER, l2=0.0).fit(FIT_X, FIT_Y)
    assert model.loss(FIT_X, FIT_Y) == pytest.approx(FIT_LOSS, abs=TOL)


def test_classifier_proba_matches_fixture():
    model = SoftmaxRegression(learning_rate=FIT_LR, n_iter=FIT_N_ITER, l2=0.0).fit(FIT_X, FIT_Y)
    assert model.predict_proba(FIT_X).ravel().tolist() == pytest.approx(FIT_PROBA, abs=TOL)


def test_classifier_separable_converges():
    # A well-separated dataset should be classified perfectly after training.
    X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 0.0]]
    y = [0, 1, 2, 2, 2, 1]
    model = SoftmaxRegression(learning_rate=0.5, n_iter=300).fit(X, y)
    assert model.predict(X).tolist() == y
    assert model.n_classes_ == 3


def test_softmax_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        softmax([])


def test_softmax_bad_temperature_raises():
    with pytest.raises(ValueError, match="temperature"):
        softmax([1.0, 2.0], temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        softmax([1.0, 2.0], temperature=-1.0)


def test_cross_entropy_label_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        cross_entropy_from_logits([1.0, 2.0], 2)
    with pytest.raises(ValueError, match="out of range"):
        cross_entropy_from_logits([1.0, 2.0], -1)


def test_fit_shape_mismatch_raises():
    with pytest.raises(ValueError, match="disagree on N"):
        SoftmaxRegression().fit([[0.0], [1.0]], [0])


def test_fit_empty_raises():
    with pytest.raises(ValueError, match="at least one row"):
        SoftmaxRegression().fit(np.empty((0, 2)), np.empty((0,), dtype=int))


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        SoftmaxRegression().predict([[1.0, 2.0]])


def test_predict_feature_mismatch_raises():
    model = SoftmaxRegression(n_iter=5).fit(FIT_X, FIT_Y)
    with pytest.raises(ValueError, match="features"):
        model.predict([[1.0, 2.0, 3.0]])


def test_bad_hyperparameters_raise():
    with pytest.raises(ValueError, match="learning_rate"):
        SoftmaxRegression(learning_rate=0.0)
    with pytest.raises(ValueError, match="n_iter"):
        SoftmaxRegression(n_iter=0)
    with pytest.raises(ValueError, match="l2"):
        SoftmaxRegression(l2=-1.0)
