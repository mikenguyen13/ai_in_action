"""Tests for aiinaction.ch086_elastic_net, including shared cross-language fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations of
the Elastic Net coordinate-descent solver at parity.
"""
from __future__ import annotations

import pytest

from aiinaction.ch086_elastic_net import (
    elastic_net_fit,
    elastic_net_predict,
    soft_threshold,
)

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
X = [
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 4.0],
    [4.0, 3.0],
    [5.0, 6.0],
]
Y = [2.0, 3.0, 5.0, 7.0, 8.0]

# Elastic Net, lam=0.5, alpha=0.5 (the canonical mixed case).
EN_COEF = [1.1076803723827306, 0.22885958106994972]
EN_INTERCEPT = 0.9446082234279691
EN_PRED = [
    2.510007757950599,
    3.3888285492633803,
    5.18308766485596,
    6.061908456168741,
    7.856167571761321,
]

# Pure Lasso (alpha=1, lam=1): second predictor is zeroed out.
LASSO_COEF = [1.1, 0.0]
LASSO_INTERCEPT = 1.7

# Pure Ridge (alpha=0, lam=1): both coefficients shrunk, neither zero.
RIDGE_COEF = [0.795939086294827, 0.4060913705581682]
RIDGE_INTERCEPT = 1.312690355329381

# lam=0 recovers ordinary least squares.
OLS_COEF = [1.6, 0.0]
OLS_INTERCEPT = 0.2

FIT_KW = {"max_iter": 10000, "tol": 1e-12}


def test_soft_threshold_basic():
    assert soft_threshold(3.0, 1.0) == pytest.approx(2.0)
    assert soft_threshold(-3.0, 1.0) == pytest.approx(-2.0)
    assert soft_threshold(0.5, 1.0) == 0.0
    assert soft_threshold(0.0, 0.0) == 0.0


def test_soft_threshold_negative_gamma_raises():
    with pytest.raises(ValueError, match="non-negative"):
        soft_threshold(1.0, -0.5)


def test_elastic_net_matches_fixture():
    coef, intercept = elastic_net_fit(X, Y, lam=0.5, alpha=0.5, **FIT_KW)
    assert coef == pytest.approx(EN_COEF, abs=1e-9)
    assert intercept == pytest.approx(EN_INTERCEPT, abs=1e-9)


def test_predict_matches_fixture():
    preds = elastic_net_predict(X, EN_COEF, EN_INTERCEPT)
    assert preds == pytest.approx(EN_PRED, abs=1e-9)


def test_lasso_limit_zeros_a_coefficient():
    coef, intercept = elastic_net_fit(X, Y, lam=1.0, alpha=1.0, **FIT_KW)
    assert coef == pytest.approx(LASSO_COEF, abs=1e-9)
    assert intercept == pytest.approx(LASSO_INTERCEPT, abs=1e-9)
    assert coef[1] == 0.0


def test_ridge_limit_shrinks_without_zeroing():
    coef, intercept = elastic_net_fit(X, Y, lam=1.0, alpha=0.0, **FIT_KW)
    assert coef == pytest.approx(RIDGE_COEF, abs=1e-9)
    assert intercept == pytest.approx(RIDGE_INTERCEPT, abs=1e-9)
    assert coef[0] != 0.0 and coef[1] != 0.0


def test_zero_lambda_recovers_ols():
    coef, intercept = elastic_net_fit(X, Y, lam=0.0, alpha=0.5, **FIT_KW)
    assert coef == pytest.approx(OLS_COEF, abs=1e-7)
    assert intercept == pytest.approx(OLS_INTERCEPT, abs=1e-7)


def test_predict_round_trip():
    coef, intercept = elastic_net_fit(X, Y, lam=0.5, alpha=0.5, **FIT_KW)
    preds = elastic_net_predict(X, coef, intercept)
    assert preds == pytest.approx(EN_PRED, abs=1e-9)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        elastic_net_fit([[1.0], [2.0]], [1.0], lam=0.1)


def test_empty_X_raises():
    with pytest.raises(ValueError, match="non-empty"):
        elastic_net_fit([[]], [1.0], lam=0.1)


def test_negative_lambda_raises():
    with pytest.raises(ValueError, match="non-negative"):
        elastic_net_fit(X, Y, lam=-1.0)


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        elastic_net_fit(X, Y, lam=0.1, alpha=1.5)


def test_predict_coef_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        elastic_net_predict(X, [1.0], 0.0)
