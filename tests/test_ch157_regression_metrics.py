"""Tests for aiinaction.ch157_regression_metrics, including shared fixtures.

The fixtures in EXPECTED are the single source of truth: the Julia and Rust test
suites assert against the same numbers, which keeps the three implementations at
parity. The residual fixture is the classic four-point set also used by the core
``metrics`` parity tests.
"""
from __future__ import annotations

import pytest

from aiinaction import ch157_regression_metrics as rm

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
Y_TRUE = [3.0, -0.5, 2.0, 7.0]
Y_PRED = [2.5, 0.0, 2.0, 8.0]
DELTA = 0.75
EXPECTED = {
    "mse": 0.375,
    "rmse": 0.6123724356957945,
    "mae": 0.5,
    # Per-observation Huber loss with delta=0.75 on the residuals [0.5, -0.5, 0, -1].
    "huber_each": [0.125, 0.125, 0.0, 0.46875],
    "huber_mean": 0.1796875,
}


def test_mse_matches_fixture():
    assert rm.mse(Y_TRUE, Y_PRED) == pytest.approx(EXPECTED["mse"])


def test_rmse_matches_fixture():
    assert rm.rmse(Y_TRUE, Y_PRED) == pytest.approx(EXPECTED["rmse"])


def test_mae_matches_fixture():
    assert rm.mae(Y_TRUE, Y_PRED) == pytest.approx(EXPECTED["mae"])


def test_rmse_is_sqrt_mse():
    assert rm.rmse(Y_TRUE, Y_PRED) == pytest.approx(rm.mse(Y_TRUE, Y_PRED) ** 0.5)


def test_huber_each_matches_fixture():
    assert rm.huber_loss(Y_TRUE, Y_PRED, delta=DELTA) == pytest.approx(
        EXPECTED["huber_each"]
    )


def test_huber_mean_matches_fixture():
    assert rm.huber_loss_mean(Y_TRUE, Y_PRED, delta=DELTA) == pytest.approx(
        EXPECTED["huber_mean"]
    )


def test_huber_continuity_at_threshold():
    # At |r| == delta the quadratic and linear branches agree: both give 0.5*delta^2.
    d = 1.5
    quad = rm.huber_loss([0.0], [d], delta=d)[0]
    lin = rm.huber_loss([0.0], [d + 1e-9], delta=d)[0]
    assert quad == pytest.approx(0.5 * d * d)
    assert lin == pytest.approx(0.5 * d * d, abs=1e-6)


def test_huber_large_delta_approaches_half_mse():
    # As delta -> inf every residual is an inlier, so mean Huber -> 0.5 * MSE.
    big = rm.huber_loss_mean(Y_TRUE, Y_PRED, delta=1000.0)
    assert big == pytest.approx(0.5 * rm.mse(Y_TRUE, Y_PRED))


def test_perfect_prediction_is_zero():
    assert rm.mse(Y_TRUE, Y_TRUE) == 0.0
    assert rm.rmse(Y_TRUE, Y_TRUE) == 0.0
    assert rm.mae(Y_TRUE, Y_TRUE) == 0.0
    assert rm.huber_loss_mean(Y_TRUE, Y_TRUE, delta=1.0) == 0.0


@pytest.mark.parametrize("fn", [rm.mse, rm.huber_loss, rm.huber_loss_mean])
def test_length_mismatch_raises(fn):
    with pytest.raises(ValueError, match="length mismatch"):
        fn([1.0, 2.0], [1.0])


@pytest.mark.parametrize("fn", [rm.mse, rm.huber_loss, rm.huber_loss_mean])
def test_empty_raises(fn):
    with pytest.raises(ValueError, match="non-empty"):
        fn([], [])


@pytest.mark.parametrize("bad_delta", [0.0, -1.0, float("inf"), float("nan")])
def test_huber_bad_delta_raises(bad_delta):
    with pytest.raises(ValueError, match="delta must be a positive"):
        rm.huber_loss(Y_TRUE, Y_PRED, delta=bad_delta)
