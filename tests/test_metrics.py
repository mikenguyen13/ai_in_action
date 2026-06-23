"""Tests for aiinaction.metrics, including the shared cross-language fixtures.

The fixtures in EXPECTED are the single source of truth: the Julia and Rust test
suites assert against the same numbers, which is what keeps the three libraries at
parity.
"""
from __future__ import annotations

import math

import pytest

from aiinaction import metrics

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/tests.
Y_TRUE = [3.0, -0.5, 2.0, 7.0]
Y_PRED = [2.5, 0.0, 2.0, 8.0]
EXPECTED = {
    "rmse": 0.6123724356957945,
    "mae": 0.5,
    "r2_score": 0.9486081370449679,
}


def test_rmse_matches_fixture():
    assert metrics.rmse(Y_TRUE, Y_PRED) == pytest.approx(EXPECTED["rmse"])


def test_mae_matches_fixture():
    assert metrics.mae(Y_TRUE, Y_PRED) == pytest.approx(EXPECTED["mae"])


def test_r2_matches_fixture():
    assert metrics.r2_score(Y_TRUE, Y_PRED) == pytest.approx(EXPECTED["r2_score"])


def test_perfect_prediction():
    assert metrics.rmse(Y_TRUE, Y_TRUE) == 0.0
    assert metrics.mae(Y_TRUE, Y_TRUE) == 0.0
    assert metrics.r2_score(Y_TRUE, Y_TRUE) == pytest.approx(1.0)


def test_accuracy():
    assert metrics.accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == pytest.approx(0.5)


@pytest.mark.parametrize("fn", [metrics.rmse, metrics.mae, metrics.r2_score])
def test_length_mismatch_raises(fn):
    with pytest.raises(ValueError, match="length mismatch"):
        fn([1.0, 2.0], [1.0])


@pytest.mark.parametrize("fn", [metrics.rmse, metrics.mae])
def test_empty_raises(fn):
    with pytest.raises(ValueError, match="non-empty"):
        fn([], [])


def test_r2_zero_variance_raises():
    with pytest.raises(ValueError, match="undefined"):
        metrics.r2_score([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
