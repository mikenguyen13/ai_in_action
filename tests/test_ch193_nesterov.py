"""Tests for aiinaction.ch193_nesterov, including the shared cross-language fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity.

The test objective is the strictly convex quadratic
``f(x) = 0.5 x^T A x - b^T x`` with

    A = [[5, 1], [1, 3]],   b = [1, 2],

whose gradient is ``grad f(x) = A x - b`` and whose unique minimizer is
``x* = A^{-1} b = [1/14, 9/14]``. The step size ``eta = 1/6`` is below ``1/L``
(``L = lambda_max(A) ~= 5.414``), and the momentum fixtures use ``beta = 1/2``;
both are clean rationals so all three languages reproduce identical floats.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aiinaction.ch193_nesterov import nesterov_convex, nesterov_momentum

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
A = np.array([[5.0, 1.0], [1.0, 3.0]])
B = np.array([1.0, 2.0])
ETA = 1.0 / 6.0
BETA = 0.5
X_STAR = [0.07142857142857142, 0.6428571428571429]


def grad(x: np.ndarray) -> np.ndarray:
    return A @ np.asarray(x, dtype=float) - B


# Deterministic state after exactly 5 iterations, tol disabled.
CONVEX5_X = [0.06917044734055175, 0.6483203299202533]
CONVEX5_GRAD_NORM = 0.015285826582542654
CONVEX5_HISTORY = [
    0.8498365855987974,
    0.4746668747398631,
    0.2123582666221123,
    0.05611771750996081,
    0.015285826582542654,
]

MOM5_X = [0.061631944444444475, 0.666087962962963]
MOM5_GRAD_NORM = 0.06519733559752104
MOM5_HISTORY = [
    0.8498365855987974,
    0.3004626062886657,
    0.021960261528947072,
    0.07158169488919522,
    0.06519733559752104,
]


def test_convex_five_iterations_matches_fixture():
    res = nesterov_convex(grad, [0.0, 0.0], step_size=ETA, max_iter=5, tol=0.0)
    assert res.n_iter == 5
    assert res.converged is False
    assert res.x[0] == pytest.approx(CONVEX5_X[0])
    assert res.x[1] == pytest.approx(CONVEX5_X[1])
    assert res.grad_norm == pytest.approx(CONVEX5_GRAD_NORM)
    assert list(res.history) == pytest.approx(CONVEX5_HISTORY)


def test_momentum_five_iterations_matches_fixture():
    res = nesterov_momentum(grad, [0.0, 0.0], step_size=ETA, momentum=BETA, max_iter=5, tol=0.0)
    assert res.n_iter == 5
    assert res.converged is False
    assert res.x[0] == pytest.approx(MOM5_X[0])
    assert res.x[1] == pytest.approx(MOM5_X[1])
    assert res.grad_norm == pytest.approx(MOM5_GRAD_NORM)
    assert list(res.history) == pytest.approx(MOM5_HISTORY)


def test_convex_converges_to_minimizer():
    res = nesterov_convex(grad, [0.0, 0.0], step_size=ETA, max_iter=10000, tol=1e-10)
    assert res.converged is True
    assert res.x[0] == pytest.approx(X_STAR[0], abs=1e-7)
    assert res.x[1] == pytest.approx(X_STAR[1], abs=1e-7)
    assert res.grad_norm <= 1e-10


def test_momentum_converges_to_minimizer():
    res = nesterov_momentum(grad, [0.0, 0.0], step_size=ETA, momentum=BETA, max_iter=10000, tol=1e-10)
    assert res.converged is True
    assert res.x[0] == pytest.approx(X_STAR[0], abs=1e-7)
    assert res.x[1] == pytest.approx(X_STAR[1], abs=1e-7)
    assert res.grad_norm <= 1e-10


def test_already_at_optimum_converges_immediately():
    # Starting at x* the very first gradient norm is ~0, so we stop at iteration 1.
    res = nesterov_convex(grad, X_STAR, step_size=ETA, max_iter=100, tol=1e-9)
    assert res.converged is True
    assert res.n_iter == 1
    assert res.grad_norm <= 1e-9


def test_zero_momentum_is_plain_gradient_descent():
    # With beta = 0 the velocity form reduces to x_{k+1} = x_k - eta grad f(x_k).
    res = nesterov_momentum(grad, [0.0, 0.0], step_size=ETA, momentum=0.0, max_iter=1, tol=0.0)
    expected = -ETA * grad(np.array([0.0, 0.0]))
    assert res.x[0] == pytest.approx(expected[0])
    assert res.x[1] == pytest.approx(expected[1])


def test_history_length_equals_n_iter():
    res = nesterov_convex(grad, [0.0, 0.0], step_size=ETA, max_iter=7, tol=0.0)
    assert len(res.history) == res.n_iter == 7


def test_bad_step_size_raises():
    with pytest.raises(ValueError, match="step_size"):
        nesterov_convex(grad, [0.0, 0.0], step_size=0.0)
    with pytest.raises(ValueError, match="step_size"):
        nesterov_momentum(grad, [0.0, 0.0], step_size=-1.0, momentum=0.5)


def test_bad_momentum_raises():
    with pytest.raises(ValueError, match="momentum"):
        nesterov_momentum(grad, [0.0, 0.0], step_size=ETA, momentum=1.0)
    with pytest.raises(ValueError, match="momentum"):
        nesterov_momentum(grad, [0.0, 0.0], step_size=ETA, momentum=-0.1)


def test_bad_max_iter_raises():
    with pytest.raises(ValueError, match="max_iter"):
        nesterov_convex(grad, [0.0, 0.0], step_size=ETA, max_iter=0)


def test_bad_tol_raises():
    with pytest.raises(ValueError, match="tol"):
        nesterov_convex(grad, [0.0, 0.0], step_size=ETA, tol=-1.0)


def test_empty_x0_raises():
    with pytest.raises(ValueError, match="at least one"):
        nesterov_convex(grad, [], step_size=ETA)


def test_non_finite_x0_raises():
    with pytest.raises(ValueError, match="non-finite"):
        nesterov_convex(grad, [math.nan, 0.0], step_size=ETA)


def test_two_dimensional_x0_raises():
    with pytest.raises(ValueError, match="1-D"):
        nesterov_convex(grad, [[0.0, 0.0]], step_size=ETA)


def test_grad_shape_mismatch_raises():
    with pytest.raises(ValueError, match="gradient"):
        nesterov_convex(lambda x: np.array([0.0]), [0.0, 0.0], step_size=ETA)
