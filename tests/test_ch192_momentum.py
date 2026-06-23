"""Tests for aiinaction.ch192_momentum, including shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. They cover one explicit heavy-ball step, a fixed-length anisotropic
quadratic run (exactly reproducible, no convergence tolerance involved), a
convergence run to the known minimum, the plain-gradient-descent (beta=0) special
case, and Polyak's optimal momentum formula.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch192_momentum as mom

# --- Shared fixtures (mirrored in julia/AIInAction/test and rust/aiinaction/src) ---

# A single heavy-ball step: v' = beta*v + g, theta' = theta - alpha*v'.
STEP_THETA = [1.0, 2.0]
STEP_VEL = [0.5, -0.5]
STEP_GRAD = [0.2, 0.4]
STEP_ALPHA = 0.1
STEP_BETA = 0.9
EXPECTED_STEP_THETA = [0.935, 2.005]
EXPECTED_STEP_VEL = [0.65, -0.04999999999999999]

# Anisotropic quadratic f(theta) = 1/2 (theta-b)^T H (theta-b); grad = H (theta-b).
QUAD_H = [[3.0, 0.2], [0.2, 1.0]]
QUAD_B = [1.0, -2.0]
QUAD_THETA0 = [0.0, 0.0]
QUAD_ALPHA = 0.2
QUAD_BETA = 0.9

# Exactly five steps from theta0 with tol=0 (fully deterministic, no early stop).
EXPECTED_5ITER_THETA = [1.2893499712, -3.1595351615999996]
EXPECTED_5ITER_VEL = [1.884893344, 3.068733408]
EXPECTED_5ITER_HIST = [
    3.1622776601683795,
    1.90275589606234,
    1.339506583783745,
    2.072914156257514,
    1.9343282392460306,
]

# Optimal momentum for lambda in [1, 100] (condition number 100).
EXPECTED_OPT_BETA = 0.6694214876033059


def test_single_step_matches_fixture():
    th, v = mom.momentum_step(STEP_THETA, STEP_VEL, STEP_GRAD, STEP_ALPHA, STEP_BETA)
    assert th.tolist() == pytest.approx(EXPECTED_STEP_THETA)
    assert v.tolist() == pytest.approx(EXPECTED_STEP_VEL)


def test_beta_zero_step_is_plain_gradient_descent():
    th, v = mom.momentum_step([1.0], [7.0], [2.0], alpha=0.1, beta=0.0)
    # velocity collapses to the gradient; theta steps by -alpha * grad.
    assert v.tolist() == pytest.approx([2.0])
    assert th.tolist() == pytest.approx([0.8])


def test_fixed_five_iterations_match_fixture():
    g = mom.quadratic_gradient(QUAD_H, QUAD_B)
    r = mom.minimize(g, QUAD_THETA0, alpha=QUAD_ALPHA, beta=QUAD_BETA, max_iter=5, tol=0.0)
    assert r.n_iter == 5
    assert r.converged is False
    assert r.theta.tolist() == pytest.approx(EXPECTED_5ITER_THETA)
    assert r.velocity.tolist() == pytest.approx(EXPECTED_5ITER_VEL)
    assert r.history.tolist() == pytest.approx(EXPECTED_5ITER_HIST)


def test_converges_to_quadratic_minimum():
    g = mom.quadratic_gradient(QUAD_H, QUAD_B)
    r = mom.minimize(g, QUAD_THETA0, alpha=QUAD_ALPHA, beta=QUAD_BETA, max_iter=5000, tol=1e-10)
    assert r.converged is True
    assert r.theta.tolist() == pytest.approx(QUAD_B, abs=1e-7)
    assert r.grad_norm <= 1e-10


def test_beta_zero_recovers_gradient_descent_minimum():
    g = mom.quadratic_gradient([[2.0]], [5.0])
    r = mom.minimize(g, [0.0], alpha=0.3, beta=0.0, max_iter=500, tol=1e-12)
    assert r.converged is True
    assert r.theta.tolist() == pytest.approx([5.0])


def test_momentum_beats_plain_gd_on_illconditioned_quadratic():
    # Steep/shallow valley: momentum should need fewer steps to a fixed tolerance.
    H = [[50.0, 0.0], [0.0, 1.0]]
    g = mom.quadratic_gradient(H, [0.0, 0.0])
    plain = mom.minimize(g, [1.0, 1.0], alpha=0.02, beta=0.0, max_iter=10000, tol=1e-6)
    fast = mom.minimize(g, [1.0, 1.0], alpha=0.02, beta=0.9, max_iter=10000, tol=1e-6)
    assert plain.converged and fast.converged
    assert fast.n_iter < plain.n_iter


def test_optimal_beta_matches_fixture():
    assert mom.optimal_beta(1.0, 100.0) == pytest.approx(EXPECTED_OPT_BETA)


def test_optimal_beta_is_zero_for_isotropic():
    assert mom.optimal_beta(2.0, 2.0) == pytest.approx(0.0, abs=1e-15)


# --- Edge cases and validation ---


def test_bad_alpha_raises():
    g = mom.quadratic_gradient([[1.0]], [0.0])
    with pytest.raises(ValueError, match="learning rate"):
        mom.minimize(g, [1.0], alpha=0.0, beta=0.5)


def test_bad_beta_raises():
    g = mom.quadratic_gradient([[1.0]], [0.0])
    with pytest.raises(ValueError, match="momentum"):
        mom.minimize(g, [1.0], alpha=0.1, beta=1.0)


def test_negative_beta_raises():
    with pytest.raises(ValueError, match="momentum"):
        mom.momentum_step([1.0], [0.0], [1.0], alpha=0.1, beta=-0.1)


def test_step_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        mom.momentum_step([1.0, 2.0], [0.0], [1.0, 1.0], alpha=0.1, beta=0.5)


def test_non_finite_theta0_raises():
    g = mom.quadratic_gradient([[1.0]], [0.0])
    with pytest.raises(ValueError, match="non-finite"):
        mom.minimize(g, [float("nan")], alpha=0.1, beta=0.5)


def test_bad_max_iter_raises():
    g = mom.quadratic_gradient([[1.0]], [0.0])
    with pytest.raises(ValueError, match="max_iter"):
        mom.minimize(g, [1.0], alpha=0.1, beta=0.5, max_iter=0)


def test_quadratic_gradient_shape_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        mom.quadratic_gradient([[1.0, 0.0], [0.0, 1.0]], [1.0])


def test_optimal_beta_bad_order_raises():
    with pytest.raises(ValueError, match="lambda_max"):
        mom.optimal_beta(10.0, 1.0)


def test_n_features_property():
    g = mom.quadratic_gradient(QUAD_H, QUAD_B)
    r = mom.minimize(g, QUAD_THETA0, alpha=QUAD_ALPHA, beta=QUAD_BETA, max_iter=3, tol=0.0)
    assert r.n_features == 2
