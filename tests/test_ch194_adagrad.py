"""Tests for aiinaction.ch194_adagrad, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which keeps the three implementations at parity.
The test objective is the separable quadratic
``f(theta) = 0.5 * sum a_i (theta_i - b_i)^2`` with the coefficients below; its
gradient is ``a_i * (theta_i - b_i)``.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch194_adagrad as ag

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
A = [1.0, 4.0, 0.25]
B = [2.0, -1.0, 5.0]
THETA0 = [0.0, 0.0, 0.0]
ETA = 0.5
EPS = 1e-8

EXPECTED = {
    # Gradient at theta0.
    "grad0": [-2.0, 4.0, -1.25],
    # State after one AdaGrad step from theta0.
    "theta_after1": [0.4999999975, -0.49999999875, 0.49999999600000006],
    "acc_after1": [4.0, 16.0, 1.5625],
    "elr_after1": [0.24999999875, 0.1249999996875, 0.39999999680000003],
    # State after three steps.
    "theta_after3": [1.0163655300219518, -0.843601262889007, 1.0977190862943662],
    "acc_after3": [7.690000015612001, 21.22229126752294, 3.9125960778289572],
    # Full minimization converges to the analytic minimizer b.
    "min_theta": [2.0, -1.0, 5.0],
    "min_steps": 641,
}


def _grad(theta):
    return ag.quadratic_grad(theta, A, B)


def test_initial_accumulator_is_zero():
    s = ag.init_state(THETA0, learning_rate=ETA, epsilon=EPS)
    assert s.accumulator.tolist() == [0.0, 0.0, 0.0]


def test_initial_effective_rate_is_eta_over_eps():
    s = ag.init_state(THETA0, learning_rate=ETA, epsilon=EPS)
    assert ag.effective_learning_rate(s).tolist() == pytest.approx([ETA / EPS] * 3)


def test_grad0_matches_fixture():
    s = ag.init_state(THETA0, learning_rate=ETA, epsilon=EPS)
    assert _grad(s.theta).tolist() == pytest.approx(EXPECTED["grad0"])


def test_single_step_theta_matches_fixture():
    s = ag.init_state(THETA0, learning_rate=ETA, epsilon=EPS)
    ag.adagrad_step(s, _grad(s.theta))
    assert s.theta.tolist() == pytest.approx(EXPECTED["theta_after1"])


def test_single_step_accumulator_matches_fixture():
    s = ag.init_state(THETA0, learning_rate=ETA, epsilon=EPS)
    ag.adagrad_step(s, _grad(s.theta))
    assert s.accumulator.tolist() == pytest.approx(EXPECTED["acc_after1"])


def test_single_step_effective_rate_matches_fixture():
    s = ag.init_state(THETA0, learning_rate=ETA, epsilon=EPS)
    ag.adagrad_step(s, _grad(s.theta))
    assert ag.effective_learning_rate(s).tolist() == pytest.approx(EXPECTED["elr_after1"])


def test_three_steps_theta_matches_fixture():
    s = ag.init_state(THETA0, learning_rate=ETA, epsilon=EPS)
    for _ in range(3):
        ag.adagrad_step(s, _grad(s.theta))
    assert s.theta.tolist() == pytest.approx(EXPECTED["theta_after3"])


def test_three_steps_accumulator_matches_fixture():
    s = ag.init_state(THETA0, learning_rate=ETA, epsilon=EPS)
    for _ in range(3):
        ag.adagrad_step(s, _grad(s.theta))
    assert s.accumulator.tolist() == pytest.approx(EXPECTED["acc_after3"])


def test_minimize_converges_to_minimizer():
    res = ag.minimize(_grad, THETA0, learning_rate=ETA, epsilon=EPS, max_iter=10000, tol=1e-9)
    assert res.converged is True
    assert res.theta.tolist() == pytest.approx(EXPECTED["min_theta"], abs=1e-6)
    assert res.n_steps == EXPECTED["min_steps"]


def test_minimize_grad_norm_below_tol():
    res = ag.minimize(_grad, THETA0, learning_rate=ETA, epsilon=EPS, max_iter=10000, tol=1e-9)
    assert res.grad_norm <= 1e-9


def test_accumulator_is_monotone_nondecreasing():
    s = ag.init_state(THETA0, learning_rate=ETA, epsilon=EPS)
    prev = s.accumulator.copy()
    for _ in range(20):
        ag.adagrad_step(s, _grad(s.theta))
        assert np.all(s.accumulator >= prev - 1e-15)
        prev = s.accumulator.copy()


def test_effective_rate_is_nonincreasing():
    s = ag.init_state(THETA0, learning_rate=ETA, epsilon=EPS)
    prev = ag.effective_learning_rate(s)
    for _ in range(20):
        ag.adagrad_step(s, _grad(s.theta))
        cur = ag.effective_learning_rate(s)
        assert np.all(cur <= prev + 1e-15)
        prev = cur


def test_constant_gradient_decays_like_sqrt_t():
    # With a constant unit gradient, G = t and elr -> eta / (sqrt(t) + eps).
    s = ag.init_state([0.0], learning_rate=1.0, epsilon=1e-8)
    g = [1.0]
    for t in range(1, 101):
        ag.adagrad_step(s, g)
        expected = 1.0 / (np.sqrt(t) + 1e-8)
        assert float(ag.effective_learning_rate(s)[0]) == pytest.approx(expected)


def test_quadratic_value_and_grad_consistency():
    th = [1.5, 0.0, 4.0]
    assert ag.quadratic_value(th, A, B) == pytest.approx(
        0.5 * (1.0 * 0.25 + 4.0 * 1.0 + 0.25 * 1.0)
    )
    assert ag.quadratic_grad(th, A, B).tolist() == pytest.approx([-0.5, 4.0, -0.25])


def test_minimize_runs_full_iters_when_tol_zero():
    res = ag.minimize(_grad, THETA0, learning_rate=ETA, epsilon=EPS, max_iter=5, tol=0.0)
    assert res.n_steps == 5
    assert res.converged is False


# --- edge cases / validation ---


def test_init_rejects_non_positive_learning_rate():
    with pytest.raises(ValueError, match="learning_rate must be a positive"):
        ag.init_state(THETA0, learning_rate=0.0)


def test_init_rejects_non_positive_epsilon():
    with pytest.raises(ValueError, match="epsilon must be a positive"):
        ag.init_state(THETA0, epsilon=-1e-8)


def test_init_rejects_empty_theta():
    with pytest.raises(ValueError, match="non-empty"):
        ag.init_state([])


def test_init_rejects_non_finite_theta():
    with pytest.raises(ValueError, match="non-finite"):
        ag.init_state([0.0, float("nan")])


def test_step_rejects_length_mismatch():
    s = ag.init_state(THETA0, learning_rate=ETA, epsilon=EPS)
    with pytest.raises(ValueError, match="length"):
        ag.adagrad_step(s, [1.0, 2.0])


def test_step_rejects_non_finite_grad():
    s = ag.init_state(THETA0, learning_rate=ETA, epsilon=EPS)
    with pytest.raises(ValueError, match="non-finite"):
        ag.adagrad_step(s, [1.0, float("inf"), 0.0])


def test_minimize_rejects_bad_max_iter():
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        ag.minimize(_grad, THETA0, max_iter=0)


def test_minimize_rejects_negative_tol():
    with pytest.raises(ValueError, match="tol must be a nonnegative"):
        ag.minimize(_grad, THETA0, tol=-1.0)


def test_quadratic_rejects_non_positive_curvature():
    with pytest.raises(ValueError, match="strictly positive"):
        ag.quadratic_grad([0.0, 0.0], [1.0, 0.0], [0.0, 0.0])
