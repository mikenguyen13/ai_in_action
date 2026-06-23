"""Tests for aiinaction.ch195_rmsprop, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. They are produced by hand-checkable RMSProp updates on small vectors.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch195_rmsprop

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
PARAMS0 = [1.0, -2.0, 0.5]
GRAD = [0.1, -0.3, 2.0]
LR = 0.01
BETA = 0.9
EPS = 1e-8

EXPECTED = {
    # After one rmsprop_step on PARAMS0 with constant gradient GRAD.
    "v1": [0.0009999999999999998, 0.008999999999999998, 0.3999999999999999],
    "params1": [0.968377233398313, -1.9683772267316493, 0.4683772238983162],
    # After a second step with the same gradient.
    "v2": [0.0018999999999999998, 0.017099999999999997, 0.7599999999999998],
    "params2": [0.9454356652744136, -1.9454356550989789, 0.44543565077441793],
    # minimize(grad = c*x with c=[1,4], params0=[2,2], 10 steps, lr=0.1).
    "min_params": [0.6027968620415073, 0.6027968532310017],
    "min_v": [0.8446178332851308, 13.513885189461563],
}


def test_init_state_zeros_v_and_count():
    s = ch195_rmsprop.init_state(PARAMS0, lr=LR, beta=BETA, eps=EPS)
    assert s.n_params == 3
    assert s.v.tolist() == [0.0, 0.0, 0.0]
    assert s.step_count == 0
    assert s.params.tolist() == PARAMS0


def test_single_step_v_matches_fixture():
    s = ch195_rmsprop.init_state(PARAMS0, lr=LR, beta=BETA, eps=EPS)
    s1 = ch195_rmsprop.rmsprop_step(s, GRAD)
    assert s1.v.tolist() == pytest.approx(EXPECTED["v1"])
    assert s1.step_count == 1


def test_single_step_params_match_fixture():
    s = ch195_rmsprop.init_state(PARAMS0, lr=LR, beta=BETA, eps=EPS)
    s1 = ch195_rmsprop.rmsprop_step(s, GRAD)
    assert s1.params.tolist() == pytest.approx(EXPECTED["params1"])


def test_two_steps_match_fixture():
    s = ch195_rmsprop.init_state(PARAMS0, lr=LR, beta=BETA, eps=EPS)
    s2 = ch195_rmsprop.rmsprop_step(ch195_rmsprop.rmsprop_step(s, GRAD), GRAD)
    assert s2.v.tolist() == pytest.approx(EXPECTED["v2"])
    assert s2.params.tolist() == pytest.approx(EXPECTED["params2"])
    assert s2.step_count == 2


def test_minimize_quadratic_matches_fixture():
    c = np.array([1.0, 4.0])
    s = ch195_rmsprop.minimize(lambda x: c * x, [2.0, 2.0], 10, lr=0.1, beta=BETA, eps=EPS)
    assert s.params.tolist() == pytest.approx(EXPECTED["min_params"])
    assert s.v.tolist() == pytest.approx(EXPECTED["min_v"])
    assert s.step_count == 10


def test_first_step_is_sign_descent_of_magnitude_lr():
    # With v_0 = 0, the first step moves each coordinate by almost exactly
    # -lr * sign(g) because sqrt(v1) = sqrt((1-beta)) * |g| and the update is
    # lr * g / (sqrt((1-beta)) g^2 ... ) -> lr / sqrt(1-beta) * sign(g), but here
    # we just check the descent direction opposes the gradient sign.
    s = ch195_rmsprop.init_state(PARAMS0, lr=LR, beta=BETA, eps=EPS)
    s1 = ch195_rmsprop.rmsprop_step(s, GRAD)
    delta = s1.params - np.asarray(PARAMS0)
    for d, g in zip(delta.tolist(), GRAD):
        assert np.sign(d) == -np.sign(g)


def test_minimize_converges_to_origin_on_quadratic():
    c = np.array([1.0, 9.0])
    s = ch195_rmsprop.minimize(lambda x: c * x, [1.0, 1.0], 300, lr=0.05)
    # RMSProp normalizes the step, so progress slows near the optimum; both
    # coordinates should nonetheless be driven well below their start of 1.0.
    assert np.all(np.abs(s.params) < 0.05)


def test_zero_steps_returns_initial_params():
    c = np.array([1.0, 4.0])
    s = ch195_rmsprop.minimize(lambda x: c * x, [2.0, 2.0], 0, lr=0.1)
    assert s.params.tolist() == [2.0, 2.0]
    assert s.step_count == 0


def test_beta_zero_is_signed_gradient_normalization():
    # With beta=0, v_t = g_t^2, so sqrt(v) = |g| and the update is
    # -lr * g / (|g| + eps) ~= -lr * sign(g).
    s = ch195_rmsprop.init_state([5.0], lr=0.1, beta=0.0, eps=1e-12)
    s1 = ch195_rmsprop.rmsprop_step(s, [2.0])
    assert s1.params.tolist() == pytest.approx([5.0 - 0.1 * 2.0 / (2.0 + 1e-12)])


def test_grad_length_mismatch_raises():
    s = ch195_rmsprop.init_state(PARAMS0)
    with pytest.raises(ValueError, match="entries but state has"):
        ch195_rmsprop.rmsprop_step(s, [0.1, 0.2])


def test_non_finite_params_raises():
    with pytest.raises(ValueError, match="non-finite"):
        ch195_rmsprop.init_state([1.0, float("nan")])


def test_non_finite_grad_raises():
    s = ch195_rmsprop.init_state(PARAMS0)
    with pytest.raises(ValueError, match="non-finite"):
        ch195_rmsprop.rmsprop_step(s, [0.1, float("inf"), 0.2])


def test_empty_params_raises():
    with pytest.raises(ValueError, match="non-empty"):
        ch195_rmsprop.init_state([])


def test_bad_lr_raises():
    with pytest.raises(ValueError, match="lr must be positive"):
        ch195_rmsprop.init_state(PARAMS0, lr=0.0)


def test_bad_beta_raises():
    with pytest.raises(ValueError, match="beta must be in"):
        ch195_rmsprop.init_state(PARAMS0, beta=1.0)


def test_bad_eps_raises():
    with pytest.raises(ValueError, match="eps must be positive"):
        ch195_rmsprop.init_state(PARAMS0, eps=-1e-8)


def test_negative_n_steps_raises():
    with pytest.raises(ValueError, match="non-negative"):
        ch195_rmsprop.minimize(lambda x: x, [1.0], -1)
