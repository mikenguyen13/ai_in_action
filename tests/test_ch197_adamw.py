"""Tests for aiinaction.ch197_adamw, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. The optimizer is exercised on (a) a single explicit step from a clean
state and (b) full minimization of a diagonal quadratic.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch197_adamw
from aiinaction.ch197_adamw import AdamWConfig, adamw_step, init_state, minimize

# ---------------------------------------------------------------------------
# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
# ---------------------------------------------------------------------------

# Single-step fixture: one AdamW update from zero state.
STEP_PARAMS = [1.0, -2.0, 0.5]
STEP_GRAD = [0.5, -1.0, 2.0]
STEP_CONFIG = dict(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)

EXPECTED = {
    "step1_theta": [0.899000002, -1.898000001, 0.3995000005],
    "step1_m": [0.04999999999999999, -0.09999999999999998, 0.19999999999999996],
    "step1_v": [0.0002500000000000002, 0.0010000000000000009, 0.0040000000000000036],
    # minimize a diagonal quadratic: grad(x) = A * (x - target).
    "quadratic_x_200": [1.999943097328645, -1.000007218001001],
    # 50 steps from (5, 5) with weight_decay = 0.01.
    "quadratic_wd_50": [1.7895443334390244, 0.604294026757625],
}

# Diagonal quadratic used by the minimization fixtures.
QUAD_A = np.array([3.0, 1.0])
QUAD_TARGET = np.array([2.0, -1.0])


def _quad_grad(x):
    return QUAD_A * (np.asarray(x) - QUAD_TARGET)


def test_single_step_theta_matches_fixture():
    st = init_state(3)
    theta = adamw_step(STEP_PARAMS, STEP_GRAD, st, AdamWConfig(**STEP_CONFIG))
    assert theta.tolist() == pytest.approx(EXPECTED["step1_theta"])


def test_single_step_moments_match_fixture():
    st = init_state(3)
    adamw_step(STEP_PARAMS, STEP_GRAD, st, AdamWConfig(**STEP_CONFIG))
    assert st.m.tolist() == pytest.approx(EXPECTED["step1_m"])
    assert st.v.tolist() == pytest.approx(EXPECTED["step1_v"])
    assert st.t == 1


def test_minimize_quadratic_matches_fixture():
    x = minimize(_quad_grad, [0.0, 0.0], AdamWConfig(lr=0.1), 200)
    assert x.tolist() == pytest.approx(EXPECTED["quadratic_x_200"])


def test_minimize_with_weight_decay_matches_fixture():
    cfg = AdamWConfig(lr=0.1, weight_decay=0.01)
    x = minimize(_quad_grad, [5.0, 5.0], cfg, 50)
    assert x.tolist() == pytest.approx(EXPECTED["quadratic_wd_50"])


# ---------------------------------------------------------------------------
# Properties and behaviour.
# ---------------------------------------------------------------------------


def test_minimize_converges_to_target():
    x = minimize(_quad_grad, [0.0, 0.0], AdamWConfig(lr=0.1), 500)
    assert x.tolist() == pytest.approx(QUAD_TARGET.tolist(), abs=1e-4)


def test_zero_weight_decay_is_plain_adam():
    # With weight_decay = 0 the update equals lr * mhat / (sqrt(vhat) + eps).
    st = init_state(2)
    params = [1.0, 2.0]
    grad = [0.3, -0.4]
    cfg = AdamWConfig(lr=0.05, weight_decay=0.0)
    theta = adamw_step(params, grad, st, cfg)
    # On step 1, mhat = g and vhat = g*g, so the adaptive step is sign(g) * lr.
    expected = [1.0 - 0.05 * 1.0, 2.0 - 0.05 * (-1.0)]
    assert theta.tolist() == pytest.approx(expected, abs=1e-6)


def test_weight_decay_shrinks_when_gradient_zero():
    # If the gradient is exactly zero, only decoupled decay acts: theta *= (1 - lr*wd).
    st = init_state(2)
    cfg = AdamWConfig(lr=0.1, weight_decay=0.2)
    theta = adamw_step([4.0, -6.0], [0.0, 0.0], st, cfg)
    assert theta.tolist() == pytest.approx([4.0 * (1 - 0.02), -6.0 * (1 - 0.02)])


def test_step_increments_t():
    st = init_state(2)
    cfg = AdamWConfig(lr=0.1)
    params = [1.0, 1.0]
    for expected_t in (1, 2, 3):
        params = adamw_step(params, [0.1, 0.1], st, cfg)
        assert st.t == expected_t


def test_params_not_mutated_in_place():
    st = init_state(2)
    params = [1.0, 2.0]
    adamw_step(params, [0.5, 0.5], st, AdamWConfig(lr=0.1))
    assert params == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Validation / edge cases.
# ---------------------------------------------------------------------------


def test_bad_lr_raises():
    with pytest.raises(ValueError, match="lr must be positive"):
        AdamWConfig(lr=0.0)


def test_bad_beta1_raises():
    with pytest.raises(ValueError, match="beta1 must be in"):
        AdamWConfig(beta1=1.0)


def test_bad_beta2_raises():
    with pytest.raises(ValueError, match="beta2 must be in"):
        AdamWConfig(beta2=-0.1)


def test_bad_eps_raises():
    with pytest.raises(ValueError, match="eps must be positive"):
        AdamWConfig(eps=0.0)


def test_negative_weight_decay_raises():
    with pytest.raises(ValueError, match="weight_decay must be non-negative"):
        AdamWConfig(weight_decay=-0.01)


def test_init_state_bad_n_raises():
    with pytest.raises(ValueError, match="n_params must be >= 1"):
        init_state(0)


def test_grad_length_mismatch_raises():
    st = init_state(3)
    with pytest.raises(ValueError, match="expected 3"):
        adamw_step([1.0, 2.0, 3.0], [0.1, 0.2], st, AdamWConfig())


def test_non_finite_grad_raises():
    st = init_state(2)
    with pytest.raises(ValueError, match="non-finite"):
        adamw_step([1.0, 2.0], [float("nan"), 0.1], st, AdamWConfig())


def test_minimize_bad_n_steps_raises():
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        minimize(_quad_grad, [0.0, 0.0], AdamWConfig(), 0)


def test_module_exports():
    assert hasattr(ch197_adamw, "adamw_step")
    assert hasattr(ch197_adamw, "minimize")
