"""Tests for aiinaction.ch201_batch_norm, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. The data matrix is a small 4x3 mini-batch with hand-picked scale/shift.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch201_batch_norm as bn

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
X = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [2.0, 0.0, 1.0],
]
GAMMA = [1.0, 2.0, 0.5]
BETA = [0.0, 1.0, -1.0]
EPS = 1e-5

DY = [
    [0.1, -0.2, 0.3],
    [0.4, 0.5, -0.6],
    [-0.7, 0.8, 0.9],
    [1.0, -1.1, 1.2],
]

RUNNING_MEAN = [3.0, 4.0, 5.0]
RUNNING_VAR = [2.0, 3.0, 4.0]

EXPECTED = {
    "mean": [3.5, 3.75, 4.75],
    "var": [5.25, 9.1875, 9.1875],
    "xhat_row0": [-1.0910884120486357, -0.5773499549856541, -0.5773499549856541],
    "y_row0": [-1.0910884120486357, -0.15469990997130822, -1.288674977492827],
    "y_row3": [-0.6546530472291814, -1.4743569499385174, -1.6185892374846294],
    "dgamma": [-1.745741459277817, 2.80427120993032, -0.6433328069840143],
    "dbeta": [0.8, 0.0, 1.8],
    "dx_row0": [-0.2514695048226982, 0.1351074538761989, -0.040061000612684985],
    "dx_row3": [0.2244527108511118, -0.1535117479685656, 0.09089478082556916],
    "yinf_row0": [-1.4142100268524473, -1.3093972277663308, -1.4999993750011718],
}


def test_forward_mean_var_match_fixture():
    _, cache = bn.batch_norm_forward(X, GAMMA, BETA, eps=EPS)
    assert cache.mean.tolist() == pytest.approx(EXPECTED["mean"])
    assert cache.var.tolist() == pytest.approx(EXPECTED["var"])


def test_forward_xhat_matches_fixture():
    _, cache = bn.batch_norm_forward(X, GAMMA, BETA, eps=EPS)
    assert cache.x_hat[0].tolist() == pytest.approx(EXPECTED["xhat_row0"])


def test_forward_output_matches_fixture():
    y, _ = bn.batch_norm_forward(X, GAMMA, BETA, eps=EPS)
    assert y[0].tolist() == pytest.approx(EXPECTED["y_row0"])
    assert y[3].tolist() == pytest.approx(EXPECTED["y_row3"])


def test_backward_param_grads_match_fixture():
    _, cache = bn.batch_norm_forward(X, GAMMA, BETA, eps=EPS)
    _, dgamma, dbeta = bn.batch_norm_backward(DY, cache)
    assert dgamma.tolist() == pytest.approx(EXPECTED["dgamma"])
    assert dbeta.tolist() == pytest.approx(EXPECTED["dbeta"])


def test_backward_input_grad_matches_fixture():
    _, cache = bn.batch_norm_forward(X, GAMMA, BETA, eps=EPS)
    dx, _, _ = bn.batch_norm_backward(DY, cache)
    assert dx[0].tolist() == pytest.approx(EXPECTED["dx_row0"])
    assert dx[3].tolist() == pytest.approx(EXPECTED["dx_row3"])


def test_inference_matches_fixture():
    y = bn.batch_norm_inference(X, GAMMA, BETA, RUNNING_MEAN, RUNNING_VAR, eps=EPS)
    assert y[0].tolist() == pytest.approx(EXPECTED["yinf_row0"])


def test_normalized_output_has_zero_mean_unit_var():
    # With gamma=1, beta=0 the output equals x_hat, which is centered and (for
    # eps -> 0) unit population variance per feature.
    d = 3
    y, _ = bn.batch_norm_forward(X, [1.0] * d, [0.0] * d, eps=1e-12)
    arr = np.asarray(y)
    assert arr.mean(axis=0).tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert arr.var(axis=0).tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)


def test_gamma_beta_recover_identity():
    # Setting gamma = sqrt(var + eps) and beta = mean recovers the input exactly.
    eps = 1e-5
    _, cache = bn.batch_norm_forward(X, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], eps=eps)
    g = np.sqrt(cache.var + eps)
    b = cache.mean
    y, _ = bn.batch_norm_forward(X, g.tolist(), b.tolist(), eps=eps)
    assert y.ravel().tolist() == pytest.approx(
        np.asarray(X, dtype=float).ravel().tolist(), abs=1e-9
    )


def test_backward_matches_numerical_gradient():
    # Finite-difference check of dx against a scalar loss L = sum(w * y).
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(5, 3))
    gamma = np.array([0.7, 1.3, -0.5])
    beta = np.array([0.2, -0.4, 1.1])
    w = rng.normal(size=(5, 3))
    eps = 1e-5

    def loss(x):
        y, _ = bn.batch_norm_forward(x, gamma, beta, eps=eps)
        return float(np.sum(w * y))

    y, cache = bn.batch_norm_forward(arr, gamma, beta, eps=eps)
    dx, _, _ = bn.batch_norm_backward(w, cache)

    h = 1e-6
    num = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            xp = arr.copy(); xp[i, j] += h
            xm = arr.copy(); xm[i, j] -= h
            num[i, j] = (loss(xp) - loss(xm)) / (2 * h)
    assert dx.ravel().tolist() == pytest.approx(num.ravel().tolist(), abs=1e-6)


def test_dbeta_equals_column_sum_of_dy():
    _, cache = bn.batch_norm_forward(X, GAMMA, BETA, eps=EPS)
    _, _, dbeta = bn.batch_norm_backward(DY, cache)
    assert dbeta.tolist() == pytest.approx(np.asarray(DY).sum(axis=0).tolist())


def test_non_finite_raises():
    with pytest.raises(ValueError, match="non-finite"):
        bn.batch_norm_forward([[1.0, 2.0], [float("nan"), 3.0]], [1.0, 1.0], [0.0, 0.0])


def test_gamma_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        bn.batch_norm_forward(X, [1.0, 2.0], BETA)


def test_eps_must_be_positive():
    with pytest.raises(ValueError, match="eps must be positive"):
        bn.batch_norm_forward(X, GAMMA, BETA, eps=0.0)


def test_backward_shape_mismatch_raises():
    _, cache = bn.batch_norm_forward(X, GAMMA, BETA, eps=EPS)
    with pytest.raises(ValueError, match="cache was built for"):
        bn.batch_norm_backward([[1.0, 2.0, 3.0]], cache)


def test_inference_negative_running_var_raises():
    with pytest.raises(ValueError, match="non-negative"):
        bn.batch_norm_inference(X, GAMMA, BETA, RUNNING_MEAN, [-1.0, 1.0, 1.0])
