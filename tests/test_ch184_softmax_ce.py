"""Tests for aiinaction.ch184_softmax_ce, including shared cross-language fixtures.

The fixtures Z184 / Y184 and the EXPECTED numbers below are the single source of
truth: the Julia and Rust suites assert against the same values, which keeps the
three implementations at parity.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction.ch184_softmax_ce import (
    cross_entropy_grad,
    cross_entropy_loss,
    log_softmax,
    softmax,
)

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
Z184 = [[2.0, 1.0, 0.1], [0.5, 2.5, 0.3], [1.0, 1.0, 1.0]]
Y184 = [0, 1, 2]

SOFTMAX_ROW0 = [0.6590011388859679, 0.24243297070471392, 0.09856589040931818]
LOG_SOFTMAX_ROW0 = [-0.41703001627783354, -1.4170300162778335, -2.3170300162778332]
LOSS = 0.5785639426554937
LOSS_SMOOTH = 0.6574528315443826
GRAD = [
    [-0.1136662870380107, 0.08081099023490464, 0.03285529680310606],
    [0.03620124343567079, -0.06584031473611691, 0.029639071300446088],
    [0.1111111111111111, 0.1111111111111111, -0.22222222222222224],
]
GRAD_SMOOTH = [
    [-0.09144406481578848, 0.06969987912379354, 0.02174418569199495],
    [0.025090132324559682, -0.0436180925138947, 0.018527960189334978],
    [0.09999999999999999, 0.09999999999999999, -0.20000000000000004],
]


def test_softmax_matches_fixture():
    p = softmax(Z184)
    assert p[0] == pytest.approx(SOFTMAX_ROW0)


def test_softmax_rows_sum_to_one():
    p = softmax(Z184)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p > 0.0)


def test_log_softmax_matches_fixture():
    ls = log_softmax(Z184)
    assert ls[0] == pytest.approx(LOG_SOFTMAX_ROW0)


def test_log_softmax_equals_log_of_softmax():
    assert np.allclose(log_softmax(Z184), np.log(softmax(Z184)))


def test_loss_matches_fixture():
    assert cross_entropy_loss(Z184, Y184) == pytest.approx(LOSS)


def test_loss_smoothed_matches_fixture():
    assert cross_entropy_loss(Z184, Y184, label_smoothing=0.1) == pytest.approx(LOSS_SMOOTH)


def test_grad_matches_fixture():
    g = cross_entropy_grad(Z184, Y184)
    assert g == pytest.approx(np.array(GRAD))


def test_grad_smoothed_matches_fixture():
    g = cross_entropy_grad(Z184, Y184, label_smoothing=0.1)
    assert g == pytest.approx(np.array(GRAD_SMOOTH))


def test_grad_rows_sum_to_zero():
    # Each row is (p - q) / N with p and q both distributions, so rows sum to 0.
    g = cross_entropy_grad(Z184, Y184)
    assert np.allclose(g.sum(axis=1), 0.0, atol=1e-12)


def test_uniform_logits_loss_is_log_k():
    # Equal logits => uniform softmax => loss = log(K).
    z = [[0.0, 0.0, 0.0]]
    assert cross_entropy_loss(z, [1]) == pytest.approx(np.log(3.0))


def test_gradient_matches_numerical_finite_difference():
    z = np.array(Z184, dtype=float)
    g = cross_entropy_grad(z, Y184)
    h = 1e-6
    num = np.zeros_like(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            zp = z.copy()
            zp[i, j] += h
            zm = z.copy()
            zm[i, j] -= h
            num[i, j] = (cross_entropy_loss(zp, Y184) - cross_entropy_loss(zm, Y184)) / (2 * h)
    assert np.allclose(g, num, atol=1e-7)


def test_stable_under_large_logits():
    # Naive exp would overflow; stable form must stay finite and correct.
    z = [[1000.0, 0.0], [0.0, 1000.0]]
    p = softmax(z)
    assert np.all(np.isfinite(p))
    assert p[0, 0] == pytest.approx(1.0)
    loss = cross_entropy_loss(z, [0, 1])
    assert np.isfinite(loss)
    assert loss == pytest.approx(0.0, abs=1e-9)


def test_bad_dimensions_raise():
    with pytest.raises(ValueError, match="2-D"):
        softmax([1.0, 2.0])


def test_too_few_classes_raises():
    with pytest.raises(ValueError, match="2 classes"):
        cross_entropy_loss([[1.0]], [0])


def test_non_finite_raises():
    with pytest.raises(ValueError, match="non-finite"):
        softmax([[1.0, float("inf")]])


def test_label_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        cross_entropy_loss(Z184, [0, 1, 3])


def test_label_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        cross_entropy_loss(Z184, [0, 1])


def test_bad_smoothing_raises():
    with pytest.raises(ValueError, match="label_smoothing"):
        cross_entropy_loss(Z184, Y184, label_smoothing=1.0)
