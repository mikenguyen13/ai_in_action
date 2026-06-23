"""Tests for aiinaction.ch202_layer_norm (LayerNorm and RMSNorm).

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. The shared example uses x = [2, 4, 6, 8] with gamma = [1.5, 0.5, 1.0, 2.0]
and beta = [0.1, -0.2, 0.0, 0.3], evaluated with eps = 1e-5.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch202_layer_norm as ln

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
X = [2.0, 4.0, 6.0, 8.0]
GAMMA = [1.5, 0.5, 1.0, 2.0]
BETA = [0.1, -0.2, 0.0, 0.3]
EPS = 1e-5

EXPECTED_LN_PLAIN = [
    -1.3416394448610998,
    -0.4472131482870333,
    0.4472131482870333,
    1.3416394448610998,
]
EXPECTED_LN_AFFINE = [
    -1.9124591672916496,
    -0.42360657414351666,
    0.4472131482870333,
    2.9832788897221993,
]
EXPECTED_RMS_PLAIN = [
    0.36514831081206406,
    0.7302966216241281,
    1.0954449324361921,
    1.4605932432482562,
]
EXPECTED_RMS_GAMMA = [
    0.5477224662180961,
    0.36514831081206406,
    1.0954449324361921,
    2.9211864864965125,
]


def test_layer_norm_plain_matches_fixture():
    got = ln.layer_norm(X, eps=EPS)
    assert got == pytest.approx(EXPECTED_LN_PLAIN)


def test_layer_norm_affine_matches_fixture():
    got = ln.layer_norm(X, GAMMA, BETA, eps=EPS)
    assert got == pytest.approx(EXPECTED_LN_AFFINE)


def test_rms_norm_plain_matches_fixture():
    got = ln.rms_norm(X, eps=EPS)
    assert got == pytest.approx(EXPECTED_RMS_PLAIN)


def test_rms_norm_gamma_matches_fixture():
    got = ln.rms_norm(X, GAMMA, eps=EPS)
    assert got == pytest.approx(EXPECTED_RMS_GAMMA)


def test_layer_norm_zero_mean_unit_variance():
    # With default affine params, normalized output has mean 0 and population
    # variance ~1 (up to eps).
    y = ln.layer_norm(X, eps=0.0)
    assert float(np.mean(y)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.mean(y * y)) == pytest.approx(1.0)


def test_layer_norm_shift_scale_invariance():
    # LN(a*x + b) == LN(x) for the plain (gamma=1, beta=0) operator.
    a, b = 3.0, 5.0
    base = ln.layer_norm(X, eps=0.0)
    shifted = ln.layer_norm([a * v + b for v in X], eps=0.0)
    assert shifted == pytest.approx(base)


def test_rms_norm_scale_invariance_direction():
    # RMSNorm is invariant to positive uniform scaling but NOT to shifts.
    a = 4.0
    base = ln.rms_norm(X, eps=0.0)
    scaled = ln.rms_norm([a * v for v in X], eps=0.0)
    assert scaled == pytest.approx(base)


def test_constant_vector_layer_norm_is_finite_with_eps():
    # Zero variance: eps keeps the denominator away from zero, output is all-beta.
    y = ln.layer_norm([3.0, 3.0, 3.0], eps=1e-5)
    assert np.all(np.isfinite(y))
    assert y == pytest.approx([0.0, 0.0, 0.0])


def test_apply_layer_norm_rows_independent():
    M = [[2.0, 4.0, 6.0, 8.0], [1.0, 1.0, 1.0, 1.0]]
    out = ln.apply_layer_norm(M, eps=EPS)
    assert out.shape == (2, 4)
    assert out[0] == pytest.approx(EXPECTED_LN_PLAIN)
    # Constant row normalizes to all zeros (gamma=1, beta=0).
    assert out[1] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_apply_rms_norm_rows_independent():
    M = [[2.0, 4.0, 6.0, 8.0], [2.0, 4.0, 6.0, 8.0]]
    out = ln.apply_rms_norm(M, eps=EPS)
    assert out.shape == (2, 4)
    assert out[0] == pytest.approx(EXPECTED_RMS_PLAIN)
    assert out[1] == pytest.approx(EXPECTED_RMS_PLAIN)


def test_empty_vector_raises():
    with pytest.raises(ValueError, match="at least one feature"):
        ln.layer_norm([])


def test_non_finite_raises():
    with pytest.raises(ValueError, match="non-finite"):
        ln.rms_norm([1.0, float("nan"), 3.0])


def test_gamma_wrong_length_raises():
    with pytest.raises(ValueError, match="gamma must have length"):
        ln.layer_norm(X, [1.0, 2.0])


def test_beta_wrong_length_raises():
    with pytest.raises(ValueError, match="beta must have length"):
        ln.layer_norm(X, GAMMA, [1.0, 2.0])


def test_negative_eps_raises():
    with pytest.raises(ValueError, match="eps must be non-negative"):
        ln.rms_norm(X, eps=-1e-6)


def test_2d_input_to_vector_raises():
    with pytest.raises(ValueError, match="1-D vector"):
        ln.layer_norm([[1.0, 2.0], [3.0, 4.0]])
