"""Tests for aiinaction.ch130_gmm_em, including the shared cross-language fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three libraries at
parity. Tolerances are 1e-9 to match the Rust/Julia suites.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aiinaction.ch130_gmm_em import (
    GMMParams,
    e_step,
    fit_gmm,
    gaussian_pdf,
    log_likelihood,
    m_step,
)

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
# A tiny 1-D, two-cluster data set: four points near 0 and four near 10.
X = [
    [0.0],
    [1.0],
    [2.0],
    [1.0],
    [9.0],
    [10.0],
    [11.0],
    [10.0],
]

INIT = GMMParams(
    weights=np.array([0.5, 0.5]),
    means=np.array([[0.0], [8.0]]),
    covariances=np.array([[[1.0]], [[1.0]]]),
)

TOL = 1e-9


def test_gaussian_pdf_standard_normal():
    # N(0|0,1) = 1/sqrt(2 pi)
    assert gaussian_pdf([0.0], [0.0], [[1.0]]) == pytest.approx(
        1.0 / math.sqrt(2.0 * math.pi), abs=TOL
    )


def test_gaussian_pdf_2d():
    # Standard bivariate normal at the origin = 1/(2 pi).
    val = gaussian_pdf([0.0, 0.0], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    assert val == pytest.approx(1.0 / (2.0 * math.pi), abs=TOL)


def test_e_step_rows_sum_to_one():
    gamma = e_step(X, INIT)
    assert gamma.shape == (8, 2)
    np.testing.assert_allclose(gamma.sum(axis=1), np.ones(8), atol=TOL)


def test_e_step_fixture_first_row():
    # Point x=0 should be assigned almost entirely to component 0.
    gamma = e_step(X, INIT)
    assert gamma[0, 0] == pytest.approx(0.9999999999999873, abs=TOL)
    assert gamma[0, 1] == pytest.approx(1.2664165549094016e-14, abs=TOL)


def test_log_likelihood_fixture_initial():
    ll = log_likelihood(X, INIT)
    assert ll == pytest.approx(-24.89668559750626, abs=TOL)


def test_m_step_recovers_two_clusters():
    gamma = e_step(X, INIT)
    updated = m_step(X, gamma, reg_covar=0.0)
    # Effective weights near 0.5/0.5; means near 1.0 and 10.0.
    np.testing.assert_allclose(updated.weights, [0.5, 0.5], atol=1e-6)
    assert updated.means[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert updated.means[1, 0] == pytest.approx(10.0, abs=1e-6)


def test_fit_gmm_converges_to_fixture():
    result = fit_gmm(X, INIT, max_iter=200, tol=1e-10, reg_covar=0.0)
    assert result.converged
    # Monotonic nondecreasing log-likelihood.
    for a, b in zip(result.history, result.history[1:]):
        assert b >= a - 1e-12
    assert result.log_likelihood == pytest.approx(-14.124096987877165, abs=1e-7)
    means = sorted(float(m[0]) for m in result.params.means)
    assert means[0] == pytest.approx(1.0, abs=1e-6)
    assert means[1] == pytest.approx(10.0, abs=1e-6)
    np.testing.assert_allclose(sorted(result.params.weights), [0.5, 0.5], atol=1e-6)


def test_gaussian_pdf_bad_cov_raises():
    with pytest.raises(ValueError, match="positive definite"):
        gaussian_pdf([0.0], [0.0], [[0.0]])


def test_gaussian_pdf_shape_mismatch_raises():
    with pytest.raises(ValueError, match="mean length"):
        gaussian_pdf([0.0, 1.0], [0.0], [[1.0, 0.0], [0.0, 1.0]])


def test_e_step_weights_not_normalized_raises():
    bad = GMMParams(
        weights=np.array([0.3, 0.3]),
        means=np.array([[0.0], [8.0]]),
        covariances=np.array([[[1.0]], [[1.0]]]),
    )
    with pytest.raises(ValueError, match="sum to 1"):
        e_step(X, bad)


def test_m_step_negative_reg_raises():
    gamma = e_step(X, INIT)
    with pytest.raises(ValueError, match="reg_covar"):
        m_step(X, gamma, reg_covar=-1.0)


def test_fit_gmm_bad_max_iter_raises():
    with pytest.raises(ValueError, match="max_iter"):
        fit_gmm(X, INIT, max_iter=0)
