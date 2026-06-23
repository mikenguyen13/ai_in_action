"""Tests for aiinaction.ch083_basis_regression and the shared fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three libraries at
parity. The polynomial data lie exactly on the parabola ``y = 1 + 2x - x^2`` so
the degree-2 OLS fit recovers the coefficients ``[1, 2, -1]`` exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch083_basis_regression as br

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
X = [-2.0, -1.0, 0.0, 1.0, 2.0]
Y = [-7.0, -2.0, 1.0, 2.0, 1.0]

EXPECTED_OLS_COEF = [1.0, 2.0, -1.0]
EXPECTED_RIDGE_COEF = [0.59090909090909061, 1.8181818181818181, -0.8545454545454545]
EXPECTED_DOF_OLS = 3.0
EXPECTED_DOF_RIDGE = 2.5363636363636362

# RBF fixture.
RBF_X = [-1.0, 0.0, 1.0]
RBF_CENTERS = [-1.0, 1.0]
RBF_WIDTH = 1.0
EXPECTED_RBF_DESIGN = [
    [1.0, 1.0, 0.1353352832366127],
    [1.0, 0.6065306597126334, 0.6065306597126334],
    [1.0, 0.1353352832366127, 1.0],
]


def test_polynomial_design_shape_and_values():
    phi = br.polynomial_design(X, 2)
    assert phi.shape == (5, 3)
    assert phi[:, 0].tolist() == [1.0] * 5
    assert phi[:, 1].tolist() == X
    assert phi[:, 2].tolist() == [4.0, 1.0, 0.0, 1.0, 4.0]


def test_ols_recovers_exact_polynomial():
    phi = br.polynomial_design(X, 2)
    beta = br.fit_ridge(phi, Y, 0.0)
    assert beta == pytest.approx(EXPECTED_OLS_COEF, abs=1e-9)


def test_predict_reproduces_targets():
    phi = br.polynomial_design(X, 2)
    beta = br.fit_ridge(phi, Y, 0.0)
    assert br.predict(phi, beta) == pytest.approx(Y, abs=1e-9)


def test_ridge_shrinks_coefficients():
    phi = br.polynomial_design(X, 2)
    beta = br.fit_ridge(phi, Y, 1.0)
    assert beta == pytest.approx(EXPECTED_RIDGE_COEF, abs=1e-9)
    # Ridge shrinks the coefficient norm relative to OLS.
    ols = br.fit_ridge(phi, Y, 0.0)
    assert np.linalg.norm(beta) < np.linalg.norm(ols)


def test_effective_dof():
    phi = br.polynomial_design(X, 2)
    assert br.effective_dof(phi, 0.0) == pytest.approx(EXPECTED_DOF_OLS, abs=1e-9)
    assert br.effective_dof(phi, 1.0) == pytest.approx(EXPECTED_DOF_RIDGE, abs=1e-9)


def test_rbf_design():
    phi = br.rbf_design(RBF_X, RBF_CENTERS, RBF_WIDTH, include_bias=True)
    assert phi.shape == (3, 3)
    np.testing.assert_allclose(phi, EXPECTED_RBF_DESIGN, atol=1e-12)


def test_estimator_fit_predict():
    model = br.BasisRegression(degree=2, penalty=0.0).fit(X, Y)
    assert model.coef_ == pytest.approx(EXPECTED_OLS_COEF, abs=1e-9)
    assert model.predict(X) == pytest.approx(Y, abs=1e-9)
    assert model.predict([3.0])[0] == pytest.approx(1.0 + 2.0 * 3.0 - 3.0 ** 2, abs=1e-9)


def test_estimator_rbf_basis():
    model = br.BasisRegression(
        basis="rbf", centers=RBF_CENTERS, width=RBF_WIDTH, penalty=0.0
    )
    model.fit(RBF_X, [0.0, 1.0, 0.0])
    # Interpolates the three points with three basis functions.
    assert model.predict(RBF_X) == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


# --- edge cases and validation ---------------------------------------------


def test_polynomial_design_negative_degree_raises():
    with pytest.raises(ValueError, match="non-negative"):
        br.polynomial_design(X, -1)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="non-empty"):
        br.polynomial_design([], 2)


def test_fit_length_mismatch_raises():
    phi = br.polynomial_design(X, 2)
    with pytest.raises(ValueError, match="length mismatch"):
        br.fit_ridge(phi, [1.0, 2.0], 0.0)


def test_negative_penalty_raises():
    phi = br.polynomial_design(X, 2)
    with pytest.raises(ValueError, match="non-negative"):
        br.fit_ridge(phi, Y, -0.5)


def test_rbf_nonpositive_width_raises():
    with pytest.raises(ValueError, match="positive"):
        br.rbf_design(X, RBF_CENTERS, 0.0)


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        br.BasisRegression(degree=2).predict(X)


def test_unknown_basis_raises():
    with pytest.raises(ValueError, match="unknown basis"):
        br.BasisRegression(basis="spline").fit(X, Y)


def test_non_finite_input_raises():
    with pytest.raises(ValueError, match="finite"):
        br.polynomial_design([1.0, float("nan"), 3.0], 2)
