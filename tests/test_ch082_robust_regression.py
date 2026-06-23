"""Tests for aiinaction.ch082_robust_regression and the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three libraries at parity.

Tolerances:
* OLS / WLS / GLS / Vandermonde / solve are exact closed forms -> 1e-9.
* Huber and quantile are iterative (IRLS); they recover the underlying clean
  parameters but only to the iteration / smoothing tolerance, so those assertions
  use the documented looser bounds (1e-7 and 1e-5 respectively).
"""
from __future__ import annotations

import pytest

from aiinaction import ch082_robust_regression as rr

# --------------------------------------------------------------------------- #
# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.   #
# --------------------------------------------------------------------------- #
# A clean line y = 1 + 2x for x = 0..5, with index 3 contaminated (7 -> 20).
X_HUBER = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]
Y_HUBER = [1.0, 3.0, 5.0, 20.0, 9.0, 11.0]
HUBER_COEF = [1.0, 2.0]              # robust fit recovers the clean line
OLS_HUBER_COEF = [2.2380952380952381, 2.3714285714285714]  # OLS is pulled toward outlier

# WLS: y = 1 + 2x exactly, arbitrary positive weights -> exact recovery.
X_WLS = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
Y_WLS = [1.0, 3.0, 5.0, 7.0]
W_WLS = [1.0, 2.0, 3.0, 4.0]
WLS_COEF = [1.0, 2.0]

# GLS with a correlated (AR(1)-like) covariance.
X_GLS = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
Y_GLS = [1.0, 2.0, 2.5]
COV_GLS = [[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]]
GLS_COEF = [1.05, 0.75]

# Quantile (median) on a clean line y = 1 + x for x = 0..3 with one outlier.
X_Q = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
Y_Q = [1.0, 2.0, 3.0, 4.0, 50.0]
Q50_COEF = [1.0, 1.0]               # median robust to the outlier


# --------------------------------------------------------------------------- #
# solve / predict / vandermonde                                               #
# --------------------------------------------------------------------------- #
def test_solve_diagonal():
    assert rr.solve([[2.0, 0.0], [0.0, 4.0]], [2.0, 8.0]) == pytest.approx([1.0, 2.0])


def test_solve_general():
    x = rr.solve([[3.0, 2.0], [1.0, 2.0]], [5.0, 5.0])
    assert x == pytest.approx([0.0, 2.5])


def test_solve_singular_raises():
    with pytest.raises(ValueError, match="singular"):
        rr.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


def test_predict_matches():
    assert rr.predict([[1.0, 2.0], [1.0, 3.0]], [1.0, 1.0]) == pytest.approx([3.0, 4.0])


def test_vandermonde_basis():
    assert rr.vandermonde([0.0, 2.0], 2) == [[1.0, 0.0, 0.0], [1.0, 2.0, 4.0]]


def test_vandermonde_no_bias():
    assert rr.vandermonde([2.0, 3.0], 2, include_bias=False) == [[2.0, 4.0], [3.0, 9.0]]


# --------------------------------------------------------------------------- #
# Estimators against the shared fixtures                                      #
# --------------------------------------------------------------------------- #
def test_ols_pulled_by_outlier():
    assert rr.fit_ols(X_HUBER, Y_HUBER) == pytest.approx(OLS_HUBER_COEF, abs=1e-9)


def test_huber_recovers_clean_line():
    res = rr.fit_huber(X_HUBER, Y_HUBER, delta=1.345)
    assert res.coef == pytest.approx(HUBER_COEF, abs=1e-7)
    assert res.converged


def test_wls_exact():
    assert rr.fit_wls(X_WLS, Y_WLS, W_WLS) == pytest.approx(WLS_COEF, abs=1e-9)


def test_gls_correlated():
    assert rr.fit_gls(X_GLS, Y_GLS, COV_GLS) == pytest.approx(GLS_COEF, abs=1e-9)


def test_quantile_median_robust():
    beta = rr.fit_quantile(X_Q, Y_Q, tau=0.5)
    assert beta == pytest.approx(Q50_COEF, abs=1e-5)


def test_quantile_tau_half_matches_lad():
    # The median fit should be far from the OLS fit on contaminated data.
    ols = rr.fit_ols(X_Q, Y_Q)
    med = rr.fit_quantile(X_Q, Y_Q, tau=0.5)
    assert abs(ols[1] - med[1]) > 1.0


def test_basis_expansion_fits_parabola():
    # y = 2 - x + 3x^2 should be recovered exactly by an OLS fit on a degree-2 basis.
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ys = [2.0 - x + 3.0 * x * x for x in xs]
    Phi = rr.vandermonde(xs, 2)
    beta = rr.fit_ols(Phi, ys)
    assert beta == pytest.approx([2.0, -1.0, 3.0], abs=1e-9)


# --------------------------------------------------------------------------- #
# Validation / edge cases                                                     #
# --------------------------------------------------------------------------- #
def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        rr.fit_ols([[1.0], [1.0]], [1.0])


def test_ragged_matrix_raises():
    with pytest.raises(ValueError, match="ragged"):
        rr.fit_ols([[1.0, 2.0], [1.0]], [1.0, 2.0])


def test_underdetermined_raises():
    with pytest.raises(ValueError, match="underdetermined"):
        rr.fit_ols([[1.0, 2.0, 3.0]], [1.0])


def test_negative_weights_raise():
    with pytest.raises(ValueError, match="non-negative"):
        rr.fit_wls(X_WLS, Y_WLS, [1.0, -1.0, 1.0, 1.0])


def test_huber_bad_delta_raises():
    with pytest.raises(ValueError, match="delta must be positive"):
        rr.fit_huber(X_WLS, Y_WLS, delta=0.0)


def test_quantile_bad_tau_raises():
    with pytest.raises(ValueError, match="open interval"):
        rr.fit_quantile(X_WLS, Y_WLS, tau=1.5)


def test_gls_wrong_cov_shape_raises():
    with pytest.raises(ValueError, match="to match"):
        rr.fit_gls(X_GLS, Y_GLS, [[1.0, 0.0], [0.0, 1.0]])


def test_vandermonde_negative_degree_raises():
    with pytest.raises(ValueError, match="non-negative"):
        rr.vandermonde([1.0, 2.0], -1)
