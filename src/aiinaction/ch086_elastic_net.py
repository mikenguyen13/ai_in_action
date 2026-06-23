"""Elastic Net regression via coordinate descent (Chapter 086).

A small, self-contained reference implementation of the Elastic Net estimator

    minimize  (1 / (2 n)) * ||y - X b||^2 + lambda * (alpha ||b||_1 + ((1 - alpha) / 2) ||b||_2^2)

solved by cyclic coordinate descent with soft thresholding. The intercept is
fit unpenalized by centering the response and predictors internally.

This mirrors the Julia (``AIInAction.Ch086ElasticNet``) and Rust
(``aiinaction::ch086_elastic_net``) implementations one-to-one; the
cross-language fixtures in the test suites assert that all three agree to within
floating-point tolerance.

The public API is intentionally tiny:

* :func:`soft_threshold` -- the scalar soft-thresholding operator.
* :func:`elastic_net_fit` -- fit coefficients and intercept.
* :func:`elastic_net_predict` -- predict from a fitted model.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["soft_threshold", "elastic_net_fit", "elastic_net_predict"]


def soft_threshold(z: float, gamma: float) -> float:
    """Soft-thresholding operator ``S(z, gamma) = sign(z) * max(|z| - gamma, 0)``.

    This is the proximal operator of ``gamma * |.|`` and the engine of the
    coordinate-descent update for the L1 part of the penalty.

    >>> soft_threshold(3.0, 1.0)
    2.0
    >>> soft_threshold(-3.0, 1.0)
    -2.0
    >>> soft_threshold(0.5, 1.0)
    0.0
    """
    gamma = float(gamma)
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    z = float(z)
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0


def _as_matrix(X: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got ndim={arr.ndim}")
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("X must be non-empty")
    return arr


def elastic_net_fit(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    lam: float,
    alpha: float = 0.5,
    *,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[list[float], float]:
    """Fit Elastic Net coefficients by cyclic coordinate descent.

    Parameters
    ----------
    X
        Design matrix of shape ``(n_samples, n_features)``.
    y
        Response vector of length ``n_samples``.
    lam
        Overall penalty strength ``lambda >= 0``. ``lam = 0`` reduces to
        ordinary least squares (up to the coordinate-descent tolerance).
    alpha
        Mixing parameter in ``[0, 1]``. ``alpha = 1`` is the Lasso, ``alpha = 0``
        is Ridge, intermediate values blend the two.
    max_iter
        Maximum number of full coordinate sweeps.
    tol
        Convergence tolerance on the largest coefficient change in a sweep.

    Returns
    -------
    (coef, intercept)
        ``coef`` is a list of ``n_features`` coefficients on the original
        (un-centered) feature scale; ``intercept`` is the unpenalized intercept.

    Notes
    -----
    Predictors are centered internally so the intercept is not penalized. The
    standardized coordinate update is

        b_j <- S( (1/n) x_j^T r_j , lambda * alpha ) / ( 1 + lambda * (1 - alpha) )

    where ``r_j`` is the partial residual excluding feature ``j``.
    """
    Xm = _as_matrix(X)
    yv = np.asarray(y, dtype=float).ravel()
    n, p = Xm.shape
    if yv.shape[0] != n:
        raise ValueError(f"length mismatch: X has {n} rows but y has {yv.shape[0]}")
    lam = float(lam)
    alpha = float(alpha)
    if lam < 0.0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")

    # Center so the intercept is unpenalized; recover it after fitting.
    x_mean = Xm.mean(axis=0)
    y_mean = float(yv.mean())
    Xc = Xm - x_mean
    yc = yv - y_mean

    # Precompute per-feature L2 norms / n (the curvature of each 1-D problem).
    col_sq = (Xc * Xc).sum(axis=0) / n  # length p

    beta = np.zeros(p, dtype=float)
    residual = yc.copy()  # residual = yc - Xc @ beta, kept in sync incrementally
    l1 = lam * alpha
    l2 = lam * (1.0 - alpha)

    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            if col_sq[j] == 0.0:
                # Constant feature contributes nothing; force coefficient to 0.
                if beta[j] != 0.0:
                    residual += Xc[:, j] * beta[j]
                    beta[j] = 0.0
                continue
            xj = Xc[:, j]
            beta_j_old = beta[j]
            # Partial residual r_j = residual + x_j * beta_j (add feature j back).
            rho = (xj @ (residual + xj * beta_j_old)) / n
            beta_j_new = soft_threshold(rho, l1) / (col_sq[j] + l2)
            if beta_j_new != beta_j_old:
                residual += xj * (beta_j_old - beta_j_new)
                beta[j] = beta_j_new
                change = abs(beta_j_new - beta_j_old)
                if change > max_change:
                    max_change = change
        if max_change < tol:
            break

    intercept = y_mean - float(x_mean @ beta)
    return beta.tolist(), intercept


def elastic_net_predict(
    X: Sequence[Sequence[float]],
    coef: Sequence[float],
    intercept: float,
) -> list[float]:
    """Predict responses ``X @ coef + intercept`` for a fitted model.

    >>> elastic_net_predict([[1.0, 2.0], [3.0, 4.0]], [1.0, 0.0], 0.5)
    [1.5, 3.5]
    """
    Xm = _as_matrix(X)
    cv = np.asarray(coef, dtype=float).ravel()
    if cv.shape[0] != Xm.shape[1]:
        raise ValueError(
            f"length mismatch: X has {Xm.shape[1]} features but coef has {cv.shape[0]}"
        )
    preds = Xm @ cv + float(intercept)
    return preds.tolist()
