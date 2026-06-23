"""Polynomial and basis-function regression with least squares and ridge.

A small, self-contained reference implementation of basis-function regression.
The estimator builds a design matrix from a chosen basis (polynomial powers or
Gaussian radial bases), then solves the (optionally ridge-regularized) normal
equations for the coefficients.

This mirrors the Julia (`AIInAction.Ch083BasisRegression`) and Rust
(`aiinaction::ch083_basis_regression`) implementations one-to-one; the
cross-language parity tests in CI assert that all three agree to within
floating-point tolerance on shared fixtures.

The public API is intentionally small:

* :func:`polynomial_design` / :func:`rbf_design` build design matrices.
* :func:`fit_ridge` solves the (ridge) least-squares problem.
* :func:`predict` evaluates a fitted coefficient vector on a design matrix.
* :func:`effective_dof` reports the effective degrees of freedom of a ridge fit.
* :class:`BasisRegression` ties a basis and a penalty together with ``fit`` /
  ``predict``.

Only :mod:`numpy` is used for the linear algebra solve; the basis construction
is written out explicitly so the mapping to Julia and Rust is transparent.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "polynomial_design",
    "rbf_design",
    "fit_ridge",
    "predict",
    "effective_dof",
    "BasisRegression",
]


def _as_1d(x: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def polynomial_design(x: Sequence[float], degree: int) -> np.ndarray:
    """Build the polynomial design matrix with columns ``1, x, x^2, ..., x^d``.

    Parameters
    ----------
    x:
        One-dimensional sequence of inputs.
    degree:
        Polynomial degree ``d >= 0``; the result has ``d + 1`` columns.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(x), degree + 1)`` whose column ``j`` is ``x**j``.

    >>> polynomial_design([0.0, 1.0, 2.0], 2).tolist()
    [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 2.0, 4.0]]
    """
    xa = _as_1d(x, "x")
    if not isinstance(degree, int):
        raise ValueError(f"degree must be an int, got {type(degree).__name__}")
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    return np.vander(xa, N=degree + 1, increasing=True)


def rbf_design(
    x: Sequence[float],
    centers: Sequence[float],
    width: float,
    include_bias: bool = True,
) -> np.ndarray:
    """Build a Gaussian radial-basis-function design matrix.

    Column ``j`` is ``exp(-(x - c_j)^2 / (2 * width^2))`` for each center
    ``c_j``. When ``include_bias`` is true a leading column of ones is prepended.

    Parameters
    ----------
    x:
        One-dimensional sequence of inputs.
    centers:
        Locations of the basis bumps.
    width:
        Gaussian length scale ``ell > 0``.
    include_bias:
        Whether to prepend a constant column.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(x), len(centers) + include_bias)``.
    """
    xa = _as_1d(x, "x")
    ca = _as_1d(centers, "centers")
    width = float(width)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(f"width must be a positive finite number, got {width}")
    diff = xa[:, None] - ca[None, :]
    phi = np.exp(-(diff ** 2) / (2.0 * width ** 2))
    if include_bias:
        phi = np.column_stack([np.ones(xa.shape[0]), phi])
    return phi


def fit_ridge(phi: np.ndarray, y: Sequence[float], penalty: float = 0.0) -> np.ndarray:
    """Solve the ridge least-squares problem for the coefficient vector.

    Minimizes ``||y - phi @ beta||^2 + penalty * ||beta||^2`` via the normal
    equations ``(phi^T phi + penalty * I) beta = phi^T y``. With ``penalty == 0``
    this is ordinary least squares; the symmetric solve is used because adding a
    positive penalty guarantees the system is well conditioned.

    Parameters
    ----------
    phi:
        Design matrix of shape ``(n, m)``.
    y:
        Target vector of length ``n``.
    penalty:
        Ridge penalty ``lambda >= 0``. The bias/intercept column is penalized
        like any other coefficient, matching the simple closed form.

    Returns
    -------
    numpy.ndarray
        Coefficient vector of length ``m``.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2:
        raise ValueError(f"phi must be two-dimensional, got shape {phi.shape}")
    ya = _as_1d(y, "y")
    n, m = phi.shape
    if ya.shape[0] != n:
        raise ValueError(f"length mismatch: phi has {n} rows but y has {ya.shape[0]}")
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi must contain only finite values")
    penalty = float(penalty)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError(f"penalty must be a non-negative finite number, got {penalty}")
    gram = phi.T @ phi + penalty * np.eye(m)
    rhs = phi.T @ ya
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:  # singular Gram matrix at penalty == 0
        raise ValueError(
            "normal equations are singular; increase penalty or reduce basis size"
        ) from exc
    return beta


def predict(phi: np.ndarray, beta: Sequence[float]) -> np.ndarray:
    """Evaluate fitted coefficients on a design matrix: ``phi @ beta``.

    Parameters
    ----------
    phi:
        Design matrix of shape ``(n, m)``.
    beta:
        Coefficient vector of length ``m``.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2:
        raise ValueError(f"phi must be two-dimensional, got shape {phi.shape}")
    ba = np.asarray(beta, dtype=float)
    if ba.ndim != 1:
        raise ValueError(f"beta must be one-dimensional, got shape {ba.shape}")
    if phi.shape[1] != ba.shape[0]:
        raise ValueError(
            f"shape mismatch: phi has {phi.shape[1]} columns but beta has {ba.shape[0]}"
        )
    return phi @ ba


def effective_dof(phi: np.ndarray, penalty: float = 0.0) -> float:
    """Effective degrees of freedom of a ridge fit.

    Equals ``tr(phi (phi^T phi + penalty I)^{-1} phi^T) = sum_j s_j^2 / (s_j^2 + penalty)``
    where ``s_j`` are the singular values of ``phi``. At ``penalty == 0`` this is
    the rank of ``phi`` (the number of basis functions when full rank).

    Parameters
    ----------
    phi:
        Design matrix.
    penalty:
        Ridge penalty ``lambda >= 0``.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2:
        raise ValueError(f"phi must be two-dimensional, got shape {phi.shape}")
    penalty = float(penalty)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError(f"penalty must be a non-negative finite number, got {penalty}")
    s = np.linalg.svd(phi, compute_uv=False)
    s2 = s ** 2
    return float(np.sum(s2 / (s2 + penalty)))


@dataclass
class BasisRegression:
    """Basis-function regression estimator with optional ridge penalty.

    Parameters
    ----------
    degree:
        Polynomial degree for the ``"poly"`` basis.
    penalty:
        Ridge penalty ``lambda >= 0``.
    basis:
        ``"poly"`` for the polynomial power basis (default) or ``"rbf"`` for a
        Gaussian radial basis. For ``"rbf"`` you must pass ``centers`` and
        ``width``.
    centers:
        RBF centers (required when ``basis == "rbf"``).
    width:
        RBF Gaussian length scale (required when ``basis == "rbf"``).

    Attributes
    ----------
    coef_:
        Fitted coefficient vector, available after :meth:`fit`.
    """

    degree: int = 2
    penalty: float = 0.0
    basis: str = "poly"
    centers: Sequence[float] | None = None
    width: float | None = None
    coef_: np.ndarray | None = field(default=None, init=False)

    def _design(self, x: Sequence[float]) -> np.ndarray:
        if self.basis == "poly":
            return polynomial_design(x, self.degree)
        if self.basis == "rbf":
            if self.centers is None or self.width is None:
                raise ValueError("rbf basis requires both 'centers' and 'width'")
            return rbf_design(x, self.centers, self.width)
        raise ValueError(f"unknown basis {self.basis!r}; expected 'poly' or 'rbf'")

    def fit(self, x: Sequence[float], y: Sequence[float]) -> "BasisRegression":
        """Fit coefficients to inputs ``x`` and targets ``y``; returns self."""
        phi = self._design(x)
        self.coef_ = fit_ridge(phi, y, self.penalty)
        return self

    def predict(self, x: Sequence[float]) -> np.ndarray:
        """Predict targets at new inputs ``x`` using the fitted coefficients."""
        if self.coef_ is None:
            raise ValueError("model is not fitted; call fit() first")
        phi = self._design(x)
        return predict(phi, self.coef_)
