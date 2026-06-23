"""Layer Normalization and RMSNorm from scratch.

Small, well-validated reference implementations of the two normalization layers
that dominate modern transformer architectures. The public API mirrors the Julia
(`AIInAction.Ch202LayerNorm`) and Rust (`aiinaction::ch202_layer_norm`)
implementations one-to-one; the cross-language parity tests assert that all three
agree to within floating-point tolerance on shared fixtures.

Both operators act on the *feature* axis of each example independently, which is
what makes them indifferent to batch size and sequence length.

LayerNorm (Ba, Kiros, Hinton 2016), for a feature vector ``x`` of length ``d``::

    mu      = mean(x)
    var     = mean((x - mu) ** 2)               # population variance, ddof=0
    x_hat   = (x - mu) / sqrt(var + eps)
    y       = gamma * x_hat + beta

RMSNorm (Zhang, Sennrich 2019) drops the mean-subtraction step entirely::

    rms     = sqrt(mean(x ** 2) + eps)
    y       = (x / rms) * gamma

The ``eps`` is added *inside* the square root, which bounds the denominator away
from zero even when the activation vector is itself near zero. Each function has a
single-vector form and a batched (2-D, ``apply_*`` ) form that normalizes every
row independently.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "layer_norm",
    "rms_norm",
    "apply_layer_norm",
    "apply_rms_norm",
]

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


def _as_vector(x: Vector, name: str = "x") -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got array with {arr.ndim} dimension(s)")
    if arr.size < 1:
        raise ValueError(f"{name} must have at least one feature")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (nan or inf)")
    return arr


def _resolve_params(
    d: int,
    gamma: Vector | None,
    beta: Vector | None,
    *,
    allow_beta: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    if gamma is None:
        g = np.ones(d, dtype=np.float64)
    else:
        g = np.asarray(gamma, dtype=np.float64)
        if g.ndim != 1 or g.size != d:
            raise ValueError(f"gamma must have length {d}, got shape {g.shape}")
        if not np.all(np.isfinite(g)):
            raise ValueError("gamma contains non-finite values (nan or inf)")
    if not allow_beta:
        return g, None
    if beta is None:
        b = np.zeros(d, dtype=np.float64)
    else:
        b = np.asarray(beta, dtype=np.float64)
        if b.ndim != 1 or b.size != d:
            raise ValueError(f"beta must have length {d}, got shape {b.shape}")
        if not np.all(np.isfinite(b)):
            raise ValueError("beta contains non-finite values (nan or inf)")
    return g, b


def _check_eps(eps: float) -> float:
    eps = float(eps)
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    return eps


def layer_norm(
    x: Vector,
    gamma: Vector | None = None,
    beta: Vector | None = None,
    *,
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """Layer-normalize a single feature vector.

    Subtracts the feature-wise mean, divides by the standard deviation (using the
    population variance, ``ddof=0``), then applies the learned affine map
    ``y = gamma * x_hat + beta``.

    Parameters
    ----------
    x:
        Feature vector of length ``d >= 1`` with finite values.
    gamma:
        Per-feature gain of length ``d``. Defaults to all ones.
    beta:
        Per-feature bias of length ``d``. Defaults to all zeros.
    eps:
        Non-negative constant added inside the square root for numerical stability.

    Returns
    -------
    numpy.ndarray
        The normalized vector, shape ``(d,)``.

    Examples
    --------
    >>> y = layer_norm([1.0, 2.0, 3.0, 4.0])
    >>> [round(float(v), 6) for v in y]
    [-1.341641, -0.447214, 0.447214, 1.341641]
    """
    arr = _as_vector(x)
    d = arr.size
    eps = _check_eps(eps)
    g, b = _resolve_params(d, gamma, beta, allow_beta=True)

    mu = float(arr.mean())
    centered = arr - mu
    var = float(np.mean(centered * centered))
    x_hat = centered / np.sqrt(var + eps)
    return g * x_hat + b


def rms_norm(
    x: Vector,
    gamma: Vector | None = None,
    *,
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """RMS-normalize a single feature vector.

    Divides by the root mean square of the features (no mean subtraction), then
    applies the per-feature gain ``y = (x / rms) * gamma``. There is no additive
    bias in the standard RMSNorm formulation.

    Parameters
    ----------
    x:
        Feature vector of length ``d >= 1`` with finite values.
    gamma:
        Per-feature gain of length ``d``. Defaults to all ones.
    eps:
        Non-negative constant added inside the square root for numerical stability.

    Returns
    -------
    numpy.ndarray
        The normalized vector, shape ``(d,)``.

    Examples
    --------
    >>> y = rms_norm([1.0, 2.0, 3.0, 4.0], eps=0.0)
    >>> [round(float(v), 6) for v in y]
    [0.365148, 0.730297, 1.095445, 1.460593]
    """
    arr = _as_vector(x)
    d = arr.size
    eps = _check_eps(eps)
    g, _ = _resolve_params(d, gamma, None, allow_beta=False)

    ms = float(np.mean(arr * arr))
    rms = np.sqrt(ms + eps)
    return (arr / rms) * g


def _as_matrix(X: Matrix) -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be a 2-D matrix, got array with {arr.ndim} dimension(s)")
    if arr.shape[0] < 1:
        raise ValueError("X must have at least one row")
    if arr.shape[1] < 1:
        raise ValueError("X must have at least one feature")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X contains non-finite values (nan or inf)")
    return arr


def apply_layer_norm(
    X: Matrix,
    gamma: Vector | None = None,
    beta: Vector | None = None,
    *,
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """Layer-normalize every row of an ``(n, d)`` matrix independently.

    Each row is treated as one example and normalized over its ``d`` features with
    :func:`layer_norm`. Returns an array of shape ``(n, d)``.
    """
    arr = _as_matrix(X)
    return np.vstack([layer_norm(row, gamma, beta, eps=eps) for row in arr])


def apply_rms_norm(
    X: Matrix,
    gamma: Vector | None = None,
    *,
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """RMS-normalize every row of an ``(n, d)`` matrix independently.

    Each row is treated as one example and normalized over its ``d`` features with
    :func:`rms_norm`. Returns an array of shape ``(n, d)``.
    """
    arr = _as_matrix(X)
    return np.vstack([rms_norm(row, gamma, eps=eps) for row in arr])
