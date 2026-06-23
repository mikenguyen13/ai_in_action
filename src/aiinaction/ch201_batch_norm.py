"""Batch Normalization forward and backward pass from scratch.

A small, well-validated reference implementation of Batch Normalization (Ioffe and
Szegedy, 2015). The public API mirrors the Julia (`AIInAction.Ch201BatchNorm`) and
Rust (`aiinaction::ch201_batch_norm`) implementations one-to-one; the
cross-language parity tests assert that all three agree to within floating-point
tolerance on shared fixtures.

For a mini-batch ``X`` of shape ``(m, d)`` (``m`` examples, ``d`` features), the
forward transform standardizes every feature over the batch and then applies a
learnable scale ``gamma`` and shift ``beta``:

.. math::

    \\mu_j = \\frac1m \\sum_i x_{ij}, \\qquad
    \\sigma_j^2 = \\frac1m \\sum_i (x_{ij} - \\mu_j)^2, \\\\
    \\hat{x}_{ij} = \\frac{x_{ij} - \\mu_j}{\\sqrt{\\sigma_j^2 + \\epsilon}}, \\qquad
    y_{ij} = \\gamma_j \\hat{x}_{ij} + \\beta_j .

The variance uses the population (biased, ``ddof=0``) convention, matching the
original paper and essentially every deep-learning framework's training-time BN.

The backward pass propagates an upstream gradient ``dY`` of shape ``(m, d)`` to
``dX``, ``dgamma`` and ``dbeta`` using the closed form

.. math::

    \\frac{\\partial \\ell}{\\partial x_{ij}}
    = \\frac{1}{\\sqrt{\\sigma_j^2 + \\epsilon}}
      \\left( g_{ij} - \\frac1m \\sum_k g_{kj}
        - \\hat{x}_{ij}\\,\\frac1m \\sum_k g_{kj}\\hat{x}_{kj} \\right),
    \\quad g_{ij} = \\gamma_j\\, \\frac{\\partial\\ell}{\\partial y_{ij}} .

An :func:`batch_norm_inference` helper applies the deterministic affine map used
at test time from frozen population statistics.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BatchNormCache",
    "batch_norm_forward",
    "batch_norm_backward",
    "batch_norm_inference",
]

Matrix = Sequence[Sequence[float]]
Vector = Sequence[float]


@dataclass(frozen=True)
class BatchNormCache:
    """Intermediate quantities saved by the forward pass for the backward pass.

    Attributes
    ----------
    x_hat:
        Standardized activations, shape ``(m, d)``.
    inv_std:
        Per-feature ``1 / sqrt(var + eps)``, shape ``(d,)``.
    gamma:
        The scale vector used in the forward pass, shape ``(d,)``.
    mean:
        Per-feature batch means, shape ``(d,)``.
    var:
        Per-feature batch variances (population, ``ddof=0``), shape ``(d,)``.
    """

    x_hat: NDArray[np.float64]
    inv_std: NDArray[np.float64]
    gamma: NDArray[np.float64]
    mean: NDArray[np.float64]
    var: NDArray[np.float64]


def _as_matrix(X: Matrix, name: str = "X") -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got array with {arr.ndim} dimension(s)")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must have at least one row")
    if arr.shape[1] < 1:
        raise ValueError(f"{name} must have at least one feature")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (nan or inf)")
    return arr


def _as_vector(v: Vector, d: int, name: str) -> NDArray[np.float64]:
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got array with {arr.ndim} dimension(s)")
    if arr.shape[0] != d:
        raise ValueError(f"{name} has length {arr.shape[0]} but X has {d} features")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (nan or inf)")
    return arr


def batch_norm_forward(
    X: Matrix,
    gamma: Vector,
    beta: Vector,
    *,
    eps: float = 1e-5,
) -> tuple[NDArray[np.float64], BatchNormCache]:
    """Training-time Batch Normalization forward pass.

    Parameters
    ----------
    X:
        Mini-batch of shape ``(m, d)`` with ``m >= 1`` examples and ``d >= 1``
        features. Must contain only finite values.
    gamma, beta:
        Learnable per-feature scale and shift, each of length ``d``.
    eps:
        Positive constant added to the variance for numerical stability.

    Returns
    -------
    (y, cache)
        ``y`` is the normalized, scaled and shifted output of shape ``(m, d)``.
        ``cache`` holds the quantities needed by :func:`batch_norm_backward`.

    Examples
    --------
    >>> y, _ = batch_norm_forward([[1.0], [3.0]], [1.0], [0.0])
    >>> [round(float(v), 6) for v in y.ravel()]
    [-0.999995, 0.999995]
    """
    if not (eps > 0.0):
        raise ValueError(f"eps must be positive, got {eps}")
    arr = _as_matrix(X)
    _m, d = arr.shape
    g = _as_vector(gamma, d, "gamma")
    b = _as_vector(beta, d, "beta")

    mean = arr.mean(axis=0)
    var = arr.var(axis=0)  # population variance (ddof=0)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (arr - mean) * inv_std
    y = g * x_hat + b

    cache = BatchNormCache(x_hat=x_hat, inv_std=inv_std, gamma=g, mean=mean, var=var)
    return y, cache


def batch_norm_backward(
    dy: Matrix,
    cache: BatchNormCache,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Backward pass: gradients with respect to ``X``, ``gamma`` and ``beta``.

    Parameters
    ----------
    dy:
        Upstream gradient ``dL/dy`` of shape ``(m, d)`` matching the forward output.
    cache:
        The :class:`BatchNormCache` returned by :func:`batch_norm_forward`.

    Returns
    -------
    (dx, dgamma, dbeta)
        ``dx`` has shape ``(m, d)``; ``dgamma`` and ``dbeta`` have shape ``(d,)``.
    """
    dY = _as_matrix(dy, "dy")
    m, d = dY.shape
    if cache.x_hat.shape != dY.shape:
        raise ValueError(
            f"dy has shape {dY.shape} but cache was built for {cache.x_hat.shape}"
        )

    dgamma = np.sum(dY * cache.x_hat, axis=0)
    dbeta = np.sum(dY, axis=0)

    g = dY * cache.gamma  # dL/dx_hat
    dx = cache.inv_std * (
        g - g.mean(axis=0) - cache.x_hat * (g * cache.x_hat).mean(axis=0)
    )
    return dx, dgamma, dbeta


def batch_norm_inference(
    X: Matrix,
    gamma: Vector,
    beta: Vector,
    running_mean: Vector,
    running_var: Vector,
    *,
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """Inference-time Batch Normalization using frozen population statistics.

    Applies the deterministic affine map
    ``y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta`` per feature.
    Each row is normalized independently of the others, so predictions do not
    depend on which examples are batched together.
    """
    if not (eps > 0.0):
        raise ValueError(f"eps must be positive, got {eps}")
    arr = _as_matrix(X)
    _m, d = arr.shape
    g = _as_vector(gamma, d, "gamma")
    b = _as_vector(beta, d, "beta")
    rm = _as_vector(running_mean, d, "running_mean")
    rv = _as_vector(running_var, d, "running_var")
    if np.any(rv < 0.0):
        raise ValueError("running_var must be non-negative")

    return g * (arr - rm) / np.sqrt(rv + eps) + b
