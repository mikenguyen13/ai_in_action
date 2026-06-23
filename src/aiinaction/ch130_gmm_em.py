"""Expectation-Maximization for Gaussian mixture models.

A small, from-scratch implementation of EM for a diagonal-free (full covariance)
Gaussian mixture, written to mirror the Julia (`AIInAction.Ch130GmmEm`) and Rust
(`aiinaction::ch130_gmm_em`) modules one-to-one. The cross-language parity tests
in CI assert that all three agree to within floating-point tolerance on shared
fixtures.

The public API is intentionally tiny:

* :func:`gaussian_pdf` -- multivariate normal density at a single point.
* :func:`e_step` -- responsibilities (soft assignments) given parameters.
* :func:`m_step` -- weighted re-estimation of weights, means, covariances.
* :func:`log_likelihood` -- incomplete-data log-likelihood of the data.
* :func:`fit_gmm` -- run EM to convergence from supplied initial parameters.

To keep the three implementations bit-comparable the algorithm avoids any
randomness: callers pass explicit initial parameters, so the same fixtures
produce the same trajectory in every language.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GMMParams",
    "GMMResult",
    "gaussian_pdf",
    "e_step",
    "m_step",
    "log_likelihood",
    "fit_gmm",
]


@dataclass
class GMMParams:
    """Parameters of a K-component, d-dimensional Gaussian mixture.

    Attributes
    ----------
    weights:
        Mixing weights, shape ``(K,)``, nonnegative and summing to one.
    means:
        Component means, shape ``(K, d)``.
    covariances:
        Component covariance matrices, shape ``(K, d, d)``, each symmetric
        positive definite.
    """

    weights: NDArray[np.float64]
    means: NDArray[np.float64]
    covariances: NDArray[np.float64]


@dataclass
class GMMResult:
    """Outcome of :func:`fit_gmm`.

    Attributes
    ----------
    params:
        Fitted :class:`GMMParams`.
    responsibilities:
        Final responsibilities, shape ``(N, K)``.
    log_likelihood:
        Final incomplete-data log-likelihood.
    n_iter:
        Number of EM iterations actually performed.
    converged:
        Whether the log-likelihood change fell below ``tol`` before ``max_iter``.
    history:
        Log-likelihood after each iteration, length ``n_iter``.
    """

    params: GMMParams
    responsibilities: NDArray[np.float64]
    log_likelihood: float
    n_iter: int
    converged: bool
    history: list[float]


def _as_float_2d(name: str, x: Sequence) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _validate_params(params: GMMParams, d: int) -> None:
    weights = np.asarray(params.weights, dtype=np.float64)
    means = np.asarray(params.means, dtype=np.float64)
    covs = np.asarray(params.covariances, dtype=np.float64)
    k = weights.shape[0]
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1-D, got ndim={weights.ndim}")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    total = float(weights.sum())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"weights must sum to 1, got {total}")
    if means.shape != (k, d):
        raise ValueError(f"means must have shape ({k}, {d}), got {means.shape}")
    if covs.shape != (k, d, d):
        raise ValueError(f"covariances must have shape ({k}, {d}, {d}), got {covs.shape}")


def gaussian_pdf(
    x: Sequence[float],
    mean: Sequence[float],
    cov: Sequence[Sequence[float]],
) -> float:
    """Density of a multivariate normal at the point ``x``.

    Parameters
    ----------
    x, mean:
        Length-``d`` vectors.
    cov:
        ``d`` by ``d`` symmetric positive-definite covariance matrix.

    Returns
    -------
    float
        The value of :math:`\\mathcal{N}(x \\mid \\text{mean}, \\text{cov})`.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or the covariance is not positive
        definite (non-positive determinant).

    Examples
    --------
    >>> round(gaussian_pdf([0.0], [0.0], [[1.0]]), 6)
    0.398942
    """
    xv = np.asarray(x, dtype=np.float64)
    mv = np.asarray(mean, dtype=np.float64)
    cm = np.asarray(cov, dtype=np.float64)
    if xv.ndim != 1 or mv.ndim != 1:
        raise ValueError("x and mean must be 1-D vectors")
    d = xv.shape[0]
    if mv.shape[0] != d:
        raise ValueError(f"mean length {mv.shape[0]} != x length {d}")
    if cm.shape != (d, d):
        raise ValueError(f"cov must have shape ({d}, {d}), got {cm.shape}")
    det = float(np.linalg.det(cm))
    if det <= 0.0:
        raise ValueError(f"covariance must be positive definite, det={det}")
    diff = xv - mv
    inv = np.linalg.inv(cm)
    quad = float(diff @ inv @ diff)
    norm = math.sqrt(((2.0 * math.pi) ** d) * det)
    return math.exp(-0.5 * quad) / norm


def e_step(x: Sequence[Sequence[float]], params: GMMParams) -> NDArray[np.float64]:
    """Compute responsibilities ``gamma[n, k]`` given current parameters.

    Returns an ``(N, K)`` array whose rows are categorical posteriors over the
    components: each row is nonnegative and sums to one.

    Raises
    ------
    ValueError
        If shapes are inconsistent, or if a data point has zero density under
        every component (responsibilities undefined).
    """
    data = _as_float_2d("x", x)
    n, d = data.shape
    _validate_params(params, d)
    weights = np.asarray(params.weights, dtype=np.float64)
    means = np.asarray(params.means, dtype=np.float64)
    covs = np.asarray(params.covariances, dtype=np.float64)
    k = weights.shape[0]
    gamma = np.zeros((n, k), dtype=np.float64)
    for ni in range(n):
        for ki in range(k):
            gamma[ni, ki] = weights[ki] * gaussian_pdf(data[ni], means[ki], covs[ki])
        row_sum = float(gamma[ni].sum())
        if row_sum <= 0.0:
            raise ValueError(
                f"data point {ni} has zero density under all components; "
                "responsibilities are undefined"
            )
        gamma[ni] /= row_sum
    return gamma


def m_step(
    x: Sequence[Sequence[float]],
    responsibilities: Sequence[Sequence[float]],
    reg_covar: float = 1e-6,
) -> GMMParams:
    """Weighted re-estimation of mixture parameters from responsibilities.

    Parameters
    ----------
    x:
        Data, shape ``(N, d)``.
    responsibilities:
        Responsibilities, shape ``(N, K)``.
    reg_covar:
        Nonnegative ridge added to the diagonal of each covariance to guard
        against the singularity pathology. Defaults to ``1e-6``.

    Returns
    -------
    GMMParams
        Updated weights, means, and covariances.
    """
    data = _as_float_2d("x", x)
    gamma = _as_float_2d("responsibilities", responsibilities)
    if reg_covar < 0.0:
        raise ValueError(f"reg_covar must be nonnegative, got {reg_covar}")
    n, d = data.shape
    if gamma.shape[0] != n:
        raise ValueError(
            f"responsibilities has {gamma.shape[0]} rows but x has {n} points"
        )
    k = gamma.shape[1]
    nk = gamma.sum(axis=0)  # effective counts, shape (K,)
    if np.any(nk <= 0.0):
        raise ValueError("a component has zero effective count; cannot re-estimate")
    weights = nk / float(n)
    means = np.zeros((k, d), dtype=np.float64)
    covs = np.zeros((k, d, d), dtype=np.float64)
    for ki in range(k):
        means[ki] = (gamma[:, ki][:, None] * data).sum(axis=0) / nk[ki]
        acc = np.zeros((d, d), dtype=np.float64)
        for ni in range(n):
            diff = data[ni] - means[ki]
            acc += gamma[ni, ki] * np.outer(diff, diff)
        covs[ki] = acc / nk[ki] + reg_covar * np.eye(d)
    return GMMParams(weights=weights, means=means, covariances=covs)


def log_likelihood(x: Sequence[Sequence[float]], params: GMMParams) -> float:
    """Incomplete-data log-likelihood of the data under the mixture.

    Computes :math:`\\sum_n \\log \\sum_k \\pi_k \\mathcal{N}(x_n \\mid \\mu_k, \\Sigma_k)`.
    """
    data = _as_float_2d("x", x)
    n, d = data.shape
    _validate_params(params, d)
    weights = np.asarray(params.weights, dtype=np.float64)
    means = np.asarray(params.means, dtype=np.float64)
    covs = np.asarray(params.covariances, dtype=np.float64)
    k = weights.shape[0]
    total = 0.0
    for ni in range(n):
        mix = 0.0
        for ki in range(k):
            mix += weights[ki] * gaussian_pdf(data[ni], means[ki], covs[ki])
        if mix <= 0.0:
            raise ValueError(
                f"data point {ni} has zero mixture density; log-likelihood is -inf"
            )
        total += math.log(mix)
    return total


def fit_gmm(
    x: Sequence[Sequence[float]],
    init: GMMParams,
    max_iter: int = 100,
    tol: float = 1e-6,
    reg_covar: float = 1e-6,
) -> GMMResult:
    """Run EM to convergence from explicit initial parameters.

    EM alternates the E step (responsibilities) and M step (weighted
    re-estimation) and stops when the increase in log-likelihood between
    successive iterations drops below ``tol`` or after ``max_iter`` iterations.
    By the standard EM argument the log-likelihood is monotonically
    nondecreasing.

    Parameters
    ----------
    x:
        Data, shape ``(N, d)``.
    init:
        Initial :class:`GMMParams`. No randomness is used, so the result is a
        deterministic function of ``x`` and ``init``.
    max_iter:
        Maximum number of EM iterations (must be positive).
    tol:
        Convergence threshold on the log-likelihood increase (nonnegative).
    reg_covar:
        Ridge passed to :func:`m_step`.

    Returns
    -------
    GMMResult
    """
    data = _as_float_2d("x", x)
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if tol < 0.0:
        raise ValueError(f"tol must be nonnegative, got {tol}")
    _validate_params(init, data.shape[1])

    params = init
    gamma = e_step(data, params)
    prev_ll = log_likelihood(data, params)
    history = [prev_ll]
    converged = False
    n_iter = 0
    for _ in range(max_iter):
        params = m_step(data, gamma, reg_covar=reg_covar)
        gamma = e_step(data, params)
        ll = log_likelihood(data, params)
        history.append(ll)
        n_iter += 1
        if abs(ll - prev_ll) < tol:
            converged = True
            prev_ll = ll
            break
        prev_ll = ll
    return GMMResult(
        params=params,
        responsibilities=gamma,
        log_likelihood=prev_ll,
        n_iter=n_iter,
        converged=converged,
        history=history,
    )
