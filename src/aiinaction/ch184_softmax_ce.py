"""Softmax cross-entropy loss and its gradient, from scratch.

A small, well-validated reference implementation of the multiclass softmax
cross-entropy objective used throughout classification with neural networks. The
public API mirrors the Julia (`AIInAction.Ch184SoftmaxCE`) and Rust
(`aiinaction::ch184_softmax_ce`) implementations one-to-one; the cross-language
parity tests assert that all three agree to within floating-point tolerance on
shared fixtures.

Everything is computed in a numerically stable way directly from *logits*:

* :func:`softmax` and :func:`log_softmax` subtract the per-row maximum before
  exponentiating, so the largest exponent argument is zero and cannot overflow.
* :func:`cross_entropy_loss` uses the fused identity
  ``log softmax(z)_k = z_k - logsumexp(z)`` rather than taking the log of a
  probability, which avoids ``log(0)`` and recomputing exponentials.
* :func:`cross_entropy_grad` returns the clean predicted-minus-target gradient
  ``(p - q) / N`` with respect to the logits.

Optional label smoothing replaces the one-hot target ``q`` with
``q'(k) = (1 - eps) * 1[k = y] + eps / K``, which curbs overconfidence.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "softmax",
    "log_softmax",
    "cross_entropy_loss",
    "cross_entropy_grad",
]

Matrix = Sequence[Sequence[float]]


def _as_logits(z: Matrix) -> NDArray[np.float64]:
    """Coerce ``z`` to a finite 2-D ``(N, K)`` float array of logits."""
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"logits must be a 2-D (N, K) matrix, got array with {arr.ndim} dimension(s)")
    if arr.shape[0] < 1:
        raise ValueError("need at least one sample (N >= 1)")
    if arr.shape[1] < 2:
        raise ValueError(f"need at least 2 classes (K >= 2), got K={arr.shape[1]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("logits contain non-finite values (nan or inf)")
    return arr


def _check_labels(labels: Sequence[int], n: int, k: int) -> NDArray[np.intp]:
    """Validate integer class labels against the batch size ``n`` and class count ``k``."""
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError(f"labels must be a 1-D array of class indices, got {y.ndim} dimension(s)")
    if y.shape[0] != n:
        raise ValueError(f"labels has length {y.shape[0]} but logits has {n} rows")
    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(y == np.floor(y)):
            raise ValueError("labels must be integer class indices")
        y = y.astype(np.intp)
    if np.any(y < 0) or np.any(y >= k):
        raise ValueError(f"labels must be in [0, {k - 1}], got values outside that range")
    return y.astype(np.intp)


def _check_smoothing(label_smoothing: float) -> float:
    ls = float(label_smoothing)
    if not (0.0 <= ls < 1.0):
        raise ValueError(f"label_smoothing must be in [0, 1), got {ls}")
    return ls


def softmax(z: Matrix) -> NDArray[np.float64]:
    """Row-wise softmax of a logit matrix.

    Maps each row of ``z`` (shape ``(N, K)``) to a probability distribution over
    ``K`` classes. Stable: the per-row maximum is subtracted before exponentiating.

    >>> p = softmax([[0.0, 0.0]])
    >>> [round(float(v), 6) for v in p[0]]
    [0.5, 0.5]
    """
    arr = _as_logits(z)
    shifted = arr - arr.max(axis=1, keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=1, keepdims=True)


def log_softmax(z: Matrix) -> NDArray[np.float64]:
    """Row-wise log-softmax of a logit matrix, computed without forming the softmax.

    Uses ``log softmax(z)_k = (z_k - m) - log sum_j exp(z_j - m)`` with
    ``m = max_j z_j`` per row, which never evaluates ``log(0)``.
    """
    arr = _as_logits(z)
    m = arr.max(axis=1, keepdims=True)
    shifted = arr - m
    lse = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted - lse


def cross_entropy_loss(
    z: Matrix,
    labels: Sequence[int],
    *,
    label_smoothing: float = 0.0,
) -> float:
    """Mean softmax cross-entropy loss over a batch of logits.

    Parameters
    ----------
    z:
        Logit matrix of shape ``(N, K)`` with ``N >= 1`` samples and ``K >= 2``
        classes. Must contain only finite values.
    labels:
        Integer class indices, one per row, each in ``[0, K - 1]``.
    label_smoothing:
        Smoothing strength ``eps`` in ``[0, 1)``. With ``eps > 0`` the one-hot
        target is replaced by ``q'(k) = (1 - eps) 1[k = y] + eps / K``. Defaults
        to ``0.0`` (standard hard-target cross-entropy).

    Returns
    -------
    float
        The average per-example loss, ``(1/N) sum_i -sum_k q'_i(k) log p_i(k)``.

    Examples
    --------
    >>> round(cross_entropy_loss([[2.0, 1.0, 0.0]], [0]), 6)
    0.407606
    """
    arr = _as_logits(z)
    n, k = arr.shape
    y = _check_labels(labels, n, k)
    ls = _check_smoothing(label_smoothing)

    ls_logp = log_softmax(arr)
    correct = ls_logp[np.arange(n), y]
    if ls == 0.0:
        return float(-correct.mean())
    # Smoothed target: weight (1 - eps) on the true class, eps/K spread over all.
    uniform = ls_logp.mean(axis=1)
    per_example = -((1.0 - ls) * correct + ls * uniform)
    return float(per_example.mean())


def cross_entropy_grad(
    z: Matrix,
    labels: Sequence[int],
    *,
    label_smoothing: float = 0.0,
) -> NDArray[np.float64]:
    """Gradient of :func:`cross_entropy_loss` with respect to the logits.

    Returns an array of shape ``(N, K)`` equal to ``(p - q') / N``, where ``p`` is
    the softmax of the logits and ``q'`` is the (optionally smoothed) target
    distribution. This is the clean predicted-minus-target gradient.

    Examples
    --------
    >>> g = cross_entropy_grad([[2.0, 1.0, 0.0]], [0])
    >>> [round(float(v), 6) for v in g[0]]
    [-0.334759, 0.244728, 0.090031]
    """
    arr = _as_logits(z)
    n, k = arr.shape
    y = _check_labels(labels, n, k)
    ls = _check_smoothing(label_smoothing)

    p = softmax(arr)
    q = np.full((n, k), ls / k, dtype=np.float64)
    q[np.arange(n), y] += 1.0 - ls
    return (p - q) / n
