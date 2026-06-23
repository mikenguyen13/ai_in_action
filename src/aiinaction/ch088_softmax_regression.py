"""Multiclass logistic regression (softmax regression) from scratch.

A small, self-contained implementation of multinomial logistic regression
trained by full-batch gradient descent on the cross-entropy loss. The model
maps a feature vector ``x`` to a probability distribution over ``K`` classes
through the numerically stable softmax of the linear logits ``z = x W + b``.

This mirrors the Julia (`AIInAction.Ch088SoftmaxRegression`) and Rust
(`aiinaction::ch088_softmax_regression`) implementations one-to-one; the
cross-language parity tests assert that all three agree to within
floating-point tolerance on shared fixtures.

Public API
----------
- ``softmax(z)``                : row-wise numerically stable softmax.
- ``cross_entropy(probs, y)``   : mean multiclass log loss.
- ``SoftmaxRegression``         : fit / predict_proba / predict on a dataset.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["softmax", "cross_entropy", "SoftmaxRegression"]


def _as_2d_float(name: str, arr: object) -> np.ndarray:
    """Coerce ``arr`` to a contiguous 2-D float array or raise ``ValueError``."""
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got ndim={a.ndim}")
    if a.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite values")
    return a


def softmax(z: Sequence[Sequence[float]]) -> np.ndarray:
    """Row-wise numerically stable softmax.

    Each row of ``z`` (the logits for one example) is mapped to a probability
    distribution. The per-row maximum is subtracted before exponentiating,
    which prevents overflow without changing the result (softmax is invariant
    to adding a constant to every logit in a row).

    Parameters
    ----------
    z : array-like of shape (n_samples, n_classes)
        Logits.

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_classes)
        Each row is non-negative and sums to one.

    >>> p = softmax([[0.0, 0.0]])
    >>> [round(v, 6) for v in p[0]]
    [0.5, 0.5]
    """
    a = _as_2d_float("z", z)
    shifted = a - a.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(probs: Sequence[Sequence[float]], y: Sequence[int]) -> float:
    """Mean multiclass cross-entropy (negative log likelihood).

    Parameters
    ----------
    probs : array-like of shape (n_samples, n_classes)
        Predicted class probabilities (rows should sum to one).
    y : array-like of shape (n_samples,)
        Integer class labels in ``[0, n_classes)``.

    Returns
    -------
    float
        ``-mean_i log probs[i, y[i]]``.
    """
    p = _as_2d_float("probs", probs)
    labels = np.asarray(y, dtype=int)
    if labels.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got ndim={labels.ndim}")
    if labels.shape[0] != p.shape[0]:
        raise ValueError(
            f"length mismatch: probs has {p.shape[0]} rows but y has {labels.shape[0]}"
        )
    n, k = p.shape
    if labels.min() < 0 or labels.max() >= k:
        raise ValueError(f"labels must lie in [0, {k}); got range [{labels.min()}, {labels.max()}]")
    eps = 1e-15
    chosen = p[np.arange(n), labels]
    return float(-np.mean(np.log(np.clip(chosen, eps, 1.0))))


class SoftmaxRegression:
    """Multinomial logistic regression trained by gradient descent.

    The model holds weights ``W`` of shape ``(n_features, n_classes)`` and a
    bias ``b`` of shape ``(n_classes,)``. It is fit by minimizing the mean
    cross-entropy plus an optional L2 penalty ``l2 * ||W||^2`` (the bias is not
    penalized), using full-batch gradient descent with a fixed learning rate.

    The clean residual gradient is used throughout::

        grad_W = X^T (P - Y_onehot) / n + 2 * l2 * W
        grad_b = mean(P - Y_onehot, axis=0)

    Parameters
    ----------
    learning_rate : float, default 0.5
        Step size for gradient descent. Must be positive.
    n_iter : int, default 500
        Number of gradient steps. Must be positive.
    l2 : float, default 0.0
        L2 regularization strength. Must be non-negative.
    """

    def __init__(self, learning_rate: float = 0.5, n_iter: int = 500, l2: float = 0.0) -> None:
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if n_iter <= 0:
            raise ValueError(f"n_iter must be positive, got {n_iter}")
        if l2 < 0.0:
            raise ValueError(f"l2 must be non-negative, got {l2}")
        self.learning_rate = float(learning_rate)
        self.n_iter = int(n_iter)
        self.l2 = float(l2)
        self.n_classes: int | None = None
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "SoftmaxRegression":
        """Fit the model on features ``X`` and integer labels ``y``.

        ``y`` must contain every class label in ``[0, n_classes)`` at least once,
        where ``n_classes = max(y) + 1``. Returns ``self``.
        """
        Xm = _as_2d_float("X", X)
        labels = np.asarray(y, dtype=int)
        if labels.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got ndim={labels.ndim}")
        if labels.shape[0] != Xm.shape[0]:
            raise ValueError(
                f"length mismatch: X has {Xm.shape[0]} rows but y has {labels.shape[0]}"
            )
        if labels.min() < 0:
            raise ValueError(f"labels must be non-negative, got min {labels.min()}")
        n, d = Xm.shape
        k = int(labels.max()) + 1
        if k < 2:
            raise ValueError(f"need at least 2 classes, got {k}")

        onehot = np.zeros((n, k), dtype=float)
        onehot[np.arange(n), labels] = 1.0

        W = np.zeros((d, k), dtype=float)
        b = np.zeros(k, dtype=float)
        for _ in range(self.n_iter):
            probs = softmax(Xm @ W + b)
            diff = probs - onehot
            grad_W = Xm.T @ diff / n + 2.0 * self.l2 * W
            grad_b = diff.mean(axis=0)
            W -= self.learning_rate * grad_W
            b -= self.learning_rate * grad_b

        self.W = W
        self.b = b
        self.n_classes = k
        return self

    def _check_fitted(self) -> None:
        if self.W is None or self.b is None:
            raise ValueError("model is not fitted; call fit() first")

    def predict_proba(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Return the predicted class-probability matrix of shape (n, n_classes)."""
        self._check_fitted()
        Xm = _as_2d_float("X", X)
        if Xm.shape[1] != self.W.shape[0]:  # type: ignore[union-attr]
            raise ValueError(
                f"X has {Xm.shape[1]} features but model was fit on {self.W.shape[0]}"  # type: ignore[union-attr]
            )
        return softmax(Xm @ self.W + self.b)

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Return the predicted integer class labels (argmax of the probabilities)."""
        return np.argmax(self.predict_proba(X), axis=1).astype(int)
