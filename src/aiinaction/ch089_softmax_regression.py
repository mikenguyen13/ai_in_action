"""Softmax regression from scratch.

A dependency-light reference implementation of the softmax function, the
numerically stable log-sum-exp / cross-entropy primitives, and a full
multinomial logistic (softmax) regression classifier trained by batch
gradient descent.

This mirrors the Julia (`AIInAction.Ch089SoftmaxRegression`) and Rust
(`aiinaction::ch089_softmax_regression`) implementations one-to-one; the
cross-language parity tests assert that all three agree to within a tight
floating-point tolerance on the shared fixtures defined in
``tests/test_ch089_softmax_regression.py``.

Only the Python module uses numpy (for the classifier's linear algebra); the
core scalar primitives are written so the three languages produce identical
numbers on the fixtures.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "softmax",
    "log_sum_exp",
    "cross_entropy_from_logits",
    "SoftmaxRegression",
]


def _as_1d(z: Sequence[float], name: str = "z") -> list[float]:
    out = [float(v) for v in z]
    if not out:
        raise ValueError(f"{name} must be non-empty")
    return out


def softmax(z: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax of a logit vector.

    Uses the max-subtraction trick so no exponential overflows. With a
    ``temperature`` ``T`` the logits are scaled by ``1/T`` before the softmax,
    sharpening (``T < 1``) or flattening (``T > 1``) the distribution.

    Args:
        z: Logit vector, length ``K >= 1``.
        temperature: Positive scaling factor ``T``.

    Returns:
        A probability vector of the same length that sums to 1.

    Raises:
        ValueError: If ``z`` is empty or ``temperature <= 0``.

    >>> [round(p, 6) for p in softmax([1.0, 2.0, 3.0])]
    [0.090031, 0.244728, 0.665241]
    """
    zl = _as_1d(z)
    if not (temperature > 0.0) or math.isnan(temperature):
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = [v / temperature for v in zl]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def log_sum_exp(z: Sequence[float]) -> float:
    """Stable ``log(sum(exp(z)))`` via the max-subtraction trick.

    >>> round(log_sum_exp([0.0, 0.0]), 6)
    0.693147

    Raises:
        ValueError: If ``z`` is empty.
    """
    zl = _as_1d(z)
    m = max(zl)
    return m + math.log(sum(math.exp(v - m) for v in zl))


def cross_entropy_from_logits(z: Sequence[float], label: int) -> float:
    """Cross-entropy loss computed directly from logits (fused, stable).

    Equals ``log_sum_exp(z) - z[label]`` and avoids forming probabilities and
    then taking a logarithm, which is the common source of inaccuracy.

    Args:
        z: Logit vector of length ``K``.
        label: Integer index of the true class, ``0 <= label < K``.

    Returns:
        The non-negative scalar loss.

    Raises:
        ValueError: If ``z`` is empty or ``label`` is out of range.

    >>> round(cross_entropy_from_logits([2.0, 1.0, 0.0], 0), 6)
    0.407606
    """
    zl = _as_1d(z)
    if not isinstance(label, (int, np.integer)):
        raise ValueError(f"label must be an integer index, got {type(label).__name__}")
    if label < 0 or label >= len(zl):
        raise ValueError(f"label {label} out of range for {len(zl)} classes")
    return log_sum_exp(zl) - zl[label]


class SoftmaxRegression:
    """Multinomial logistic (softmax) regression trained by gradient descent.

    The model holds a weight matrix ``W`` of shape ``(K, d)`` and a bias
    vector ``b`` of shape ``(K,)``. Logits are the affine map ``z = W x + b``,
    and class probabilities are ``softmax(z)``. Training minimises the mean
    cross-entropy loss with optional L2 regularisation; the gradient of the
    cross-entropy with respect to the logits is the residual ``p - y``.

    Args:
        learning_rate: Positive step size for gradient descent.
        n_iter: Number of full-batch gradient steps.
        l2: Non-negative L2 penalty on the weights (not the bias).

    The fitted parameters are deterministic: weights and biases start at zero,
    so a given dataset and hyperparameters always yield the same model.
    """

    def __init__(self, learning_rate: float = 0.5, n_iter: int = 200, l2: float = 0.0) -> None:
        if not (learning_rate > 0.0):
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        if l2 < 0.0:
            raise ValueError(f"l2 must be non-negative, got {l2}")
        self.learning_rate = float(learning_rate)
        self.n_iter = int(n_iter)
        self.l2 = float(l2)
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.n_classes_: int | None = None

    @staticmethod
    def _softmax_rows(Z: np.ndarray) -> np.ndarray:
        """Row-wise stable softmax of an ``(N, K)`` logit matrix."""
        m = Z.max(axis=1, keepdims=True)
        e = np.exp(Z - m)
        return e / e.sum(axis=1, keepdims=True)

    def _validate_xy(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, d), got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D (N,), got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y disagree on N: {X.shape[0]} != {y.shape[0]}")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one row")
        if not np.issubdtype(y.dtype, np.integer):
            if not np.all(y == y.astype(np.int64)):
                raise ValueError("y must contain integer class labels")
            y = y.astype(np.int64)
        if y.min() < 0:
            raise ValueError("class labels must be non-negative")
        return X, y

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "SoftmaxRegression":
        """Fit the model to features ``X`` (N, d) and integer labels ``y`` (N,)."""
        X, y = self._validate_xy(np.asarray(X, dtype=float), np.asarray(y))
        n, d = X.shape
        k = int(y.max()) + 1
        self.n_classes_ = k
        self.W = np.zeros((k, d), dtype=float)
        self.b = np.zeros(k, dtype=float)

        # One-hot target matrix (N, K).
        Y = np.zeros((n, k), dtype=float)
        Y[np.arange(n), y] = 1.0

        for _ in range(self.n_iter):
            Z = X @ self.W.T + self.b          # (N, K)
            P = self._softmax_rows(Z)          # (N, K)
            G = P - Y                          # (N, K) residual
            grad_W = (G.T @ X) / n + self.l2 * self.W
            grad_b = G.mean(axis=0)
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b
        return self

    def _check_fitted(self) -> None:
        if self.W is None or self.b is None:
            raise ValueError("model is not fitted; call fit() first")

    def predict_proba(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Return the ``(N, K)`` matrix of class probabilities."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, d), got shape {X.shape}")
        if X.shape[1] != self.W.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features but model expects {self.W.shape[1]}"
            )
        Z = X @ self.W.T + self.b
        return self._softmax_rows(Z)

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Return the ``(N,)`` vector of predicted class indices (argmax)."""
        return self.predict_proba(X).argmax(axis=1)

    def loss(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> float:
        """Mean cross-entropy loss (excluding the L2 term) on ``(X, y)``."""
        self._check_fitted()
        X, y = self._validate_xy(np.asarray(X, dtype=float), np.asarray(y))
        Z = X @ self.W.T + self.b
        m = Z.max(axis=1, keepdims=True)
        lse = (m + np.log(np.exp(Z - m).sum(axis=1, keepdims=True))).ravel()
        true_logit = Z[np.arange(Z.shape[0]), y]
        return float(np.mean(lse - true_logit))
