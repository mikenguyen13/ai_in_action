"""Regression and classification metrics.

Small, dependency-free reference implementations with explicit input validation.
These mirror the Julia (`AIInAction.Metrics`) and Rust (`aiinaction::metrics`)
implementations one-to-one; the cross-language parity tests in CI assert that all
three agree to within floating-point tolerance on shared fixtures.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["rmse", "mae", "r2_score", "accuracy"]


def _validate_pair(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[list[float], list[float]]:
    yt = [float(v) for v in y_true]
    yp = [float(v) for v in y_pred]
    if len(yt) != len(yp):
        raise ValueError(f"length mismatch: len(y_true)={len(yt)} != len(y_pred)={len(yp)}")
    if not yt:
        raise ValueError("inputs must be non-empty")
    return yt, yp


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error.

    >>> round(rmse([3.0, 5.0], [2.0, 5.0]), 6)
    0.707107
    """
    yt, yp = _validate_pair(y_true, y_pred)
    mse = sum((t - p) ** 2 for t, p in zip(yt, yp)) / len(yt)
    return math.sqrt(mse)


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error."""
    yt, yp = _validate_pair(y_true, y_pred)
    return sum(abs(t - p) for t, p in zip(yt, yp)) / len(yt)


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Coefficient of determination R^2.

    Raises ``ValueError`` if the target variance is zero (R^2 undefined).
    """
    yt, yp = _validate_pair(y_true, y_pred)
    mean = sum(yt) / len(yt)
    ss_tot = sum((t - mean) ** 2 for t in yt)
    if ss_tot == 0.0:
        raise ValueError("R^2 is undefined when all y_true values are equal (zero variance)")
    ss_res = sum((t - p) ** 2 for t, p in zip(yt, yp))
    return 1.0 - ss_res / ss_tot


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Classification accuracy: fraction of exactly matching labels."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} != {len(y_pred)}")
    if not y_true:
        raise ValueError("inputs must be non-empty")
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)
