"""Regression metrics: MSE, RMSE, MAE, and the Huber loss.

Small, dependency-free reference implementations with explicit input validation.
The public API mirrors the Julia (`AIInAction.Ch157RegressionMetrics`) and Rust
(`aiinaction::ch157_regression_metrics`) implementations one-to-one; the
cross-language parity tests assert that all three agree to within floating-point
tolerance on shared fixtures.

RMSE and MAE already live in :mod:`aiinaction.metrics`; this module *reuses* those
rather than re-deriving them, and adds the squared-error (:func:`mse`) and robust
Huber-loss (:func:`huber_loss`, :func:`huber_loss_mean`) pieces that the chapter
introduces. ``rmse`` and ``mae`` are re-exported for convenience so a reader can
import the whole regression-metrics toolkit from one place.

Definitions (residual ``r_i = y_i - yhat_i``, ``n`` observations):

* ``MSE  = (1/n) sum r_i^2``
* ``RMSE = sqrt(MSE)``
* ``MAE  = (1/n) sum |r_i|``
* Huber, with threshold ``delta > 0``:
  ``L_delta(r) = 0.5 r^2``                     if ``|r| <= delta``
  ``L_delta(r) = delta (|r| - 0.5 delta)``     if ``|r| >  delta``
"""
from __future__ import annotations

import math
from collections.abc import Sequence

# Reuse the validated rmse/mae already shipped in aiinaction.metrics rather than
# duplicating them. They are re-exported below.
from aiinaction.metrics import mae, rmse

__all__ = ["mse", "rmse", "mae", "huber_loss", "huber_loss_mean"]


def _validate_pair(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> tuple[list[float], list[float]]:
    yt = [float(v) for v in y_true]
    yp = [float(v) for v in y_pred]
    if len(yt) != len(yp):
        raise ValueError(
            f"length mismatch: len(y_true)={len(yt)} != len(y_pred)={len(yp)}"
        )
    if not yt:
        raise ValueError("inputs must be non-empty")
    return yt, yp


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean squared error: the average of the squared residuals.

    Lives in the *squared* units of the target and weights large residuals
    quadratically, so a single outlier can dominate the score.

    >>> round(mse([3.0, 5.0], [2.0, 5.0]), 6)
    0.5
    """
    yt, yp = _validate_pair(y_true, y_pred)
    return sum((t - p) ** 2 for t, p in zip(yt, yp)) / len(yt)


def huber_loss(
    y_true: Sequence[float], y_pred: Sequence[float], delta: float = 1.0
) -> list[float]:
    """Per-observation Huber loss with threshold ``delta``.

    Quadratic for residuals with ``|r| <= delta`` and linear beyond, the two
    branches meeting with a continuous value and derivative at ``|r| = delta``.
    Returns one loss value per observation (the elementwise loss, *not* averaged).

    Parameters
    ----------
    y_true, y_pred:
        Equal-length, non-empty sequences of targets and predictions.
    delta:
        Positive threshold separating the quadratic (inlier) and linear (outlier)
        regimes. Must be ``> 0``.

    Raises
    ------
    ValueError
        If the inputs are mismatched/empty or ``delta <= 0``.

    >>> huber_loss([0.0, 0.0], [0.5, 5.0], delta=1.5)
    [0.125, 6.375]
    """
    yt, yp = _validate_pair(y_true, y_pred)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta must be a positive finite number, got {delta}")
    out: list[float] = []
    for t, p in zip(yt, yp):
        a = abs(t - p)
        if a <= delta:
            out.append(0.5 * a * a)
        else:
            out.append(delta * (a - 0.5 * delta))
    return out


def huber_loss_mean(
    y_true: Sequence[float], y_pred: Sequence[float], delta: float = 1.0
) -> float:
    """Mean Huber loss over all observations.

    The scalar objective minimized by Huber (robust) regression: efficient like
    MSE on inliers, bounded-influence like MAE on outliers.

    >>> round(huber_loss_mean([0.0, 0.0], [0.5, 5.0], delta=1.5), 6)
    3.25
    """
    losses = huber_loss(y_true, y_pred, delta)
    return sum(losses) / len(losses)
