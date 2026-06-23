"""Calibration metrics: reliability curves and Expected Calibration Error.

Small, dependency-light reference implementations of the confidence-calibration
diagnostics from Chapter 161. The public API mirrors the Julia module
(`AIInAction.Ch161Calibration`) and the Rust module (`aiinaction::ch161_calibration`)
one-to-one; the cross-language parity tests assert that all three agree to within
floating-point tolerance on shared fixtures.

The setup is *confidence calibration*. For each held-out example we are given a
scalar confidence ``p_i = max_k f_k(x_i)`` in ``[0, 1]`` and a binary correctness
indicator ``c_i = 1[argmax_k f_k(x_i) == y_i]``. We partition ``[0, 1]`` into ``M``
bins, and within each occupied bin ``B_m`` compute

    acc(B_m)  = mean of c_i over i in B_m      (empirical accuracy)
    conf(B_m) = mean of p_i over i in B_m       (average confidence)

The Expected Calibration Error is the occupancy-weighted mean absolute gap, and the
Maximum Calibration Error is the largest gap over occupied bins:

    ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|
    MCE = max_m |acc(B_m) - conf(B_m)|

Binning convention: example ``i`` lands in bin ``b = floor(p_i * M)``, with the
right endpoint ``p_i = 1`` folded into the last bin so every confidence in
``[0, 1]`` maps to exactly one of the ``M`` bins ``[m/M, (m+1)/M)`` (the final bin
is closed on the right). Empty bins contribute nothing to ECE or MCE.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "Bin",
    "ReliabilityCurve",
    "reliability_curve",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
]


@dataclass(frozen=True)
class Bin:
    """Summary of one reliability-diagram bin.

    Attributes
    ----------
    lower, upper:
        Half-open confidence interval ``[lower, upper)`` of this bin (the final
        bin is closed on the right).
    count:
        Number of examples whose confidence fell in this bin.
    accuracy:
        Empirical accuracy ``acc(B_m)`` of the bin, or ``0.0`` if empty.
    confidence:
        Average confidence ``conf(B_m)`` of the bin, or ``0.0`` if empty.
    """

    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float

    @property
    def gap(self) -> float:
        """Signed calibration gap ``acc(B_m) - conf(B_m)``."""
        return self.accuracy - self.confidence


@dataclass(frozen=True)
class ReliabilityCurve:
    """A binned reliability curve plus the sample size it was built from."""

    bins: tuple[Bin, ...]
    n_samples: int

    @property
    def n_bins(self) -> int:
        return len(self.bins)

    @property
    def occupied(self) -> tuple[Bin, ...]:
        """The bins that contain at least one example."""
        return tuple(b for b in self.bins if b.count > 0)


def _validate(
    confidences: Sequence[float], correct: Sequence[float], n_bins: int
) -> tuple[list[float], list[float]]:
    conf = [float(p) for p in confidences]
    corr = [float(c) for c in correct]
    if len(conf) != len(corr):
        raise ValueError(
            f"length mismatch: len(confidences)={len(conf)} != len(correct)={len(corr)}"
        )
    if not conf:
        raise ValueError("inputs must be non-empty")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    for p in conf:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"confidences must lie in [0, 1], got {p}")
    for c in corr:
        if c not in (0.0, 1.0):
            raise ValueError(f"correct must be 0 or 1, got {c}")
    return conf, corr


def _bin_index(p: float, n_bins: int) -> int:
    """Map a confidence to its bin: floor(p * M), with p == 1 folded into the last bin."""
    b = int(p * n_bins)
    if b >= n_bins:
        b = n_bins - 1
    return b


def reliability_curve(
    confidences: Sequence[float], correct: Sequence[float], n_bins: int = 10
) -> ReliabilityCurve:
    """Build an equal-width reliability curve.

    Parameters
    ----------
    confidences:
        Per-example confidences ``p_i = max_k f_k(x_i)``, each in ``[0, 1]``.
    correct:
        Per-example correctness indicators ``c_i`` in ``{0, 1}``.
    n_bins:
        Number of equal-width bins partitioning ``[0, 1]`` (default 10). Must be
        a positive integer.

    Returns
    -------
    ReliabilityCurve
        The ``n_bins`` bins (including any empty ones) and the sample size.

    Examples
    --------
    >>> rc = reliability_curve([0.2, 0.8], [0, 1], n_bins=2)
    >>> [b.count for b in rc.bins]
    [1, 1]
    """
    conf, corr = _validate(confidences, correct, n_bins)
    n = len(conf)
    counts = [0] * n_bins
    acc_sum = [0.0] * n_bins
    conf_sum = [0.0] * n_bins
    for p, c in zip(conf, corr):
        b = _bin_index(p, n_bins)
        counts[b] += 1
        acc_sum[b] += c
        conf_sum[b] += p

    bins = []
    for m in range(n_bins):
        lower = m / n_bins
        upper = (m + 1) / n_bins
        if counts[m] > 0:
            acc = acc_sum[m] / counts[m]
            cf = conf_sum[m] / counts[m]
        else:
            acc = 0.0
            cf = 0.0
        bins.append(Bin(lower=lower, upper=upper, count=counts[m], accuracy=acc, confidence=cf))
    return ReliabilityCurve(bins=tuple(bins), n_samples=n)


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[float], n_bins: int = 10
) -> float:
    """Expected Calibration Error (ECE), the occupancy-weighted mean absolute gap.

    ``ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|`` over the ``M`` equal-width
    bins. Lies in ``[0, 1]``; smaller is better. Empty bins contribute nothing.

    >>> round(expected_calibration_error([0.9, 0.1], [1, 0], n_bins=10), 6)
    0.1
    """
    rc = reliability_curve(confidences, correct, n_bins)
    total = 0.0
    for b in rc.bins:
        if b.count > 0:
            total += (b.count / rc.n_samples) * abs(b.gap)
    return total


def maximum_calibration_error(
    confidences: Sequence[float], correct: Sequence[float], n_bins: int = 10
) -> float:
    """Maximum Calibration Error (MCE), the largest absolute gap over occupied bins.

    ``MCE = max_m |acc(B_m) - conf(B_m)|`` over the occupied equal-width bins.
    Returns ``0.0`` only when every occupied bin is perfectly calibrated.

    >>> round(maximum_calibration_error([0.9, 0.1], [1, 0], n_bins=10), 6)
    0.1
    """
    rc = reliability_curve(confidences, correct, n_bins)
    worst = 0.0
    for b in rc.bins:
        if b.count > 0:
            worst = max(worst, abs(b.gap))
    return worst


def brier_score(confidences: Sequence[float], correct: Sequence[float]) -> float:
    """Binary Brier score of the confidence-vs-correctness forecasts.

    ``BS = (1 / n) * sum_i (p_i - c_i)^2``. A strictly proper score that rewards
    calibration and sharpness jointly; reported alongside ECE so a calibrated but
    unsharp forecaster cannot look good on calibration alone.

    >>> round(brier_score([0.9, 0.1], [1, 0]), 6)
    0.01
    """
    conf, corr = _validate(confidences, correct, 1)
    n = len(conf)
    return sum((p - c) ** 2 for p, c in zip(conf, corr)) / n
