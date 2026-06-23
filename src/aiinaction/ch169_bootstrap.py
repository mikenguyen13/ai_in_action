"""Bootstrap resampling with percentile and BCa confidence intervals (from scratch).

A small, well-validated reference implementation of the nonparametric bootstrap for
a scalar statistic of a one-dimensional sample. The public API mirrors the Julia
module ``AIInAction.Ch169Bootstrap`` and the Rust module
``aiinaction::ch169_bootstrap`` one-to-one; the cross-language parity tests assert
that all three agree to within floating-point tolerance on shared fixtures.

To make the three languages produce *bit-for-bit identical* resamples we cannot rely
on each platform's native RNG. Instead this module ships a fully specified 64-bit
linear congruential generator (the constants are Knuth's MMIX values) together with
a deterministic "integer in ``[0, m)``" routine. Given the same seed and sample,
Python, Julia, and Rust draw exactly the same bootstrap indices, so the replicate
statistics, standard error, and interval endpoints match.

Two confidence-interval methods are provided:

* ``"percentile"`` reads the empirical ``alpha`` and ``1 - alpha`` quantiles of the
  replicate distribution directly.
* ``"bca"`` (bias-corrected and accelerated, Efron 1987) shifts those quantiles using
  a median-bias correction ``z0`` and a jackknife acceleration ``a``, giving an
  interval that is second-order accurate.

The statistic is the sample mean. Because every common per-example evaluation metric
(accuracy, mean score, mean loss) reduces to a mean over cached per-example values,
bootstrapping the mean is the workhorse case; the same machinery extends to any
statistic by swapping the aggregation.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "BootstrapResult",
    "norm_cdf",
    "norm_ppf",
    "quantile",
    "bootstrap_mean_ci",
]

# MMIX (Knuth) LCG constants, computed modulo 2**64.
_LCG_A = 6364136223846793005
_LCG_C = 1442695040888963407
_LCG_M = 1 << 64


@dataclass(frozen=True)
class BootstrapResult:
    """The outcome of a bootstrap interval computation.

    Attributes
    ----------
    estimate:
        The statistic (sample mean) on the original data.
    standard_error:
        Bootstrap standard error: the sample standard deviation (``ddof=1``) of the
        replicate statistics.
    ci_low, ci_high:
        Lower and upper confidence-interval endpoints at level ``1 - 2*alpha``.
    method:
        ``"percentile"`` or ``"bca"``.
    alpha:
        Per-tail probability used for the interval.
    replicates:
        The ``n_resamples`` bootstrap replicate statistics, in draw order.
    """

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    method: str
    alpha: float
    replicates: list[float] = field(default_factory=list)


def _next_state(state: int) -> int:
    """Advance the 64-bit LCG one step, returning the new state."""
    return (_LCG_A * state + _LCG_C) % _LCG_M


def _rand_below(state: int, bound: int) -> tuple[int, int]:
    """Draw a uniform integer in ``[0, bound)`` using Lemire's multiplicative map.

    Returns ``(value, new_state)``. The low bits of a linear congruential generator
    have short periods, so we advance the state and use its *high* 64 bits: the value
    is ``(state * bound) >> 64``, which spreads the high-entropy top bits uniformly
    across ``[0, bound)``. The arithmetic is fully specified on 64-bit integers, so
    Python, Julia, and Rust draw the identical index sequence.
    """
    if bound <= 0:
        raise ValueError("bound must be positive")
    state = _next_state(state)
    value = (state * bound) >> 64
    return value, state


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (the ``"linear"``/type-7 rule) of sorted data.

    ``sorted_values`` must be sorted ascending and non-empty. ``q`` is clamped to
    ``[0, 1]``. With ``n`` points the quantile sits at fractional index ``q*(n-1)``.
    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError("cannot take a quantile of an empty sequence")
    if q <= 0.0:
        return float(sorted_values[0])
    if q >= 1.0:
        return float(sorted_values[-1])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo]) * (1.0 - frac) + float(sorted_values[hi]) * frac


def _erf(x: float) -> float:
    """Error function via the Numerical Recipes rational/exponential form.

    Accurate to about ``1.2e-7``. We deliberately use this approximation (rather than
    :func:`math.erf`) so that ``norm_cdf`` is *bitwise reproducible* against the Julia
    and Rust ports, which have no standard ``erf`` and use this same series. That keeps
    the BCa endpoints in agreement across all three languages.
    """
    t = 1.0 / (1.0 + 0.5 * abs(x))
    tau = t * math.exp(
        -x * x - 1.26551223
        + t * (1.00002368
        + t * (0.37409196
        + t * (0.09678418
        + t * (-0.18628806
        + t * (0.27886807
        + t * (-1.13520398
        + t * (1.48851587
        + t * (-0.82215223
        + t * 0.17087277)))))))))
    return 1.0 - tau if x >= 0.0 else tau - 1.0


def norm_cdf(x: float) -> float:
    """Standard normal CDF ``Phi(x)``.

    Built on :func:`_erf` (a ~1.2e-7 approximation) rather than ``math.erfc`` so the
    value matches the Julia and Rust ports exactly; see :func:`_erf`.
    """
    return 0.5 * (1.0 - _erf(-x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Standard normal quantile ``Phi^{-1}(p)`` via Acklam's rational approximation.

    Accurate to about ``1.15e-9`` over the open interval ``(0, 1)``. ``p`` must lie
    strictly inside ``(0, 1)``.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in the open interval (0, 1), got {p}")

    # Coefficients (Acklam, 2003).
    a = (
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    )
    b = (
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std_sample(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def _validate(data: Sequence[float]) -> list[float]:
    arr = [float(v) for v in data]
    if len(arr) < 2:
        raise ValueError(f"need at least 2 observations to bootstrap, got {len(arr)}")
    if any(not math.isfinite(v) for v in arr):
        raise ValueError("data contains non-finite values (nan or inf)")
    return arr


def _jackknife_acceleration(data: list[float]) -> float:
    """Acceleration ``a`` from the leave-one-out jackknife of the mean."""
    n = len(data)
    total = sum(data)
    # Leave-one-out means: (total - x_i) / (n - 1).
    loo = [(total - data[i]) / (n - 1) for i in range(n)]
    mean_loo = _mean(loo)
    diffs = [mean_loo - v for v in loo]
    num = sum(d**3 for d in diffs)
    den = 6.0 * (sum(d**2 for d in diffs)) ** 1.5
    if den == 0.0:
        return 0.0
    return num / den


def bootstrap_mean_ci(
    data: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.025,
    method: str = "bca",
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap a confidence interval for the mean of ``data``.

    Parameters
    ----------
    data:
        One-dimensional sample with at least 2 finite observations.
    n_resamples:
        Number of bootstrap resamples ``B``. Must be ``>= 1``.
    alpha:
        Per-tail probability; the interval has confidence level ``1 - 2*alpha``. For
        a 95% interval use ``alpha=0.025``. Must lie in ``(0, 0.5)``.
    method:
        ``"percentile"`` for the plain percentile interval or ``"bca"`` for the
        bias-corrected and accelerated interval.
    seed:
        Non-negative seed for the built-in LCG. The same seed and data produce the
        same resamples in every language.

    Returns
    -------
    BootstrapResult

    Examples
    --------
    >>> r = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_resamples=1000, method="percentile", seed=42)
    >>> round(r.estimate, 6)
    3.0
    >>> r.ci_low < r.estimate < r.ci_high
    True
    """
    arr = _validate(data)
    n = len(arr)

    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"alpha must be in the open interval (0, 0.5), got {alpha}")
    if method not in ("percentile", "bca"):
        raise ValueError(f"method must be 'percentile' or 'bca', got {method!r}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    estimate = _mean(arr)

    # Draw B resamples with the deterministic LCG and record each resample mean.
    state = (seed + _LCG_C) % _LCG_M  # one warm-up mix so seed=0 is not degenerate
    replicates: list[float] = []
    for _ in range(n_resamples):
        acc = 0.0
        for _ in range(n):
            idx, state = _rand_below(state, n)
            acc += arr[idx]
        replicates.append(acc / n)

    standard_error = _std_sample(replicates)
    ordered = sorted(replicates)

    if method == "percentile":
        lo_q, hi_q = alpha, 1.0 - alpha
    else:
        # BCa: bias correction z0 from the fraction of replicates below the estimate.
        below = sum(1 for v in replicates if v < estimate)
        frac = below / n_resamples
        # Clamp the fraction off the 0/1 boundary so the quantile is finite.
        frac = min(max(frac, 0.5 / n_resamples), 1.0 - 0.5 / n_resamples)
        z0 = norm_ppf(frac)
        a = _jackknife_acceleration(arr)

        def _adjust(tail: float) -> float:
            z = norm_ppf(tail)
            num = z0 + z
            return norm_cdf(z0 + num / (1.0 - a * num))

        lo_q = _adjust(alpha)
        hi_q = _adjust(1.0 - alpha)

    ci_low = quantile(ordered, lo_q)
    ci_high = quantile(ordered, hi_q)

    return BootstrapResult(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        method=method,
        alpha=alpha,
        replicates=replicates,
    )
