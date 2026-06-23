"""Statistical anomaly detection from scratch.

A small, well-validated reference implementation of four classical statistical
anomaly detectors. The public API mirrors the Julia (`AIInAction.Ch145Anomaly`)
and Rust (`aiinaction::ch145_anomaly`) implementations one-to-one; the
cross-language parity tests assert that all three agree to within floating-point
tolerance on shared fixtures.

The four detectors are:

1. **z-score** (univariate Gaussian). Standardize each point by the sample mean
   and (corrected, ``ddof=1``) standard deviation; flag ``|z| > threshold``.
2. **Mahalanobis distance** (multivariate Gaussian). Score a point by its squared
   Mahalanobis distance ``D^2 = (x - mu)^T S^{-1} (x - mu)``, which follows a
   chi-square law with ``d`` degrees of freedom under the Gaussian null.
3. **Kernel density estimation** (nonparametric). Estimate the density with a
   Gaussian kernel and a fixed bandwidth (Silverman's rule by default); low
   density means high anomaly score.
4. **Grubbs test** (order-statistic outlier test). Test whether the single most
   extreme observation is incompatible with a normal sample, returning the
   statistic ``G``, the critical value, and a reject decision.

Everything is implemented from scratch on top of numpy (no scipy): the
chi-square and Student-t critical values used by the calibrated detectors are
computed by inverting their CDFs with a self-contained incomplete-beta /
incomplete-gamma routine so the three languages compute identical numbers.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "zscores",
    "zscore_flags",
    "mahalanobis_sq",
    "gaussian_kde",
    "kde_scores",
    "grubbs_test",
    "GrubbsResult",
    "chi2_ppf",
    "student_t_ppf",
]

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


# ---------------------------------------------------------------------------
# Special functions (self-contained, no scipy) used for calibrated thresholds.
# ---------------------------------------------------------------------------
def _ln_gamma(x: float) -> float:
    """Natural log of the gamma function (Lanczos approximation, g=7, n=9)."""
    g = 7.0
    coef = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    if x < 0.5:
        # Reflection formula.
        return math.log(math.pi / math.sin(math.pi * x)) - _ln_gamma(1.0 - x)
    x -= 1.0
    a = coef[0]
    t = x + g + 0.5
    for i in range(1, 9):
        a += coef[i] / (x + i)
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def _reg_lower_gamma(s: float, x: float) -> float:
    """Regularized lower incomplete gamma P(s, x) = gamma(s, x) / Gamma(s)."""
    if x < 0.0 or s <= 0.0:
        raise ValueError("invalid arguments to incomplete gamma")
    if x == 0.0:
        return 0.0
    if x < s + 1.0:
        # Series expansion.
        ap = s
        total = 1.0 / s
        term = total
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * math.exp(-x + s * math.log(x) - _ln_gamma(s))
    # Continued fraction (Lentz) for the upper incomplete gamma, then complement.
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    q = math.exp(-x + s * math.log(x) - _ln_gamma(s)) * h
    return 1.0 - q


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def _reg_inc_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = _ln_gamma(a) + _ln_gamma(b) - _ln_gamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def chi2_ppf(p: float, df: int) -> float:
    """Inverse CDF (quantile) of the chi-square distribution with ``df`` dof.

    Returns the value ``q`` such that ``P(X <= q) = p`` for ``X ~ chi^2_df``.
    Solved by bisection on the regularized lower incomplete gamma CDF.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    s = df / 2.0

    def cdf(q: float) -> float:
        return _reg_lower_gamma(s, q / 2.0)

    lo, hi = 0.0, 1.0
    while cdf(hi) < p:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def student_t_ppf(p: float, df: int) -> float:
    """Inverse CDF (quantile) of the Student-t distribution with ``df`` dof.

    Returns ``q`` such that ``P(T <= q) = p`` for ``T ~ t_df``. Uses the
    relation between the t CDF and the regularized incomplete beta function.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    dff = float(df)

    def cdf(t: float) -> float:
        x = dff / (dff + t * t)
        ib = _reg_inc_beta(dff / 2.0, 0.5, x)
        if t > 0.0:
            return 1.0 - 0.5 * ib
        return 0.5 * ib

    lo, hi = -1.0, 1.0
    while cdf(lo) > p:
        lo *= 2.0
        if lo < -1e12:
            break
    while cdf(hi) < p:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Input helpers.
# ---------------------------------------------------------------------------
def _as_vector(x: Vector) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got array with {arr.ndim} dimension(s)")
    if arr.size < 2:
        raise ValueError(f"need at least 2 observations, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("input contains non-finite values (nan or inf)")
    return arr


def _as_matrix(X: Matrix) -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got array with {arr.ndim} dimension(s)")
    if arr.shape[0] < 2:
        raise ValueError(f"need at least 2 samples, got {arr.shape[0]}")
    if arr.shape[1] < 1:
        raise ValueError("matrix must have at least one feature")
    if not np.all(np.isfinite(arr)):
        raise ValueError("input contains non-finite values (nan or inf)")
    return arr


# ---------------------------------------------------------------------------
# 1. z-score (univariate Gaussian).
# ---------------------------------------------------------------------------
def zscores(x: Vector) -> NDArray[np.float64]:
    """Standardized scores ``z_i = (x_i - mean) / std`` (``ddof=1``).

    Raises ``ValueError`` when the sample standard deviation is zero.

    >>> [round(v, 4) for v in zscores([1.0, 2.0, 3.0, 4.0, 100.0])]
    [-0.503, -0.4783, -0.4535, -0.4288, 1.8636]
    """
    arr = _as_vector(x)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std == 0.0:
        raise ValueError("standard deviation is zero; z-scores are undefined")
    return (arr - mean) / std


def zscore_flags(x: Vector, threshold: float = 3.0) -> NDArray[np.bool_]:
    """Boolean mask flagging points with ``|z_i| > threshold``."""
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    return np.abs(zscores(x)) > threshold


# ---------------------------------------------------------------------------
# 2. Mahalanobis distance (multivariate Gaussian).
# ---------------------------------------------------------------------------
def mahalanobis_sq(X: Matrix, points: Matrix | None = None) -> NDArray[np.float64]:
    """Squared Mahalanobis distances of ``points`` to the model fit on ``X``.

    The mean and sample covariance (``ddof=1``) are estimated from ``X``. When
    ``points`` is ``None`` the distances of the training rows themselves are
    returned. Under a Gaussian null with ``d`` features each squared distance is
    approximately ``chi^2_d``, so a calibrated threshold is ``chi2_ppf(1 - alpha, d)``.

    Raises ``ValueError`` if the covariance matrix is singular.
    """
    arr = _as_matrix(X)
    n, d = arr.shape
    mean = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    det = float(np.linalg.det(cov))
    if not math.isfinite(det) or abs(det) < 1e-300:
        raise ValueError("covariance matrix is singular; cannot invert")
    inv = np.linalg.inv(cov)

    if points is None:
        pts = arr
    else:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2:
            raise ValueError("points must be a 2-D matrix")
        if pts.shape[1] != d:
            raise ValueError(f"points have {pts.shape[1]} features but model was fit on {d}")
        if not np.all(np.isfinite(pts)):
            raise ValueError("points contains non-finite values (nan or inf)")

    diff = pts - mean
    # Row-wise quadratic form (diff @ inv * diff).sum(axis=1).
    return np.einsum("ij,jk,ik->i", diff, inv, diff)


# ---------------------------------------------------------------------------
# 3. Kernel density estimation (Gaussian kernel).
# ---------------------------------------------------------------------------
def silverman_bandwidth(x: Vector) -> float:
    """Silverman's rule-of-thumb bandwidth ``1.06 * std * n^{-1/5}`` (``ddof=1``)."""
    arr = _as_vector(x)
    n = arr.size
    std = float(arr.std(ddof=1))
    if std == 0.0:
        raise ValueError("standard deviation is zero; bandwidth is undefined")
    return 1.06 * std * n ** (-1.0 / 5.0)


def gaussian_kde(x: Vector, query: Vector, bandwidth: float | None = None) -> NDArray[np.float64]:
    """Gaussian-kernel density estimate of ``x`` evaluated at ``query``.

    ``p_hat(q) = (1 / (n h)) * sum_i phi((q - x_i) / h)`` with the standard normal
    kernel ``phi``. When ``bandwidth`` is ``None`` Silverman's rule is used. The
    returned array has the same length as ``query``.
    """
    arr = _as_vector(x)
    q = np.asarray(query, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError("query must be a 1-D vector")
    if not np.all(np.isfinite(q)):
        raise ValueError("query contains non-finite values (nan or inf)")
    h = silverman_bandwidth(arr) if bandwidth is None else float(bandwidth)
    if h <= 0.0:
        raise ValueError(f"bandwidth must be positive, got {h}")
    n = arr.size
    coef = 1.0 / (n * h * math.sqrt(2.0 * math.pi))
    u = (q[:, None] - arr[None, :]) / h
    return coef * np.exp(-0.5 * u * u).sum(axis=1)


def kde_scores(x: Vector, query: Vector | None = None, bandwidth: float | None = None) -> NDArray[np.float64]:
    """Anomaly scores ``-log p_hat(q)`` from a Gaussian KDE (higher means rarer).

    When ``query`` is ``None`` the training points are scored. A small floor is
    applied to the density before taking the log so empty regions stay finite.
    """
    arr = _as_vector(x)
    q = arr if query is None else np.asarray(query, dtype=np.float64)
    dens = gaussian_kde(arr, q, bandwidth)
    floor = 1e-300
    return -np.log(np.maximum(dens, floor))


# ---------------------------------------------------------------------------
# 4. Grubbs test for a single outlier.
# ---------------------------------------------------------------------------
class GrubbsResult:
    """Outcome of a two-sided Grubbs test for one outlier.

    Attributes
    ----------
    statistic:
        The Grubbs statistic ``G = max_i |x_i - mean| / std``.
    critical_value:
        The two-sided critical value at level ``alpha``.
    index:
        Index of the most extreme observation.
    is_outlier:
        ``True`` when ``statistic > critical_value`` (reject the null).
    alpha:
        The significance level used.
    """

    __slots__ = ("statistic", "critical_value", "index", "is_outlier", "alpha")

    def __init__(self, statistic: float, critical_value: float, index: int, is_outlier: bool, alpha: float):
        self.statistic = statistic
        self.critical_value = critical_value
        self.index = index
        self.is_outlier = is_outlier
        self.alpha = alpha

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"GrubbsResult(statistic={self.statistic!r}, critical_value={self.critical_value!r}, "
            f"index={self.index!r}, is_outlier={self.is_outlier!r}, alpha={self.alpha!r})"
        )


def grubbs_critical_value(n: int, alpha: float = 0.05) -> float:
    """Two-sided Grubbs critical value for ``n`` observations at level ``alpha``."""
    if n < 3:
        raise ValueError(f"Grubbs test needs at least 3 observations, got {n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    # Upper alpha/(2n) quantile of t with n-2 dof.
    t = student_t_ppf(1.0 - alpha / (2.0 * n), n - 2)
    t2 = t * t
    return (n - 1) / math.sqrt(n) * math.sqrt(t2 / (n - 2 + t2))


def grubbs_test(x: Vector, alpha: float = 0.05) -> GrubbsResult:
    """Two-sided Grubbs test for a single outlier in approximately normal data.

    Computes ``G = max_i |x_i - mean| / std`` and compares it to the
    order-statistic critical value. Needs at least 3 observations and a positive
    sample standard deviation.

    >>> r = grubbs_test([1.0, 2.0, 1.5, 1.8, 2.2, 50.0])
    >>> r.is_outlier
    True
    >>> r.index
    5
    """
    arr = _as_vector(x)
    n = arr.size
    if n < 3:
        raise ValueError(f"Grubbs test needs at least 3 observations, got {n}")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std == 0.0:
        raise ValueError("standard deviation is zero; Grubbs statistic is undefined")
    abs_dev = np.abs(arr - mean)
    idx = int(np.argmax(abs_dev))
    g = float(abs_dev[idx] / std)
    crit = grubbs_critical_value(n, alpha)
    return GrubbsResult(g, crit, idx, g > crit, alpha)
