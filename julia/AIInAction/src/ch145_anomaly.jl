"""
    Ch145Anomaly

Statistical anomaly detection from scratch (Julia).

Mirrors the Python module `aiinaction.ch145_anomaly` and the Rust module
`aiinaction::ch145_anomaly`. The shared fixtures in `test/test_ch145_anomaly.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

Four classical detectors: the univariate z-score, the multivariate Mahalanobis
distance, a Gaussian-kernel density estimate, and the Grubbs single-outlier
test. The chi-square and Student-t critical values used by the calibrated
detectors are computed by inverting their CDFs with a self-contained
incomplete-gamma / incomplete-beta routine so the three languages produce
identical numbers.
"""
module Ch145Anomaly

using LinearAlgebra: det, inv

export zscores, zscore_flags, mahalanobis_sq, gaussian_kde, kde_scores,
    grubbs_test, GrubbsResult, chi2_ppf, student_t_ppf, silverman_bandwidth

# ---------------------------------------------------------------------------
# Special functions (self-contained) for calibrated thresholds.
# ---------------------------------------------------------------------------
const _LANCZOS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)

"""Natural log of the gamma function (Lanczos approximation, g=7, n=9)."""
function _ln_gamma(x::Float64)
    g = 7.0
    if x < 0.5
        return log(pi / sin(pi * x)) - _ln_gamma(1.0 - x)
    end
    x -= 1.0
    a = _LANCZOS[1]
    t = x + g + 0.5
    for i in 2:9
        a += _LANCZOS[i] / (x + (i - 1))
    end
    return 0.5 * log(2.0 * pi) + (x + 0.5) * log(t) - t + log(a)
end

"""Regularized lower incomplete gamma P(s, x)."""
function _reg_lower_gamma(s::Float64, x::Float64)
    x <= 0.0 && return 0.0
    if x < s + 1.0
        ap = s
        total = 1.0 / s
        term = total
        for _ in 1:1000
            ap += 1.0
            term *= x / ap
            total += term
            abs(term) < abs(total) * 1e-16 && break
        end
        return total * exp(-x + s * log(x) - _ln_gamma(s))
    else
        tiny = 1e-300
        b = x + 1.0 - s
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in 1:1000
            an = -i * (i - s)
            b += 2.0
            d = an * d + b
            abs(d) < tiny && (d = tiny)
            c = b + an / c
            abs(c) < tiny && (c = tiny)
            d = 1.0 / d
            delta = d * c
            h *= delta
            abs(delta - 1.0) < 1e-16 && break
        end
        q = exp(-x + s * log(x) - _ln_gamma(s)) * h
        return 1.0 - q
    end
end

"""Continued fraction for the incomplete beta function (Lentz's method)."""
function _betacf(a::Float64, b::Float64, x::Float64)
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    abs(d) < tiny && (d = tiny)
    d = 1.0 / d
    h = d
    for m in 1:1000
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        abs(d) < tiny && (d = tiny)
        c = 1.0 + aa / c
        abs(c) < tiny && (c = tiny)
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        abs(d) < tiny && (d = tiny)
        c = 1.0 + aa / c
        abs(c) < tiny && (c = tiny)
        d = 1.0 / d
        delta = d * c
        h *= delta
        abs(delta - 1.0) < 1e-16 && break
    end
    return h
end

"""Regularized incomplete beta I_x(a, b)."""
function _reg_inc_beta(a::Float64, b::Float64, x::Float64)
    x <= 0.0 && return 0.0
    x >= 1.0 && return 1.0
    lbeta = _ln_gamma(a) + _ln_gamma(b) - _ln_gamma(a + b)
    front = exp(log(x) * a + log(1.0 - x) * b - lbeta)
    if x < (a + 1.0) / (a + b + 2.0)
        return front * _betacf(a, b, x) / a
    else
        return 1.0 - front * _betacf(b, a, 1.0 - x) / b
    end
end

"""Inverse CDF of the chi-square distribution with `df` degrees of freedom."""
function chi2_ppf(p::Real, df::Integer)
    (0.0 < p < 1.0) || throw(ArgumentError("p must be in (0, 1), got $p"))
    df >= 1 || throw(ArgumentError("df must be >= 1, got $df"))
    s = df / 2.0
    cdf(q) = _reg_lower_gamma(s, q / 2.0)
    lo = 0.0
    hi = 1.0
    while cdf(hi) < p
        hi *= 2.0
        hi > 1e12 && break
    end
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        cdf(mid) < p ? (lo = mid) : (hi = mid)
    end
    return 0.5 * (lo + hi)
end

"""Inverse CDF of the Student-t distribution with `df` degrees of freedom."""
function student_t_ppf(p::Real, df::Integer)
    (0.0 < p < 1.0) || throw(ArgumentError("p must be in (0, 1), got $p"))
    df >= 1 || throw(ArgumentError("df must be >= 1, got $df"))
    dff = Float64(df)
    function cdf(t)
        x = dff / (dff + t * t)
        ib = _reg_inc_beta(dff / 2.0, 0.5, x)
        return t > 0.0 ? 1.0 - 0.5 * ib : 0.5 * ib
    end
    lo = -1.0
    hi = 1.0
    while cdf(lo) > p
        lo *= 2.0
        lo < -1e12 && break
    end
    while cdf(hi) < p
        hi *= 2.0
        hi > 1e12 && break
    end
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        cdf(mid) < p ? (lo = mid) : (hi = mid)
    end
    return 0.5 * (lo + hi)
end

# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------
function _check_vector(x)
    v = Vector{Float64}(x)
    length(v) >= 2 || throw(ArgumentError("need at least 2 observations, got $(length(v))"))
    all(isfinite, v) || throw(ArgumentError("input contains non-finite values (nan or inf)"))
    return v
end

_mean(x) = sum(x) / length(x)

function _std_ddof1(x)
    m = _mean(x)
    n = length(x)
    return sqrt(sum((v - m)^2 for v in x) / (n - 1))
end

# ---------------------------------------------------------------------------
# 1. z-score.
# ---------------------------------------------------------------------------
"""Standardized scores `z_i = (x_i - mean) / std` (ddof = 1)."""
function zscores(x)
    v = _check_vector(x)
    m = _mean(v)
    s = _std_ddof1(v)
    s == 0 && throw(ArgumentError("standard deviation is zero; z-scores are undefined"))
    return (v .- m) ./ s
end

"""Boolean mask flagging points with `|z_i| > threshold`."""
function zscore_flags(x; threshold::Real=3.0)
    threshold > 0 || throw(ArgumentError("threshold must be positive, got $threshold"))
    return abs.(zscores(x)) .> threshold
end

# ---------------------------------------------------------------------------
# 2. Mahalanobis distance.
# ---------------------------------------------------------------------------
"""
    mahalanobis_sq(X, points=nothing)

Squared Mahalanobis distances of `points` to the Gaussian fit on the `n x d`
matrix `X`. When `points === nothing` the training rows are scored. Throws if
the covariance matrix is singular.
"""
function mahalanobis_sq(X, points=nothing)
    A = Matrix{Float64}(X)
    n, d = size(A)
    n >= 2 || throw(ArgumentError("need at least 2 samples, got $n"))
    d >= 1 || throw(ArgumentError("matrix must have at least one feature"))
    all(isfinite, A) || throw(ArgumentError("input contains non-finite values (nan or inf)"))

    mu = vec(sum(A; dims=1)) ./ n
    Xc = A .- mu'
    cov = (Xc' * Xc) ./ (n - 1)
    dt = det(cov)
    (isfinite(dt) && abs(dt) >= 1e-300) ||
        throw(ArgumentError("covariance matrix is singular; cannot invert"))
    S = inv(cov)

    P = points === nothing ? A : Matrix{Float64}(points)
    size(P, 2) == d ||
        throw(ArgumentError("points have $(size(P, 2)) features but model was fit on $d"))
    all(isfinite, P) || throw(ArgumentError("points contains non-finite values (nan or inf)"))

    out = Vector{Float64}(undef, size(P, 1))
    for i in 1:size(P, 1)
        diff = vec(P[i, :]) .- mu
        out[i] = dot(diff, S * diff)
    end
    return out
end

# A tiny dot used above (avoid importing LinearAlgebra.dot under a name clash).
dot(a, b) = sum(a .* b)

# ---------------------------------------------------------------------------
# 3. Kernel density estimation.
# ---------------------------------------------------------------------------
"""Silverman's rule-of-thumb bandwidth `1.06 * std * n^{-1/5}` (ddof = 1)."""
function silverman_bandwidth(x)
    v = _check_vector(x)
    s = _std_ddof1(v)
    s == 0 && throw(ArgumentError("standard deviation is zero; bandwidth is undefined"))
    return 1.06 * s * length(v)^(-1.0 / 5.0)
end

"""
    gaussian_kde(x, query; bandwidth=nothing)

Gaussian-kernel density estimate of `x` evaluated at `query`. Uses Silverman's
rule when `bandwidth === nothing`.
"""
function gaussian_kde(x, query; bandwidth=nothing)
    v = _check_vector(x)
    q = Vector{Float64}(query)
    all(isfinite, q) || throw(ArgumentError("query contains non-finite values (nan or inf)"))
    h = bandwidth === nothing ? silverman_bandwidth(v) : Float64(bandwidth)
    h > 0 || throw(ArgumentError("bandwidth must be positive, got $h"))
    n = length(v)
    coef = 1.0 / (n * h * sqrt(2.0 * pi))
    return [coef * sum(exp(-0.5 * ((qi - xi) / h)^2) for xi in v) for qi in q]
end

"""Anomaly scores `-log p_hat(q)` from a Gaussian KDE (higher means rarer)."""
function kde_scores(x, query=nothing; bandwidth=nothing)
    v = _check_vector(x)
    q = query === nothing ? v : Vector{Float64}(query)
    dens = gaussian_kde(v, q; bandwidth=bandwidth)
    floor = 1e-300
    return [-log(max(d, floor)) for d in dens]
end

# ---------------------------------------------------------------------------
# 4. Grubbs test.
# ---------------------------------------------------------------------------
"""Outcome of a two-sided Grubbs test for one outlier."""
struct GrubbsResult
    statistic::Float64
    critical_value::Float64
    index::Int
    is_outlier::Bool
    alpha::Float64
end

"""Two-sided Grubbs critical value for `n` observations at level `alpha`."""
function grubbs_critical_value(n::Integer, alpha::Real=0.05)
    n >= 3 || throw(ArgumentError("Grubbs test needs at least 3 observations, got $n"))
    (0.0 < alpha < 1.0) || throw(ArgumentError("alpha must be in (0, 1), got $alpha"))
    t = student_t_ppf(1.0 - alpha / (2.0 * n), n - 2)
    t2 = t * t
    return (n - 1) / sqrt(n) * sqrt(t2 / (n - 2 + t2))
end

"""
    grubbs_test(x; alpha=0.05)

Two-sided Grubbs test for a single outlier in approximately normal data. The
returned `index` is 1-based.
"""
function grubbs_test(x; alpha::Real=0.05)
    v = _check_vector(x)
    n = length(v)
    n >= 3 || throw(ArgumentError("Grubbs test needs at least 3 observations, got $n"))
    m = _mean(v)
    s = _std_ddof1(v)
    s == 0 && throw(ArgumentError("standard deviation is zero; Grubbs statistic is undefined"))
    devs = abs.(v .- m)
    idx = argmax(devs)
    g = devs[idx] / s
    crit = grubbs_critical_value(n, alpha)
    return GrubbsResult(g, crit, idx, g > crit, alpha)
end

end # module Ch145Anomaly
