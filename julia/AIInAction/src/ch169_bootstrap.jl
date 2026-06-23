"""
    Ch169Bootstrap

Bootstrap resampling with percentile and BCa confidence intervals (Julia, Base-only).

Mirrors the Python module `aiinaction.ch169_bootstrap` and the Rust module
`aiinaction::ch169_bootstrap`. The shared fixtures in `test/test_ch169_bootstrap.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

Cross-language reproducibility comes from a fully specified 64-bit linear congruential
generator (Knuth's MMIX constants) and Lemire's multiplicative `[0, bound)` map using
the high 64 bits of a 128-bit product. With the same seed and sample, all three
languages draw the identical resample indices, so every replicate, the standard error,
and both interval endpoints agree.

`norm_ppf` is Acklam's rational approximation; `norm_cdf` uses a self-contained
Numerical-Recipes `erfc`, so the implementation needs no SpecialFunctions dependency.
"""
module Ch169Bootstrap

export BootstrapResult, norm_cdf, norm_ppf, quantile, bootstrap_mean_ci

const LCG_A = 0x5851f42d4c957f2d  # 6364136223846793005
const LCG_C = 0x14057b7ef767814f  # 1442695040888963407

"""The outcome of a bootstrap interval computation."""
struct BootstrapResult
    estimate::Float64
    standard_error::Float64
    ci_low::Float64
    ci_high::Float64
    method::String
    alpha::Float64
    replicates::Vector{Float64}
end

# Advance the 64-bit LCG one step (UInt64 arithmetic wraps mod 2^64).
@inline _next_state(state::UInt64) = LCG_A * state + LCG_C

# Uniform integer in [0, bound) via Lemire's multiplicative map on the high 64 bits.
@inline function _rand_below(state::UInt64, bound::UInt64)
    s = _next_state(state)
    value = UInt64(widemul(s, bound) >> 64)
    return value, s
end

"""Linear-interpolation quantile (type-7) of an ascending, non-empty vector."""
function quantile(sorted_values::AbstractVector{<:Real}, q::Real)
    n = length(sorted_values)
    n > 0 || throw(ArgumentError("cannot take a quantile of an empty vector"))
    q <= 0 && return Float64(sorted_values[1])
    q >= 1 && return Float64(sorted_values[end])
    pos = q * (n - 1)
    lo = floor(Int, pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    # 0-based positions -> 1-based indices.
    return Float64(sorted_values[lo + 1]) * (1 - frac) + Float64(sorted_values[hi + 1]) * frac
end

# Error function (Numerical Recipes form, ~1.2e-7 accurate); matches the Rust erf.
function _erf(x::Float64)
    t = 1.0 / (1.0 + 0.5 * abs(x))
    tau = t * exp(-x * x - 1.26551223 +
        t * (1.00002368 +
        t * (0.37409196 +
        t * (0.09678418 +
        t * (-0.18628806 +
        t * (0.27886807 +
        t * (-1.13520398 +
        t * (1.48851587 +
        t * (-0.82215223 +
        t * 0.17087277)))))))))
    return x >= 0 ? 1.0 - tau : tau - 1.0
end

_erfc(x::Float64) = 1.0 - _erf(x)

"""Standard normal CDF `Phi(x)`."""
norm_cdf(x::Real) = 0.5 * _erfc(-Float64(x) / sqrt(2.0))

"""Standard normal quantile `Phi^{-1}(p)` via Acklam's rational approximation.

`p` must lie strictly inside `(0, 1)`."""
function norm_ppf(p::Real)
    (0 < p < 1) || throw(ArgumentError("p must be in the open interval (0, 1), got $p"))
    p = Float64(p)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00)
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low
        q = sqrt(-2.0 * log(p))
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
               ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
    elseif p <= p_high
        q = p - 0.5
        r = q * q
        return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
               (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1.0)
    else
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
                ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
    end
end

_mean(v) = sum(v) / length(v)

function _std_sample(v)
    n = length(v)
    n < 2 && return 0.0
    m = _mean(v)
    return sqrt(sum((x - m)^2 for x in v) / (n - 1))
end

# Acceleration `a` from the leave-one-out jackknife of the mean.
function _jackknife_acceleration(data::Vector{Float64})
    n = length(data)
    total = sum(data)
    loo = [(total - data[i]) / (n - 1) for i in 1:n]
    mean_loo = _mean(loo)
    diffs = [mean_loo - v for v in loo]
    num = sum(d^3 for d in diffs)
    den = 6.0 * (sum(d^2 for d in diffs))^1.5
    den == 0 && return 0.0
    return num / den
end

"""
    bootstrap_mean_ci(data; n_resamples=2000, alpha=0.025, method="bca", seed=0)

Bootstrap a confidence interval for the mean of `data`. `method` is `"percentile"`
or `"bca"`; the interval has confidence level `1 - 2*alpha`; `seed` drives the
built-in LCG so resamples are reproducible across languages.
"""
function bootstrap_mean_ci(data; n_resamples::Integer=2000, alpha::Real=0.025,
        method::AbstractString="bca", seed::Integer=0)
    arr = Vector{Float64}(data)
    n = length(arr)
    n >= 2 || throw(ArgumentError("need at least 2 observations to bootstrap, got $n"))
    all(isfinite, arr) || throw(ArgumentError("data contains non-finite values (nan or inf)"))
    n_resamples >= 1 || throw(ArgumentError("n_resamples must be >= 1, got $n_resamples"))
    (0 < alpha < 0.5) || throw(ArgumentError("alpha must be in the open interval (0, 0.5), got $alpha"))
    (method == "percentile" || method == "bca") ||
        throw(ArgumentError("method must be 'percentile' or 'bca', got $(repr(method))"))
    seed >= 0 || throw(ArgumentError("seed must be non-negative, got $seed"))

    estimate = _mean(arr)
    boundu = UInt64(n)
    # One warm-up mix so seed=0 is not degenerate (matches Python/Rust).
    state = UInt64(seed) + LCG_C
    replicates = Vector{Float64}(undef, n_resamples)
    for b in 1:n_resamples
        acc = 0.0
        for _ in 1:n
            idx, state = _rand_below(state, boundu)
            acc += arr[Int(idx) + 1]
        end
        replicates[b] = acc / n
    end

    standard_error = _std_sample(replicates)
    ordered = sort(replicates)

    local lo_q, hi_q
    if method == "percentile"
        lo_q, hi_q = alpha, 1.0 - alpha
    else
        below = count(v -> v < estimate, replicates)
        frac = below / n_resamples
        eps = 0.5 / n_resamples
        frac = min(max(frac, eps), 1.0 - eps)
        z0 = norm_ppf(frac)
        a = _jackknife_acceleration(arr)
        adjust(tail) = begin
            z = norm_ppf(tail)
            num = z0 + z
            norm_cdf(z0 + num / (1.0 - a * num))
        end
        lo_q = adjust(alpha)
        hi_q = adjust(1.0 - alpha)
    end

    return BootstrapResult(estimate, standard_error, quantile(ordered, lo_q),
        quantile(ordered, hi_q), String(method), Float64(alpha), replicates)
end

end # module Ch169Bootstrap
