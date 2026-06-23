"""
    Ch165Mcnemar

McNemar's test for comparing two classifiers on a shared test set (Julia).

Mirrors the Python module `aiinaction.ch165_mcnemar` and the Rust module
`aiinaction::ch165_mcnemar`. The shared fixtures in `test/test_ch165_mcnemar.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

Given two classifiers evaluated on the same test set, the 2x2 correctness table is

    ```
                     B correct   B wrong
        A correct        a           b
        A wrong          c           d
    ```

Only the discordant cells `b` and `c` carry information. Under the null hypothesis
of equal error rates, `b ~ Binomial(b + c, 1/2)`. Two variants are provided: the
chi-squared approximation with Edwards' continuity correction, and the exact
two-sided binomial test on `min(b, c)`. The chi-squared survival function is
computed from the regularized upper incomplete gamma `Q(1/2, x/2)` (Base-only, no
SpecialFunctions dependency), agreeing with the Python/Rust results to machine
precision.
"""
module Ch165Mcnemar

export ContingencyTable, McNemarResult, contingency_table, mcnemar_test, n_total, n_discordant

"""The 2x2 correctness contingency table for two classifiers."""
struct ContingencyTable
    a::Int
    b::Int
    c::Int
    d::Int
end

n_total(t::ContingencyTable) = t.a + t.b + t.c + t.d
n_discordant(t::ContingencyTable) = t.b + t.c

"""The outcome of McNemar's test.

`statistic` is the chi-squared statistic, or `min(b, c)` for the exact variant.
`method` is either `"chi2"` or `"exact"`."""
struct McNemarResult
    statistic::Float64
    p_value::Float64
    method::String
    b::Int
    c::Int
end

"""Build the 2x2 correctness table from two per-example correctness vectors."""
function contingency_table(correct_a, correct_b)
    length(correct_a) == length(correct_b) || throw(ArgumentError(
        "length mismatch: len(correct_a)=$(length(correct_a)) != len(correct_b)=$(length(correct_b))"))
    isempty(correct_a) && throw(ArgumentError("inputs must be non-empty"))

    a = b = c = d = 0
    for (ca_raw, cb_raw) in zip(correct_a, correct_b)
        ca = Bool(ca_raw != 0)
        cb = Bool(cb_raw != 0)
        if ca && cb
            a += 1
        elseif ca && !cb
            b += 1
        elseif !ca && cb
            c += 1
        else
            d += 1
        end
    end
    return ContingencyTable(a, b, c, d)
end

"""Regularized upper incomplete gamma function `Q(s, x) = 1 - P(s, x)`."""
function _gammq(s::Float64, x::Float64)
    x <= 0.0 && return 1.0
    if x < s + 1.0
        # Series expansion for the lower regularized gamma P, then return 1 - P.
        ap = s
        summ = 1.0 / s
        del = summ
        for _ in 1:1000
            ap += 1.0
            del *= x / ap
            summ += del
            abs(del) < abs(summ) * 1e-16 && break
        end
        p = summ * exp(-x + s * log(x) - lgamma_(s))
        return 1.0 - p
    else
        # Lentz's continued fraction for the upper regularized gamma Q.
        tiny = 1e-300
        b = x + 1.0 - s
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in 1:1000
            an = -Float64(i) * (Float64(i) - s)
            b += 2.0
            d = an * d + b
            abs(d) < tiny && (d = tiny)
            c = b + an / c
            abs(c) < tiny && (c = tiny)
            d = 1.0 / d
            del = d * c
            h *= del
            abs(del - 1.0) < 1e-16 && break
        end
        return exp(-x + s * log(x) - lgamma_(s)) * h
    end
end

# Base.lgamma was removed from Base; use a self-contained Lanczos log-gamma so this
# module needs no SpecialFunctions dependency and matches Python/Rust exactly.
function lgamma_(x::Float64)
    cof = (76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5)
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * log(tmp)
    ser = 1.000000000190015
    for c in cof
        y += 1.0
        ser += c / y
    end
    return -tmp + log(2.5066282746310005 * ser / x)
end

"""Survival function of the chi-squared distribution with one degree of freedom."""
_chi2_sf_1dof(x::Float64) = x <= 0.0 ? 1.0 : _gammq(0.5, x / 2.0)

"""Probability mass `C(n, k) (1/2)^n`, evaluated in log space for stability."""
function _binom_pmf_half(k::Int, n::Int)
    log_coef = lgamma_(Float64(n + 1)) - lgamma_(Float64(k + 1)) - lgamma_(Float64(n - k + 1))
    return exp(log_coef - n * log(2.0))
end

"""Two-sided exact binomial p-value for `min(b, c)` under `Binomial(b + c, 1/2)`."""
function _exact_two_sided_p(b::Int, c::Int)
    n = b + c
    n == 0 && return 1.0
    k = min(b, c)
    lower_tail = 0.0
    for i in 0:k
        lower_tail += _binom_pmf_half(i, n)
    end
    return min(1.0, 2.0 * lower_tail)
end

"""
    mcnemar_test(b, c; exact=nothing, correction=true)

Run McNemar's test on the discordant counts `b` and `c`. `exact=nothing` selects
the exact test when `b + c < 25` and the chi-squared approximation otherwise.
`correction` toggles Edwards' continuity correction (ignored by the exact variant).
"""
function mcnemar_test(b::Integer, c::Integer; exact=nothing, correction::Bool=true)
    (b >= 0 && c >= 0) || throw(ArgumentError("b and c must be non-negative, got b=$b, c=$c"))
    b = Int(b)
    c = Int(c)
    n = b + c
    n == 0 && throw(ArgumentError("McNemar's test is undefined when b + c = 0 (no discordant pairs)"))

    use_exact = exact === nothing ? (n < 25) : Bool(exact)

    if use_exact
        p = _exact_two_sided_p(b, c)
        return McNemarResult(Float64(min(b, c)), p, "exact", b, c)
    end

    diff = Float64(abs(b - c))
    delta = correction ? max(0.0, diff - 1.0) : diff
    chi2 = (delta * delta) / n
    p = _chi2_sf_1dof(chi2)
    return McNemarResult(chi2, p, "chi2", b, c)
end

end # module Ch165Mcnemar
