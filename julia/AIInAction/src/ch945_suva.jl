"""
    Ch945Suva

SUVA: auditing LLM decisions through state, understanding, value, and action (Julia).

Mirrors the numeric core of the Python module `aiinaction.ch945_suva` and the Rust module
`aiinaction::ch945_suva`, following Leng and Yuan (2026), *SUVA: A Probabilistic
Framework for Auditing LLMs with an Application to Social Preferences*, Information
Systems Research 37(3), 1805 to 1830. The shared fixtures in `test/test_ch945_suva.jl`
match the Python and Rust suites, which keeps the three implementations at parity. The
audit harness (states and prompts, the codebook and coders, the simulated LLM, the
analyses of the paper, the DPO adapter, and audit specifications) is Python only.

* Game design: `payoff_grid`, `motive_features`.
* Estimation: `ols_cluster` (OLS with CR1 cluster-robust standard errors),
  `demean_by_group` (the within transform behind prompt fixed effects),
  `diff_proportions` (the reciprocity statistic), `normal_cdf`.
* Measurement quality: `cohen_kappa` (coder agreement), `auc_score` (CoT
  predictability).
* Audit decisions: `cohens_d`, `tost_p_value` (equivalence tests for "this cue must not
  matter" requirements), `design_effect` and `samples_per_arm` (power planning),
  `dpo_smoothed_margin`.
* Reasoning trees: `collapse_repeats`, `ReasoningTree` with `add!`, `tree_from_paths`
  and `tree_node`, and `path_probability` (Algorithm 1 of the paper).

Naming note: the Python `OlsResult.z` property is the function `zscores` here, the
Python methods `ReasoningTree.add`, `ReasoningTree.from_paths` and `ReasoningTree.node`
are the functions `add!`, `tree_from_paths` and `tree_node`, and the payoff grid game is
the keyword `game` (a `Symbol` or a string).
"""
module Ch945Suva

export Z_975, Z_80, MOTIVES, normal_cdf, payoff_grid, motive_features,
    OlsResult, zscores, p_values, ci, ols_cluster, demean_by_group, diff_proportions,
    cohen_kappa, auc_score, cohens_d, tost_p_value, design_effect, samples_per_arm,
    dpo_smoothed_margin, collapse_repeats, TreeNode, ReasoningTree, add!,
    tree_from_paths, tree_node, path_probability

"""Two-sided 5 percent critical value of the standard normal."""
const Z_975 = 1.959963984540054

"""Standard normal quantile for 80 percent power."""
const Z_80 = 0.8416212335729143

"""Regressors of Eq. (3) in Leng and Yuan, in order."""
const MOTIVES = ("self_interest", "competition", "difference_aversion", "social_welfare")

# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

function _check_finite(name::AbstractString, value::Real)
    isfinite(value) || throw(ArgumentError("$name must be finite, got $value"))
    return nothing
end

function _check_prob(name::AbstractString, value::Real; open_low::Bool=false, open_high::Bool=false)
    _check_finite(name, value)
    if value < 0 || value > 1 || (open_low && value == 0) || (open_high && value == 1)
        throw(ArgumentError("$name must lie in the unit interval, got $value"))
    end
    return nothing
end

# Sign as an Int (Bool minus Bool is an Int in Julia).
_sgn(x::Real) = (x > 0) - (x < 0)

# Plain left-to-right sum (no pairwise or SIMD reordering) to mirror Python's sum().
function _seqsum(v)
    s = 0.0
    for x in v
        s += x
    end
    return s
end

# --------------------------------------------------------------------------
# Normal CDF (Hart 1968 as given by West 2005), identical in Python, Julia, and Rust
# --------------------------------------------------------------------------

"""
    normal_cdf(x) -> Float64

Standard normal CDF, accurate to about 1e-14 everywhere.

Uses the double-precision rational approximation of Hart (1968) as given by West (2005),
with the same constants and operation order as the Python and Rust ports, so p-values
agree across languages without a special-function dependency.
"""
function normal_cdf(x::Real)
    _check_finite("x", x)
    xa = abs(Float64(x))
    if xa > 37.0
        c = 0.0
    else
        e = exp(-xa * xa / 2.0)
        if xa < 7.07106781186547
            num = 3.52624965998911e-02 * xa + 0.700383064443688
            num = num * xa + 6.37396220353165
            num = num * xa + 33.912866078383
            num = num * xa + 112.079291497871
            num = num * xa + 221.213596169931
            num = num * xa + 220.206867912376
            den = 8.83883476483184e-02 * xa + 1.75566716318264
            den = den * xa + 16.064177579207
            den = den * xa + 86.7807322029461
            den = den * xa + 296.564248779674
            den = den * xa + 637.333633378831
            den = den * xa + 793.826512519948
            den = den * xa + 440.413735824752
            c = e * num / den
        else
            b = xa + 0.65
            b = xa + 4.0 / b
            b = xa + 3.0 / b
            b = xa + 2.0 / b
            b = xa + 1.0 / b
            c = e / b / 2.506628274631
        end
    end
    return x > 0 ? 1.0 - c : c
end

# --------------------------------------------------------------------------
# Game design
# --------------------------------------------------------------------------

"""
    payoff_grid(levels=(0, 200, 400, 600); game=:distributional) -> Vector{NTuple{4,Int}}

Enumerate dictator-game payoff configurations `(a1, b1, a2, b2)`.

`a1, b1` are the payoffs to person A (the simulated match) and person B (the audited
model) under option B1; `a2, b2` under option B2. The order is lexicographic in
`(a1, b1, a2, b2)`.

* `game=:distributional`: every configuration except identical options
  (`length(levels)^4 - length(levels)^2` of them, 240 for four levels).
* `game=:reciprocity`: only configurations in which one option is better for A and no
  better for B, so that helping A is costly or free but never also self-serving (Leng
  and Yuan, Section 3.2).

`levels` must hold at least two distinct integer values (integer-valued floats are
accepted). Throws `ArgumentError` for an unknown game or invalid levels.
"""
function payoff_grid(levels=(0, 200, 400, 600);
                     game::Union{Symbol,AbstractString}=:distributional)
    mode = Symbol(game)
    (mode === :distributional || mode === :reciprocity) ||
        throw(ArgumentError("unknown game $(repr(game))"))
    lv = Int[]
    for v in levels
        (v isa Real && isfinite(v) && isinteger(v)) ||
            throw(ArgumentError("levels must be integers, got $(repr(v))"))
        push!(lv, Int(v))
    end
    (length(lv) < 2 || length(unique(lv)) != length(lv)) &&
        throw(ArgumentError("levels must contain at least two distinct values"))
    out = NTuple{4,Int}[]
    for a1 in lv, b1 in lv, a2 in lv, b2 in lv
        if mode === :distributional
            keep = (a1, b1) != (a2, b2)
        else
            keep = (a1 > a2 && b1 <= b2) || (a1 < a2 && b1 >= b2)
        end
        keep && push!(out, (a1, b1, a2, b2))
    end
    return out
end

"""
    motive_features(a1, b1, a2, b2) -> NTuple{4,Int}

Sign regressors of Eq. (3): self interest, competition, difference aversion, welfare.

Each is +1 if the motive favors B1, -1 if it favors B2, and 0 if it is indifferent:

* self interest: `sgn(b1 - b2)`;
* competition: `sgn((b1 - a1) - (b2 - a2))`;
* difference aversion: `sgn(|b2 - a2| - |b1 - a1|)`;
* social welfare: `sgn((a1 + b1) - (a2 + b2))`.
"""
function motive_features(a1::Real, b1::Real, a2::Real, b2::Real)
    _check_finite("a1", a1)
    _check_finite("b1", b1)
    _check_finite("a2", a2)
    _check_finite("b2", b2)
    return (_sgn(b1 - b2),
            _sgn((b1 - a1) - (b2 - a2)),
            _sgn(abs(b2 - a2) - abs(b1 - a1)),
            _sgn((a1 + b1) - (a2 + b2)))
end

# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------

"""
OLS fit with cluster-robust (CR1) standard errors: coefficients `coef`, standard errors
`se`, the number of observations `n`, and the number of clusters `n_clusters`.
"""
struct OlsResult
    coef::Vector{Float64}
    se::Vector{Float64}
    n::Int
    n_clusters::Int
end

"""
    zscores(r::OlsResult) -> Vector{Float64}

Ratios `coef / se`, or `NaN` where the standard error is not positive.
"""
zscores(r::OlsResult) = [s > 0 ? c / s : NaN for (c, s) in zip(r.coef, r.se)]

"""
    p_values(r::OlsResult) -> Vector{Float64}

Two-sided p-values from the normal reference distribution (`NaN` where the z-score is
not finite).
"""
p_values(r::OlsResult) = [isfinite(z) ? 2.0 * normal_cdf(-abs(z)) : NaN for z in zscores(r)]

"""
    ci(r::OlsResult, z=Z_975) -> Vector{Tuple{Float64,Float64}}

Normal confidence intervals `(coef - z se, coef + z se)`, one per coefficient.
"""
function ci(r::OlsResult, z::Real=Z_975)
    zf = Float64(z)
    return [(c - zf * s, c + zf * s) for (c, s) in zip(r.coef, r.se)]
end

# Gauss-Jordan inverse with partial pivoting (same algorithm in all ports).
function _invert(m::Matrix{Float64})
    k = size(m, 1)
    a = zeros(Float64, k, 2 * k)
    for i in 1:k
        for j in 1:k
            a[i, j] = m[i, j]
        end
        a[i, k + i] = 1.0
    end
    scale = 0.0
    for v in m
        scale = max(scale, abs(v))
    end
    tol = 1e-12 * max(scale, 1.0)
    for col in 1:k
        # First row with the largest absolute pivot, as Python's max() picks it.
        piv = col
        best = abs(a[col, col])
        for r in (col + 1):k
            if abs(a[r, col]) > best
                best = abs(a[r, col])
                piv = r
            end
        end
        best <= tol &&
            throw(ArgumentError("design matrix is singular (collinear or constant regressors)"))
        if piv != col
            for j in 1:(2 * k)
                a[col, j], a[piv, j] = a[piv, j], a[col, j]
            end
        end
        p = a[col, col]
        for j in 1:(2 * k)
            a[col, j] = a[col, j] / p
        end
        for r in 1:k
            if r != col && a[r, col] != 0.0
                f = a[r, col]
                for j in 1:(2 * k)
                    a[r, j] = a[r, j] - f * a[col, j]
                end
            end
        end
    end
    return a[:, (k + 1):(2 * k)]
end

"""
    ols_cluster(x, y, clusters) -> OlsResult

Least squares with CR1 cluster-robust standard errors.

`x` is an `n x K` matrix or a vector of `n` rows of equal length `K` (include a column
of ones for an intercept). The covariance is the sandwich

`V = c (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1`, `c = G/(G-1) * (N-1)/(N-K)`,

the default of Stata's `vce(cluster)` and statsmodels' `cov_type="cluster"`. Clusters
are taken in order of first appearance. Replicates of one prompt share its sampling
noise, so the prompt (or the payoff configuration) is the natural cluster.

Throws `ArgumentError` for mismatched or empty inputs, `n <= K`, a singular design, or
fewer than two clusters.
"""
function ols_cluster(x::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
                     clusters::AbstractVector)
    n = length(y)
    (n == 0 || size(x, 1) != n || length(clusters) != n) &&
        throw(ArgumentError("x, y, and clusters must be non-empty and of equal length"))
    k = size(x, 2)
    k == 0 && throw(ArgumentError("every row of x must have the same positive length"))
    n <= k && throw(ArgumentError("need more observations than regressors"))
    for v in y
        _check_finite("y", v)
    end
    X = Matrix{Float64}(x)
    Y = Vector{Float64}(y)
    cl = collect(clusters)
    xtx = zeros(Float64, k, k)
    xty = zeros(Float64, k)
    for t in 1:n
        for i in 1:k
            xty[i] += X[t, i] * Y[t]
            for j in 1:k
                xtx[i, j] += X[t, i] * X[t, j]
            end
        end
    end
    xtx_inv = _invert(xtx)
    beta = zeros(Float64, k)
    for i in 1:k
        acc = 0.0
        for j in 1:k
            acc += xtx_inv[i, j] * xty[j]
        end
        beta[i] = acc
    end
    # Cluster scores X_g' u_g, clusters in order of first appearance.
    index = Dict{eltype(cl),Int}()
    scores = Vector{Vector{Float64}}()
    for t in 1:n
        fitted = 0.0
        for i in 1:k
            fitted += beta[i] * X[t, i]
        end
        u = Y[t] - fitted
        g = cl[t]
        gi = get(index, g, 0)
        if gi == 0
            push!(scores, zeros(Float64, k))
            gi = length(scores)
            index[g] = gi
        end
        score = scores[gi]
        for i in 1:k
            score[i] += X[t, i] * u
        end
    end
    n_g = length(scores)
    n_g < 2 && throw(ArgumentError("need at least two clusters"))
    meat = zeros(Float64, k, k)
    for score in scores
        for i in 1:k
            for j in 1:k
                meat[i, j] += score[i] * score[j]
            end
        end
    end
    c = n_g / (n_g - 1) * (n - 1) / (n - k)
    tmp = zeros(Float64, k, k)
    for i in 1:k
        for j in 1:k
            acc = 0.0
            for m in 1:k
                acc += xtx_inv[i, m] * meat[m, j]
            end
            tmp[i, j] = acc
        end
    end
    cov = zeros(Float64, k, k)
    for i in 1:k
        for j in 1:k
            acc = 0.0
            for m in 1:k
                acc += tmp[i, m] * xtx_inv[m, j]
            end
            cov[i, j] = c * acc
        end
    end
    se = [sqrt(max(cov[i, i], 0.0)) for i in 1:k]
    return OlsResult(beta, se, n, n_g)
end

function ols_cluster(x::AbstractVector{<:AbstractVector{<:Real}}, y::AbstractVector{<:Real},
                     clusters::AbstractVector)
    n = length(y)
    (n == 0 || length(x) != n || length(clusters) != n) &&
        throw(ArgumentError("x, y, and clusters must be non-empty and of equal length"))
    k = length(first(x))
    (k == 0 || any(row -> length(row) != k, x)) &&
        throw(ArgumentError("every row of x must have the same positive length"))
    X = Matrix{Float64}(undef, n, k)
    for (t, row) in enumerate(x)
        for (i, v) in enumerate(row)
            X[t, i] = v
        end
    end
    return ols_cluster(X, y, clusters)
end

"""
    demean_by_group(values, groups) -> Vector{Float64}

Subtract each group's mean: the within transform behind fixed effects.

By the Frisch-Waugh-Lovell theorem, regressing demeaned `y` on demeaned `W` (no
intercept) gives the same slopes as including one dummy per group, which is how prompt
fixed effects are absorbed.
"""
function demean_by_group(values::AbstractVector{<:Real}, groups::AbstractVector)
    length(values) == length(groups) ||
        throw(ArgumentError("values and groups must have equal length"))
    total = Dict{eltype(groups),Float64}()
    counts = Dict{eltype(groups),Int}()
    for (v, g) in zip(values, groups)
        _check_finite("values", v)
        total[g] = get(total, g, 0.0) + v
        counts[g] = get(counts, g, 0) + 1
    end
    return [Float64(v) - total[g] / counts[g] for (v, g) in zip(values, groups)]
end

"""
    diff_proportions(k1, n1, k0, n0) -> Tuple{Float64,Float64}

Difference of proportions `k1/n1 - k0/n0` and its unpooled Wald standard error.

With `k1` prosocial choices out of `n1` after the match helped and `k0` of `n0` after it
misbehaved, the difference is the reciprocity statistic of Eq. (5).
"""
function diff_proportions(k1::Real, n1::Real, k0::Real, n0::Real)
    for (name, v) in (("k1", k1), ("n1", n1), ("k0", k0), ("n0", n0))
        (isfinite(v) && isinteger(v) && v >= 0) ||
            throw(ArgumentError("$name must be a non-negative integer"))
    end
    (n1 == 0 || n0 == 0 || k1 > n1 || k0 > n0) &&
        throw(ArgumentError("need 0 <= k <= n and n > 0 in both arms"))
    m1 = Float64(n1)
    m0 = Float64(n0)
    p1 = Float64(k1) / m1
    p0 = Float64(k0) / m0
    return (p1 - p0, sqrt(p1 * (1.0 - p1) / m1 + p0 * (1.0 - p0) / m0))
end

# --------------------------------------------------------------------------
# Measurement quality
# --------------------------------------------------------------------------

"""
    cohen_kappa(a, b) -> Float64

Cohen's kappa between two coders, `(p_o - p_e) / (1 - p_e)`. Labels may be of any type
that supports `isequal` and `hash` (integers, strings, symbols).
"""
function cohen_kappa(a::AbstractVector, b::AbstractVector)
    n = length(a)
    (n == 0 || length(b) != n) &&
        throw(ArgumentError("label sequences must be non-empty and of equal length"))
    agree = 0
    for (x, y) in zip(a, b)
        isequal(x, y) && (agree += 1)
    end
    po = agree / n
    ca = Dict{Any,Int}()
    cb = Dict{Any,Int}()
    for x in a
        ca[x] = get(ca, x, 0) + 1
    end
    for y in b
        cb[y] = get(cb, y, 0) + 1
    end
    # Integer numerator, so the result does not depend on Dict iteration order.
    num = 0
    for (lab, cnt) in ca
        num += cnt * get(cb, lab, 0)
    end
    pe = num / (n * n)
    if pe == 1.0
        return po == 1.0 ? 1.0 : 0.0
    end
    return (po - pe) / (1.0 - pe)
end

"""
    auc_score(scores, labels) -> Float64

Area under the ROC curve via the Mann-Whitney statistic with midranks for ties. Labels
must be 0 or 1 and both classes must be present.
"""
function auc_score(scores::AbstractVector{<:Real}, labels::AbstractVector{<:Real})
    n = length(scores)
    (n == 0 || length(labels) != n) &&
        throw(ArgumentError("scores and labels must be non-empty and of equal length"))
    all(lab -> lab == 0 || lab == 1, labels) ||
        throw(ArgumentError("labels must be 0 or 1"))
    sc = Vector{Float64}(scores)
    lb = [lab == 1 for lab in labels]
    n_pos = count(lb)
    n_neg = n - n_pos
    (n_pos == 0 || n_neg == 0) && throw(ArgumentError("need both classes"))
    order = sortperm(sc)
    ranks = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j + 1 <= n && sc[order[j + 1]] == sc[order[i]]
            j += 1
        end
        # 1-based positions i..j share the midrank (i + j) / 2.
        mid = (i + j) / 2.0
        for t in i:j
            ranks[order[t]] = mid
        end
        i = j + 1
    end
    r_pos = 0.0
    for t in 1:n
        lb[t] && (r_pos += ranks[t])
    end
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
end

# --------------------------------------------------------------------------
# Audit decisions
# --------------------------------------------------------------------------

"""
    cohens_d(x, y) -> Float64

Standardized mean difference `(mean(x) - mean(y)) / s_pooled`.
"""
function cohens_d(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    nx = length(x)
    ny = length(y)
    (nx < 2 || ny < 2) && throw(ArgumentError("each sample needs at least two values"))
    mx = _seqsum(x) / nx
    my = _seqsum(y) / ny
    vx = _seqsum((v - mx)^2 for v in x) / (nx - 1)
    vy = _seqsum((v - my)^2 for v in y) / (ny - 1)
    sp = sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    sp == 0.0 && throw(ArgumentError("pooled standard deviation is zero"))
    return (mx - my) / sp
end

"""
    tost_p_value(estimate, se, margin) -> Float64

Two one-sided tests (TOST) p-value for `|true effect| < margin`.

Equivalence is declared at level alpha when the returned p-value is below alpha, that
is, when the `1 - 2 alpha` confidence interval lies inside `(-margin, margin)`. Normal
reference distribution.
"""
function tost_p_value(estimate::Real, se::Real, margin::Real)
    _check_finite("estimate", estimate)
    (se > 0 && isfinite(se)) || throw(ArgumentError("se must be positive and finite"))
    (margin > 0 && isfinite(margin)) ||
        throw(ArgumentError("margin must be positive and finite"))
    est = Float64(estimate)
    s = Float64(se)
    m = Float64(margin)
    p_lower = 1.0 - normal_cdf((est + m) / s)
    p_upper = normal_cdf((est - m) / s)
    return max(p_lower, p_upper)
end

"""
    design_effect(cluster_size, icc) -> Float64

Kish design effect `1 + (m - 1) rho` for `m` replicates per prompt.
"""
function design_effect(cluster_size::Real, icc::Real)
    (cluster_size >= 1 && isfinite(cluster_size)) ||
        throw(ArgumentError("cluster_size must be at least 1"))
    _check_prob("icc", icc)
    return 1.0 + (Float64(cluster_size) - 1.0) * Float64(icc)
end

"""
    samples_per_arm(p1, p0; z_alpha=Z_975, z_beta=Z_80, deff=1.0) -> Int

Responses per arm to detect `p1 - p0` with a two-sample test:
`n = deff (z_alpha + z_beta)^2 (p1 (1 - p1) + p0 (1 - p0)) / (p1 - p0)^2`, rounded up
(after subtracting `1e-12`, as in the Python and Rust ports). `deff` inflates for
replicates clustered within prompts.
"""
function samples_per_arm(p1::Real, p0::Real; z_alpha::Real=Z_975, z_beta::Real=Z_80,
                         deff::Real=1.0)
    _check_prob("p1", p1)
    _check_prob("p0", p0)
    p1 == p0 && throw(ArgumentError("p1 and p0 must differ"))
    (deff >= 1 && isfinite(deff)) || throw(ArgumentError("deff must be at least 1"))
    (isfinite(z_alpha) && isfinite(z_beta) && z_alpha > 0 && z_beta >= 0) ||
        throw(ArgumentError("z_alpha must be positive and z_beta non-negative"))
    a = Float64(p1)
    b = Float64(p0)
    zs = Float64(z_alpha) + Float64(z_beta)
    var = a * (1.0 - a) + b * (1.0 - b)
    return ceil(Int, Float64(deff) * zs^2 * var / (a - b)^2 - 1e-12)
end

"""
    dpo_smoothed_margin(epsilon, beta=1.0) -> Float64

Optimal implicit-reward margin of label-smoothed DPO on one pair.

Minimizing `-(1 - eps) log sigma(beta m) - eps log sigma(-beta m)` over the log-ratio
margin `m` gives `m* = log((1 - eps) / eps) / beta`. Without smoothing (`eps` going to
0) the optimum diverges, which is why hard-preference DPO keeps pushing the policy
toward determinism. Requires `0 < epsilon < 0.5` and `beta > 0`.
"""
function dpo_smoothed_margin(epsilon::Real, beta::Real=1.0)
    _check_prob("epsilon", epsilon; open_low=true)
    epsilon >= 0.5 && throw(ArgumentError("epsilon must be below 0.5"))
    (beta > 0 && isfinite(beta)) || throw(ArgumentError("beta must be positive"))
    e = Float64(epsilon)
    return log((1.0 - e) / e) / Float64(beta)
end

# --------------------------------------------------------------------------
# Reasoning trees (Algorithm 1)
# --------------------------------------------------------------------------

"""
    collapse_repeats(labels) -> Vector

Merge consecutive duplicate codes, as Algorithm 1 does before building the tree.
"""
function collapse_repeats(labels)
    out = Vector{eltype(labels)}()
    for lab in labels
        if isempty(out) || out[end] != lab
            push!(out, lab)
        end
    end
    return out
end

"""
One node of a `ReasoningTree`: its `label`, how many responses pass through it
(`count`), how those responses end (`actions`, action => count), and its `children`
(label => node).
"""
mutable struct TreeNode
    label::String
    count::Int
    actions::Dict{String,Int}
    children::Dict{String,TreeNode}
end

TreeNode(label::AbstractString) =
    TreeNode(String(label), 0, Dict{String,Int}(), Dict{String,TreeNode}())

"""
    ReasoningTree()

Probabilistic decision tree over coded reasoning paths (Algorithm 1).

Each path is the collapsed sequence of codes followed by the action. A node stores how
many responses pass through it and how those responses end, so `P(next | prefix, S)` is
a ratio of child to parent counts and `P(action | prefix, S)` is the node's action
share. Build it with `add!` or `tree_from_paths`; look nodes up with `tree_node`.
"""
struct ReasoningTree
    root::TreeNode
end

ReasoningTree() = ReasoningTree(TreeNode("root"))

function _bump!(node::TreeNode, action::String)
    node.count += 1
    node.actions[action] = get(node.actions, action, 0) + 1
    return nothing
end

"""
    add!(tree::ReasoningTree, codes, action) -> ReasoningTree

Add one response: its codes (consecutive repeats are collapsed) followed by its action.
"""
function add!(tree::ReasoningTree, codes, action::AbstractString)
    act = String(action)
    path = String[String(c) for c in collapse_repeats(codes)]
    push!(path, act)
    node = tree.root
    _bump!(node, act)
    for lab in path
        child = get(node.children, lab, nothing)
        if child === nothing
            child = TreeNode(lab)
            node.children[lab] = child
        end
        node = child
        _bump!(node, act)
    end
    return tree
end

"""
    tree_from_paths(paths) -> ReasoningTree

Build a tree from an iterable of `(codes, action)` pairs.
"""
function tree_from_paths(paths)
    tree = ReasoningTree()
    for (codes, action) in paths
        add!(tree, codes, action)
    end
    return tree
end

"""
    tree_node(tree::ReasoningTree, prefix) -> TreeNode

The node reached by following the labels in `prefix` from the root. Throws `KeyError`
if the path does not exist.
"""
function tree_node(tree::ReasoningTree, prefix)
    node = tree.root
    for lab in prefix
        haskey(node.children, lab) || throw(KeyError(collect(prefix)))
        node = node.children[lab]
    end
    return node
end

"""
    path_probability(tree::ReasoningTree, path) -> Float64

Product of conditional transition probabilities along `path`:
`P(C1, ..., Cn, A | S) = P(C1|S) * prod_j P(Cj | C1:j-1, S) * P(A | C1:n, S)`,
estimated by child-to-parent count ratios. Returns 0 for unseen paths and throws
`ArgumentError` for an empty tree.
"""
function path_probability(tree::ReasoningTree, path)
    node = tree.root
    node.count == 0 && throw(ArgumentError("empty tree"))
    prob = 1.0
    for lab in path
        child = get(node.children, lab, nothing)
        child === nothing && return 0.0
        prob *= child.count / node.count
        node = child
    end
    return prob
end

end # module Ch945Suva
