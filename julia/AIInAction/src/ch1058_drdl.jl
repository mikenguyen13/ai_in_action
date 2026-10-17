"""
    Ch1058Drdl

Numeric core of DRDL for dynamic financial risk prediction (Julia).

Reference implementations of the deterministic building blocks of the discrete and
regularized deep learning (DRDL) method of Wang, Che, Jiang, and Zhao (2026),
"Leveraging Multiview Data Through Discrete and Regularized Deep Learning for Dynamic
Financial Risk Prediction", Information Systems Research 37(3), together with the
discrete-time survival metrics used to evaluate it.

Mirrors the Python module `aiinaction.ch1058_drdl` and the Rust module
`aiinaction::ch1058_drdl`. The shared fixtures in `test/test_ch1058_drdl.jl` match the
Python/Rust suites, which keeps the three implementations at parity.

Notation. There are `m` prediction horizons `t_1 < ... < t_m`. Each company `i` (a row)
has an observed horizon index `A_i` in `1:m` and an event flag `delta_i`: `delta_i = 1`
means the first risk event happened in horizon `A_i`; `delta_i = 0` means no event was
seen through `A_i` (censored). The model outputs a cumulative risk matrix `F` with
`F[i, u] = P(T_i <= t_u)`.

* `risk_accumulation`: the risk accumulation function (paper Eq. 8),
  `p_u = 1 - exp(-sum_{j<=u} exp(o_j))`, the cumulative incidence of a discrete-time
  hazard model with a complementary log-log link, so monotone in `u` by construction.
* `nearest_codes` and `first_occurrence_mask`: the discrete multiview recoder's
  nearest-codebook assignment (Eq. 2) and the mask-based fusion redundancy mask (Eq. 3).
* `orthogonal_loss`: the codebook orthogonality regularizer (Eq. 7).
* `focal_sigma` and `focal_survival_loss`: the focal survival loss (Eqs. 9-10), which
  penalizes only wrongly ranked comparable pairs.
* `cumulative_labels` and `multiperiod_bce`: per-horizon binary cross-entropy on
  cumulative labels.
* `project_conflicting`: adaptive gradient balancing (Eq. 11).
* `td_concordance`, `td_auc`, `brier_scores`: Antolini time-dependent C-index,
  cumulative/dynamic AUC at a horizon, and per-horizon Brier scores.

Indexing note: `nearest_codes` returns 0-based code indices (matching the shared
fixtures and the input convention of `first_occurrence_mask`), while
`comparable_pairs` returns 1-based row indices (Python reports 0-based). Horizons
(`times`, `horizon`) are 1-based in all three languages.
"""
module Ch1058Drdl

export risk_accumulation, nearest_codes, first_occurrence_mask, orthogonal_loss,
    focal_sigma, comparable_pairs, focal_survival_loss, cumulative_labels,
    multiperiod_bce, project_conflicting, td_concordance, td_auc, brier_scores

# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

function _check_finite(name::AbstractString, x::AbstractArray{<:Real})
    all(isfinite, x) || throw(ArgumentError("$name contains non-finite values"))
    return nothing
end

# Validate (times, events) against an n x m risk matrix and return them as Vector{Int}.
function _survival_targets(times::AbstractVector{<:Real}, events::AbstractVector{<:Real},
                           n::Integer, m::Integer)
    (length(times) == n && length(events) == n) ||
        throw(ArgumentError("times and events must have length $n"))
    all(isinteger, times) || throw(ArgumentError("times must be integer horizon indices"))
    for x in times
        (1 <= x <= m) || throw(ArgumentError("times must lie in 1..$m, got $x"))
    end
    for x in events
        (x == 0 || x == 1) || throw(ArgumentError("events must be 0 or 1, got $x"))
    end
    t = Int[Int(x) for x in times]
    e = Int[Int(x) for x in events]
    return t, e
end

# --------------------------------------------------------------------------
# Risk accumulation function (Eq. 8)
# --------------------------------------------------------------------------

"""
    risk_accumulation(outputs::AbstractVector) -> Vector{Float64}
    risk_accumulation(outputs::AbstractMatrix) -> Matrix{Float64}

Map unconstrained outputs to monotone cumulative risk probabilities,
`p_u = 1 - exp(-sum_{j=1}^{u} exp(o_j))`, computed as `-expm1(-cum)` for accuracy at
small risks. Works on a vector of `m` outputs or row-wise on an `n x m` matrix.
"""
function risk_accumulation(outputs::AbstractVector{<:Real})
    isempty(outputs) && throw(ArgumentError("outputs must be non-empty"))
    _check_finite("outputs", outputs)
    o = collect(Float64, outputs)
    out = Vector{Float64}(undef, length(o))
    cum = 0.0
    for u in 1:length(o)
        cum += exp(o[u])
        out[u] = -expm1(-cum)
    end
    return out
end

function risk_accumulation(outputs::AbstractMatrix{<:Real})
    size(outputs, 2) == 0 && throw(ArgumentError("outputs must have at least one column"))
    _check_finite("outputs", outputs)
    n, m = size(outputs)
    out = Matrix{Float64}(undef, n, m)
    for i in 1:n
        cum = 0.0
        for u in 1:m
            cum += exp(Float64(outputs[i, u]))
            out[i, u] = -expm1(-cum)
        end
    end
    return out
end

# --------------------------------------------------------------------------
# Discrete multiview recoder and mask-based fusion (Eqs. 2-4)
# --------------------------------------------------------------------------

"""
    nearest_codes(embeddings, codebook) -> Vector{Int}

0-based index of the nearest codebook row (squared L2 distance) for every embedding
row. Ties go to the lowest index, matching Python's `argmin`.
"""
function nearest_codes(embeddings::AbstractMatrix{<:Real}, codebook::AbstractMatrix{<:Real})
    _check_finite("embeddings", embeddings)
    _check_finite("codebook", codebook)
    size(embeddings, 2) == size(codebook, 2) ||
        throw(ArgumentError("embeddings and codebook must have the same dimension"))
    size(codebook, 1) == 0 && throw(ArgumentError("codebook must be non-empty"))
    n, d = size(embeddings)
    k = size(codebook, 1)
    out = Vector{Int}(undef, n)
    for i in 1:n
        best = 1
        best_d2 = Inf
        for j in 1:k
            d2 = 0.0
            for c in 1:d
                diff = Float64(embeddings[i, c]) - Float64(codebook[j, c])
                d2 += diff * diff
            end
            if d2 < best_d2  # strict, so the lowest index wins a tie
                best = j
                best_d2 = d2
            end
        end
        out[i] = best - 1
    end
    return out
end

"""
    first_occurrence_mask(codes) -> Vector{Int}

`m_i = 0` if code `z_i` already appeared among `z_1..z_{i-1}`, else `1`. Codes are
non-negative (0-based, as returned by `nearest_codes`).
"""
function first_occurrence_mask(codes::AbstractVector{<:Integer})
    seen = Set{Int}()
    out = zeros(Int, length(codes))
    for (i, z) in enumerate(codes)
        z < 0 && throw(ArgumentError("codes must be non-negative, got $z"))
        zi = Int(z)
        if !(zi in seen)
            out[i] = 1
            push!(seen, zi)
        end
    end
    return out
end

"""
    orthogonal_loss(codebook) -> Float64

Frobenius norm of the strict upper triangle of the Gram matrix of the row-normalized
codebook, `sqrt(sum_{k<l} (c_k . c_l)^2)` for unit codes `c_k`; zero exactly when the
codes are mutually orthogonal.
"""
function orthogonal_loss(codebook::AbstractMatrix{<:Real})
    _check_finite("codebook", codebook)
    k, d = size(codebook)
    u = Matrix{Float64}(undef, k, d)
    for r in 1:k
        s = 0.0
        for c in 1:d
            s += Float64(codebook[r, c])^2
        end
        nr = sqrt(s)
        nr == 0.0 && throw(ArgumentError("codebook rows must be non-zero"))
        for c in 1:d
            u[r, c] = codebook[r, c] / nr
        end
    end
    total = 0.0
    for a in 1:k
        for b in (a + 1):k
            g = 0.0
            for c in 1:d
                g += u[a, c] * u[b, c]
            end
            total += g * g
        end
    end
    return sqrt(total)
end

# --------------------------------------------------------------------------
# Focal survival loss (Eqs. 9-10)
# --------------------------------------------------------------------------

"""
    focal_sigma(x, gamma) -> Float64

`sigma(x) = 2 / (1 + exp(-x^gamma)) - 1` for `x >= 0`: zero at zero and strictly below
one. Broadcast with `focal_sigma.(xs, gamma)` for arrays.
"""
function focal_sigma(x::Real, gamma::Real)
    (isfinite(gamma) && gamma > 0) ||
        throw(ArgumentError("gamma must be a positive finite number, got $gamma"))
    (isfinite(x) && x >= 0) || throw(ArgumentError("x must be finite and non-negative, got $x"))
    return 2.0 / (1.0 + exp(-(Float64(x)^Float64(gamma)))) - 1.0
end

"""
    comparable_pairs(times, events) -> Vector{Tuple{Int,Int}}

Ordered pairs `(i, j)` of 1-based row indices where company `i` is known to fail
before `j`: `delta_i = 1` and either `A_i < A_j`, or `A_i == A_j` with `delta_j = 0`
(`j` survived the horizon in which `i` failed). Ordered by `i`, then `j`.
"""
function comparable_pairs(times::AbstractVector{<:Real}, events::AbstractVector{<:Real})
    length(times) == length(events) ||
        throw(ArgumentError("times and events must have the same length"))
    n = length(times)
    pairs = Tuple{Int,Int}[]
    for i in 1:n
        events[i] == 1 || continue
        for j in 1:n
            i == j && continue
            if times[i] < times[j] || (times[i] == times[j] && events[j] == 0)
                push!(pairs, (i, j))
            end
        end
    end
    return pairs
end

"""
    focal_survival_loss(cum_probs, times, events; gamma=1.0) -> Float64

Mean focal penalty over comparable pairs (Eq. 9). For a comparable pair `(i, j)`
both companies are compared at the horizon `A_i` where `i` failed. The pair is
wrongly ranked when `F[j, A_i] > F[i, A_i]`; its penalty is
`sigma(max(0, F[j, A_i] - F[i, A_i]))`, so correctly ranked pairs contribute exactly
zero. Returns `0.0` if no pair is comparable.
"""
function focal_survival_loss(cum_probs::AbstractMatrix{<:Real}, times::AbstractVector{<:Real},
                             events::AbstractVector{<:Real}; gamma::Real=1.0)
    _check_finite("cum_probs", cum_probs)
    (isfinite(gamma) && gamma > 0) ||
        throw(ArgumentError("gamma must be a positive finite number, got $gamma"))
    t, e = _survival_targets(times, events, size(cum_probs, 1), size(cum_probs, 2))
    pairs = comparable_pairs(t, e)
    isempty(pairs) && return 0.0
    total = 0.0
    for (i, j) in pairs
        u = t[i]
        gap = Float64(cum_probs[j, u]) - Float64(cum_probs[i, u])
        total += focal_sigma(max(0.0, gap), gamma)
    end
    return total / length(pairs)
end

# --------------------------------------------------------------------------
# Multiperiod BCE
# --------------------------------------------------------------------------

"""
    cumulative_labels(times, events, m) -> Matrix{Float64}

`y[i, u] = 1` if company `i` had its first event at or before horizon `u`, else `0`.
"""
function cumulative_labels(times::AbstractVector{<:Real}, events::AbstractVector{<:Real},
                           m::Integer)
    (m isa Bool || m < 1) && throw(ArgumentError("m must be a positive integer, got $m"))
    n = length(times)
    t, e = _survival_targets(times, events, n, m)
    y = zeros(Float64, n, Int(m))
    for i in 1:n
        e[i] == 1 || continue
        for u in t[i]:Int(m)
            y[i, u] = 1.0
        end
    end
    return y
end

"""
    multiperiod_bce(cum_probs, times, events; eps=1e-12) -> Float64

Mean over companies of the summed per-horizon binary cross-entropy between the
cumulative labels and `cum_probs` (clipped to `[eps, 1 - eps]`). Assumes censoring
only at the end of the window (every company is observed through `t_m` unless it
fails earlier), the setting of the paper's yearly, quarterly, and monthly windows.
"""
function multiperiod_bce(cum_probs::AbstractMatrix{<:Real}, times::AbstractVector{<:Real},
                         events::AbstractVector{<:Real}; eps::Real=1e-12)
    _check_finite("cum_probs", cum_probs)
    all(x -> 0 <= x <= 1, cum_probs) || throw(ArgumentError("cum_probs must lie in [0, 1]"))
    n, m = size(cum_probs)
    y = cumulative_labels(times, events, m)
    lo = Float64(eps)
    hi = 1.0 - lo
    total = 0.0
    for i in 1:n
        row = 0.0
        for u in 1:m
            p = clamp(Float64(cum_probs[i, u]), lo, hi)
            row += -(y[i, u] * log(p) + (1.0 - y[i, u]) * log(1.0 - p))
        end
        total += row
    end
    return total / n
end

# --------------------------------------------------------------------------
# Adaptive gradient balancing (Eq. 11)
# --------------------------------------------------------------------------

"""
    project_conflicting(g_main, g_aux) -> Vector{Float64}

Remove the component of `g_aux` that conflicts with `g_main`. If
`cos(g_main, g_aux) < 0` return `g_aux - (g_aux . g_main / ||g_main||^2) g_main`
(orthogonal to `g_main`); otherwise return a copy of `g_aux`. Inputs of any shape are
flattened, as in Python's `ravel`.
"""
function project_conflicting(g_main::AbstractArray{<:Real}, g_aux::AbstractArray{<:Real})
    length(g_main) == length(g_aux) ||
        throw(ArgumentError("gradients must have the same number of elements"))
    (all(isfinite, g_main) && all(isfinite, g_aux)) ||
        throw(ArgumentError("gradients must be finite"))
    a = vec(collect(Float64, g_main))
    b = vec(collect(Float64, g_aux))
    nn = 0.0
    dot_ab = 0.0
    for k in 1:length(a)
        nn += a[k] * a[k]
        dot_ab += a[k] * b[k]
    end
    (nn == 0.0 || dot_ab >= 0.0) && return b
    scale = dot_ab / nn
    return Float64[b[k] - scale * a[k] for k in 1:length(a)]
end

# --------------------------------------------------------------------------
# Evaluation metrics
# --------------------------------------------------------------------------

"""
    td_concordance(cum_probs, times, events) -> Float64

Antolini's time-dependent concordance index: over comparable pairs `(i, j)`, the
fraction with `F[i, A_i] > F[j, A_i]`, counting ties as one half. Throws if no pair is
comparable.
"""
function td_concordance(cum_probs::AbstractMatrix{<:Real}, times::AbstractVector{<:Real},
                        events::AbstractVector{<:Real})
    _check_finite("cum_probs", cum_probs)
    t, e = _survival_targets(times, events, size(cum_probs, 1), size(cum_probs, 2))
    pairs = comparable_pairs(t, e)
    isempty(pairs) && throw(ArgumentError("no comparable pairs"))
    score = 0.0
    for (i, j) in pairs
        u = t[i]
        if cum_probs[i, u] > cum_probs[j, u]
            score += 1.0
        elseif cum_probs[i, u] == cum_probs[j, u]
            score += 0.5
        end
    end
    return score / length(pairs)
end

"""
    td_auc(cum_probs, times, events, horizon) -> Float64

Cumulative/dynamic AUC at the 1-based `horizon`. Cases had their event at or before
`horizon`; controls are known to be event-free through `horizon` (`A_j > horizon`, or
`A_j == horizon` and censored). Ties count one half.
"""
function td_auc(cum_probs::AbstractMatrix{<:Real}, times::AbstractVector{<:Real},
                events::AbstractVector{<:Real}, horizon::Integer)
    _check_finite("cum_probs", cum_probs)
    n, m = size(cum_probs)
    t, e = _survival_targets(times, events, n, m)
    (1 <= horizon <= m) || throw(ArgumentError("horizon must lie in 1..$m, got $horizon"))
    cases = Float64[]
    controls = Float64[]
    for i in 1:n
        v = Float64(cum_probs[i, horizon])
        if e[i] == 1 && t[i] <= horizon
            push!(cases, v)
        elseif t[i] > horizon || (t[i] == horizon && e[i] == 0)
            push!(controls, v)
        end
    end
    (isempty(cases) || isempty(controls)) &&
        throw(ArgumentError("need at least one case and one control"))
    gt = 0
    eq = 0
    for a in cases
        for b in controls
            if a > b
                gt += 1
            elseif a == b
                eq += 1
            end
        end
    end
    return (gt + 0.5 * eq) / (length(cases) * length(controls))
end

"""
    brier_scores(cum_probs, times, events) -> Vector{Float64}

Per-horizon Brier score `mean_i (y[i, u] - F[i, u])^2` against the cumulative labels
(end-of-window censoring).
"""
function brier_scores(cum_probs::AbstractMatrix{<:Real}, times::AbstractVector{<:Real},
                      events::AbstractVector{<:Real})
    _check_finite("cum_probs", cum_probs)
    n, m = size(cum_probs)
    y = cumulative_labels(times, events, m)
    out = zeros(Float64, m)
    for u in 1:m
        s = 0.0
        for i in 1:n
            s += (y[i, u] - Float64(cum_probs[i, u]))^2
        end
        out[u] = s / n
    end
    return out
end

end # module Ch1058Drdl
