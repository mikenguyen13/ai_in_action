"""
    ChMigrationFlows

Stock-flow accounting, scaling, balancing, loss and calibration pieces of Gaskin and
Abel (2026), "Deep learning four decades of human migration" (Julia).

Mirrors the Python module `aiinaction.ch_migration_flows` and the Rust module
`aiinaction::ch_migration_flows`. The shared fixtures in
`test/test_ch_migration_flows.jl` match the Python/Rust suites, which keeps the three
implementations at parity. The synthetic world generator, the torch recurrent model
and the training loop are Python only; every numeric function is ported.

* Accounting: `stock_step`, `od_flows`, `net_migration` implement
  `S_ij(t+1) = S_ij + delta_ij B_j - mu_j S_ij + sum_k (T_ikj - T_ijk)` with the flow
  tensor `T[i, j, k]` (born in i, moving from j to k).
* Scaling: `yj_sym`, `yj_sym_inv`, `yj_sym_deriv`, `yj_profile_loglik`,
  `fit_yj_lambda`, `elasticity_chain`.
* Balancing: `ipf` and `demographic_balance`, then `uncertainty_weights`.
* Loss and calibration: `weighted_l2`, `survival_fraction`, `calibrate_offset`.
* Baseline and metrics: `stock_difference_baseline`, `gravity_log_rate`, `log_rmse`,
  `pearson`, `coverage`, `ensemble_mean_std`.

Tensors are 1-based `T[i, j, k]`; the sample axis of `ensemble_mean_std` is the first.
"""
module ChMigrationFlows

export yj_sym, yj_sym_inv, yj_sym_deriv, yj_profile_loglik, fit_yj_lambda,
    elasticity_chain, stock_step, od_flows, net_migration, ipf, demographic_balance,
    uncertainty_weights, weighted_l2, survival_fraction, calibrate_offset,
    stock_difference_baseline, gravity_log_rate, log_rmse, pearson, coverage,
    ensemble_mean_std

# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

function _check_finite(name::AbstractString, x)
    all(isfinite, x) || throw(ArgumentError("$name must be finite"))
    return nothing
end

function _check_lambda(lam::Real)
    isfinite(lam) || throw(ArgumentError("lambda must be finite, got $lam"))
    return nothing
end

_mean(x) = sum(x) / length(x)

# --------------------------------------------------------------------------
# Symmetrised Yeo-Johnson transform
# --------------------------------------------------------------------------

"""
    yj_sym(x, lam)

`sgn(x) ((|x|+1)^lam - 1)/lam`, or `sgn(x) log(|x|+1)` when `lam` is (near) zero.
`lam = 1` is the identity.
"""
function yj_sym(x::Real, lam::Real)
    _check_lambda(lam)
    isfinite(x) || throw(ArgumentError("x must be finite"))
    mag = abs(x)
    body = abs(lam) < 1e-12 ? log1p(mag) : expm1(lam * log1p(mag)) / lam
    return sign(x) * body
end
yj_sym(x::AbstractArray, lam::Real) = [yj_sym(v, lam) for v in x]

"""Inverse of [`yj_sym`](@ref); throws if `y` is outside the transform's range."""
function yj_sym_inv(y::Real, lam::Real)
    _check_lambda(lam)
    isfinite(y) || throw(ArgumentError("y must be finite"))
    mag = abs(y)
    if abs(lam) < 1e-12
        body = expm1(mag)
    else
        inner = 1.0 + lam * mag
        inner <= 0.0 &&
            throw(ArgumentError("y is outside the range of the transform for this lambda"))
        body = expm1(log(inner) / lam)
    end
    return sign(y) * body
end
yj_sym_inv(y::AbstractArray, lam::Real) = [yj_sym_inv(v, lam) for v in y]

"""`d phi / d x = (|x|+1)^(lam-1)`."""
function yj_sym_deriv(x::Real, lam::Real)
    _check_lambda(lam)
    isfinite(x) || throw(ArgumentError("x must be finite"))
    return exp((lam - 1.0) * log1p(abs(x)))
end

"""
    yj_profile_loglik(x, lam)

Profile Gaussian log-likelihood `-n/2 log var(phi(x)) + (lam - 1) sum log(|x|+1)`
(population variance), up to a constant.
"""
function yj_profile_loglik(x::AbstractVector{<:Real}, lam::Real)
    length(x) >= 2 || throw(ArgumentError("need at least two observations"))
    phi = yj_sym(x, lam)
    m = _mean(phi)
    v = _mean((phi .- m) .^ 2)
    v > 0 || throw(ArgumentError("transformed sample has zero variance"))
    jac = (lam - 1.0) * sum(log1p.(abs.(x)))
    return -0.5 * length(x) * log(v) + jac
end

"""Grid-search maximum likelihood lambda (default grid -1 to 2 in steps of 0.01)."""
function fit_yj_lambda(x::AbstractVector{<:Real}; grid=((-100):200) ./ 100.0)
    best = nothing
    best_ll = -Inf
    for lam in grid
        ll = try
            yj_profile_loglik(x, lam)
        catch err
            err isa ArgumentError ? continue : rethrow()
        end
        if ll > best_ll
            best, best_ll = float(lam), ll
        end
    end
    best === nothing && throw(ArgumentError("no admissible lambda on the grid"))
    return best
end

"""`eps = (du/dphi) phi'(x) |x|`: elasticity of `u = log T` with respect to raw `x`."""
function elasticity_chain(du_dphi::Real, x::Real, lam::Real)
    isfinite(du_dphi) || throw(ArgumentError("du_dphi must be finite"))
    return du_dphi * yj_sym_deriv(x, lam) * abs(x)
end

# --------------------------------------------------------------------------
# Stock-flow accounting
# --------------------------------------------------------------------------

"""
    stock_step(stock, births, mortality, flows)

One annual step: `S_ij + delta_ij B_j - mu_j S_ij + sum_k T[i,k,j] - sum_k T[i,j,k]`.
"""
function stock_step(stock::AbstractMatrix{<:Real}, births::AbstractVector{<:Real},
                    mortality::AbstractVector{<:Real}, flows::AbstractArray{<:Real,3})
    n = size(stock, 1)
    size(stock, 2) == n || throw(ArgumentError("stock must be square"))
    (length(births) == n && length(mortality) == n && size(flows) == (n, n, n)) ||
        throw(ArgumentError("births, mortality, flows must match the stock dimension"))
    for a in (stock, births, mortality, flows)
        _check_finite("input", a)
    end
    (any(<(0), births) || any(<(0), mortality) || any(>(1), mortality)) &&
        throw(ArgumentError("births must be non-negative and mortality in [0, 1]"))
    any(<(0), flows) && throw(ArgumentError("flows must be non-negative"))
    out = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        arrivals = 0.0
        departures = 0.0
        for k in 1:n
            arrivals += flows[i, k, j]
            departures += flows[i, j, k]
        end
        out[i, j] = stock[i, j] + (i == j ? births[j] : 0.0) - mortality[j] * stock[i, j] +
                    arrivals - departures
    end
    return out
end

"""Total origin-destination flow `F[j, k] = sum_i T[i, j, k]`."""
function od_flows(flows::AbstractArray{<:Real,3})
    _check_finite("flows", flows)
    n1, n2, n3 = size(flows)
    f = zeros(Float64, n2, n3)
    for i in 1:n1, j in 1:n2, k in 1:n3
        f[j, k] += flows[i, j, k]
    end
    return f
end

"""`mu_j = sum_k F[k, j] - sum_k F[j, k]`; the entries sum to zero."""
function net_migration(f::AbstractMatrix{<:Real})
    size(f, 1) == size(f, 2) || throw(ArgumentError("F must be square"))
    _check_finite("F", f)
    n = size(f, 1)
    return [sum(f[k, j] for k in 1:n) - sum(f[j, k] for k in 1:n) for j in 1:n]
end

"""Classical stock differencing: `max(S2 - S1, 0)/interval` with a zero diagonal."""
function stock_difference_baseline(s_prev::AbstractMatrix{<:Real},
                                   s_next::AbstractMatrix{<:Real}, interval::Real)
    interval > 0 || throw(ArgumentError("interval must be positive"))
    size(s_prev) == size(s_next) ||
        throw(ArgumentError("stock tables must have the same shape"))
    _check_finite("s_prev", s_prev)
    _check_finite("s_next", s_next)
    out = max.(s_next .- s_prev, 0.0) ./ interval
    for i in 1:min(size(out)...)
        out[i, i] = 0.0
    end
    return out
end

# --------------------------------------------------------------------------
# Iterative proportional fitting and demographic balancing
# --------------------------------------------------------------------------

"""
    ipf(table, row_targets, col_targets; tol=1e-10, max_iter=10_000)

Iterative proportional fitting (Deming and Stephan, 1940). Returns
`(fitted, iterations)`.
"""
function ipf(table::AbstractMatrix{<:Real}, row_targets::AbstractVector{<:Real},
             col_targets::AbstractVector{<:Real}; tol::Real=1e-10, max_iter::Integer=10_000)
    x = Matrix{Float64}(table)
    r = Vector{Float64}(row_targets)
    c = Vector{Float64}(col_targets)
    for a in (x, r, c)
        _check_finite("input", a)
    end
    size(x) == (length(r), length(c)) ||
        throw(ArgumentError("target lengths must match the table shape"))
    (any(<(0), x) || any(<(0), r) || any(<(0), c)) &&
        throw(ArgumentError("table and targets must be non-negative"))
    abs(sum(r) - sum(c)) > 1e-9 * max(1.0, sum(r)) &&
        throw(ArgumentError("row and column targets must have the same total"))
    for i in axes(x, 1)
        sum(x[i, :]) == 0 && r[i] > 0 &&
            throw(ArgumentError("a target is positive but the table row or column is empty"))
    end
    for j in axes(x, 2)
        sum(x[:, j]) == 0 && c[j] > 0 &&
            throw(ArgumentError("a target is positive but the table row or column is empty"))
    end
    scale = max(1.0, sum(r))
    for it in 1:max_iter
        for i in axes(x, 1)
            rs = sum(x[i, :])
            f = rs > 0 ? r[i] / rs : 0.0
            x[i, :] .*= f
        end
        for j in axes(x, 2)
            cs = sum(x[:, j])
            f = cs > 0 ? c[j] / cs : 0.0
            x[:, j] .*= f
        end
        err = 0.0
        for i in axes(x, 1)
            err = max(err, abs(sum(x[i, :]) - r[i]))
        end
        for j in axes(x, 2)
            err = max(err, abs(sum(x[:, j]) - c[j]))
        end
        err <= tol * scale && return x, it
    end
    return x, Int(max_iter)
end

"""
    demographic_balance(s1, s2, births, deaths) -> (bal1, bal2, rel1, rel2)

Add births (diagonal) to `s1` and deaths to `s2`, fit both to the average of their
marginals with [`ipf`](@ref), subtract births and deaths again. `rel` is the relative
adjustment of each cell.
"""
function demographic_balance(s1::AbstractMatrix{<:Real}, s2::AbstractMatrix{<:Real},
                             births::AbstractVector{<:Real}, deaths::AbstractMatrix{<:Real})
    n = size(s1, 1)
    (size(s1) == (n, n) && size(s2) == (n, n) && size(deaths) == (n, n) &&
     length(births) == n) || throw(ArgumentError("shape mismatch"))
    a1 = Matrix{Float64}(s1)
    for i in 1:n
        a1[i, i] += births[i]
    end
    a2 = Matrix{Float64}(s2) .+ deaths
    rows = [0.5 * (sum(a1[i, :]) + sum(a2[i, :])) for i in 1:n]
    cols = [0.5 * (sum(a1[:, j]) + sum(a2[:, j])) for j in 1:n]
    f1, _ = ipf(a1, rows, cols)
    f2, _ = ipf(a2, rows, cols)
    bal1 = copy(f1)
    for i in 1:n
        bal1[i, i] -= births[i]
    end
    bal2 = f2 .- deaths
    eps = 1e-12
    rel1 = abs.(bal1 .- s1) ./ max.(abs.(s1), eps)
    rel2 = abs.(bal2 .- s2) ./ max.(abs.(s2), eps)
    return bal1, bal2, rel1, rel2
end

# --------------------------------------------------------------------------
# Loss weights, loss, calibration
# --------------------------------------------------------------------------

"""Inverse relative uncertainty, normalised to mean 1, clamped to `[lo, hi]`."""
function uncertainty_weights(rel_unc::AbstractVector{<:Real}, lo::Real=0.5, hi::Real=2.0;
                             floor::Real=1e-6)
    _check_finite("rel_unc", rel_unc)
    any(<(0), rel_unc) && throw(ArgumentError("relative uncertainty must be non-negative"))
    (0 < lo <= hi) || throw(ArgumentError("need 0 < lo <= hi"))
    raw = 1.0 ./ max.(rel_unc, floor)
    w = raw ./ _mean(raw)
    return clamp.(w, lo, hi)
end

"""`mean_k w_k (pred_k - target_k)^2`."""
function weighted_l2(pred::AbstractArray{<:Real}, target::AbstractArray{<:Real},
                     weights::AbstractArray{<:Real})
    (size(pred) == size(target) && size(pred) == size(weights)) ||
        throw(ArgumentError("pred, target, weights must have the same shape"))
    for a in (pred, target, weights)
        _check_finite("input", a)
    end
    any(<(0), weights) && throw(ArgumentError("weights must be non-negative"))
    return _mean(weights .* (pred .- target) .^ 2)
end

"""`s(t0) = 1`, `s(t) = prod (1 - mu(tau))`; the result has `length(mortality) + 1` entries."""
function survival_fraction(mortality::AbstractVector{<:Real})
    _check_finite("mortality", mortality)
    (any(<(0), mortality) || any(>(1), mortality)) &&
        throw(ArgumentError("mortality must lie in [0, 1]"))
    out = Vector{Float64}(undef, length(mortality) + 1)
    out[1] = 1.0
    for t in eachindex(mortality)
        out[t + 1] = out[t] * (1.0 - mortality[t])
    end
    return out
end

"""
    calibrate_offset(observed, predicted, weights, survival)

Closed-form initial-stock offset `sum s w (S - S_hat) / sum w s^2`, raised to the
smallest value keeping every shifted stock non-negative.
"""
function calibrate_offset(observed::AbstractVector{<:Real}, predicted::AbstractVector{<:Real},
                          weights::AbstractVector{<:Real}, survival::AbstractVector{<:Real})
    length(observed) == length(predicted) == length(weights) == length(survival) ||
        throw(ArgumentError("all series must have the same length"))
    for a in (observed, predicted, weights, survival)
        _check_finite("input", a)
    end
    (any(<(0), weights) || any(<=(0), survival)) &&
        throw(ArgumentError("weights must be non-negative and survival positive"))
    den = sum(weights .* survival .^ 2)
    den > 0 || throw(ArgumentError("weights have no mass"))
    b = sum(survival .* weights .* (observed .- predicted)) / den
    lower = maximum(-predicted ./ survival)
    return max(b, lower)
end

# --------------------------------------------------------------------------
# Gravity rate and metrics
# --------------------------------------------------------------------------

"""`b0 + bg (log gdp_d - log gdp_o) + bd log dist + bl lang + bp log pop_d`."""
function gravity_log_rate(beta::AbstractVector{<:Real}, log_gdp_o::Real, log_gdp_d::Real,
                          log_dist::Real, lang::Real, log_pop_d::Real)
    length(beta) == 5 || throw(ArgumentError("beta must have five entries"))
    return beta[1] + beta[2] * (log_gdp_d - log_gdp_o) + beta[3] * log_dist +
           beta[4] * lang + beta[5] * log_pop_d
end

"""Root mean squared error on the natural-log scale."""
function log_rmse(truth::AbstractArray{<:Real}, estimate::AbstractArray{<:Real};
                  floor::Real=1e-9)
    size(truth) == size(estimate) || throw(ArgumentError("shape mismatch"))
    d = log.(max.(estimate, floor)) .- log.(max.(truth, floor))
    return sqrt(_mean(d .* d))
end

"""Pearson correlation (`NaN` when either sample has zero variance)."""
function pearson(x::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    (length(x) == length(y) && length(x) >= 2) ||
        throw(ArgumentError("need two samples of equal length >= 2"))
    xc = vec(x) .- _mean(x)
    yc = vec(y) .- _mean(y)
    den = sqrt(sum(xc .^ 2) * sum(yc .^ 2))
    return den > 0 ? sum(xc .* yc) / den : NaN
end

"""Share of truths inside `mean +/- z std`."""
function coverage(truth::AbstractArray{<:Real}, m::AbstractArray{<:Real},
                  s::AbstractArray{<:Real}, z::Real=1.96)
    (size(truth) == size(m) && size(truth) == size(s)) ||
        throw(ArgumentError("shape mismatch"))
    return _mean(abs.(truth .- m) .<= z .* s)
end

"""Mean and sample standard deviation (n - 1) down the first axis of a matrix."""
function ensemble_mean_std(samples::AbstractMatrix{<:Real})
    d = size(samples, 1)
    d >= 2 || throw(ArgumentError("need at least two samples"))
    _check_finite("samples", samples)
    m = [_mean(samples[:, c]) for c in axes(samples, 2)]
    s = [sqrt(sum((samples[:, c] .- m[c]) .^ 2) / (d - 1)) for c in axes(samples, 2)]
    return m, s
end

end # module ChMigrationFlows
