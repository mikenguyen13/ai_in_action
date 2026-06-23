"""
    Ch130GmmEm

Expectation-Maximization for Gaussian mixture models (Julia).

Mirrors the Python module `aiinaction.ch130_gmm_em` and the Rust module
`aiinaction::ch130_gmm_em`. The shared fixtures in `test/test_ch130_gmm_em.jl`
match the Python/Rust suites to keep the three libraries at parity.

Data is an `N` by `d` matrix (rows are points). Means are `K` by `d`. Covariances
are a length-`K` vector of `d` by `d` matrices.
"""
module Ch130GmmEm

using LinearAlgebra

export GMMParams, GMMResult, gaussian_pdf, e_step, m_step, log_likelihood, fit_gmm

"""Parameters of a K-component, d-dimensional Gaussian mixture."""
struct GMMParams
    weights::Vector{Float64}
    means::Matrix{Float64}                 # K by d
    covariances::Vector{Matrix{Float64}}   # length K, each d by d
end

"""Outcome of [`fit_gmm`](@ref)."""
struct GMMResult
    params::GMMParams
    responsibilities::Matrix{Float64}
    log_likelihood::Float64
    n_iter::Int
    converged::Bool
    history::Vector{Float64}
end

function _validate_params(params::GMMParams, d::Int)
    k = length(params.weights)
    any(params.weights .< 0) && throw(ArgumentError("weights must be nonnegative"))
    total = sum(params.weights)
    isapprox(total, 1.0; atol=1e-8) ||
        throw(ArgumentError("weights must sum to 1, got $total"))
    size(params.means) == (k, d) ||
        throw(ArgumentError("means must have shape ($k, $d)"))
    length(params.covariances) == k && all(size(c) == (d, d) for c in params.covariances) ||
        throw(ArgumentError("covariances must have shape ($k, $d, $d)"))
    return nothing
end

"""Density of a multivariate normal at the point `x` (a length-`d` vector)."""
function gaussian_pdf(x::AbstractVector{<:Real}, mean::AbstractVector{<:Real},
                      cov::AbstractMatrix{<:Real})
    d = length(x)
    length(mean) == d || throw(ArgumentError("mean length $(length(mean)) != x length $d"))
    size(cov) == (d, d) || throw(ArgumentError("cov must have shape ($d, $d)"))
    det_ = det(cov)
    det_ > 0 || throw(ArgumentError("covariance must be positive definite, det=$det_"))
    diff = collect(Float64, x) .- collect(Float64, mean)
    quad = dot(diff, cov \ diff)
    norm = sqrt(((2.0 * pi)^d) * det_)
    return exp(-0.5 * quad) / norm
end

"""Compute responsibilities `gamma[n, k]` given current parameters."""
function e_step(x::AbstractMatrix{<:Real}, params::GMMParams)
    n, d = size(x)
    n > 0 || throw(ArgumentError("x must be non-empty"))
    _validate_params(params, d)
    k = length(params.weights)
    gamma = zeros(Float64, n, k)
    for ni in 1:n
        xi = @view x[ni, :]
        row_sum = 0.0
        for ki in 1:k
            p = params.weights[ki] *
                gaussian_pdf(xi, @view(params.means[ki, :]), params.covariances[ki])
            gamma[ni, ki] = p
            row_sum += p
        end
        row_sum > 0 ||
            throw(ArgumentError("data point $ni has zero density under all components"))
        for ki in 1:k
            gamma[ni, ki] /= row_sum
        end
    end
    return gamma
end

"""Weighted re-estimation of mixture parameters from responsibilities."""
function m_step(x::AbstractMatrix{<:Real}, responsibilities::AbstractMatrix{<:Real};
                reg_covar::Real=1e-6)
    n, d = size(x)
    n > 0 || throw(ArgumentError("x must be non-empty"))
    reg_covar >= 0 || throw(ArgumentError("reg_covar must be nonnegative, got $reg_covar"))
    size(responsibilities, 1) == n ||
        throw(ArgumentError("responsibilities has $(size(responsibilities,1)) rows but x has $n points"))
    k = size(responsibilities, 2)
    nk = vec(sum(responsibilities; dims=1))
    any(nk .<= 0) &&
        throw(ArgumentError("a component has zero effective count; cannot re-estimate"))
    weights = nk ./ n
    means = zeros(Float64, k, d)
    covs = [zeros(Float64, d, d) for _ in 1:k]
    for ki in 1:k
        for ni in 1:n
            @views means[ki, :] .+= responsibilities[ni, ki] .* x[ni, :]
        end
        means[ki, :] ./= nk[ki]
        acc = zeros(Float64, d, d)
        for ni in 1:n
            diff = collect(Float64, @view(x[ni, :])) .- @view(means[ki, :])
            acc .+= responsibilities[ni, ki] .* (diff * diff')
        end
        covs[ki] = acc ./ nk[ki] + reg_covar * Matrix{Float64}(I, d, d)
    end
    return GMMParams(weights, means, covs)
end

"""Incomplete-data log-likelihood of the data under the mixture."""
function log_likelihood(x::AbstractMatrix{<:Real}, params::GMMParams)
    n, d = size(x)
    n > 0 || throw(ArgumentError("x must be non-empty"))
    _validate_params(params, d)
    k = length(params.weights)
    total = 0.0
    for ni in 1:n
        xi = @view x[ni, :]
        mix = 0.0
        for ki in 1:k
            mix += params.weights[ki] *
                   gaussian_pdf(xi, @view(params.means[ki, :]), params.covariances[ki])
        end
        mix > 0 || throw(ArgumentError("data point $ni has zero mixture density"))
        total += log(mix)
    end
    return total
end

"""Run EM to convergence from explicit initial parameters."""
function fit_gmm(x::AbstractMatrix{<:Real}, init::GMMParams;
                 max_iter::Int=100, tol::Real=1e-6, reg_covar::Real=1e-6)
    n, d = size(x)
    n > 0 || throw(ArgumentError("x must be non-empty"))
    max_iter > 0 || throw(ArgumentError("max_iter must be positive, got $max_iter"))
    tol >= 0 || throw(ArgumentError("tol must be nonnegative, got $tol"))
    _validate_params(init, d)

    params = init
    gamma = e_step(x, params)
    prev_ll = log_likelihood(x, params)
    history = Float64[prev_ll]
    converged = false
    n_iter = 0
    for _ in 1:max_iter
        params = m_step(x, gamma; reg_covar=reg_covar)
        gamma = e_step(x, params)
        ll = log_likelihood(x, params)
        push!(history, ll)
        n_iter += 1
        if abs(ll - prev_ll) < tol
            converged = true
            prev_ll = ll
            break
        end
        prev_ll = ll
    end
    return GMMResult(params, gamma, prev_ll, n_iter, converged, history)
end

end # module Ch130GmmEm
