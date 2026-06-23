"""
    Ch202LayerNorm

Layer Normalization and RMSNorm from scratch (Julia).

Mirrors the Python module `aiinaction.ch202_layer_norm` and the Rust module
`aiinaction::ch202_layer_norm`. The shared fixtures in `test/test_ch202_layer_norm.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

Both operators act on the feature axis of a single example (Base-only):

  LayerNorm:  y = gamma * (x - mean) / sqrt(var + eps) + beta   (var is ddof=0)
  RMSNorm:    y = gamma * x / sqrt(mean(x^2) + eps)             (no mean, no beta)

The `eps` is added inside the square root so the denominator stays bounded away
from zero even when the activation vector is itself near zero.
"""
module Ch202LayerNorm

export layer_norm, rms_norm, apply_layer_norm, apply_rms_norm

function _check_inputs(x::AbstractVector{Float64}, eps::Float64, name::AbstractString)
    isempty(x) && throw(ArgumentError("$name must have at least one feature"))
    all(isfinite, x) || throw(ArgumentError("$name contains non-finite values (nan or inf)"))
    eps >= 0 || throw(ArgumentError("eps must be non-negative, got $eps"))
    return nothing
end

function _resolve_gamma(d::Int, gamma)
    gamma === nothing && return ones(Float64, d)
    g = Vector{Float64}(gamma)
    length(g) == d || throw(ArgumentError("gamma must have length $d, got $(length(g))"))
    all(isfinite, g) || throw(ArgumentError("gamma contains non-finite values (nan or inf)"))
    return g
end

function _resolve_beta(d::Int, beta)
    beta === nothing && return zeros(Float64, d)
    b = Vector{Float64}(beta)
    length(b) == d || throw(ArgumentError("beta must have length $d, got $(length(b))"))
    all(isfinite, b) || throw(ArgumentError("beta contains non-finite values (nan or inf)"))
    return b
end

"""
    layer_norm(x; gamma=nothing, beta=nothing, eps=1e-5)

Layer-normalize a single feature vector. Subtracts the feature mean, divides by
the standard deviation (population variance, ddof=0), then applies the affine map
`gamma * x_hat + beta`. Defaults: `gamma` all ones, `beta` all zeros.
"""
function layer_norm(x; gamma=nothing, beta=nothing, eps::Real=1e-5)
    xv = Vector{Float64}(x)
    epsf = Float64(eps)
    _check_inputs(xv, epsf, "x")
    d = length(xv)
    g = _resolve_gamma(d, gamma)
    b = _resolve_beta(d, beta)

    n = Float64(d)
    mu = sum(xv) / n
    var = sum((v - mu)^2 for v in xv) / n
    denom = sqrt(var + epsf)
    return [g[i] * ((xv[i] - mu) / denom) + b[i] for i in 1:d]
end

"""
    rms_norm(x; gamma=nothing, eps=1e-5)

RMS-normalize a single feature vector. Divides by the root mean square of the
features (no mean subtraction), then applies the gain `gamma`. There is no bias.
Default `gamma` is all ones.
"""
function rms_norm(x; gamma=nothing, eps::Real=1e-5)
    xv = Vector{Float64}(x)
    epsf = Float64(eps)
    _check_inputs(xv, epsf, "x")
    d = length(xv)
    g = _resolve_gamma(d, gamma)

    n = Float64(d)
    ms = sum(v^2 for v in xv) / n
    rms = sqrt(ms + epsf)
    return [(xv[i] / rms) * g[i] for i in 1:d]
end

"""
    apply_layer_norm(X; gamma=nothing, beta=nothing, eps=1e-5)

Layer-normalize every row of an `n x d` matrix independently. Returns `n x d`.
"""
function apply_layer_norm(X; gamma=nothing, beta=nothing, eps::Real=1e-5)
    A = Matrix{Float64}(X)
    size(A, 1) >= 1 || throw(ArgumentError("X must have at least one row"))
    out = Matrix{Float64}(undef, size(A, 1), size(A, 2))
    for i in 1:size(A, 1)
        out[i, :] = layer_norm(A[i, :]; gamma=gamma, beta=beta, eps=eps)
    end
    return out
end

"""
    apply_rms_norm(X; gamma=nothing, eps=1e-5)

RMS-normalize every row of an `n x d` matrix independently. Returns `n x d`.
"""
function apply_rms_norm(X; gamma=nothing, eps::Real=1e-5)
    A = Matrix{Float64}(X)
    size(A, 1) >= 1 || throw(ArgumentError("X must have at least one row"))
    out = Matrix{Float64}(undef, size(A, 1), size(A, 2))
    for i in 1:size(A, 1)
        out[i, :] = rms_norm(A[i, :]; gamma=gamma, eps=eps)
    end
    return out
end

end # module Ch202LayerNorm
