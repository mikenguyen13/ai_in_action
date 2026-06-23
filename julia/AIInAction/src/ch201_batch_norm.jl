"""
    Ch201BatchNorm

Batch Normalization forward and backward pass from scratch (Julia).

Mirrors the Python module `aiinaction.ch201_batch_norm` and the Rust module
`aiinaction::ch201_batch_norm`. The shared fixtures in `test/test_ch201_batch_norm.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

For a mini-batch `X` of shape `(m, d)`, the forward transform standardizes each
feature over the batch (population variance, `ddof = 0`) and applies a learnable
scale `gamma` and shift `beta`:

    mu_j    = mean_i X[i, j]
    var_j   = mean_i (X[i, j] - mu_j)^2
    xhat    = (X .- mu') ./ sqrt.(var .+ eps)'
    Y       = gamma' .* xhat .+ beta'
"""
module Ch201BatchNorm

export BatchNormCache, batch_norm_forward, batch_norm_backward, batch_norm_inference

"""Intermediate quantities saved by the forward pass for the backward pass."""
struct BatchNormCache
    x_hat::Matrix{Float64}
    inv_std::Vector{Float64}
    gamma::Vector{Float64}
    mean::Vector{Float64}
    var::Vector{Float64}
end

function _as_matrix(X, name="X")
    A = Matrix{Float64}(X)
    n, d = size(A)
    n >= 1 || throw(ArgumentError("$name must have at least one row"))
    d >= 1 || throw(ArgumentError("$name must have at least one feature"))
    all(isfinite, A) || throw(ArgumentError("$name contains non-finite values (nan or inf)"))
    return A
end

function _as_vector(v, d, name)
    a = Vector{Float64}(v)
    length(a) == d || throw(ArgumentError("$name has length $(length(a)) but X has $d features"))
    all(isfinite, a) || throw(ArgumentError("$name contains non-finite values (nan or inf)"))
    return a
end

"""
    batch_norm_forward(X, gamma, beta; eps=1e-5)

Training-time Batch Normalization forward pass. Returns `(Y, cache)` where `Y` is
the `m x d` output and `cache::BatchNormCache` holds the backward-pass state.
"""
function batch_norm_forward(X, gamma, beta; eps::Float64=1e-5)
    eps > 0 || throw(ArgumentError("eps must be positive, got $eps"))
    A = _as_matrix(X)
    n, d = size(A)
    g = _as_vector(gamma, d, "gamma")
    b = _as_vector(beta, d, "beta")

    mu = vec(sum(A; dims=1)) ./ n
    Xc = A .- mu'
    var = vec(sum(abs2, Xc; dims=1)) ./ n  # population variance (ddof=0)
    inv_std = 1.0 ./ sqrt.(var .+ eps)
    x_hat = Xc .* inv_std'
    Y = x_hat .* g' .+ b'

    return Y, BatchNormCache(x_hat, inv_std, g, mu, var)
end

"""
    batch_norm_backward(dY, cache)

Backward pass. Returns `(dX, dgamma, dbeta)` with `dX` of shape `(m, d)` and the
parameter gradients of length `d`.
"""
function batch_norm_backward(dY, cache::BatchNormCache)
    D = Matrix{Float64}(dY)
    size(D) == size(cache.x_hat) ||
        throw(ArgumentError("dY has shape $(size(D)) but cache was built for $(size(cache.x_hat))"))
    all(isfinite, D) || throw(ArgumentError("dY contains non-finite values (nan or inf)"))
    n = size(D, 1)

    dgamma = vec(sum(D .* cache.x_hat; dims=1))
    dbeta = vec(sum(D; dims=1))

    g = D .* cache.gamma'                       # dL/dxhat
    g_mean = vec(sum(g; dims=1)) ./ n
    gxhat_mean = vec(sum(g .* cache.x_hat; dims=1)) ./ n
    dX = cache.inv_std' .* (g .- g_mean' .- cache.x_hat .* gxhat_mean')

    return dX, dgamma, dbeta
end

"""
    batch_norm_inference(X, gamma, beta, running_mean, running_var; eps=1e-5)

Inference-time Batch Normalization using frozen population statistics. Applies
`y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta` per feature.
"""
function batch_norm_inference(X, gamma, beta, running_mean, running_var; eps::Float64=1e-5)
    eps > 0 || throw(ArgumentError("eps must be positive, got $eps"))
    A = _as_matrix(X)
    n, d = size(A)
    g = _as_vector(gamma, d, "gamma")
    b = _as_vector(beta, d, "beta")
    rm = _as_vector(running_mean, d, "running_mean")
    rv = _as_vector(running_var, d, "running_var")
    any(rv .< 0) && throw(ArgumentError("running_var must be non-negative"))

    return g' .* (A .- rm') ./ sqrt.(rv .+ eps)' .+ b'
end

end # module Ch201BatchNorm
