"""
    Ch197Adamw

AdamW: Adam with decoupled weight decay, from scratch (Julia).

Mirrors the Python module `aiinaction.ch197_adamw` and the Rust module
`aiinaction::ch197_adamw`. The shared fixtures in `test/test_ch197_adamw.jl` match
the Python/Rust suites, which keeps the three implementations at parity.

The update rule for a parameter vector `theta` at step `t` with gradient `g`:

1. `m = beta1 * m + (1 - beta1) * g`
2. `v = beta2 * v + (1 - beta2) * g .* g`
3. `mhat = m / (1 - beta1^t)`, `vhat = v / (1 - beta2^t)`
4. `theta = theta - lr * (mhat ./ (sqrt.(vhat) .+ eps) + wd * theta)`

The weight-decay term `wd * theta` is applied directly to the parameters, outside
the adaptive preconditioner, which is the defining feature of AdamW.
"""
module Ch197Adamw

export AdamWConfig, AdamWState, init_state, adamw_step!, minimize

"""Hyperparameters for the AdamW optimizer."""
struct AdamWConfig
    lr::Float64
    beta1::Float64
    beta2::Float64
    eps::Float64
    weight_decay::Float64

    function AdamWConfig(; lr::Real=1e-3, beta1::Real=0.9, beta2::Real=0.999,
                         eps::Real=1e-8, weight_decay::Real=0.0)
        lr > 0 || throw(ArgumentError("lr must be positive, got $lr"))
        (0 <= beta1 < 1) || throw(ArgumentError("beta1 must be in [0, 1), got $beta1"))
        (0 <= beta2 < 1) || throw(ArgumentError("beta2 must be in [0, 1), got $beta2"))
        eps > 0 || throw(ArgumentError("eps must be positive, got $eps"))
        weight_decay >= 0 ||
            throw(ArgumentError("weight_decay must be non-negative, got $weight_decay"))
        return new(Float64(lr), Float64(beta1), Float64(beta2), Float64(eps),
                   Float64(weight_decay))
    end
end

"""Mutable optimizer state for one parameter vector."""
mutable struct AdamWState
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end

"""Create a zero-initialized optimizer state for `n_params` parameters."""
function init_state(n_params::Integer)
    n_params >= 1 || throw(ArgumentError("n_params must be >= 1, got $n_params"))
    return AdamWState(zeros(Float64, n_params), zeros(Float64, n_params), 0)
end

function _check_vector(x, name, expected_len)
    isempty(x) && throw(ArgumentError("$name must be non-empty"))
    all(isfinite, x) ||
        throw(ArgumentError("$name contains non-finite values (nan or inf)"))
    if expected_len !== nothing && length(x) != expected_len
        throw(ArgumentError("$name has length $(length(x)) but expected $expected_len"))
    end
    return nothing
end

"""
    adamw_step!(params, grad, state, config) -> Vector{Float64}

Apply one AdamW update, advancing `state` in place, and return the new
parameters. `params` itself is not modified.
"""
function adamw_step!(params, grad, state::AdamWState, config::AdamWConfig)
    theta = Vector{Float64}(params)
    d = length(theta)
    _check_vector(theta, "params", nothing)
    g = Vector{Float64}(grad)
    _check_vector(g, "grad", d)
    (length(state.m) == d && length(state.v) == d) ||
        throw(ArgumentError("state has length m=$(length(state.m)), v=$(length(state.v)) but params has length $d"))

    b1 = config.beta1
    b2 = config.beta2
    state.t += 1
    t = state.t

    @inbounds for i in 1:d
        state.m[i] = b1 * state.m[i] + (1 - b1) * g[i]
        state.v[i] = b2 * state.v[i] + (1 - b2) * (g[i] * g[i])
    end

    bc1 = 1 - b1^t
    bc2 = 1 - b2^t

    out = Vector{Float64}(undef, d)
    @inbounds for i in 1:d
        mhat = state.m[i] / bc1
        vhat = state.v[i] / bc2
        adaptive = mhat / (sqrt(vhat) + config.eps)
        out[i] = theta[i] - config.lr * (adaptive + config.weight_decay * theta[i])
    end
    return out
end

"""
    minimize(grad_fn, x0, config, n_steps) -> Vector{Float64}

Run AdamW for `n_steps` iterations from `x0`. `grad_fn` maps a parameter vector
to its gradient of the same length.
"""
function minimize(grad_fn, x0, config::AdamWConfig, n_steps::Integer)
    n_steps >= 1 || throw(ArgumentError("n_steps must be >= 1, got $n_steps"))
    x = Vector{Float64}(x0)
    _check_vector(x, "x0", nothing)
    state = init_state(length(x))
    for _ in 1:n_steps
        g = grad_fn(x)
        _check_vector(g, "grad_fn(x)", length(x))
        x = adamw_step!(x, g, state, config)
    end
    return x
end

end # module Ch197Adamw
