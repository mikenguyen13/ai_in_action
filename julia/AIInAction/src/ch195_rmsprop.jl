"""
    Ch195Rmsprop

RMSProp adaptive-learning-rate optimizer from scratch (Julia).

Mirrors the Python module `aiinaction.ch195_rmsprop` and the Rust module
`aiinaction::ch195_rmsprop`. The shared fixtures in `test/test_ch195_rmsprop.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

Plain (uncentered, no-momentum) RMSProp maintains, per coordinate `i`, an
exponential moving average of the squared gradient

    v_t,i = beta * v_{t-1,i} + (1 - beta) * g_t,i^2,   v_0 = 0,

and updates each coordinate with its own effective step size

    theta_{t+1,i} = theta_t,i - eta * g_t,i / (sqrt(v_t,i) + eps).

The epsilon is placed outside the square root, matching the chapter.
"""
module Ch195Rmsprop

export RMSPropState, init_state, rmsprop_step, minimize

"""State of an RMSProp optimizer. `params` and `v` are length-`d` vectors."""
struct RMSPropState
    params::Vector{Float64}
    v::Vector{Float64}
    lr::Float64
    beta::Float64
    eps::Float64
    step_count::Int
end

n_params(s::RMSPropState) = length(s.params)

function _as_vector(x, name)
    v = Vector{Float64}(x)
    length(v) >= 1 || throw(ArgumentError("$name must be non-empty"))
    all(isfinite, v) || throw(ArgumentError("$name contains non-finite values (nan or inf)"))
    return v
end

function _validate_hparams(lr, beta, eps)
    lr > 0 || throw(ArgumentError("lr must be positive, got $lr"))
    (0 <= beta < 1) || throw(ArgumentError("beta must be in [0, 1), got $beta"))
    eps > 0 || throw(ArgumentError("eps must be positive, got $eps"))
    return nothing
end

"""
    init_state(params; lr=1e-2, beta=0.9, eps=1e-8)

Initialize an RMSProp state from a starting parameter vector. `v` is set to zeros
and `step_count` to 0.
"""
function init_state(params; lr::Real=1e-2, beta::Real=0.9, eps::Real=1e-8)
    p = _as_vector(params, "params")
    _validate_hparams(lr, beta, eps)
    return RMSPropState(p, zeros(Float64, length(p)), Float64(lr), Float64(beta),
        Float64(eps), 0)
end

"""
    rmsprop_step(state, grad)

Apply one RMSProp update and return the new state. Coordinate-wise:
`v <- beta*v + (1-beta)*g^2` then `params <- params - lr*g/(sqrt(v)+eps)`.
"""
function rmsprop_step(state::RMSPropState, grad)
    g = _as_vector(grad, "grad")
    length(g) == n_params(state) ||
        throw(ArgumentError("grad has $(length(g)) entries but state has $(n_params(state)) parameters"))
    v_new = state.beta .* state.v .+ (1 - state.beta) .* g .^ 2
    params_new = state.params .- state.lr .* g ./ (sqrt.(v_new) .+ state.eps)
    return RMSPropState(params_new, v_new, state.lr, state.beta, state.eps,
        state.step_count + 1)
end

"""
    minimize(grad_fn, params0, n_steps; lr=1e-2, beta=0.9, eps=1e-8)

Run RMSProp for `n_steps` iterations on the objective whose gradient is `grad_fn`.
`grad_fn` maps the current parameter vector to the gradient at that point.
"""
function minimize(grad_fn, params0, n_steps::Integer; lr::Real=1e-2,
                  beta::Real=0.9, eps::Real=1e-8)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative, got $n_steps"))
    state = init_state(params0; lr=lr, beta=beta, eps=eps)
    for _ in 1:n_steps
        g = Vector{Float64}(grad_fn(state.params))
        state = rmsprop_step(state, g)
    end
    return state
end

end # module Ch195Rmsprop
