"""
    Ch194Adagrad

AdaGrad adaptive learning rates from scratch (Julia).

Mirrors the Python module `aiinaction.ch194_adagrad` and the Rust module
`aiinaction::ch194_adagrad`. The shared fixtures in `test/test_ch194_adagrad.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

AdaGrad gives every coordinate its own learning rate by accumulating the sum of
squared gradients `G[i] = sum_tau g[tau, i]^2` and stepping
`theta[i] -= eta / (sqrt(G[i]) + eps) * g[i]`. Base-only, no external dependency.
"""
module Ch194Adagrad

export AdaGradState, AdaGradResult, init_state, adagrad_step,
    effective_learning_rate, minimize, quadratic_value, quadratic_grad

"""Mutable optimizer state: parameters, gradient-square accumulator, and hyperparameters."""
mutable struct AdaGradState
    theta::Vector{Float64}
    accumulator::Vector{Float64}
    learning_rate::Float64
    epsilon::Float64
end

"""The outcome of a [`minimize`] run."""
struct AdaGradResult
    theta::Vector{Float64}
    accumulator::Vector{Float64}
    n_steps::Int
    grad_norm::Float64
    converged::Bool
end

function _check_vector(v, name)
    isempty(v) && throw(ArgumentError("$name must be non-empty"))
    all(isfinite, v) || throw(ArgumentError("$name contains non-finite values (nan or inf)"))
    return nothing
end

function _check_hyperparams(learning_rate, epsilon)
    (isfinite(learning_rate) && learning_rate > 0) ||
        throw(ArgumentError("learning_rate must be a positive finite number, got $learning_rate"))
    (isfinite(epsilon) && epsilon > 0) ||
        throw(ArgumentError("epsilon must be a positive finite number, got $epsilon"))
    return nothing
end

"""
    init_state(theta0; learning_rate=0.1, epsilon=1e-8)

Create a fresh state with a zero accumulator.
"""
function init_state(theta0; learning_rate::Real=0.1, epsilon::Real=1e-8)
    th = Vector{Float64}(theta0)
    _check_vector(th, "theta0")
    _check_hyperparams(learning_rate, epsilon)
    return AdaGradState(copy(th), zeros(Float64, length(th)),
        Float64(learning_rate), Float64(epsilon))
end

"""Per-coordinate effective learning rate `eta / (sqrt(G) + eps)`."""
function effective_learning_rate(state::AdaGradState)
    return state.learning_rate ./ (sqrt.(state.accumulator) .+ state.epsilon)
end

"""Apply one in-place AdaGrad update for the gradient `grad`; mutates and returns `state`."""
function adagrad_step(state::AdaGradState, grad)
    g = Vector{Float64}(grad)
    _check_vector(g, "grad")
    length(g) == length(state.theta) ||
        throw(ArgumentError("grad has length $(length(g)) but theta has length $(length(state.theta))"))
    @inbounds for i in eachindex(state.theta)
        state.accumulator[i] += g[i] * g[i]
        step = state.learning_rate / (sqrt(state.accumulator[i]) + state.epsilon)
        state.theta[i] -= step * g[i]
    end
    return state
end

"""
    minimize(grad_fn, theta0; learning_rate=0.1, epsilon=1e-8, max_iter=1000, tol=1e-8)

Minimize an objective by AdaGrad using only its gradient function `grad_fn`, which
maps a parameter vector to its gradient. Stops when the gradient norm drops to
`tol` or `max_iter` steps have been taken.
"""
function minimize(grad_fn, theta0; learning_rate::Real=0.1, epsilon::Real=1e-8,
        max_iter::Integer=1000, tol::Real=1e-8)
    max_iter >= 1 || throw(ArgumentError("max_iter must be a positive integer, got $max_iter"))
    (isfinite(tol) && tol >= 0) ||
        throw(ArgumentError("tol must be a nonnegative finite number, got $tol"))

    state = init_state(theta0; learning_rate=learning_rate, epsilon=epsilon)
    d = length(state.theta)

    grad_norm = Inf
    steps = 0
    converged = false
    for _ in 1:max_iter
        g = Vector{Float64}(grad_fn(state.theta))
        length(g) == d ||
            throw(ArgumentError("grad_fn returned length $(length(g)) but expected $d"))
        all(isfinite, g) || throw(ArgumentError("grad_fn returned non-finite values (nan or inf)"))
        grad_norm = sqrt(sum(abs2, g))
        if grad_norm <= tol
            converged = true
            break
        end
        adagrad_step(state, g)
        steps += 1
    end

    return AdaGradResult(state.theta, state.accumulator, steps, grad_norm, converged)
end

function _check_quad(theta, a, b)
    (length(theta) == length(a) == length(b)) ||
        throw(ArgumentError("theta, a, b must share length; got $(length(theta)), $(length(a)), $(length(b))"))
    all(x -> x > 0, a) || throw(ArgumentError("curvatures a must be strictly positive"))
    return nothing
end

"""Gradient of the separable quadratic `f = 0.5 * sum a_i (theta_i - b_i)^2`."""
function quadratic_grad(theta, a, b)
    th = Vector{Float64}(theta)
    av = Vector{Float64}(a)
    bv = Vector{Float64}(b)
    _check_quad(th, av, bv)
    return av .* (th .- bv)
end

"""Value of `f = 0.5 * sum a_i (theta_i - b_i)^2` (the test objective)."""
function quadratic_value(theta, a, b)
    th = Vector{Float64}(theta)
    av = Vector{Float64}(a)
    bv = Vector{Float64}(b)
    _check_quad(th, av, bv)
    return 0.5 * sum(av .* (th .- bv) .^ 2)
end

end # module Ch194Adagrad
