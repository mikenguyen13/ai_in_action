"""
    Ch193Nesterov

Nesterov Accelerated Gradient (NAG) from scratch (Julia).

Mirrors the Python module `aiinaction.ch193_nesterov` and the Rust module
`aiinaction::ch193_nesterov`. The shared fixtures in `test/test_ch193_nesterov.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

The gradient is supplied as a callable mapping a point (`Vector{Float64}`) to its
gradient (a vector of the same length). Two schedules are provided:

* `nesterov_convex` -- the two-sequence (FISTA-style) form for smooth convex
  objectives, with `t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2`.
* `nesterov_momentum` -- the constant-momentum velocity form for strongly convex
  objectives, with lookahead `y_k = x_k + beta * v_k`.
"""
module Ch193Nesterov

export OptimizeResult, nesterov_convex, nesterov_momentum

"""The outcome of a Nesterov optimization run. `history` holds the gradient norm
recorded after each iteration."""
struct OptimizeResult
    x::Vector{Float64}
    n_iter::Int
    grad_norm::Float64
    converged::Bool
    history::Vector{Float64}
end

function _prepare(x0)
    x = Vector{Float64}(x0)
    isempty(x) && throw(ArgumentError("x0 must have at least one entry"))
    all(isfinite, x) || throw(ArgumentError("x0 contains non-finite values (nan or inf)"))
    return x
end

function _validate(step_size::Real, max_iter::Integer, tol::Real)
    (step_size > 0 && isfinite(step_size)) ||
        throw(ArgumentError("step_size must be a positive finite number, got $step_size"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be a positive integer, got $max_iter"))
    (tol >= 0 && isfinite(tol)) ||
        throw(ArgumentError("tol must be a non-negative finite number, got $tol"))
    return nothing
end

function _eval_grad(grad, y::Vector{Float64})
    g = Vector{Float64}(grad(y))
    length(g) == length(y) ||
        throw(ArgumentError("gradient has length $(length(g)) but the point has length $(length(y)); the gradient oracle must return a vector matching x0"))
    all(isfinite, g) || throw(ArgumentError("gradient returned non-finite values (nan or inf)"))
    return g
end

_norm(v::Vector{Float64}) = sqrt(sum(abs2, v))

"""
    nesterov_convex(grad, x0, step_size; max_iter=1000, tol=1e-8)

Minimize a smooth convex function with the two-sequence Nesterov schedule.
`step_size` is the constant `eta` (`1/L` is canonical).
"""
function nesterov_convex(grad, x0, step_size::Real; max_iter::Integer=1000, tol::Real=1e-8)
    x = _prepare(x0)
    _validate(step_size, max_iter, tol)
    eta = Float64(step_size)
    y = copy(x)
    t = 1.0
    history = Float64[]
    converged = false
    n_iter = 0

    for k in 1:max_iter
        n_iter = k
        g = _eval_grad(grad, y)
        x_next = y .- eta .* g
        t_next = (1.0 + sqrt(1.0 + 4.0 * t * t)) / 2.0
        gamma = (t - 1.0) / t_next
        y = x_next .+ gamma .* (x_next .- x)
        x = x_next
        t = t_next

        gn = _norm(_eval_grad(grad, x))
        push!(history, gn)
        if gn <= tol
            converged = true
            break
        end
    end

    grad_norm = isempty(history) ? _norm(_eval_grad(grad, x)) : last(history)
    return OptimizeResult(x, n_iter, grad_norm, converged, history)
end

"""
    nesterov_momentum(grad, x0, step_size, momentum; max_iter=1000, tol=1e-8)

Minimize with the constant-momentum (velocity) Nesterov form. The lookahead point
is `y_k = x_k + beta * v_k`. `momentum` must lie in `[0, 1)`.
"""
function nesterov_momentum(grad, x0, step_size::Real, momentum::Real;
                           max_iter::Integer=1000, tol::Real=1e-8)
    x = _prepare(x0)
    _validate(step_size, max_iter, tol)
    (0.0 <= momentum < 1.0) ||
        throw(ArgumentError("momentum must lie in [0, 1), got $momentum"))
    eta = Float64(step_size)
    beta = Float64(momentum)
    v = zeros(Float64, length(x))
    history = Float64[]
    converged = false
    n_iter = 0

    for k in 1:max_iter
        n_iter = k
        y = x .+ beta .* v
        g = _eval_grad(grad, y)
        v = beta .* v .- eta .* g
        x = x .+ v

        gn = _norm(_eval_grad(grad, x))
        push!(history, gn)
        if gn <= tol
            converged = true
            break
        end
    end

    grad_norm = isempty(history) ? _norm(_eval_grad(grad, x)) : last(history)
    return OptimizeResult(x, n_iter, grad_norm, converged, history)
end

end # module Ch193Nesterov
