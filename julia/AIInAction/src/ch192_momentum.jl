"""
    Ch192Momentum

Momentum (heavy-ball) gradient descent from scratch (Julia).

Mirrors the Python module `aiinaction.ch192_momentum` and the Rust module
`aiinaction::ch192_momentum`. The shared fixtures in `test/test_ch192_momentum.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

The optimizer maintains a velocity vector that is an exponentially weighted
accumulation of past gradients and steps the parameters along it:

    v <- beta * v + grad(theta)
    theta <- theta - alpha * v

`beta = 0` recovers plain gradient descent. The driver `minimize` takes a gradient
function, so it works for any differentiable objective. `quadratic_gradient` builds
the gradient of `f(theta) = 1/2 (theta - b)' H (theta - b)`.
"""
module Ch192Momentum

export MomentumResult, momentum_step, minimize, quadratic_gradient, optimal_beta

"""The outcome of a momentum optimization run. `components` are stored as plain vectors."""
struct MomentumResult
    theta::Vector{Float64}
    velocity::Vector{Float64}
    n_iter::Int
    converged::Bool
    grad_norm::Float64
    history::Vector{Float64}
end

n_features(r::MomentumResult) = length(r.theta)

function _check_vector(x::AbstractVector, name::AbstractString)
    isempty(x) && throw(ArgumentError("$name must have at least one entry"))
    all(isfinite, x) || throw(ArgumentError("$name contains non-finite values (nan or inf)"))
    return nothing
end

function _check_hyperparams(alpha::Real, beta::Real)
    (isfinite(alpha) && alpha > 0) ||
        throw(ArgumentError("alpha (learning rate) must be a positive finite number, got $alpha"))
    (isfinite(beta) && 0 <= beta < 1) ||
        throw(ArgumentError("beta (momentum) must be in [0, 1), got $beta"))
    return nothing
end

"""
    momentum_step(theta, velocity, grad, alpha, beta)

Apply a single heavy-ball update and return `(new_theta, new_velocity)`:
`v' = beta*v + g`, `theta' = theta - alpha*v'`.
"""
function momentum_step(theta, velocity, grad, alpha::Real, beta::Real)
    _check_hyperparams(alpha, beta)
    th = Vector{Float64}(theta)
    v = Vector{Float64}(velocity)
    g = Vector{Float64}(grad)
    _check_vector(th, "theta")
    _check_vector(v, "velocity")
    _check_vector(g, "grad")
    (length(th) == length(v) == length(g)) ||
        throw(ArgumentError("length mismatch: theta=$(length(th)), velocity=$(length(v)), grad=$(length(g))"))
    new_v = beta .* v .+ g
    new_theta = th .- alpha .* new_v
    return new_theta, new_v
end

"""
    minimize(grad_fn, theta0; alpha, beta=0.9, max_iter=1000, tol=1e-8)

Minimize an objective with heavy-ball momentum from `theta0`. `grad_fn` maps a
parameter vector to its gradient. Stops as soon as `norm(grad(theta)) <= tol`.
"""
function minimize(grad_fn, theta0; alpha::Real, beta::Real=0.9,
                  max_iter::Integer=1000, tol::Real=1e-8)
    _check_hyperparams(alpha, beta)
    theta = Vector{Float64}(theta0)
    _check_vector(theta, "theta0")
    max_iter >= 1 || throw(ArgumentError("max_iter must be a positive integer, got $max_iter"))
    (isfinite(tol) && tol >= 0) ||
        throw(ArgumentError("tol must be a non-negative finite number, got $tol"))

    d = length(theta)
    velocity = zeros(Float64, d)
    history = Float64[]
    grad_norm = Inf
    converged = false
    n_iter = 0

    for _ in 1:max_iter
        g = Vector{Float64}(grad_fn(theta))
        length(g) == d ||
            throw(ArgumentError("grad_fn returned length $(length(g)) but theta has length $d"))
        all(isfinite, g) ||
            throw(ArgumentError("grad_fn returned non-finite values (nan or inf)"))
        grad_norm = sqrt(sum(abs2, g))
        push!(history, grad_norm)
        if grad_norm <= tol
            converged = true
            break
        end
        velocity = beta .* velocity .+ g
        theta = theta .- alpha .* velocity
        n_iter += 1
    end

    return MomentumResult(theta, velocity, n_iter, converged, grad_norm, history)
end

"""
    quadratic_gradient(H, b)

Build the gradient function of `f(theta) = 1/2 (theta - b)' H (theta - b)`, namely
`grad f(theta) = H (theta - b)`. `H` must be square `d x d` and `b` length `d`.
"""
function quadratic_gradient(H, b)
    Hm = Matrix{Float64}(H)
    size(Hm, 1) == size(Hm, 2) ||
        throw(ArgumentError("H must be a square matrix, got size $(size(Hm))"))
    all(isfinite, Hm) || throw(ArgumentError("H contains non-finite values (nan or inf)"))
    bv = Vector{Float64}(b)
    _check_vector(bv, "b")
    size(Hm, 1) == length(bv) ||
        throw(ArgumentError("H is $(size(Hm, 1))x$(size(Hm, 2)) but b has length $(length(bv))"))
    return theta -> Hm * (Vector{Float64}(theta) .- bv)
end

"""
    optimal_beta(lambda_min, lambda_max)

Polyak's optimal momentum for a quadratic with curvature in `[lambda_min, lambda_max]`:
`((sqrt(hi) - sqrt(lo)) / (sqrt(hi) + sqrt(lo)))^2`.
"""
function optimal_beta(lambda_min::Real, lambda_max::Real)
    (isfinite(lambda_min) && isfinite(lambda_max) && lambda_min > 0 && lambda_max > 0) ||
        throw(ArgumentError("lambda_min and lambda_max must be positive finite numbers"))
    lambda_max >= lambda_min ||
        throw(ArgumentError("lambda_max ($lambda_max) must be >= lambda_min ($lambda_min)"))
    r = (sqrt(lambda_max) - sqrt(lambda_min)) / (sqrt(lambda_max) + sqrt(lambda_min))
    return r^2
end

end # module Ch192Momentum
