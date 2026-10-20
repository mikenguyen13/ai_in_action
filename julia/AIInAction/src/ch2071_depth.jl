"""
    Ch2071Depth

Why depth is not free: signal propagation and depth diagnostics from scratch
(Julia).

Mirrors the Python module `aiinaction.ch2071_depth` and the Rust module
`aiinaction::ch2071_depth`. The shared fixtures in `test/test_ch2071_depth.jl`
match the Python and Rust suites, which keeps the three implementations at
parity.

Contents: the mean field length map, its fixed point `q*`, the correlation
map, the slope `chi_1` and the correlation depth scale (Poole et al. 2016,
Schoenholz et al. 2017); the closed form ReLU correlation map (the normalized
arc cosine kernel of Cho and Saul 2009), its iteration in plain and residual
networks, the `9 pi^2 / (2 l^2)` collapse law and the continuous depth limit;
residual stream variance growth; the learning time of a scalar deep linear
network (Saxe et al. 2014); residual path statistics (Veit et al. 2016); and
Telgarsky's sawtooth. Activations are named by the strings `"linear"`,
`"relu"` and `"tanh"`. Everything is deterministic: quadrature is composite
Simpson with fixed nodes and the ODE uses fixed step RK4.
"""
module Ch2071Depth

export ACTIVATIONS, activation, activation_derivative, gaussian_expectation,
       gaussian_expectation_2d, length_map, length_fixed_point, correlation_map,
       chi1, correlation_depth_scale, relu_correlation_map,
       relu_correlation_trajectory, relu_collapse_asymptotic, relu_kernel_ode,
       residual_variance, deep_linear_learning_time, deep_linear_time_asymptotic,
       path_length_distribution, effective_path_distribution, sawtooth,
       count_linear_pieces

const ACTIVATIONS = ("linear", "relu", "tanh")

# Quadrature defaults. Composite Simpson on [-LIM, LIM] with N (even) panels.
const LIM = 10.0
const N1 = 4000
const N2 = 400

function _check_activation(name::AbstractString)
    name in ACTIVATIONS ||
        throw(ArgumentError("unknown activation \"$name\"; expected one of $ACTIVATIONS"))
    return nothing
end

"""
    activation(name, x)

Evaluate `phi(x)` for `name` in `ACTIVATIONS`.
"""
function activation(name::AbstractString, x::Real)
    _check_activation(name)
    x = Float64(x)
    name == "linear" && return x
    name == "relu" && return x > 0.0 ? x : 0.0
    return tanh(x)
end

"""
    activation_derivative(name, x)

Evaluate `phi'(x)`; the ReLU derivative at 0 is taken as 0.
"""
function activation_derivative(name::AbstractString, x::Real)
    _check_activation(name)
    x = Float64(x)
    name == "linear" && return 1.0
    name == "relu" && return x > 0.0 ? 1.0 : 0.0
    t = tanh(x)
    return 1.0 - t * t
end

function _simpson_weights(n::Integer)
    (n >= 2 && iseven(n)) ||
        throw(ArgumentError("number of Simpson panels must be an even integer >= 2"))
    w = [iseven(i) ? 2.0 : 4.0 for i in 0:n]
    w[1] = 1.0
    w[end] = 1.0
    return w
end

"""
    gaussian_expectation(f; n=4000, lim=10.0)

`E[f(z)]` for `z ~ N(0, 1)` by composite Simpson quadrature of
`f(z) * exp(-z^2 / 2) / sqrt(2 pi)` on `[-lim, lim]`.
"""
function gaussian_expectation(f; n::Integer=N1, lim::Real=LIM)
    w = _simpson_weights(n)
    lim = Float64(lim)
    h = 2.0 * lim / n
    c = 1.0 / sqrt(2.0 * pi)
    total = 0.0
    for i in 0:n
        z = -lim + i * h
        total += w[i + 1] * f(z) * exp(-0.5 * z * z)
    end
    return total * h / 3.0 * c
end

"""
    gaussian_expectation_2d(f; n=400, lim=10.0)

`E[f(z1, z2)]` for independent standard normals by tensor Simpson quadrature.
"""
function gaussian_expectation_2d(f; n::Integer=N2, lim::Real=LIM)
    w = _simpson_weights(n)
    lim = Float64(lim)
    h = 2.0 * lim / n
    c = 1.0 / (2.0 * pi)
    nodes = [-lim + i * h for i in 0:n]
    dens = [exp(-0.5 * z * z) for z in nodes]
    total = 0.0
    for i in 1:(n + 1)
        z1 = nodes[i]
        wi = w[i] * dens[i]
        row = 0.0
        for j in 1:(n + 1)
            row += w[j] * dens[j] * f(z1, nodes[j])
        end
        total += wi * row
    end
    return total * (h / 3.0)^2 * c
end

function _check_variances(sigma_w2::Real, sigma_b2::Real)
    (sigma_w2 > 0.0 && sigma_b2 >= 0.0) ||
        throw(ArgumentError("need sigma_w2 > 0 and sigma_b2 >= 0"))
    return nothing
end

"""
    length_map(q, sigma_w2, sigma_b2, phi)

One step of the mean field length map
`q_{l+1} = sigma_w^2 E[phi(sqrt(q_l) z)^2] + sigma_b^2`.
"""
function length_map(q::Real, sigma_w2::Real, sigma_b2::Real, phi::AbstractString)
    _check_variances(sigma_w2, sigma_b2)
    q < 0.0 && throw(ArgumentError("q must be nonnegative"))
    _check_activation(phi)
    s = sqrt(Float64(q))
    return sigma_w2 * gaussian_expectation(z -> activation(phi, s * z)^2) + sigma_b2
end

"""
    length_fixed_point(sigma_w2, sigma_b2, phi; q0=1.0, tol=1e-12, max_iter=10_000)

Iterate the length map from `q0` to its fixed point `q*`. Throws an
`ErrorException` if the map diverges or has not converged.
"""
function length_fixed_point(sigma_w2::Real, sigma_b2::Real, phi::AbstractString;
                            q0::Real=1.0, tol::Real=1e-12, max_iter::Integer=10_000)
    q = Float64(q0)
    for _ in 1:max_iter
        q_next = length_map(q, sigma_w2, sigma_b2, phi)
        (!isfinite(q_next) || q_next > 1e12) &&
            error("length map diverged; no finite fixed point")
        abs(q_next - q) <= tol * max(1.0, q) && return q_next
        q = q_next
    end
    error("length map did not converge")
end

"""
    correlation_map(c, q, sigma_w2, sigma_b2, phi; n=400)

One step of the mean field correlation map at fixed length `q`:
`c_{l+1} = (sigma_w^2 E[phi(u1) phi(u2)] + sigma_b^2) / q` with
`u1 = sqrt(q) z1` and `u2 = sqrt(q) (c z1 + sqrt(1 - c^2) z2)`.
"""
function correlation_map(c::Real, q::Real, sigma_w2::Real, sigma_b2::Real,
                         phi::AbstractString; n::Integer=N2)
    _check_variances(sigma_w2, sigma_b2)
    (-1.0 <= c <= 1.0) || throw(ArgumentError("correlation must lie in [-1, 1]"))
    q > 0.0 || throw(ArgumentError("q must be positive"))
    _check_activation(phi)
    c = Float64(c)
    s = sqrt(Float64(q))
    r = sqrt(max(0.0, 1.0 - c * c))
    e = gaussian_expectation_2d(
        (z1, z2) -> activation(phi, s * z1) * activation(phi, s * (c * z1 + r * z2)); n=n)
    return (sigma_w2 * e + sigma_b2) / q
end

"""
    chi1(q, sigma_w2, phi)

Slope of the correlation map at `c = 1`, `chi_1 = sigma_w^2 E[phi'(sqrt(q) z)^2]`.
`chi_1 < 1` is the ordered phase and `chi_1 > 1` the chaotic phase.
"""
function chi1(q::Real, sigma_w2::Real, phi::AbstractString)
    (sigma_w2 > 0.0 && q >= 0.0) || throw(ArgumentError("need sigma_w2 > 0 and q >= 0"))
    _check_activation(phi)
    s = sqrt(Float64(q))
    return sigma_w2 * gaussian_expectation(z -> activation_derivative(phi, s * z)^2)
end

"""
    correlation_depth_scale(chi_c)

`xi_c = -1 / log(chi_c)`; infinite at criticality `chi_c = 1`.
"""
function correlation_depth_scale(chi_c::Real)
    chi_c <= 0.0 && throw(ArgumentError("chi_c must be positive"))
    chi_c == 1.0 && return Inf
    return -1.0 / log(Float64(chi_c))
end

"""
    relu_correlation_map(c)

Closed form ReLU correlation map at criticality,
`f(c) = (sqrt(1 - c^2) + (pi - acos(c)) c) / pi`.
"""
function relu_correlation_map(c::Real)
    (-1.0 <= c <= 1.0) || throw(ArgumentError("correlation must lie in [-1, 1]"))
    c = Float64(c)
    return (sqrt(max(0.0, 1.0 - c * c)) + (pi - acos(c)) * c) / pi
end

"""
    relu_correlation_trajectory(c0, depth; alpha=nothing)

Correlation after each of `depth` layers, returned as `[c_0, ..., c_depth]`.
`alpha = nothing` is a plain critical ReLU network, `c_{l+1} = f(c_l)`; a
residual network with branch scale `alpha` follows
`c_{l+1} = (c_l + alpha^2 f(c_l)) / (1 + alpha^2)`.
"""
function relu_correlation_trajectory(c0::Real, depth::Integer; alpha::Union{Nothing,Real}=nothing)
    depth < 0 && throw(ArgumentError("depth must be nonnegative"))
    (alpha !== nothing && alpha < 0.0) && throw(ArgumentError("alpha must be nonnegative"))
    c = Float64(c0)
    out = [c]
    for _ in 1:depth
        fc = relu_correlation_map(c)
        if alpha === nothing
            c = fc
        else
            a2 = Float64(alpha) * Float64(alpha)
            c = (c + a2 * fc) / (1.0 + a2)
        end
        c = min(1.0, max(-1.0, c))
        push!(out, c)
    end
    return out
end

"""
    relu_collapse_asymptotic(depth; alpha=nothing)

Asymptotic `1 - c_l` for large depth: `9 pi^2 / (2 l^2)` for a plain network
and `9 pi^2 (1 + alpha^2)^2 / (2 alpha^4 l^2)` for a residual one.
"""
function relu_collapse_asymptotic(depth::Real; alpha::Union{Nothing,Real}=nothing)
    depth <= 0 && throw(ArgumentError("depth must be positive"))
    l = Float64(depth)
    base = 9.0 * pi^2 / (2.0 * l^2)
    alpha === nothing && return base
    alpha <= 0.0 && throw(ArgumentError("alpha must be positive"))
    a2 = Float64(alpha) * Float64(alpha)
    return base * (1.0 + a2)^2 / (a2 * a2)
end

"""
    relu_kernel_ode(c0, t; steps=10_000)

Integrate `dc/dt = f(c) - c` from `c(0) = c0` to time `t` with fixed step
RK4. This is the continuous depth limit of a residual ReLU network with branch
scale `alpha = 1 / sqrt(L)`.
"""
function relu_kernel_ode(c0::Real, t::Real; steps::Integer=10_000)
    (steps > 0 && t >= 0.0) || throw(ArgumentError("need steps > 0 and t >= 0"))
    g(c) = (cc = min(1.0, max(-1.0, c)); relu_correlation_map(cc) - cc)
    h = Float64(t) / steps
    c = Float64(c0)
    for _ in 1:steps
        k1 = g(c)
        k2 = g(c + 0.5 * h * k1)
        k3 = g(c + 0.5 * h * k2)
        k4 = g(c + h * k3)
        c += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    end
    return c
end

"""
    residual_variance(q0, depth, alpha)

Residual stream variance `q_0 (1 + alpha^2)^depth` for ReLU branches.
"""
function residual_variance(q0::Real, depth::Integer, alpha::Real)
    (q0 >= 0.0 && depth >= 0 && alpha >= 0.0) ||
        throw(ArgumentError("need q0 >= 0, depth >= 0, alpha >= 0"))
    return Float64(q0) * (1.0 + Float64(alpha) * Float64(alpha))^depth
end

"""
    deep_linear_learning_time(depth, a0; s=1.0, frac=0.5, n=4000)

Gradient flow time for the scalar depth `N` linear net `u = a_1 ... a_N`,
started at `a_i = a0`, to reach `u = frac * s`:
`T = int_{u0}^{frac s} du / (N u^{2 - 2/N} (s - u))` with `u0 = a0^N`, by
Simpson's rule in `v = log u`.
"""
function deep_linear_learning_time(depth::Integer, a0::Real; s::Real=1.0,
                                   frac::Real=0.5, n::Integer=4000)
    depth < 1 && throw(ArgumentError("depth must be at least 1"))
    (a0 > 0.0 && s > 0.0 && 0.0 < frac < 1.0) ||
        throw(ArgumentError("need a0 > 0, s > 0 and 0 < frac < 1"))
    s = Float64(s)
    u0 = Float64(a0)^depth
    u1 = Float64(frac) * s
    u0 >= u1 && return 0.0
    v0, v1 = log(u0), log(u1)
    w = _simpson_weights(n)
    h = (v1 - v0) / n
    p = 2.0 - 2.0 / depth
    total = 0.0
    for i in 0:n
        v = v0 + i * h
        u = exp(v)
        total += w[i + 1] * u / (depth * u^p * (s - u))
    end
    return total * h / 3.0
end

"""
    deep_linear_time_asymptotic(depth, a0; s=1.0)

Leading order learning time `a0^{-(N-2)} / (s (N - 2))` for `N > 2`.
"""
function deep_linear_time_asymptotic(depth::Integer, a0::Real; s::Real=1.0)
    depth <= 2 && throw(ArgumentError("asymptotic formula needs depth > 2"))
    (a0 > 0.0 && s > 0.0) || throw(ArgumentError("need a0 > 0 and s > 0"))
    return Float64(a0)^(-(depth - 2)) / (Float64(s) * (depth - 2))
end

"""
    path_length_distribution(n_blocks)

Fraction of the `2^n` paths through `n` residual blocks that cross exactly
`k` branches, `C(n, k) / 2^n` for `k = 0, ..., n`.
"""
function path_length_distribution(n_blocks::Integer)
    n_blocks < 0 && throw(ArgumentError("n_blocks must be nonnegative"))
    total = 2.0^n_blocks
    return [Float64(binomial(big(n_blocks), k)) / total for k in 0:n_blocks]
end

"""
    effective_path_distribution(n_blocks, rho)

Share of the gradient carried by paths of each length when each residual
branch scales a backpropagated signal by `rho`: Binomial(n, rho / (1 + rho)).
"""
function effective_path_distribution(n_blocks::Integer, rho::Real)
    (n_blocks >= 0 && rho >= 0.0) || throw(ArgumentError("need n_blocks >= 0 and rho >= 0"))
    p = Float64(rho) / (1.0 + Float64(rho))
    return [Float64(binomial(big(n_blocks), k)) * p^k * (1.0 - p)^(n_blocks - k)
            for k in 0:n_blocks]
end

"""
    sawtooth(x, depth)

Telgarsky's construction: `depth` compositions of the tent map
`m(x) = 2 relu(x) - 4 relu(x - 1/2)`, giving `2^depth` linear pieces on
`[0, 1]`.
"""
function sawtooth(x::Real, depth::Integer)
    depth < 0 && throw(ArgumentError("depth must be nonnegative"))
    y = Float64(x)
    for _ in 1:depth
        y = 2.0 * max(y, 0.0) - 4.0 * max(y - 0.5, 0.0)
    end
    return y
end

"""
    count_linear_pieces(ys; tol=1e-9)

Count maximal linear pieces of a function sampled on a uniform grid: each
maximal run of grid points whose second difference exceeds `tol` in magnitude
marks one breakpoint.
"""
function count_linear_pieces(ys::AbstractVector{<:Real}; tol::Real=1e-9)
    length(ys) < 3 && throw(ArgumentError("need at least three samples"))
    pieces = 1
    in_kink = false
    for i in 2:(length(ys) - 1)
        d2 = ys[i + 1] - 2.0 * ys[i] + ys[i - 1]
        kink = abs(d2) > tol
        (kink && !in_kink) && (pieces += 1)
        in_kink = kink
    end
    return pieces
end

end # module Ch2071Depth
