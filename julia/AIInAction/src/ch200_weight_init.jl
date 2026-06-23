"""
    Ch200WeightInit

Xavier (Glorot) and He (Kaiming) weight initialization from scratch (Julia).

Mirrors the Python module `aiinaction.ch200_weight_init` and the Rust module
`aiinaction::ch200_weight_init`. The shared fixtures in `test/test_ch200_weight_init.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

A layer's weight variance must scale inversely with its fan to keep activations
stable forward and gradients stable backward. Xavier uses
`Var(W) = gain^2 * 2 / (fan_in + fan_out)`; He uses `Var(W) = gain^2 / fan` with
`gain = sqrt(2)` for ReLU. Sampling uses a self-contained, deterministic
SplitMix64 PRNG plus the Box-Muller transform so that, for a fixed seed, the
Python, Julia, and Rust implementations emit the identical weight matrix (to
floating-point tolerance). Base-only, no external dependency.
"""
module Ch200WeightInit

export InitScale, calculate_gain, xavier_scale, he_scale,
    xavier_normal, xavier_uniform, he_normal, he_uniform, relu_gain

"""The theoretical spread of an initialization scheme.

`std` is the gaussian standard deviation `sqrt(Var(W))`; `bound` is the half-width
`r` of the matching uniform support `U(-r, r)`, with `r = std * sqrt(3)`."""
struct InitScale
    std::Float64
    bound::Float64
end

"""
    calculate_gain(nonlinearity; param=nothing)

Recommended gain `g` for a nonlinearity, matching PyTorch conventions. Supported:
`linear`, `sigmoid`, `tanh`, `relu`, `leaky_relu` (uses `param` as the negative
slope, default 0.01) and `selu`.
"""
function calculate_gain(nonlinearity::AbstractString; param=nothing)
    name = lowercase(nonlinearity)
    if name in ("linear", "conv1d", "conv2d", "conv3d", "sigmoid")
        return 1.0
    elseif name == "tanh"
        return 5.0 / 3.0
    elseif name == "relu"
        return sqrt(2.0)
    elseif name == "leaky_relu"
        slope = param === nothing ? 0.01 : Float64(param)
        slope <= -1.0 &&
            throw(ArgumentError("leaky_relu negative slope must be > -1, got $slope"))
        return sqrt(2.0 / (1.0 + slope * slope))
    elseif name == "selu"
        return 3.0 / 4.0
    end
    throw(ArgumentError("unsupported nonlinearity: $(repr(nonlinearity))"))
end

function _check_fan(fan_in::Integer, fan_out::Integer)
    fan_in >= 1 || throw(ArgumentError("fan_in must be a positive integer, got $fan_in"))
    fan_out >= 1 || throw(ArgumentError("fan_out must be a positive integer, got $fan_out"))
    return nothing
end

"""Xavier/Glorot scale: `std = gain * sqrt(2 / (fan_in + fan_out))`."""
function xavier_scale(fan_in::Integer, fan_out::Integer; gain::Real=1.0)
    _check_fan(fan_in, fan_out)
    gain > 0 || throw(ArgumentError("gain must be positive, got $gain"))
    std = gain * sqrt(2.0 / (fan_in + fan_out))
    return InitScale(std, std * sqrt(3.0))
end

"""He/Kaiming scale: `std = gain / sqrt(fan)`. `fan` is the chosen mode."""
function he_scale(fan::Integer; gain::Real=sqrt(2.0))
    fan >= 1 || throw(ArgumentError("fan must be a positive integer, got $fan"))
    gain > 0 || throw(ArgumentError("gain must be positive, got $gain"))
    std = gain / sqrt(fan)
    return InitScale(std, std * sqrt(3.0))
end

"""The default ReLU gain `sqrt(2)`."""
relu_gain() = sqrt(2.0)

# --- Deterministic SplitMix64 PRNG (identical to Python/Rust) ----------------

mutable struct _SplitMix64
    state::UInt64
end

function _next_u64!(rng::_SplitMix64)
    rng.state += 0x9E3779B97F4A7C15
    z = rng.state
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    return z ⊻ (z >> 31)
end

function _next_double!(rng::_SplitMix64)
    # Top 53 bits give a uniform double in [0, 1).
    return Float64(_next_u64!(rng) >> 11) * (1.0 / Float64(UInt64(1) << 53))
end

function _next_normal!(rng::_SplitMix64)
    u1 = _next_double!(rng)
    u2 = _next_double!(rng)
    if u1 <= 0.0
        u1 = 5e-324  # smallest positive subnormal double
    end
    r = sqrt(-2.0 * log(u1))
    return r * cos(2.0 * pi * u2)
end

# Weight matrices are returned (fan_out, fan_in) (rows = output units).
function _fill_normal(fan_in::Integer, fan_out::Integer, std::Float64, seed::Integer)
    rng = _SplitMix64(UInt64(seed))
    out = Matrix{Float64}(undef, fan_out, fan_in)
    # Row-major fill order to match Python/Rust exactly.
    for i in 1:fan_out
        for j in 1:fan_in
            out[i, j] = _next_normal!(rng) * std
        end
    end
    return out
end

function _fill_uniform(fan_in::Integer, fan_out::Integer, bound::Float64, seed::Integer)
    rng = _SplitMix64(UInt64(seed))
    out = Matrix{Float64}(undef, fan_out, fan_in)
    for i in 1:fan_out
        for j in 1:fan_in
            out[i, j] = (_next_double!(rng) * 2.0 - 1.0) * bound
        end
    end
    return out
end

"""Sample a `(fan_out, fan_in)` weight matrix from Xavier normal."""
function xavier_normal(fan_in::Integer, fan_out::Integer; gain::Real=1.0, seed::Integer=0)
    s = xavier_scale(fan_in, fan_out; gain=gain)
    return _fill_normal(fan_in, fan_out, s.std, seed)
end

"""Sample a `(fan_out, fan_in)` weight matrix from Xavier uniform `U(-r, r)`."""
function xavier_uniform(fan_in::Integer, fan_out::Integer; gain::Real=1.0, seed::Integer=0)
    s = xavier_scale(fan_in, fan_out; gain=gain)
    return _fill_uniform(fan_in, fan_out, s.bound, seed)
end

"""Sample a `(fan_out, fan_in)` weight matrix from He normal. `mode` is
`:fan_in` (default) or `:fan_out`."""
function he_normal(fan_in::Integer, fan_out::Integer; gain::Real=sqrt(2.0),
        mode::Symbol=:fan_in, seed::Integer=0)
    _check_fan(fan_in, fan_out)
    gain > 0 || throw(ArgumentError("gain must be positive, got $gain"))
    mode in (:fan_in, :fan_out) ||
        throw(ArgumentError("mode must be :fan_in or :fan_out, got $(repr(mode))"))
    fan = mode === :fan_in ? fan_in : fan_out
    s = he_scale(fan; gain=gain)
    return _fill_normal(fan_in, fan_out, s.std, seed)
end

"""Sample a `(fan_out, fan_in)` weight matrix from He uniform `U(-r, r)`."""
function he_uniform(fan_in::Integer, fan_out::Integer; gain::Real=sqrt(2.0),
        mode::Symbol=:fan_in, seed::Integer=0)
    _check_fan(fan_in, fan_out)
    gain > 0 || throw(ArgumentError("gain must be positive, got $gain"))
    mode in (:fan_in, :fan_out) ||
        throw(ArgumentError("mode must be :fan_in or :fan_out, got $(repr(mode))"))
    fan = mode === :fan_in ? fan_in : fan_out
    s = he_scale(fan; gain=gain)
    return _fill_uniform(fan_in, fan_out, s.bound, seed)
end

end # module Ch200WeightInit
