"""
    Ch207Densenet

Dense connectivity (DenseNet) mechanics, from scratch (Julia).

Mirrors the Python module `aiinaction.ch207_densenet` and the Rust module
`aiinaction::ch207_densenet`. The shared fixtures in
`test/test_ch207_densenet.jl` match the Python/Rust suites, which keeps the
three implementations at parity.

Channel and parameter arithmetic (`dense_block_channel_sizes`,
`dense_block_param_count`, `transition_output_channels`,
`plain_block_param_count`, `densenet_dense_param_total`) is exact integer
arithmetic and needs no randomness. The toy `dense_block_forward` stands in
for a real dense block's batch-norm/ReLU/convolution composite `H_l` with a
single seeded linear-plus-ReLU layer, using the same 64-bit linear
congruential generator (LCG) used elsewhere in this book so that weights are
reproducible bit for bit across languages given the same seed.
"""
module Ch207Densenet

export Lcg, next_uniform!, dense_block_channel_sizes, dense_block_param_count,
       transition_output_channels, plain_block_param_count, init_layer_weights,
       dense_layer_forward, dense_block_forward, DENSENET_VARIANTS,
       densenet_dense_param_total

const LCG_A = 0x5851f42d4c957f2d  # 6364136223846793005
const LCG_C = 0x14057b7ef767814f  # 1442695040888963407
const UNIT = 9007199254740992.0   # 2^53

"""A minimal, fully reproducible 64-bit linear congruential generator.

Mirrors the Python `Lcg` and Rust `Lcg` bit for bit using native `UInt64`
wrapping arithmetic."""
mutable struct Lcg
    state::UInt64
end

function Lcg(seed::Integer)
    seed >= 0 || throw(ArgumentError("seed must be non-negative, got $seed"))
    return Lcg(UInt64(seed))
end

"""Advance `rng` and return the next uniform draw in `[0, 1)`."""
function next_uniform!(rng::Lcg)
    rng.state = LCG_A * rng.state + LCG_C
    top = rng.state >> 11
    return Float64(top) / UNIT
end

# --------------------------------------------------------------------------
# Channel and parameter arithmetic
# --------------------------------------------------------------------------

"""
    dense_block_channel_sizes(c0, growth_rate, num_layers)

Return the input-channel count seen by each layer of a dense block, plus the
block's total output width as the final entry. Layer `l` (0-indexed) sees
`c0 + growth_rate * l` channels.
"""
function dense_block_channel_sizes(c0::Integer, growth_rate::Integer, num_layers::Integer)
    c0 >= 0 || throw(ArgumentError("c0 must be >= 0, got $c0"))
    growth_rate >= 1 || throw(ArgumentError("growth_rate must be >= 1, got $growth_rate"))
    num_layers >= 1 || throw(ArgumentError("num_layers must be >= 1, got $num_layers"))
    return [c0 + growth_rate * l for l in 0:num_layers]
end

"""
    dense_block_param_count(c0, growth_rate, num_layers, bn_size=4)

Count parameters in a DenseNet-BC style dense block: each layer applies a
`1x1` bottleneck to `bn_size * growth_rate` channels followed by a `3x3`
convolution down to `growth_rate` new channels (biases and batch norm
omitted). Sums `params_l = c_l * (b*k) + (b*k) * k * 9` over the block.
"""
function dense_block_param_count(c0::Integer, growth_rate::Integer, num_layers::Integer, bn_size::Integer=4)
    bn_size >= 1 || throw(ArgumentError("bn_size must be >= 1, got $bn_size"))
    sizes = dense_block_channel_sizes(c0, growth_rate, num_layers)
    k = growth_rate
    bw = bn_size * k
    total = 0
    for c_l in sizes[1:end-1]
        total += c_l * bw
        total += bw * k * 9
    end
    return total
end

"""
    transition_output_channels(c_in, theta)

Return the compressed channel count `floor(theta * c_in)` after a DenseNet
transition layer, `0 < theta <= 1`.
"""
function transition_output_channels(c_in::Integer, theta::Real)
    (theta > 0 && theta <= 1) || throw(ArgumentError("theta must satisfy 0 < theta <= 1, got $theta"))
    c_in >= 1 || throw(ArgumentError("c_in must be >= 1, got $c_in"))
    return Int(floor(theta * c_in))
end

"""
    plain_block_param_count(c0, width, num_layers)

Parameter count of a plain (additive, ResNet-style) stack matched on output
width: the first `3x3` layer maps `c0 -> width`, every later layer maps
`width -> width`. The parameter-cost baseline for Section 4.2's efficiency
comparison.
"""
function plain_block_param_count(c0::Integer, width::Integer, num_layers::Integer)
    c0 >= 1 || throw(ArgumentError("c0 must be >= 1, got $c0"))
    width >= 1 || throw(ArgumentError("width must be >= 1, got $width"))
    num_layers >= 1 || throw(ArgumentError("num_layers must be >= 1, got $num_layers"))
    return c0 * width * 9 + (num_layers - 1) * (width * width * 9)
end

# --------------------------------------------------------------------------
# Toy forward pass demonstrating the concatenation mechanic
# --------------------------------------------------------------------------

"""
    init_layer_weights(c_in, growth_rate, seed)

Deterministically initialize one dense layer's weight matrix (`growth_rate x
c_in`) and zero bias, drawing every entry from `Lcg(seed)` mapped to
`[-lim, lim]` with `lim = sqrt(6 / (c_in + growth_rate))`.
"""
function init_layer_weights(c_in::Integer, growth_rate::Integer, seed::Integer)
    c_in >= 1 || throw(ArgumentError("c_in must be >= 1, got $c_in"))
    growth_rate >= 1 || throw(ArgumentError("growth_rate must be >= 1, got $growth_rate"))
    lim = sqrt(6.0 / (c_in + growth_rate))
    rng = Lcg(seed)
    w = Matrix{Float64}(undef, growth_rate, c_in)
    for i in 1:growth_rate, j in 1:c_in
        u = next_uniform!(rng)
        w[i, j] = (2.0 * u - 1.0) * lim
    end
    b = zeros(Float64, growth_rate)
    return w, b
end

"""
    dense_layer_forward(x, w, b)

Apply one dense-block layer `H_l`: `relu(w * x + b)`.
"""
function dense_layer_forward(x::AbstractVector{<:Real}, w::AbstractMatrix{<:Real}, b::AbstractVector{<:Real})
    z = w * x .+ b
    return max.(z, 0.0)
end

"""
    dense_block_forward(x0, growth_rate, num_layers, seed)

Run a toy dense block forward pass, literally implementing `x_l =
H_l([x_0, ..., x_{l-1}])` by concatenating each layer's output onto the
running feature vector. Layer `l` uses `init_layer_weights` seeded with
`seed + l`. Returns `(final_features, channel_sizes)`.
"""
function dense_block_forward(x0::AbstractVector{<:Real}, growth_rate::Integer, num_layers::Integer, seed::Integer)
    isempty(x0) && throw(ArgumentError("x0 must be non-empty"))
    all(isfinite, x0) || throw(ArgumentError("x0 contains non-finite values (nan or inf)"))
    c0 = length(x0)
    sizes = dense_block_channel_sizes(c0, growth_rate, num_layers)
    features = Float64.(collect(x0))
    for l in 0:(num_layers - 1)
        w, b = init_layer_weights(length(features), growth_rate, seed + l)
        new_features = dense_layer_forward(features, w, b)
        features = vcat(features, new_features)
        @assert length(features) == sizes[l + 2]
    end
    return features, sizes
end

# --------------------------------------------------------------------------
# DenseNet-BC architecture variants (Huang et al. 2017, Table 1)
# --------------------------------------------------------------------------

const DENSENET_VARIANTS = Dict(
    "121" => (6, 12, 24, 16),
    "169" => (6, 12, 32, 32),
    "201" => (6, 12, 48, 32),
    "264" => (6, 12, 64, 48),
)

"""
    densenet_dense_param_total(variant; growth_rate=32, k0=64, theta=0.5, bn_size=4)

Sum dense-block and transition-layer parameters for a DenseNet-BC variant
(`"121"`, `"169"`, `"201"`, or `"264"`). Excludes the stem convolution, final
batch norm, and classifier head; see the Python docstring for details.
"""
function densenet_dense_param_total(variant::AbstractString; growth_rate::Integer=32, k0::Integer=64, theta::Real=0.5, bn_size::Integer=4)
    haskey(DENSENET_VARIANTS, variant) || throw(ArgumentError("unknown variant $variant"))
    blocks = DENSENET_VARIANTS[variant]
    c = k0
    total = 0
    for (i, num_layers) in enumerate(blocks)
        total += dense_block_param_count(c, growth_rate, num_layers, bn_size)
        c = c + growth_rate * num_layers
        if i < length(blocks)
            c_out = transition_output_channels(c, theta)
            total += c * c_out
            c = c_out
        end
    end
    return total
end

end # module Ch207Densenet
