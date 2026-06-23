"""
    Ch118Smote

SMOTE: Synthetic Minority Over-sampling Technique (Chapter 118).

Mirrors the Python module `aiinaction.ch118_smote` and the Rust module
`aiinaction::ch118_smote`. SMOTE rebalances an imbalanced training set by
synthesizing minority-class examples through linear interpolation between a
minority point and one of its `k` nearest minority neighbors:

    x_new = x_i + lambda * (x_nn - x_i),   lambda ~ Uniform(0, 1).

All randomness flows through a fixed linear-congruential generator (the Numerical
Recipes constants) so the three languages emit identical synthetic points to
floating-point tolerance for a given seed.
"""
module Ch118Smote

export LCG, next_uint!, next_float!, next_index!, euclidean, k_nearest, smote_sample, smote

const LCG_A = UInt64(1664525)
const LCG_C = UInt64(1013904223)
const LCG_M = UInt64(1) << 32

"""Deterministic linear-congruential generator shared across all three languages."""
mutable struct LCG
    state::UInt64
    function LCG(seed::Integer)
        seed < 0 && throw(ArgumentError("seed must be non-negative, got $seed"))
        new(UInt64(seed) % LCG_M)
    end
end

"""Advance the generator and return the raw 32-bit state."""
function next_uint!(rng::LCG)
    rng.state = (LCG_A * rng.state + LCG_C) % LCG_M
    return rng.state
end

"""Next pseudo-random float in [0, 1)."""
next_float!(rng::LCG) = Float64(next_uint!(rng)) / Float64(LCG_M)

"""Next pseudo-random integer in [0, n)."""
function next_index!(rng::LCG, n::Integer)
    n <= 0 && throw(ArgumentError("n must be positive, got $n"))
    return Int(next_uint!(rng) % UInt64(n))
end

"""Euclidean distance between two equal-length feature vectors."""
function euclidean(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    length(a) == length(b) ||
        throw(ArgumentError("dimension mismatch: $(length(a)) != $(length(b))"))
    isempty(a) && throw(ArgumentError("vectors must be non-empty"))
    return sqrt(sum((ai - bi)^2 for (ai, bi) in zip(a, b)))
end

"""
    k_nearest(points, idx, k)

Indices (1-based) of the `k` nearest neighbors of `points[idx]`, excluding itself.
Ties are broken by the lower point index, so the result is deterministic.
"""
function k_nearest(points::AbstractVector{<:AbstractVector{<:Real}}, idx::Integer, k::Integer)
    n = length(points)
    n == 0 && throw(ArgumentError("points must be non-empty"))
    (1 <= idx <= n) || throw(ArgumentError("idx $idx out of range for $n points"))
    k <= 0 && throw(ArgumentError("k must be positive, got $k"))
    k > n - 1 && throw(ArgumentError("k=$k exceeds available neighbors $(n - 1)"))
    dists = [(euclidean(points[idx], points[j]), j) for j in 1:n if j != idx]
    sort!(dists, by = dj -> (dj[1], dj[2]))
    return [j for (_, j) in dists[1:k]]
end

"""One synthetic point on the segment from `x_i` toward `x_nn`: x_i + lam*(x_nn - x_i)."""
function smote_sample(x_i::AbstractVector{<:Real}, x_nn::AbstractVector{<:Real}, lam::Real)
    length(x_i) == length(x_nn) ||
        throw(ArgumentError("dimension mismatch: $(length(x_i)) != $(length(x_nn))"))
    (0.0 <= lam <= 1.0) || throw(ArgumentError("lam must be in [0, 1], got $lam"))
    return [Float64(xi) + lam * (Float64(xn) - Float64(xi)) for (xi, xn) in zip(x_i, x_nn)]
end

"""
    smote(minority, n_synthetic; k=5, seed=0)

Generate `n_synthetic` synthetic minority examples via SMOTE. For each point: pick a
base minority index round-robin, draw one of its `k` nearest neighbors via the LCG,
then draw `lambda` via the LCG. The draw order (neighbor first, then lambda) is fixed
and shared across languages. The same seed yields identical output in Python and Rust.
"""
function smote(minority::AbstractVector{<:AbstractVector{<:Real}}, n_synthetic::Integer;
               k::Integer = 5, seed::Integer = 0)
    n_synthetic < 0 && throw(ArgumentError("n_synthetic must be non-negative, got $n_synthetic"))
    n = length(minority)
    n == 0 && throw(ArgumentError("minority set must be non-empty"))
    dim = length(minority[1])
    dim == 0 && throw(ArgumentError("feature vectors must be non-empty"))
    for row in minority
        length(row) == dim || throw(ArgumentError("all minority rows must have the same dimension"))
    end
    n_synthetic == 0 && return Vector{Vector{Float64}}()
    k <= 0 && throw(ArgumentError("k must be positive, got $k"))
    k > n - 1 && throw(ArgumentError(
        "k=$k exceeds available neighbors $(n - 1); need at least k+1 minority points"))

    neighbors = [k_nearest(minority, i, k) for i in 1:n]

    rng = LCG(seed)
    synthetic = Vector{Vector{Float64}}()
    for s in 0:(n_synthetic - 1)
        base = (s % n) + 1                    # 1-based round-robin
        nn_choice = next_index!(rng, k) + 1   # 1-based neighbor index
        nn = neighbors[base][nn_choice]
        lam = next_float!(rng)
        push!(synthetic, smote_sample(minority[base], minority[nn], lam))
    end
    return synthetic
end

end # module Ch118Smote
