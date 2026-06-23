"""
    Ch204Dropout

Inverted dropout with Bernoulli masking, from scratch (Julia).

Mirrors the Python module `aiinaction.ch204_dropout` and the Rust module
`aiinaction::ch204_dropout`. The shared fixtures in `test/test_ch204_dropout.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

Randomness comes from a tiny 64-bit linear congruential generator using the
Numerical Recipes constants. Fixing this generator across all three languages
makes the masks reproducible: given the same seed and length, the same units are
dropped everywhere. Each draw uses the top 53 bits of the 64-bit state, so every
uniform is an exact multiple of `2^-53`.

Convention: `p` is the retention probability (`1 - p` is the drop probability). A
retained unit is scaled by `1/p`; a dropped unit becomes `0.0`. This is inverted
dropout, for which `E[mask] = 1` and `E[mask .* h] = h`.
"""
module Ch204Dropout

export Lcg, next_uniform!, bernoulli_mask, inverted_dropout, expected_scale

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
    rng.state = LCG_A * rng.state + LCG_C   # UInt64 arithmetic wraps mod 2^64
    top = rng.state >> 11                   # top 53 bits
    return Float64(top) / UNIT
end

"""Return `1 / p`, the survivor scaling that keeps the mask mean at 1.

Throws unless `0 < p <= 1`."""
function expected_scale(p::Real)
    (p > 0 && p <= 1) ||
        throw(ArgumentError("retention probability p must satisfy 0 < p <= 1, got $p"))
    return 1.0 / p
end

"""
    bernoulli_mask(n, p, seed)

Build a length-`n` inverted-dropout mask with retention probability `p`. Unit `i`
is retained when `u_i < p` (the `i`-th draw of `Lcg(seed)`) and assigned `1/p`;
otherwise it is dropped and assigned `0.0`.
"""
function bernoulli_mask(n::Integer, p::Real, seed::Integer)
    n >= 1 || throw(ArgumentError("n must be >= 1, got $n"))
    scale = expected_scale(p)
    rng = Lcg(seed)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        u = next_uniform!(rng)
        out[i] = u < p ? scale : 0.0
    end
    return out
end

"""
    inverted_dropout(h, p, seed)

Apply inverted dropout to the activation vector `h`. Returns `(masked, mask)`
where `masked[i] = mask[i] * h[i]`. Dropped units are forced to exactly `0.0`;
survivors are scaled by `1/p`.
"""
function inverted_dropout(h::AbstractVector{<:Real}, p::Real, seed::Integer)
    isempty(h) && throw(ArgumentError("h must be non-empty"))
    all(isfinite, h) || throw(ArgumentError("h contains non-finite values (nan or inf)"))
    mask = bernoulli_mask(length(h), p, seed)
    masked = Float64.(h) .* mask
    return masked, mask
end

end # module Ch204Dropout
