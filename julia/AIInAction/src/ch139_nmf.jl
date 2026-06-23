"""
    Ch139Nmf

Non-Negative Matrix Factorization (NMF) from scratch (Julia).

Mirrors the Python module `aiinaction.ch139_nmf` and the Rust module
`aiinaction::ch139_nmf`. The shared fixtures in `test/test_ch139_nmf.jl` match the
Python/Rust suites, which keeps the three implementations at parity.

Given a non-negative matrix `V` of shape `(n, m)`, NMF seeks non-negative factors
`W` (`n x r`) and `H` (`r x m`) with `V ~= W H`, refined by the Lee-Seung
multiplicative updates for the squared Frobenius objective:

    H <- H .* (W' V) ./ (W' W H .+ eps)
    W <- W .* (V H') ./ (W H H' .+ eps)

Determinism: the factors are seeded by a self-contained 32-bit linear congruential
generator (Numerical Recipes constants), filled row-major. The identical LCG, fill
order, and fixed iteration count are reproduced in the Python and Rust ports, so
all three agree bit-for-bit to 1e-9.
"""
module Ch139Nmf

export NMFResult, fit_nmf, transform, reconstruct, reconstruction_error

# Numerical floor added to denominators. Shared across languages.
const _EPS = 1e-10

"""The fitted state of an NMF model. `W` is `n x r`, `H` is `r x m`."""
struct NMFResult
    W::Matrix{Float64}
    H::Matrix{Float64}
    n_iter::Int
    error::Float64
end

n_components(r::NMFResult) = size(r.W, 2)
n_features(r::NMFResult) = size(r.W, 1)

function _validate_input(A::Matrix{Float64})
    (size(A, 1) >= 1 && size(A, 2) >= 1) ||
        throw(ArgumentError("V must be non-empty, got shape $(size(A))"))
    all(isfinite, A) || throw(ArgumentError("V contains non-finite values (nan or inf)"))
    all(x -> x >= 0.0, A) || throw(ArgumentError("V must be non-negative (all entries >= 0)"))
    return A
end

"""Deterministic uniform fill in `(0, 1]` via a 32-bit LCG, row-major.

Entry `(i, j)` is the `((i-1) * cols + j)`-th draw, matching the Python/Rust
initializers exactly."""
function _seeded_uniform(rows::Int, cols::Int, seed::Integer)
    out = Matrix{Float64}(undef, rows, cols)
    state = UInt64(seed) & 0xFFFFFFFF
    for i in 1:rows
        for j in 1:cols
            state = (UInt64(1664525) * state + UInt64(1013904223)) & 0xFFFFFFFF
            out[i, j] = (Float64(state) + 1.0) / 4294967297.0
        end
    end
    return out
end

"""
    fit_nmf(V, n_components; max_iter=200, seed=0)

Factor the non-negative `n x m` matrix `V` as `W H` via multiplicative updates.
`n_components` is the rank `r` in `[1, min(n, m)]`.
"""
function fit_nmf(V, n_components::Integer; max_iter::Integer=200, seed::Integer=0)
    A = _validate_input(Matrix{Float64}(V))
    n, m = size(A)
    max_components = min(n, m)
    r = Int(n_components)
    (1 <= r <= max_components) ||
        throw(ArgumentError("n_components must be in [1, $max_components] for a $(n)x$(m) matrix, got $r"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be a positive integer, got $max_iter"))
    seed >= 0 || throw(ArgumentError("seed must be a non-negative integer, got $seed"))

    W = _seeded_uniform(n, r, seed)
    H = _seeded_uniform(r, m, seed + 1)

    for _ in 1:max_iter
        # Update H: H .*= (W' V) ./ (W' W H).
        Wt = transpose(W)
        WtV = Wt * A
        WtWH = (Wt * W) * H
        H = H .* (WtV ./ (WtWH .+ _EPS))
        # Update W: W .*= (V H') ./ (W H H').
        Ht = transpose(H)
        VHt = A * Ht
        WHHt = W * (H * Ht)
        W = W .* (VHt ./ (WHHt .+ _EPS))
    end

    err = sqrt(sum(abs2, A .- W * H))
    return NMFResult(W, H, Int(max_iter), err)
end

"""Encode new data under the fixed model basis `W`, returning `H` (`r x m`)."""
function transform(model::NMFResult, V; max_iter::Integer=200)
    A = _validate_input(Matrix{Float64}(V))
    size(A, 1) == n_features(model) ||
        throw(ArgumentError("V has $(size(A, 1)) features but model basis has $(n_features(model))"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be a positive integer, got $max_iter"))

    r = n_components(model)
    m = size(A, 2)
    W = model.W
    H = _seeded_uniform(r, m, 1)
    Wt = transpose(W)
    WtV = Wt * A
    WtW = Wt * W
    for _ in 1:max_iter
        H = H .* (WtV ./ (WtW * H .+ _EPS))
    end
    return H
end

"""Return the low-rank reconstruction `W H` of the fitted model."""
reconstruct(model::NMFResult) = model.W * model.H

"""Frobenius reconstruction error `||V - W H||_F` for given factors."""
function reconstruction_error(V, W, H)
    Va = _validate_input(Matrix{Float64}(V))
    Wa = Matrix{Float64}(W)
    Ha = Matrix{Float64}(H)
    size(Wa, 1) == size(Va, 1) ||
        throw(ArgumentError("W has $(size(Wa, 1)) rows but V has $(size(Va, 1))"))
    size(Ha, 2) == size(Va, 2) ||
        throw(ArgumentError("H has $(size(Ha, 2)) columns but V has $(size(Va, 2))"))
    size(Wa, 2) == size(Ha, 1) ||
        throw(ArgumentError("inner dimensions disagree: W is $(size(Wa)), H is $(size(Ha))"))
    return sqrt(sum(abs2, Va .- Wa * Ha))
end

end # module Ch139Nmf
