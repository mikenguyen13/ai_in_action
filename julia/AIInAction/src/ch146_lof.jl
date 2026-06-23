"""
    Ch146Lof

Local Outlier Factor (LOF) from scratch (Julia).

Mirrors the Python module `aiinaction.ch146_lof` and the Rust module
`aiinaction::ch146_lof`. The shared fixtures in `test/test_ch146_lof.jl` match the
Python/Rust suites, which keeps the three implementations at parity.

This is a Base-only implementation of the Local Outlier Factor of Breunig,
Kriegel, Ng, and Sander (2000). Each point is scored by how much sparser its local
neighborhood is than the neighborhoods of its own `k` nearest neighbors. Distance
ties are broken by ascending point index, so neighborhoods are deterministic and
identical across the three languages. Input matrices are `n x d` (one point per
row), matching the Python and Rust conventions.
"""
module Ch146Lof

export euclidean, knn_distances, k_distance, lrd, lof_scores, top_anomalies

function _as_matrix(X)
    A = Matrix{Float64}(X)
    n, d = size(A)
    n >= 1 || throw(ArgumentError("X must have at least one row"))
    d >= 1 || throw(ArgumentError("X must have at least one feature"))
    all(isfinite, A) || throw(ArgumentError("X contains non-finite values (nan or inf)"))
    return A
end

"""Euclidean (L2) distance between two equal-length vectors."""
function euclidean(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    length(a) == length(b) ||
        throw(ArgumentError("length mismatch: $(length(a)) != $(length(b))"))
    return sqrt(sum((Float64(x) - Float64(y))^2 for (x, y) in zip(a, b)))
end

function _check_k(k::Integer, n::Integer)
    (1 <= k <= n - 1) ||
        throw(ArgumentError("k must be in [1, $(n - 1)] for $n points, got $k"))
    return nothing
end

function _pairwise(A::Matrix{Float64})
    n = size(A, 1)
    dist = zeros(Float64, n, n)
    @inbounds for i in 1:n, j in (i + 1):n
        d = euclidean(@view(A[i, :]), @view(A[j, :]))
        dist[i, j] = d
        dist[j, i] = d
    end
    return dist
end

"""Indices (1-based) of the `k` nearest neighbors of point `i`, ties by index."""
function _neighbors(dist::Matrix{Float64}, i::Int, k::Int, n::Int)
    others = [j for j in 1:n if j != i]
    sort!(others; by=j -> (dist[i, j], j))
    return others[1:k]
end

"""Return, for each point, the 1-based indices of its `k` nearest neighbors."""
function knn_distances(X, k::Integer)
    A = _as_matrix(X)
    n = size(A, 1)
    _check_k(k, n)
    dist = _pairwise(A)
    return [_neighbors(dist, i, Int(k), n) for i in 1:n]
end

"""Return the `k`-distance (distance to the `k`-th nearest neighbor) of each point."""
function k_distance(X, k::Integer)
    A = _as_matrix(X)
    n = size(A, 1)
    _check_k(k, n)
    dist = _pairwise(A)
    return [dist[i, _neighbors(dist, i, Int(k), n)[end]] for i in 1:n]
end

function _lrd_from(dist::Matrix{Float64}, neighbors::Vector{Vector{Int}}, kdist::Vector{Float64})
    n = size(dist, 1)
    lrd_vals = zeros(Float64, n)
    @inbounds for i in 1:n
        nbrs = neighbors[i]
        total = 0.0
        for y in nbrs
            total += max(kdist[y], dist[i, y])
        end
        mean_reach = total / length(nbrs)
        lrd_vals[i] = mean_reach == 0.0 ? Inf : 1.0 / mean_reach
    end
    return lrd_vals
end

"""Local reachability density of each point."""
function lrd(X, k::Integer)
    A = _as_matrix(X)
    n = size(A, 1)
    _check_k(k, n)
    dist = _pairwise(A)
    neighbors = [_neighbors(dist, i, Int(k), n) for i in 1:n]
    kdist = [dist[i, neighbors[i][end]] for i in 1:n]
    return _lrd_from(dist, neighbors, kdist)
end

"""
    lof_scores(X, k)

Local Outlier Factor score of every point in the `n x d` matrix `X`. Values near
`1` are inliers; values much greater than `1` are local outliers.
"""
function lof_scores(X, k::Integer)
    A = _as_matrix(X)
    n = size(A, 1)
    _check_k(k, n)
    dist = _pairwise(A)
    neighbors = [_neighbors(dist, i, Int(k), n) for i in 1:n]
    kdist = [dist[i, neighbors[i][end]] for i in 1:n]
    lrd_vals = _lrd_from(dist, neighbors, kdist)

    scores = zeros(Float64, n)
    @inbounds for i in 1:n
        nbrs = neighbors[i]
        li = lrd_vals[i]
        if isinf(li)
            scores[i] = 0.0
        else
            total = 0.0
            for y in nbrs
                ly = lrd_vals[y]
                total += isinf(ly) ? Inf : ly / li
            end
            scores[i] = total / length(nbrs)
        end
    end
    return scores
end

"""
    top_anomalies(X, k, m)

1-based indices of the `m` highest-LOF points, most anomalous first. Ties in score
are broken by ascending point index.
"""
function top_anomalies(X, k::Integer, m::Integer)
    m >= 1 || throw(ArgumentError("m must be a positive integer, got $m"))
    scores = lof_scores(X, k)
    n = length(scores)
    m <= n || throw(ArgumentError("m must be in [1, $n] for $n points, got $m"))
    order = sortperm(1:n; by=i -> (-scores[i], i))
    return order[1:m]
end

end # module Ch146Lof
