"""
    Ch133ClusteringAtScale

Clustering at scale: mini-batch k-means, BIRCH clustering features, canopy
clustering, and k-means|| seeding (Julia).

Mirrors the Python module `aiinaction.ch133_clustering_at_scale` and the Rust
module `aiinaction::ch133_clustering_at_scale`. A tiny self-contained 32-bit LCG
makes sampling reproducible byte-for-byte across the three languages, so the
shared fixtures agree to 1e-9.
"""
module Ch133ClusteringAtScale

export Lcg, next_u32!, next_below!, next_unit!,
       squared_distance, nearest_centroid, inertia,
       ClusteringFeature, cf_from_points, merge_cf, centroid, radius,
       mini_batch_kmeans, canopy_clustering, kmeans_parallel_init

# --------------------------------------------------------------------------- #
# Deterministic 32-bit LCG (Numerical Recipes constants).                      #
# --------------------------------------------------------------------------- #
mutable struct Lcg
    state::UInt32
    Lcg(seed::Integer) = new(UInt32(seed & 0xFFFFFFFF))
end

const _LCG_A = UInt32(1664525)
const _LCG_C = UInt32(1013904223)

"""Advance the state and return the new 32-bit value."""
function next_u32!(rng::Lcg)
    rng.state = _LCG_A * rng.state + _LCG_C  # wraps mod 2^32 in UInt32
    return rng.state
end

"""Return an integer in `[0, bound)`."""
function next_below!(rng::Lcg, bound::Integer)
    bound > 0 || throw(ArgumentError("bound must be positive, got $bound"))
    return Int(next_u32!(rng) % UInt32(bound))
end

"""Uniform draw in `[0, 1)` from the 32-bit stream."""
next_unit!(rng::Lcg) = Float64(next_u32!(rng)) / 4294967296.0

# --------------------------------------------------------------------------- #
# Geometry helpers.                                                           #
# --------------------------------------------------------------------------- #
function _validate_matrix(points::Vector{<:Vector{<:Real}}, name::AbstractString)
    isempty(points) && throw(ArgumentError("$name must be non-empty"))
    dim = length(points[1])
    dim == 0 && throw(ArgumentError("$name rows must have at least one dimension"))
    for (i, row) in enumerate(points)
        length(row) == dim || throw(ArgumentError(
            "$name are ragged: row 1 has $dim dims but row $i has $(length(row))"))
    end
    return dim
end

"""Squared Euclidean distance between two equal-length vectors."""
function squared_distance(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    length(a) == length(b) ||
        throw(ArgumentError("dimension mismatch: $(length(a)) != $(length(b))"))
    isempty(a) && throw(ArgumentError("vectors must be non-empty"))
    return sum((x - y)^2 for (x, y) in zip(a, b))
end

"""Return `(index, squared_distance)` of the nearest centroid. Ties -> lowest index."""
function nearest_centroid(point::AbstractVector{<:Real}, centroids::Vector{<:Vector{<:Real}})
    _validate_matrix(centroids, "centroids")
    best_j = 1
    best_d = squared_distance(point, centroids[1])
    for j in 2:length(centroids)
        d = squared_distance(point, centroids[j])
        if d < best_d
            best_d = d
            best_j = j
        end
    end
    return (best_j, best_d)
end

"""Total within-cluster sum of squared distances (the k-means objective)."""
function inertia(points::Vector{<:Vector{<:Real}}, centroids::Vector{<:Vector{<:Real}})
    _validate_matrix(points, "points")
    _validate_matrix(centroids, "centroids")
    return sum(nearest_centroid(p, centroids)[2] for p in points)
end

# --------------------------------------------------------------------------- #
# BIRCH clustering feature.                                                   #
# --------------------------------------------------------------------------- #
struct ClusteringFeature
    n::Int
    ls::Vector{Float64}
    ss::Float64
    function ClusteringFeature(n::Integer, ls::AbstractVector{<:Real}, ss::Real)
        n >= 0 || throw(ArgumentError("count N must be non-negative, got $n"))
        isempty(ls) && throw(ArgumentError("linear sum LS must be non-empty"))
        return new(Int(n), Float64.(ls), Float64(ss))
    end
end

dim(cf::ClusteringFeature) = length(cf.ls)

"""Build a CF from raw points."""
function cf_from_points(points::Vector{<:Vector{<:Real}})
    d = _validate_matrix(points, "points")
    ls = zeros(Float64, d)
    ss = 0.0
    for p in points
        for i in 1:d
            ls[i] += p[i]
        end
        ss += sum(v * v for v in p)
    end
    return ClusteringFeature(length(points), ls, ss)
end

"""Return the additive union `a + b`."""
function merge_cf(a::ClusteringFeature, b::ClusteringFeature)
    dim(a) == dim(b) || throw(ArgumentError("dimension mismatch: $(dim(a)) != $(dim(b))"))
    return ClusteringFeature(a.n + b.n, a.ls .+ b.ls, a.ss + b.ss)
end

"""Centroid `LS / N`."""
function centroid(cf::ClusteringFeature)
    cf.n == 0 && throw(ArgumentError("centroid is undefined for an empty CF (N=0)"))
    return cf.ls ./ cf.n
end

"""RMS distance of members to the centroid, from `(N, LS, SS)` alone."""
function radius(cf::ClusteringFeature)
    cf.n == 0 && throw(ArgumentError("radius is undefined for an empty CF (N=0)"))
    mean_ss = cf.ss / cf.n
    centroid_norm_sq = sum(v * v for v in cf.ls) / (cf.n * cf.n)
    var = mean_ss - centroid_norm_sq
    return var > 0.0 ? sqrt(var) : 0.0
end

# --------------------------------------------------------------------------- #
# Mini-batch k-means (Sculley 2010).                                          #
# --------------------------------------------------------------------------- #
"""Run mini-batch k-means and return the final centroids."""
function mini_batch_kmeans(points::Vector{<:Vector{<:Real}},
                           centroids::Vector{<:Vector{<:Real}};
                           batch_size::Integer, n_iter::Integer, seed::Integer)
    d = _validate_matrix(points, "points")
    cd = _validate_matrix(centroids, "centroids")
    cd == d || throw(ArgumentError("centroid dim $cd does not match point dim $d"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive, got $batch_size"))
    n_iter >= 0 || throw(ArgumentError("n_iter must be non-negative, got $n_iter"))

    cs = [Float64.(c) for c in centroids]
    counts = zeros(Int, length(cs))
    rng = Lcg(seed)
    n = length(points)
    for _ in 1:n_iter
        batch = [points[next_below!(rng, n) + 1] for _ in 1:batch_size]
        assignments = [nearest_centroid(x, cs)[1] for x in batch]
        for (x, j) in zip(batch, assignments)
            counts[j] += 1
            eta = 1.0 / counts[j]
            for i in 1:d
                cs[j][i] = (1.0 - eta) * cs[j][i] + eta * x[i]
            end
        end
    end
    return cs
end

# --------------------------------------------------------------------------- #
# Canopy clustering (McCallum, Nigam, Ungar 2000).                            #
# --------------------------------------------------------------------------- #
"""Partition points into overlapping canopies using two squared-distance thresholds."""
function canopy_clustering(points::Vector{<:Vector{<:Real}};
                           t1::Real, t2::Real, seed::Integer)
    _validate_matrix(points, "points")
    (t1 >= 0 && t2 >= 0) ||
        throw(ArgumentError("thresholds must be non-negative, got t1=$t1, t2=$t2"))
    t2 <= t1 || throw(ArgumentError("require t2 <= t1, got t1=$t1, t2=$t2"))

    n = length(points)
    pool = collect(0:(n - 1))  # zero-based indices to match Python/Rust output
    rng = Lcg(seed)
    canopies = Vector{Vector{Int}}()
    while !isempty(pool)
        pick = next_below!(rng, length(pool)) + 1
        center_idx = pool[pick]
        center = points[center_idx + 1]
        members = Int[]
        survivors = Int[]
        for idx in pool
            d = squared_distance(center, points[idx + 1])
            d <= t1 && push!(members, idx)
            d > t2 && push!(survivors, idx)
        end
        push!(canopies, sort(members))
        pool = survivors
    end
    return canopies
end

# --------------------------------------------------------------------------- #
# k-means|| seeding (Bahmani et al. 2012).                                    #
# --------------------------------------------------------------------------- #
"""Seed `k` centers with the scalable k-means|| oversampling scheme."""
function kmeans_parallel_init(points::Vector{<:Vector{<:Real}}, k::Integer;
                              oversampling::Real, n_rounds::Integer, seed::Integer)
    _validate_matrix(points, "points")
    n = length(points)
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    k <= n || throw(ArgumentError("k=$k exceeds number of points n=$n"))
    oversampling > 0 || throw(ArgumentError("oversampling must be positive, got $oversampling"))
    n_rounds >= 0 || throw(ArgumentError("n_rounds must be non-negative, got $n_rounds"))

    rng = Lcg(seed)
    chosen = Int[next_below!(rng, n)]  # zero-based index

    min_sq(idx) = minimum(squared_distance(points[idx + 1], points[c + 1]) for c in chosen)

    for _ in 1:n_rounds
        dists = [min_sq(i) for i in 0:(n - 1)]
        phi = sum(dists)
        phi <= 0.0 && break
        for i in 0:(n - 1)
            i in chosen && continue
            prob = oversampling * dists[i + 1] / phi
            prob > 1.0 && (prob = 1.0)
            next_unit!(rng) < prob && push!(chosen, i)
        end
    end

    candidates = Int[]
    for c in chosen
        c in candidates || push!(candidates, c)
    end

    seeds = Int[candidates[1]]
    while length(seeds) < k
        best_idx = -1
        best_gain = -1.0
        for c in candidates
            c in seeds && continue
            d = minimum(squared_distance(points[c + 1], points[s + 1]) for s in seeds)
            if d > best_gain
                best_gain = d
                best_idx = c
            end
        end
        if best_idx < 0
            for i in 0:(n - 1)
                if !(i in seeds)
                    best_idx = i
                    break
                end
            end
        end
        push!(seeds, best_idx)
    end

    return [Float64.(points[s + 1]) for s in seeds[1:k]]
end

end # module Ch133ClusteringAtScale
