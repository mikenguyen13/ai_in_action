"""
    Ch124KMeansVariants

K-Means variants and extensions (Chapter 124). Mirrors the Python module
`aiinaction.ch124_kmeans_variants` and the Rust module
`aiinaction::ch124_kmeans_variants`. The shared fixtures in
`test/test_ch124_kmeans_variants.jl` match the Python/Rust suites to keep the
three implementations at parity.

Every routine is deterministic given its inputs; initialization is supplied by
the caller as explicit centroids or indices.
"""
module Ch124KMeansVariants

export lloyd_step, inertia, mini_batch_update, kmedians_centroid, pam_assign_cost,
    rbf_kernel_matrix, kernel_assignment_distances, fuzzy_centroids, fuzzy_memberships,
    bisecting_split

_sqdist(a, b) = sum((a .- b) .^ 2)

function _check_matrix(X::AbstractMatrix, name::AbstractString)
    size(X, 1) == 0 && throw(ArgumentError("$name must contain at least one sample"))
    size(X, 2) == 0 && throw(ArgumentError("$name must have at least one feature"))
    return size(X, 2)
end

"""
    lloyd_step(X, centroids) -> (labels, new_centroids)

One iteration of Lloyd's algorithm. `X` is `n x d`, `centroids` is `k x d`.
Assigns to the nearest centroid (ties to the lowest index, 1-based labels) then
recomputes each centroid as the mean of its members; an empty centroid is left
unchanged.
"""
function lloyd_step(X::AbstractMatrix, centroids::AbstractMatrix)
    d = _check_matrix(X, "X")
    k = _check_matrix(centroids, "centroids")
    k == d || throw(ArgumentError("centroid dimension $k does not match data dimension $d"))
    n = size(X, 1)
    kc = size(centroids, 1)
    labels = Vector{Int}(undef, n)
    for i in 1:n
        best, bestd = 1, Inf
        for j in 1:kc
            dd = _sqdist(view(X, i, :), view(centroids, j, :))
            if dd < bestd
                bestd, best = dd, j
            end
        end
        labels[i] = best
    end
    new_c = copy(Matrix{Float64}(centroids))
    for j in 1:kc
        idx = findall(==(j), labels)
        if !isempty(idx)
            new_c[j, :] = vec(sum(view(X, idx, :), dims=1)) ./ length(idx)
        end
    end
    return labels, new_c
end

"""Within-cluster sum of squared distances to the nearest centroid."""
function inertia(X::AbstractMatrix, centroids::AbstractMatrix)
    d = _check_matrix(X, "X")
    k = _check_matrix(centroids, "centroids")
    k == d || throw(ArgumentError("centroid dimension $k does not match data dimension $d"))
    total = 0.0
    for i in 1:size(X, 1)
        best = Inf
        for j in 1:size(centroids, 1)
            dd = _sqdist(view(X, i, :), view(centroids, j, :))
            dd < best && (best = dd)
        end
        total += best
    end
    return total
end

"""
    mini_batch_update(centroids, counts, batch) -> (new_centroids, new_counts)

One mini-batch K-Means update (Sculley, 2010). Points are processed in order with
per-centroid learning rate `1 / count`.
"""
function mini_batch_update(centroids::AbstractMatrix, counts::AbstractVector, batch::AbstractMatrix)
    k = _check_matrix(centroids, "centroids")
    length(counts) == size(centroids, 1) ||
        throw(ArgumentError("counts length $(length(counts)) does not match number of centroids $(size(centroids,1))"))
    any(c -> c < 0, counts) && throw(ArgumentError("counts must be non-negative"))
    bd = _check_matrix(batch, "batch")
    bd == k || throw(ArgumentError("batch dimension $bd does not match centroid dimension $k"))
    c = copy(Matrix{Float64}(centroids))
    cnt = copy(Vector{Float64}(counts))
    for i in 1:size(batch, 1)
        x = view(batch, i, :)
        best, bestd = 1, Inf
        for j in 1:size(c, 1)
            dd = _sqdist(x, view(c, j, :))
            if dd < bestd
                bestd, best = dd, j
            end
        end
        cnt[best] += 1.0
        eta = 1.0 / cnt[best]
        c[best, :] = (1.0 - eta) .* c[best, :] .+ eta .* x
    end
    return c, cnt
end

"""Coordinatewise (lower) median: the L1-optimal k-medians representative."""
function kmedians_centroid(members::AbstractMatrix)
    d = _check_matrix(members, "members")
    n = size(members, 1)
    idx = (n - 1) ÷ 2 + 1  # lower median (1-based)
    out = Vector{Float64}(undef, d)
    for t in 1:d
        out[t] = sort(collect(view(members, :, t)))[idx]
    end
    return out
end

"""
    pam_assign_cost(distances, medoid_indices) -> (labels, total_cost)

Assign each point to its nearest medoid given an `n x n` dissimilarity matrix.
`medoid_indices` and returned labels are 1-based; labels index into
`medoid_indices`.
"""
function pam_assign_cost(distances::AbstractMatrix, medoid_indices::AbstractVector{<:Integer})
    n = size(distances, 1)
    (n > 0 && size(distances, 2) == n) ||
        throw(ArgumentError("distances must be a square n x n dissimilarity matrix"))
    isempty(medoid_indices) && throw(ArgumentError("medoid_indices must be non-empty"))
    for m in medoid_indices
        (1 <= m <= n) || throw(ArgumentError("medoid index $m out of range [1, $n]"))
    end
    labels = Vector{Int}(undef, n)
    total = 0.0
    for i in 1:n
        bestpos, bestcost = 1, Inf
        for (pos, m) in enumerate(medoid_indices)
            c = distances[i, m]
            if c < bestcost
                bestcost, bestpos = c, pos
            end
        end
        labels[i] = bestpos
        total += bestcost
    end
    return labels, total
end

"""Gaussian (RBF) kernel matrix `K[i,j] = exp(-gamma ||x_i - x_j||^2)`."""
function rbf_kernel_matrix(X::AbstractMatrix, gamma::Real)
    gamma > 0 || throw(ArgumentError("gamma must be positive, got $gamma"))
    _check_matrix(X, "X")
    n = size(X, 1)
    K = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        K[i, j] = exp(-gamma * _sqdist(view(X, i, :), view(X, j, :)))
    end
    return K
end

"""
    kernel_assignment_distances(kernel, labels, n_clusters) -> Matrix

Feature-space squared distances from each point to every cluster mean via the
kernel trick. `labels` are 1-based; an empty cluster yields `Inf`.
"""
function kernel_assignment_distances(kernel::AbstractMatrix, labels::AbstractVector{<:Integer}, n_clusters::Integer)
    n = size(kernel, 1)
    (n > 0 && size(kernel, 2) == n) ||
        throw(ArgumentError("kernel must be a square n x n matrix"))
    length(labels) == n ||
        throw(ArgumentError("labels length $(length(labels)) does not match kernel size $n"))
    n_clusters > 0 || throw(ArgumentError("n_clusters must be positive"))
    for c in labels
        (1 <= c <= n_clusters) || throw(ArgumentError("label $c out of range [1, $n_clusters]"))
    end
    members = [findall(==(j), labels) for j in 1:n_clusters]
    third = [isempty(S) ? Inf : sum(kernel[S, S]) / length(S)^2 for S in members]
    out = Matrix{Float64}(undef, n, n_clusters)
    for i in 1:n, j in 1:n_clusters
        S = members[j]
        if isempty(S)
            out[i, j] = Inf
        else
            cross = sum(kernel[i, l] for l in S)
            out[i, j] = kernel[i, i] - 2.0 * cross / length(S) + third[j]
        end
    end
    return out
end

"""Membership-weighted fuzzy c-means centroids."""
function fuzzy_centroids(X::AbstractMatrix, memberships::AbstractMatrix, m::Real)
    m > 1 || throw(ArgumentError("fuzziness exponent m must be greater than 1, got $m"))
    d = _check_matrix(X, "X")
    size(memberships, 1) == size(X, 1) ||
        throw(ArgumentError("memberships must have shape (n_samples, n_clusters)"))
    k = size(memberships, 2)
    num = zeros(Float64, k, d)
    den = zeros(Float64, k)
    for i in 1:size(X, 1), j in 1:k
        w = memberships[i, j]^m
        den[j] += w
        num[j, :] .+= w .* view(X, i, :)
    end
    for j in 1:k
        den[j] == 0 &&
            throw(ArgumentError("a cluster has zero total membership; cannot form its centroid"))
        num[j, :] ./= den[j]
    end
    return num
end

"""Update fuzzy memberships from distances to all centroids (Bezdek FCM)."""
function fuzzy_memberships(X::AbstractMatrix, centroids::AbstractMatrix, m::Real)
    m > 1 || throw(ArgumentError("fuzziness exponent m must be greater than 1, got $m"))
    d = _check_matrix(X, "X")
    kd = _check_matrix(centroids, "centroids")
    kd == d || throw(ArgumentError("centroid dimension $kd does not match data dimension $d"))
    k = size(centroids, 1)
    p = 2.0 / (m - 1.0)
    out = zeros(Float64, size(X, 1), k)
    for i in 1:size(X, 1)
        dist = [sqrt(_sqdist(view(X, i, :), view(centroids, j, :))) for j in 1:k]
        zeros_idx = findall(==(0.0), dist)
        if !isempty(zeros_idx)
            for j in zeros_idx
                out[i, j] = 1.0 / length(zeros_idx)
            end
            continue
        end
        for j in 1:k
            denom = sum((dist[j] / dist[l])^p for l in 1:k)
            out[i, j] = 1.0 / denom
        end
    end
    return out
end

"""
    bisecting_split(X, init_two_centroids) -> (labels, two_centroids, sse)

Run 2-means to convergence to bisect a cluster. Labels are 1-based in {1, 2}.
"""
function bisecting_split(X::AbstractMatrix, init_two_centroids::AbstractMatrix)
    size(init_two_centroids, 1) == 2 ||
        throw(ArgumentError("bisecting requires exactly 2 initial centroids, got $(size(init_two_centroids,1))"))
    centroids = Matrix{Float64}(init_two_centroids)
    labels = Int[]
    prev = nothing
    for _ in 1:1000
        labels, centroids = lloyd_step(X, centroids)
        if prev !== nothing && prev == labels
            break
        end
        prev = copy(labels)
    end
    return labels, centroids, inertia(X, centroids)
end

end # module Ch124KMeansVariants
