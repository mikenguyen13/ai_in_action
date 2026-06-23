"""
    Ch125AgglomerativeClustering

Agglomerative hierarchical clustering from scratch (Julia). Mirrors the Python
module `aiinaction.ch125_agglomerative_clustering` and the Rust module
`aiinaction::ch125_agglomerative_clustering`. The shared fixtures in
`test/test_ch125_agglomerative_clustering.jl` match the Python/Rust suites to keep
the three at parity.

`linkage_matrix` returns SciPy-style rows `[node_a, node_b, height, size]`:
original points have ids `0..n-1` and the merge formed at step `t` (0-indexed)
has id `n + t`. Supported linkages: "single", "complete", "average", "ward".
"""
module Ch125AgglomerativeClustering

export LINKAGES, linkage_matrix, fcluster, cophenetic_distances

const LINKAGES = ("single", "complete", "average", "ward")

function _lance_williams(d_ak, d_bk, d_ab, n_a, n_b, n_k, linkage)
    if linkage == "single"
        return 0.5 * d_ak + 0.5 * d_bk - 0.5 * abs(d_ak - d_bk)
    elseif linkage == "complete"
        return 0.5 * d_ak + 0.5 * d_bk + 0.5 * abs(d_ak - d_bk)
    elseif linkage == "average"
        total = n_a + n_b
        return (n_a / total) * d_ak + (n_b / total) * d_bk
    elseif linkage == "ward"
        total = n_a + n_b + n_k
        return (n_a + n_k) / total * d_ak + (n_b + n_k) / total * d_bk -
               n_k / total * d_ab
    else
        throw(ArgumentError("unknown linkage $(linkage); expected one of $(LINKAGES)"))
    end
end

function _validate(points::AbstractVector)
    isempty(points) && throw(ArgumentError("inputs must be non-empty"))
    width = length(points[1])
    width == 0 && throw(ArgumentError("points must have at least one feature"))
    for row in points
        length(row) == width ||
            throw(ArgumentError("all points must have the same number of features"))
    end
    return width
end

function _pairwise_sq_euclidean(points::AbstractVector)
    n = length(points)
    dist = zeros(Float64, n, n)
    for i in 1:n, j in (i+1):n
        s = sum((a - b)^2 for (a, b) in zip(points[i], points[j]))
        dist[i, j] = s
        dist[j, i] = s
    end
    return dist
end

"""
    linkage_matrix(points, linkage="ward")

Agglomerative clustering of `points` (a vector of coordinate vectors). Returns a
vector of `n - 1` rows `[node_a, node_b, height, size]` with `node_a < node_b`,
using 0-based node ids to match the Python/Rust output. For Ward the height is
the Euclidean (non-squared) merge distance.
"""
function linkage_matrix(points::AbstractVector, linkage::AbstractString = "ward")
    linkage in LINKAGES ||
        throw(ArgumentError("unknown linkage $(linkage); expected one of $(LINKAGES)"))
    _validate(points)
    n = length(points)
    n < 2 && throw(ArgumentError("need at least 2 points to cluster"))

    # Ward works in squared-Euclidean space; the others use raw distances. We use
    # a dictionary keyed by 0-based cluster id so the matrix can grow past n.
    sq = _pairwise_sq_euclidean(points)
    dist = Dict{Tuple{Int,Int},Float64}()
    for i in 1:n, j in 1:n
        if i != j
            v = linkage == "ward" ? sq[i, j] : sqrt(sq[i, j])
            dist[(i - 1, j - 1)] = v
        end
    end

    active = collect(0:(n - 1))
    sizes = Dict(i => 1 for i in 0:(n - 1))
    next_id = n
    result = Vector{Vector{Float64}}()

    for _ in 1:(n - 1)
        best = Inf
        bi = -1
        bj = -1
        for a_idx in 1:length(active), b_idx in (a_idx+1):length(active)
            d = dist[(active[a_idx], active[b_idx])]
            if d < best
                best = d
                bi, bj = a_idx, b_idx
            end
        end
        ca, cb = active[bi], active[bj]
        n_a, n_b = sizes[ca], sizes[cb]

        height = linkage == "ward" ? sqrt(best) : best
        node_a, node_b = ca < cb ? (ca, cb) : (cb, ca)
        push!(result, [Float64(node_a), Float64(node_b), height, Float64(n_a + n_b)])

        new_id = next_id
        next_id += 1
        for ck in active
            (ck == ca || ck == cb) && continue
            d_new = _lance_williams(
                dist[(ca, ck)], dist[(cb, ck)], best, n_a, n_b, sizes[ck], linkage,
            )
            dist[(new_id, ck)] = d_new
            dist[(ck, new_id)] = d_new
        end

        active = filter(c -> c != ca && c != cb, active)
        push!(active, new_id)
        sizes[new_id] = n_a + n_b
    end

    return result
end

"""
    fcluster(linkage_mat, n_clusters)

Cut the tree to obtain exactly `n_clusters` flat clusters. Returns a vector of
0-based labels, assigned in increasing order of the smallest original-point id in
each cluster.
"""
function fcluster(linkage_mat::AbstractVector, n_clusters::Integer)
    n = length(linkage_mat) + 1
    (1 <= n_clusters <= n) ||
        throw(ArgumentError("n_clusters must be in 1..$(n), got $(n_clusters)"))

    members = Dict{Int,Vector{Int}}(i => [i] for i in 0:(n - 1))
    merges_to_apply = n - n_clusters
    for t in 0:(merges_to_apply - 1)
        a = Int(linkage_mat[t + 1][1])
        b = Int(linkage_mat[t + 1][2])
        members[n + t] = vcat(members[a], members[b])
        delete!(members, a)
        delete!(members, b)
    end

    clusters = sort(collect(values(members)), by = minimum)
    labels = zeros(Int, n)
    for (cid, group) in enumerate(clusters)
        for p in group
            labels[p + 1] = cid - 1
        end
    end
    return labels
end

"""
    cophenetic_distances(linkage_mat)

Cophenetic distance matrix induced by a linkage matrix: a symmetric `n x n` matrix
whose `(i, j)` entry is the height of the merge at which points `i` and `j` first
share a cluster.
"""
function cophenetic_distances(linkage_mat::AbstractVector)
    n = length(linkage_mat) + 1
    members = Dict{Int,Vector{Int}}(i => [i] for i in 0:(n - 1))
    coph = zeros(Float64, n, n)
    for (t, row) in enumerate(linkage_mat)
        a = Int(row[1])
        b = Int(row[2])
        height = Float64(row[3])
        ma = members[a]
        mb = members[b]
        for x in ma, y in mb
            coph[x + 1, y + 1] = height
            coph[y + 1, x + 1] = height
        end
        members[n + (t - 1)] = vcat(ma, mb)
    end
    return coph
end

end # module Ch125AgglomerativeClustering
