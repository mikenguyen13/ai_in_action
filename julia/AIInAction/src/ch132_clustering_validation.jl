"""
    Ch132ClusteringValidation

Clustering validation metrics (Chapter 132): silhouette coefficient,
Davies-Bouldin index, Calinski-Harabasz index, and the adjusted Rand index.

From-scratch implementations mirroring the Python module
`aiinaction.ch132_clustering_validation` and the Rust module
`aiinaction::ch132_clustering_validation`. The shared fixtures in
`test/test_ch132_clustering_validation.jl` match the Python/Rust suites.

Points are rows of a matrix `X` of size `(n_samples, n_features)`; labels are an
integer vector of length `n_samples`. Label values are arbitrary; only the
induced partition matters.
"""
module Ch132ClusteringValidation

export silhouette_score, davies_bouldin_index, calinski_harabasz_index,
    adjusted_rand_index

function _check_labels(X::AbstractMatrix{<:Real}, labels::AbstractVector{<:Integer})
    n = size(X, 1)
    n == 0 && throw(ArgumentError("X must contain at least one sample"))
    length(labels) == n ||
        throw(ArgumentError("length mismatch: len(labels)=$(length(labels)) != n_samples=$n"))
    uniq = sort(unique(labels))
    k = length(uniq)
    k < 2 && throw(ArgumentError("need at least 2 clusters, got $k"))
    k > n && throw(ArgumentError("number of clusters ($k) cannot exceed number of samples ($n)"))
    return uniq
end

_euclidean(a::AbstractVector, b::AbstractVector) = sqrt(sum((a .- b) .^ 2))

"""Mean silhouette coefficient over all samples (Euclidean distance)."""
function silhouette_score(X::AbstractMatrix{<:Real}, labels::AbstractVector{<:Integer})
    uniq = _check_labels(X, labels)
    n = size(X, 1)

    # Pairwise distance matrix.
    dist = zeros(Float64, n, n)
    for i in 1:n, j in (i + 1):n
        d = _euclidean(view(X, i, :), view(X, j, :))
        dist[i, j] = d
        dist[j, i] = d
    end

    members = Dict(c => findall(==(c), labels) for c in uniq)

    total = 0.0
    for i in 1:n
        own = labels[i]
        own_members = members[own]
        if length(own_members) <= 1
            continue  # s = 0
        end
        a_i = sum(dist[i, j] for j in own_members) / (length(own_members) - 1)
        b_i = Inf
        for c in uniq
            c == own && continue
            mem = members[c]
            mean_to_c = sum(dist[i, j] for j in mem) / length(mem)
            b_i = min(b_i, mean_to_c)
        end
        denom = max(a_i, b_i)
        if denom != 0.0
            total += (b_i - a_i) / denom
        end
    end
    return total / n
end

"""Davies-Bouldin index (lower is better; 0 is ideal)."""
function davies_bouldin_index(X::AbstractMatrix{<:Real}, labels::AbstractVector{<:Integer})
    uniq = _check_labels(X, labels)
    k = length(uniq)
    dim = size(X, 2)

    centroids = zeros(Float64, k, dim)
    scatter = zeros(Float64, k)
    for (j, c) in enumerate(uniq)
        idx = findall(==(c), labels)
        pts = X[idx, :]
        mu = vec(sum(pts, dims = 1)) ./ length(idx)
        centroids[j, :] = mu
        scatter[j] = sum(_euclidean(view(pts, r, :), mu) for r in 1:length(idx)) / length(idx)
    end

    total = 0.0
    for j in 1:k
        worst = 0.0
        for m in 1:k
            m == j && continue
            sep = _euclidean(view(centroids, j, :), view(centroids, m, :))
            sep == 0.0 && throw(ArgumentError(
                "clusters $(uniq[j]) and $(uniq[m]) have identical centroids; " *
                "Davies-Bouldin is undefined (zero separation)"))
            ratio = (scatter[j] + scatter[m]) / sep
            worst = max(worst, ratio)
        end
        total += worst
    end
    return total / k
end

"""Calinski-Harabasz index, a.k.a. variance ratio criterion (higher is better)."""
function calinski_harabasz_index(X::AbstractMatrix{<:Real}, labels::AbstractVector{<:Integer})
    uniq = _check_labels(X, labels)
    n = size(X, 1)
    k = length(uniq)
    grand = vec(sum(X, dims = 1)) ./ n

    between = 0.0
    within = 0.0
    for c in uniq
        idx = findall(==(c), labels)
        pts = X[idx, :]
        mu = vec(sum(pts, dims = 1)) ./ length(idx)
        between += length(idx) * sum((mu .- grand) .^ 2)
        for r in 1:length(idx)
            within += sum((view(pts, r, :) .- mu) .^ 2)
        end
    end
    within == 0.0 &&
        throw(ArgumentError("within-cluster scatter is zero; Calinski-Harabasz is undefined"))
    return (between / within) * (n - k) / (k - 1)
end

"""Adjusted Rand Index between two labelings (external, chance-corrected)."""
function adjusted_rand_index(labels_true::AbstractVector{<:Integer},
                             labels_pred::AbstractVector{<:Integer})
    length(labels_true) == length(labels_pred) || throw(ArgumentError(
        "length mismatch: len(labels_true)=$(length(labels_true)) != " *
        "len(labels_pred)=$(length(labels_pred))"))
    n = length(labels_true)
    n == 0 && throw(ArgumentError("inputs must be non-empty"))

    rows = sort(unique(labels_true))
    cols = sort(unique(labels_pred))
    row_of = Dict(v => i for (i, v) in enumerate(rows))
    col_of = Dict(v => i for (i, v) in enumerate(cols))

    table = zeros(Int, length(rows), length(cols))
    for i in 1:n
        table[row_of[labels_true[i]], col_of[labels_pred[i]]] += 1
    end

    comb2(x) = x * (x - 1) / 2.0

    sum_ij = sum(comb2(c) for c in table)
    sum_a = sum(comb2(s) for s in sum(table, dims = 2))
    sum_b = sum(comb2(s) for s in sum(table, dims = 1))
    total_pairs = n * (n - 1) / 2.0

    expected = total_pairs > 0 ? sum_a * sum_b / total_pairs : 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    denom == 0.0 && return 1.0
    return (sum_ij - expected) / denom
end

end # module Ch132ClusteringValidation
