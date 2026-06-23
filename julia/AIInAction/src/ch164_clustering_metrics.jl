"""
    Ch164ClusteringMetrics

External clustering-comparison metrics (Chapter 164, Julia).

Mirrors the Python module `aiinaction.ch164_clustering_metrics` and the Rust
module `aiinaction::ch164_clustering_metrics`. The shared fixtures in
`test/test_ch164_clustering_metrics.jl` match the Python/Rust suites (1e-9
tolerance), which keeps the three implementations at parity.

Implemented here: the contingency table, Shannon entropy, mutual information,
normalized mutual information (four averaging methods), homogeneity, completeness,
the V-measure, and the Fowlkes-Mallows index. The silhouette coefficient and the
adjusted Rand index already live in `AIInAction.Ch132ClusteringValidation` and are
not duplicated here.

Labels are integer vectors of equal length; only the induced partition matters.
All logarithms are natural (nats); the base cancels in every reported ratio.
"""
module Ch164ClusteringMetrics

export contingency_matrix, entropy, mutual_information,
    normalized_mutual_information, homogeneity, completeness, v_measure,
    fowlkes_mallows_index

function _validate(labels_true, labels_pred)
    length(labels_true) == length(labels_pred) ||
        throw(ArgumentError("length mismatch: $(length(labels_true)) != $(length(labels_pred))"))
    isempty(labels_true) && throw(ArgumentError("inputs must be non-empty"))
    return nothing
end

# Maps each distinct label to a contiguous 1-based index in ascending order.
function _index_map(labels)
    keys = sort(unique(labels))
    return Dict(v => i for (i, v) in enumerate(keys))
end

"""
    contingency_matrix(labels_true, labels_pred)

Contingency table `n_ij = |U_i ∩ V_j|`. Rows index the sorted distinct values of
`labels_true`, columns those of `labels_pred`.
"""
function contingency_matrix(labels_true, labels_pred)
    _validate(labels_true, labels_pred)
    row_idx = _index_map(labels_true)
    col_idx = _index_map(labels_pred)
    table = zeros(Int, length(row_idx), length(col_idx))
    for (a, b) in zip(labels_true, labels_pred)
        table[row_idx[a], col_idx[b]] += 1
    end
    return table
end

"""Shannon entropy (in nats) of the partition induced by `labels`."""
function entropy(labels)
    isempty(labels) && throw(ArgumentError("inputs must be non-empty"))
    n = length(labels)
    counts = Dict{Any,Int}()
    for v in labels
        counts[v] = get(counts, v, 0) + 1
    end
    h = 0.0
    for c in values(counts)
        p = c / n
        h -= p * log(p)
    end
    return h
end

"""Mutual information `I(U; V)` (in nats) between two labelings."""
function mutual_information(labels_true, labels_pred)
    table = contingency_matrix(labels_true, labels_pred)
    n = float(length(labels_true))
    a = vec(sum(table; dims=2))  # true class sizes
    b = vec(sum(table; dims=1))  # predicted cluster sizes

    mi = 0.0
    rows, cols = size(table)
    for i in 1:rows, j in 1:cols
        nij = float(table[i, j])
        nij == 0.0 && continue
        mi += (nij / n) * log((n * nij) / (a[i] * b[j]))
    end
    return max(mi, 0.0)  # MI is provably nonnegative.
end

"""
    normalized_mutual_information(labels_true, labels_pred; average_method=:arithmetic)

Normalized mutual information `I(U; V) / mean(H(U), H(V))` in `[0, 1]`. The
`average_method` is one of `:arithmetic`, `:geometric`, `:min`, `:max`. When both
labelings are trivial (a single cluster each) the result is defined to be 1.0.
"""
function normalized_mutual_information(labels_true, labels_pred; average_method::Symbol=:arithmetic)
    h_true = entropy(labels_true)
    h_pred = entropy(labels_pred)
    mi = mutual_information(labels_true, labels_pred)

    denom = if average_method === :arithmetic
        (h_true + h_pred) / 2
    elseif average_method === :geometric
        sqrt(h_true * h_pred)
    elseif average_method === :min
        min(h_true, h_pred)
    elseif average_method === :max
        max(h_true, h_pred)
    else
        throw(ArgumentError("average_method must be one of :arithmetic, :geometric, :min, :max; got $(average_method)"))
    end

    denom == 0.0 && return 1.0
    return mi / denom
end

function _homogeneity_completeness(labels_true, labels_pred)
    h_true = entropy(labels_true)
    h_pred = entropy(labels_pred)
    mi = mutual_information(labels_true, labels_pred)
    homog = h_true == 0.0 ? 1.0 : mi / h_true
    compl = h_pred == 0.0 ? 1.0 : mi / h_pred
    return homog, compl
end

"""Homogeneity: `1 - H(true | pred) / H(true) = I(U; V) / H(true)` in `[0, 1]`."""
homogeneity(labels_true, labels_pred) = _homogeneity_completeness(labels_true, labels_pred)[1]

"""Completeness: `1 - H(pred | true) / H(pred) = I(U; V) / H(pred)` in `[0, 1]`."""
completeness(labels_true, labels_pred) = _homogeneity_completeness(labels_true, labels_pred)[2]

"""
    v_measure(labels_true, labels_pred; beta=1.0)

V-measure: weighted harmonic mean `(1 + beta) h c / (beta h + c)` of homogeneity
`h` and completeness `c`. Returns 0.0 when both components are 0.
"""
function v_measure(labels_true, labels_pred; beta::Real=1.0)
    beta < 0 && throw(ArgumentError("beta must be nonnegative, got $beta"))
    h, c = _homogeneity_completeness(labels_true, labels_pred)
    denom = beta * h + c
    denom == 0.0 && return 0.0
    return (1 + beta) * h * c / denom
end

"""
    fowlkes_mallows_index(labels_true, labels_pred)

Fowlkes-Mallows index `a / sqrt((a + b)(a + c))`, the geometric mean of pairwise
precision and recall. Returns 0.0 when no pair is co-clustered in either partition.
"""
function fowlkes_mallows_index(labels_true, labels_pred)
    table = contingency_matrix(labels_true, labels_pred)
    comb2(x) = x * (x - 1) / 2

    a = sum(comb2(float(table[i, j])) for i in 1:size(table, 1), j in 1:size(table, 2))
    a_plus_b = sum(comb2(float(s)) for s in vec(sum(table; dims=1)))  # cluster sizes
    a_plus_c = sum(comb2(float(s)) for s in vec(sum(table; dims=2)))  # class sizes

    denom = a_plus_b * a_plus_c
    denom == 0.0 && return 0.0
    return a / sqrt(denom)
end

end # module Ch164ClusteringMetrics
