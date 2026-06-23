"""
    Ch162RankingMetrics

Ranking metrics from scratch: MRR, MAP, and NDCG (Julia).

Mirrors the Python module `aiinaction.ch162_ranking_metrics` and the Rust module
`aiinaction::ch162_ranking_metrics`. The shared fixtures in
`test/test_ch162_ranking_metrics.jl` match the Python/Rust suites, which keeps
the three implementations at parity. Base-only (no extra dependencies).

Conventions: a *relevance list* gives the per-position relevance of items in
ranked order (position 1 is the top). An item is a hit when its relevance is
strictly positive. The optional cutoff `k` (`nothing` means no cutoff) restricts
to the top `k` positions. Reciprocal rank, average precision, and NDCG are all
`0` for a query with no relevant item.
"""
module Ch162RankingMetrics

export reciprocal_rank, mean_reciprocal_rank, precision_at_k,
    average_precision, mean_average_precision, dcg, ndcg, mean_ndcg

function _validate_rel(relevances)
    for r in relevances
        isfinite(r) || throw(ArgumentError("relevances must be finite"))
        r >= 0 || throw(ArgumentError("relevances must be non-negative"))
    end
    return nothing
end

function _cutoff(len::Int, k)
    k === nothing && return len
    k < 1 && throw(ArgumentError("k must be a positive integer or nothing, got $k"))
    return min(Int(k), len)
end

"""Reciprocal rank of the first relevant item within the top `k`, or `0.0`."""
function reciprocal_rank(relevances; k=nothing)
    _validate_rel(relevances)
    cut = _cutoff(length(relevances), k)
    @inbounds for i in 1:cut
        if relevances[i] > 0
            return 1.0 / i
        end
    end
    return 0.0
end

"""Mean reciprocal rank over a non-empty set of queries."""
function mean_reciprocal_rank(queries; k=nothing)
    isempty(queries) && throw(ArgumentError("queries must be non-empty"))
    return sum(reciprocal_rank(q; k=k) for q in queries) / length(queries)
end

"""Precision at cutoff `k`: fraction of the top `k` items that are relevant.

`k` may exceed the list length; the denominator is still `k`."""
function precision_at_k(relevances, k::Integer)
    k >= 1 || throw(ArgumentError("k must be a positive integer, got $k"))
    _validate_rel(relevances)
    cut = min(Int(k), length(relevances))
    hits = count(i -> relevances[i] > 0, 1:cut)
    return hits / k
end

"""Average precision for a single query under binary relevance.

`n_relevant === nothing` infers `R` from the positive entries; otherwise `R` must
be at least the number of hits observed within the cutoff. Returns `0.0` for
`R == 0`."""
function average_precision(relevances; n_relevant=nothing, k=nothing)
    _validate_rel(relevances)
    cut = _cutoff(length(relevances), k)

    observed_hits = count(i -> relevances[i] > 0, 1:cut)
    if n_relevant === nothing
        R = count(r -> r > 0, relevances)
    else
        n_relevant >= 0 || throw(ArgumentError("n_relevant must be non-negative, got $n_relevant"))
        n_relevant < observed_hits &&
            throw(ArgumentError("n_relevant=$n_relevant is smaller than the $observed_hits relevant items observed"))
        R = Int(n_relevant)
    end
    R == 0 && return 0.0

    hits = 0
    score = 0.0
    @inbounds for i in 1:cut
        if relevances[i] > 0
            hits += 1
            score += hits / i
        end
    end
    return score / R
end

"""Mean average precision over a non-empty set of queries.

`n_relevant`, when not `nothing`, must supply one `R` per query."""
function mean_average_precision(queries; n_relevant=nothing, k=nothing)
    isempty(queries) && throw(ArgumentError("queries must be non-empty"))
    if n_relevant !== nothing && length(n_relevant) != length(queries)
        throw(ArgumentError("n_relevant has length $(length(n_relevant)) but there are $(length(queries)) queries"))
    end
    total = 0.0
    for (idx, q) in enumerate(queries)
        R = n_relevant === nothing ? nothing : n_relevant[idx]
        total += average_precision(q; n_relevant=R, k=k)
    end
    return total / length(queries)
end

"""Discounted cumulative gain at cutoff `k`.

`exponential=true` uses gain `2^rel - 1`; `false` uses raw `rel`. The discount at
rank `j` (1-based) is `1 / log2(j + 1)`."""
function dcg(relevances; k=nothing, exponential::Bool=true)
    _validate_rel(relevances)
    cut = _cutoff(length(relevances), k)
    total = 0.0
    @inbounds for j in 1:cut
        gain = exponential ? (2.0^relevances[j] - 1.0) : float(relevances[j])
        total += gain / log2(j + 1)
    end
    return total
end

"""Normalized discounted cumulative gain at cutoff `k`.

Divides `dcg` by the ideal DCG (relevances sorted descending). Returns `0.0` when
the ideal DCG is `0`. Result lies in `[0, 1]`."""
function ndcg(relevances; k=nothing, exponential::Bool=true)
    actual = dcg(relevances; k=k, exponential=exponential)
    ideal = dcg(sort(collect(relevances); rev=true); k=k, exponential=exponential)
    ideal == 0 && return 0.0
    return actual / ideal
end

"""Mean NDCG over a non-empty set of queries."""
function mean_ndcg(queries; k=nothing, exponential::Bool=true)
    isempty(queries) && throw(ArgumentError("queries must be non-empty"))
    return sum(ndcg(q; k=k, exponential=exponential) for q in queries) / length(queries)
end

end # module Ch162RankingMetrics
