"""
    Ch163IrMetrics

Information retrieval (IR) ranking metrics from scratch (Julia).

Mirrors the Python module `aiinaction.ch163_ir_metrics` and the Rust module
`aiinaction::ch163_ir_metrics`. The shared fixtures in `test/test_ch163_ir_metrics.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

Binary ranked metrics (`precision_at_k`, `recall_at_k`, `average_precision`,
`reciprocal_rank` and the means) take a rank-ordered vector of `0/1` labels.
Graded metrics (`dcg_at_k`, `ndcg_at_k`) take a vector of non-negative integer
relevance grades. Recall and MAP additionally take `num_relevant = |R_q|`.
"""
module Ch163IrMetrics

export precision_at_k, recall_at_k, average_precision, mean_average_precision,
    dcg_at_k, ndcg_at_k, reciprocal_rank, mean_reciprocal_rank

function _check_binary(relevances::AbstractVector{<:Integer})
    isempty(relevances) && throw(ArgumentError("relevances must be non-empty"))
    for (i, r) in enumerate(relevances)
        (r == 0 || r == 1) ||
            throw(ArgumentError("relevances must be binary 0/1 labels, got $r at index $(i - 1)"))
    end
    return nothing
end

function _check_grades(grades::AbstractVector{<:Integer})
    isempty(grades) && throw(ArgumentError("grades must be non-empty"))
    for (i, g) in enumerate(grades)
        g >= 0 || throw(ArgumentError("grades must be non-negative, got $g at index $(i - 1)"))
    end
    return nothing
end

function _check_k(k::Integer, n::Integer)
    k >= 1 || throw(ArgumentError("k must be a positive integer, got $k"))
    return min(k, n)
end

"""Precision@k: fraction of the top `k` ranked items that are relevant. `k` is clamped."""
function precision_at_k(relevances::AbstractVector{<:Integer}, k::Integer)
    _check_binary(relevances)
    kk = _check_k(k, length(relevances))
    return sum(@view relevances[1:kk]) / kk
end

"""Recall@k: fraction of all `num_relevant` relevant documents in the top `k`."""
function recall_at_k(relevances::AbstractVector{<:Integer}, k::Integer, num_relevant::Integer)
    _check_binary(relevances)
    num_relevant > 0 ||
        throw(ArgumentError("num_relevant must be a positive integer, got $num_relevant"))
    found_total = sum(relevances)
    num_relevant >= found_total ||
        throw(ArgumentError("num_relevant=$num_relevant is smaller than the $found_total relevant labels present"))
    kk = _check_k(k, length(relevances))
    return sum(@view relevances[1:kk]) / num_relevant
end

"""Average Precision (AP) for a single query. `num_relevant=nothing` uses the labels present."""
function average_precision(relevances::AbstractVector{<:Integer}; num_relevant=nothing)
    _check_binary(relevances)
    found_total = sum(relevances)
    nrel = num_relevant === nothing ? found_total : Int(num_relevant)
    nrel > 0 || throw(ArgumentError("num_relevant must be a positive integer, got $nrel"))
    nrel >= found_total ||
        throw(ArgumentError("num_relevant=$nrel is smaller than the $found_total relevant labels present"))
    hits = 0
    precision_sum = 0.0
    for (i, r) in enumerate(relevances)
        if r == 1
            hits += 1
            precision_sum += hits / i
        end
    end
    return precision_sum / nrel
end

"""Mean Average Precision (MAP): AP averaged over a set of queries."""
function mean_average_precision(rankings::AbstractVector; num_relevant=nothing)
    isempty(rankings) && throw(ArgumentError("rankings must contain at least one query"))
    if num_relevant === nothing
        aps = [average_precision(r) for r in rankings]
    else
        length(num_relevant) == length(rankings) ||
            throw(ArgumentError("length mismatch: $(length(rankings)) rankings != $(length(num_relevant)) num_relevant entries"))
        aps = [average_precision(r; num_relevant=n) for (r, n) in zip(rankings, num_relevant)]
    end
    return sum(aps) / length(aps)
end

"""Discounted Cumulative Gain at cutoff `k` (exponential gain `2^rel - 1`, discount `log2(i+1)`)."""
function dcg_at_k(grades::AbstractVector{<:Integer}, k::Integer)
    _check_grades(grades)
    kk = _check_k(k, length(grades))
    total = 0.0
    for i in 1:kk
        gain = 2.0^grades[i] - 1.0
        total += gain / log2(i + 1)
    end
    return total
end

"""Normalized DCG at cutoff `k`, in `[0, 1]`. `ideal_grades=nothing` sorts `grades` descending."""
function ndcg_at_k(grades::AbstractVector{<:Integer}, k::Integer; ideal_grades=nothing)
    _check_grades(grades)
    if ideal_grades === nothing
        ideal = sort(collect(grades); rev=true)
    else
        _check_grades(ideal_grades)
        ideal = sort(collect(ideal_grades); rev=true)
    end
    dcg = dcg_at_k(grades, k)
    idcg = dcg_at_k(ideal, k)
    idcg == 0.0 &&
        throw(ArgumentError("IDCG@k is zero (no relevant documents); NDCG is undefined"))
    return dcg / idcg
end

"""Reciprocal Rank: `1 / rank` of the first relevant document, or `0.0` if none appear."""
function reciprocal_rank(relevances::AbstractVector{<:Integer})
    _check_binary(relevances)
    for (i, r) in enumerate(relevances)
        r == 1 && return 1.0 / i
    end
    return 0.0
end

"""Mean Reciprocal Rank (MRR): reciprocal rank averaged over queries."""
function mean_reciprocal_rank(rankings::AbstractVector)
    isempty(rankings) && throw(ArgumentError("rankings must contain at least one query"))
    return sum(reciprocal_rank(r) for r in rankings) / length(rankings)
end

end # module Ch163IrMetrics
