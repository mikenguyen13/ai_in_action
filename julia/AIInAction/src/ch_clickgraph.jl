"""
    ChClickgraph

Clickstream graphs: node embeddings, burstiness, graph attention, and quantile loss (Julia).

Mirrors the Python module `aiinaction.ch_clickgraph` and the Rust module
`aiinaction::ch_clickgraph`. The shared fixtures in `test/test_ch_clickgraph.jl` match the
Python/Rust suites. The seeded simulator `simulate_clickstream` is Python only; every
numeric function is ported.

* Graph construction: `session_graph`, `average_graph`, `normalize_edge_weights`,
  `upstream_neighbors`.
* Scaling and burstiness: `robust_median_scale`, `burstiness`, `neighbor_burstiness`,
  `ema_last`, `burst_forecast`.
* Embeddings: `lstm_embed` (time series to vector) and `attention_neighbor_embedding`
  (one-head graph transformer layer with an edge term).
* Forecast assembly and loss: `gated_forecast`, `pinball_loss`, `total_quantile_loss`,
  `weighted_quantile_loss`, `interval_coverage`.

Indexing note: node ids are 1-based here (Python and Rust use 0-based ids), so
`upstream_neighbors` also returns 1-based ids. `det_matrix` keeps the 0-based formula so
that all three languages generate identical test matrices.
"""
module ChClickgraph

export RobustScaled, robust_median_scale, burstiness, session_graph, average_graph,
    normalize_edge_weights, upstream_neighbors, neighbor_burstiness, ema_last,
    burst_forecast, softmax, sigmoid, det_matrix, lstm_embed, attention_neighbor_embedding,
    gated_forecast, pinball_loss, total_quantile_loss, weighted_quantile_loss,
    interval_coverage

"""Result of [`robust_median_scale`](@ref)."""
struct RobustScaled
    median::Float64
    mad::Float64
    scaled::Vector{Float64}
end

function _median(a::AbstractVector{<:Real})
    s = sort(Float64.(a))
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

"""
    robust_median_scale(x)

Center on the median and divide by the mean absolute deviation about the median. A zero
deviation (constant series) uses a divisor of 1.
"""
function robust_median_scale(x::AbstractVector{<:Real})
    isempty(x) && throw(ArgumentError("x must be non-empty"))
    a = Float64.(x)
    med = _median(a)
    mad = sum(abs.(a .- med)) / length(a)
    div = mad > 0 ? mad : 1.0
    return RobustScaled(med, mad, (a .- med) ./ div)
end

"""
    burstiness(x)

Lappas et al. burstiness of each single interval, `x_t / sum(x) - 1/T`. An all-zero window
returns zeros.
"""
function burstiness(x::AbstractVector{<:Real})
    isempty(x) && throw(ArgumentError("x must be non-empty"))
    any(<(0), x) && throw(ArgumentError("counts must be non-negative"))
    a = Float64.(x)
    total = sum(a)
    total <= 0 && return zeros(length(a))
    return a ./ total .- 1.0 / length(a)
end

"""
    session_graph(sessions, n_nodes)

Node counts and directed transition matrix from sessions (vectors of 1-based node ids).
Consecutive repeats count as occurrences but not as transitions.
"""
function session_graph(sessions::AbstractVector, n_nodes::Integer)
    n_nodes > 0 || throw(ArgumentError("n_nodes must be positive"))
    counts = zeros(n_nodes)
    W = zeros(n_nodes, n_nodes)
    for s in sessions
        prev = 0
        for node in s
            (1 <= node <= n_nodes) ||
                throw(ArgumentError("node id $node outside [1, $n_nodes]"))
            counts[node] += 1.0
            if prev > 0 && prev != node
                W[prev, node] += 1.0
            end
            prev = node
        end
    end
    return counts, W
end

"""Elementwise mean of daily adjacency matrices."""
function average_graph(graphs::AbstractVector)
    isempty(graphs) && throw(ArgumentError("need at least one graph"))
    acc = zeros(size(graphs[1]))
    for gm in graphs
        acc .+= gm
    end
    return acc ./ length(graphs)
end

"""Divide each edge `i -> j` by the average quantity of target `j` (zero if that is not positive)."""
function normalize_edge_weights(W::AbstractMatrix{<:Real}, mean_counts::AbstractVector{<:Real})
    (size(W, 1) == size(W, 2) && length(mean_counts) == size(W, 2)) ||
        throw(ArgumentError("W must be square and mean_counts must match its size"))
    out = zeros(size(W))
    for j in axes(W, 2)
        if mean_counts[j] > 0
            out[:, j] .= W[:, j] ./ mean_counts[j]
        end
    end
    return out
end

"""Nodes `i != j` with a positive edge `i -> j`, ascending (1-based)."""
function upstream_neighbors(W::AbstractMatrix{<:Real}, j::Integer)
    return [i for i in axes(W, 1) if i != j && W[i, j] > 0]
end

"""Average the burstiness series (rows of `B`) over the upstream neighbors; zeros if none."""
function neighbor_burstiness(B::AbstractMatrix{<:Real}, upstream::AbstractVector{<:Integer})
    isempty(upstream) && return zeros(size(B, 2))
    return vec(sum(Float64.(B[upstream, :]); dims=1)) ./ length(upstream)
end

"""Final value of the EMA `s_t = a x_t + (1 - a) s_(t-1)`."""
function ema_last(x::AbstractVector{<:Real}, alpha::Real)
    (0 < alpha <= 1) || throw(ArgumentError("alpha must be in (0, 1]"))
    isempty(x) && throw(ArgumentError("x must be non-empty"))
    s = Float64(x[1])
    for v in x[2:end]
        s = alpha * v + (1 - alpha) * s
    end
    return s
end

"""Burst forecasts `tanh(raw) * trend`."""
burst_forecast(raw::AbstractVector{<:Real}, trend::Real) = tanh.(Float64.(raw)) .* Float64(trend)

"""Numerically stable softmax."""
function softmax(v::AbstractVector{<:Real})
    isempty(v) && throw(ArgumentError("v must be non-empty"))
    a = Float64.(v)
    e = exp.(a .- maximum(a))
    return e ./ sum(e)
end

"""Logistic function, stable for large `|z|`."""
function sigmoid(z::Real)
    ez = exp(-abs(z))
    return z >= 0 ? 1.0 / (1.0 + ez) : ez / (1.0 + ez)
end
sigmoid(z::AbstractArray{<:Real}) = sigmoid.(z)

"""Deterministic test matrix `0.4 sin(0.7 (i cols + j + 1) + salt)` with 0-based `i`, `j`."""
function det_matrix(rows::Integer, cols::Integer, salt::Real=0.0)
    return [0.4 * sin(0.7 * ((i - 1) * cols + (j - 1) + 1) + salt) for i in 1:rows, j in 1:cols]
end

"""
    lstm_embed(x_seq, W, b)

Final hidden state of a single-layer LSTM. `x_seq` is `(T, d)`, `W` is `(4H, d + H)` over
`[x_t; h_(t-1)]` with gate blocks input, forget, cell candidate, output, and `b` has length
`4H`.
"""
function lstm_embed(x_seq::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, b::AbstractVector{<:Real})
    T, d = size(x_seq)
    T > 0 || throw(ArgumentError("x_seq must be non-empty"))
    H = size(W, 1) ÷ 4
    (size(W) == (4H, d + H) && length(b) == 4H) ||
        throw(ArgumentError("W must be (4H, d+H) and b must be (4H,)"))
    h = zeros(H)
    c = zeros(H)
    for t in 1:T
        z = W * vcat(Float64.(x_seq[t, :]), h) .+ b
        i = sigmoid.(z[1:H])
        f = sigmoid.(z[H+1:2H])
        gg = tanh.(z[2H+1:3H])
        o = sigmoid.(z[3H+1:4H])
        c = f .* c .+ i .* gg
        h = o .* tanh.(c)
    end
    return h
end

"""
    attention_neighbor_embedding(h_i, H_nb, e_nb, Wq, Wk, Wv, we) -> (z, alpha)

One-head graph transformer layer. `q = Wq h_i`, `k_j = Wk h_j + we * e_ij`, `v_j = Wv h_j`,
`alpha = softmax(q . k_j / sqrt(d_k))`, `z = sum_j alpha_j v_j`. `H_nb` holds one neighbor
per row. No neighbors gives `z = 0` and an empty `alpha`.
"""
function attention_neighbor_embedding(h_i, H_nb::AbstractMatrix{<:Real}, e_nb, Wq, Wk, Wv, we)
    dk = size(Wq, 1)
    (size(Wk, 1) == dk && length(we) == dk) ||
        throw(ArgumentError("Wq, Wk and we must share the key dimension"))
    n = size(H_nb, 1)
    n == 0 && return zeros(size(Wv, 1)), zeros(0)
    length(e_nb) == n || throw(ArgumentError("e_nb must have one entry per neighbor"))
    q = Wq * Float64.(h_i)
    K = H_nb * Wk' .+ Float64.(e_nb) * Float64.(we)'
    V = H_nb * Wv'
    alpha = softmax(K * q ./ sqrt(dk))
    return vec(alpha' * V), alpha
end

"""Final forecast `base + burst + sigmoid(gate_logits) * neigh` (elementwise)."""
function gated_forecast(base, burst, neigh, gate_logits)
    n = length(base)
    (length(burst) == n && length(neigh) == n && length(gate_logits) == n) ||
        throw(ArgumentError("all inputs must have the same length"))
    return Float64.(base) .+ Float64.(burst) .+ sigmoid.(Float64.(gate_logits)) .* Float64.(neigh)
end

"""Pinball loss `q (y - yhat)_+ + (1 - q) (yhat - y)_+`."""
function pinball_loss(y::Real, yhat::Real, q::Real)
    (0 < q < 1) || throw(ArgumentError("q must be in (0, 1)"))
    return q * max(0.0, y - yhat) + (1 - q) * max(0.0, yhat - y)
end

"""Sum of pinball losses over horizon steps and quantiles (`preds` is `(H, Q)`)."""
function total_quantile_loss(y::AbstractVector{<:Real}, preds::AbstractMatrix{<:Real},
                             quantiles::AbstractVector{<:Real})
    size(preds) == (length(y), length(quantiles)) ||
        throw(ArgumentError("preds must have shape (length(y), length(quantiles))"))
    total = 0.0
    for h in eachindex(y), (k, q) in enumerate(quantiles)
        total += pinball_loss(y[h], preds[h, k], q)
    end
    return total
end

"""Scale-free summary `2 sum pinball_q / sum |y|`."""
function weighted_quantile_loss(y::AbstractVector{<:Real}, yhat::AbstractVector{<:Real}, q::Real)
    (length(y) == length(yhat) && !isempty(y)) ||
        throw(ArgumentError("y and yhat must be non-empty and the same length"))
    denom = sum(abs, y)
    denom == 0 && throw(ArgumentError("sum |y| is zero; wQL is undefined"))
    total = sum(pinball_loss(a, b, q) for (a, b) in zip(y, yhat))
    return 2.0 * total / denom
end

"""Fraction of observations inside `[lo, hi]` (inclusive)."""
function interval_coverage(y::AbstractVector{<:Real}, lo::AbstractVector{<:Real}, hi::AbstractVector{<:Real})
    (length(y) == length(lo) == length(hi) && !isempty(y)) ||
        throw(ArgumentError("y, lo and hi must be non-empty and the same length"))
    return count(i -> lo[i] <= y[i] <= hi[i], eachindex(y)) / length(y)
end

end # module ChClickgraph
