"""
    Ch526HitlDiscovery

Human-in-the-loop sequential experiments for drug discovery (Julia).

Mirrors the Python module `aiinaction.ch526_hitl_discovery` and the Rust module
`aiinaction::ch526_hitl_discovery`, after He, Hua, Wang, and Zheng (2026), Collaborative
Intelligence in Sequential Experiments (Information Systems Research 37(3), 1390-1415).
The shared fixtures in `test/test_ch526_hitl_discovery.jl` match the Python/Rust suites.
The SATPdb loader, ESM2 embedding, and the torch `BayesianMLP` are Python only; every
numeric function is ported.

A pool of `N` molecules, a budget of `B` lab tests per round for `R` rounds, and a target
is a molecule whose `K` binary properties are all positive. Each round a surrogate is
trained on the labels gathered so far and the algorithm hands the expert two shortlists:
an uncertainty list (highest epistemic uncertainty, which teaches the model the most) and
a search list (highest upper confidence bound, the best bets for a target). The expert
picks the `B` molecules that go to the lab.

* Uncertainty: `decompose_uncertainty` splits predictive entropy into aleatoric (expected
  entropy) and epistemic (mutual information) parts from an `S x N x K` array of draws.
* Scores and selection: `ucb_scores`, `top_k`, `merge_shortlists`.
* The expert: `Expert`, `reveal!`, `expert_select`.
* Metrics: `is_target`, `hit_rate`, `recall_rate`.
* Surrogate: `BootstrapLogisticEnsemble` with `fit!` and `predict_samples`.
* Simulator: `simulate` over `STRATEGIES`, returning a `SimResult`.

Indexing note: every molecule index (`top_k`, `merge_shortlists`, `expert_select`, the
shortlists passed to them, and `Expert` masks) is 1-based, so Python index `i` is Julia
index `i + 1`. Matrices are row-per-molecule, as in Python. `reveal!`, the bootstrap
ensemble, and `simulate` draw from Julia's `Random.Xoshiro`, so results agree with Python
statistically, not bit for bit.
"""
module Ch526HitlDiscovery

using Random

export binary_entropy, decompose_uncertainty, ucb_scores, top_k, merge_shortlists,
    Expert, reveal!, expert_select, is_target, hit_rate, recall_rate,
    BootstrapLogisticEnsemble, fit!, predict_samples, STRATEGIES, SimResult,
    recall_by_round, simulate

"""Natural log of 2."""
const LN2 = log(2.0)

"""Clipping constant that keeps `0 log 0 = 0` numerically."""
const EPS = 1e-12

# --------------------------------------------------------------------------
# Uncertainty decomposition and scores
# --------------------------------------------------------------------------

"""
    binary_entropy(p) -> Float64

Entropy of a Bernoulli(`p`) in nats, with `0 log 0 = 0`. Arrays are handled elementwise.
"""
function binary_entropy(p::Real)
    q = clamp(Float64(p), EPS, 1.0 - EPS)
    return -(q * log(q) + (1.0 - q) * log(1.0 - q))
end
binary_entropy(p::AbstractArray{<:Real}) = binary_entropy.(p)

"""
    decompose_uncertainty(prob_samples) -> (mu, aleatoric, epistemic)

Split predictive uncertainty into data and model parts. `prob_samples` has shape
`(S, N, K)` (or `(S, N)` for one property): `S` posterior draws of P(property k is
positive) for `N` molecules. Returns three `N x K` matrices:

* `mu`, the posterior predictive mean;
* `aleatoric`, the expected entropy `E_theta H[p(y|x,theta)]` (noise more data cannot
  remove);
* `epistemic`, the mutual information `H[E p] - E H[p]` (what more data can teach).
"""
function decompose_uncertainty(prob_samples::AbstractArray{<:Real,3})
    ps = Array{Float64,3}(prob_samples)
    s = size(ps, 1)
    mu = dropdims(sum(ps; dims=1); dims=1) ./ s
    aleatoric = dropdims(sum(binary_entropy.(ps); dims=1); dims=1) ./ s
    epistemic = max.(binary_entropy.(mu) .- aleatoric, 0.0)
    return mu, aleatoric, epistemic
end
function decompose_uncertainty(prob_samples::AbstractMatrix{<:Real})
    return decompose_uncertainty(reshape(Float64.(prob_samples), size(prob_samples)..., 1))
end

"""
    ucb_scores(mu, sigma_d, beta) -> Vector{Float64}

Search score `sum_k [mu_k + sqrt(beta) sigma_d_k]` (paper Eq. 16-18) per molecule. Data
uncertainty, not model uncertainty, is the bonus. For a Bernoulli head pass
`sigma_d = sqrt.(mu .* (1 .- mu))`. `mu` and `sigma_d` are `N x K` (a vector is read as
`N x 1`).
"""
function ucb_scores(mu::AbstractMatrix{<:Real}, sigma_d::AbstractMatrix{<:Real}, beta::Real)
    size(mu) == size(sigma_d) || throw(ArgumentError("mu and sigma_d must have the same shape"))
    beta >= 0 || throw(ArgumentError("beta must be non-negative"))
    sb = sqrt(Float64(beta))
    return vec(sum(Float64.(mu) .+ sb .* Float64.(sigma_d); dims=2))
end
ucb_scores(mu::AbstractVector{<:Real}, sigma_d::AbstractVector{<:Real}, beta::Real) =
    ucb_scores(reshape(mu, :, 1), reshape(sigma_d, :, 1), beta)

"""
    top_k(scores, k, exclude=nothing) -> Vector{Int}

1-based indices of the `k` largest scores, ties broken by lower index. `exclude` is a
boolean mask of molecules that are not available (already tested). Fewer than `k`
indices come back only if fewer are available.
"""
function top_k(scores::AbstractVector{<:Real}, k::Integer,
               exclude::Union{Nothing,AbstractVector{Bool}}=nothing)
    s = collect(Float64, scores)
    if exclude !== nothing
        length(exclude) == length(s) || throw(ArgumentError("exclude must match scores"))
        for i in eachindex(s)
            exclude[i] && (s[i] = -Inf)
        end
    end
    n_avail = count(isfinite, s)
    kk = max(0, min(Int(k), n_avail))
    order = sortperm(s; rev=true, alg=MergeSort)  # stable, so ties keep the lower index
    return order[1:kk]
end

"""
    merge_shortlists(uncertainty_idx, search_idx) -> Vector{Int}

Union of the two shortlists, search list first, order preserved, no repeats.
"""
function merge_shortlists(uncertainty_idx::AbstractVector{<:Integer},
                          search_idx::AbstractVector{<:Integer})
    seen = Set{Int}()
    out = Int[]
    for i in Iterators.flatten((search_idx, uncertainty_idx))
        if !(Int(i) in seen)
            push!(seen, Int(i))
            push!(out, Int(i))
        end
    end
    return out
end

# --------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------

"""
Simulated domain expert with partial, private knowledge.

The expert knows the true target status of a molecule with probability `knowledge`
(decided once per molecule by `reveal!`, so it is consistent across rounds).
`explore_share` is the fraction of the molecules the expert has no private information
about that are filled from the uncertainty list rather than the search list.
"""
mutable struct Expert
    knowledge::Float64
    explore_share::Float64
    known_mask::Union{Nothing,Vector{Bool}}
    known_target::Union{Nothing,Vector{Bool}}
end

Expert(knowledge::Real, explore_share::Real=0.3) =
    Expert(Float64(knowledge), Float64(explore_share), nothing, nothing)

"""
    reveal!(expert, is_target_vec, rng)

Fix, once, which molecules the expert has private knowledge about.
"""
function reveal!(expert::Expert, is_target_vec::AbstractVector{Bool}, rng::AbstractRNG)
    n = length(is_target_vec)
    expert.known_mask = rand(rng, n) .< expert.knowledge
    expert.known_target = collect(Bool, is_target_vec)
    return expert
end

"""
    expert_select(expert, search_list, uncertainty_list, budget) -> Vector{Int}

Choose the molecules sent to the lab (1-based indices). Rules, in order:

1. every molecule the expert knows to be a target is taken first;
2. molecules the expert knows to be non-targets are dropped (override);
3. the remaining slots go to molecules the expert knows nothing about, split between the
   search list and the uncertainty list in the proportion `explore_share`;
4. if slots are still empty, they are filled from the dropped molecules so the budget is
   always spent.
"""
function expert_select(expert::Expert, search_list::AbstractVector{<:Integer},
                       uncertainty_list::AbstractVector{<:Integer}, budget::Integer)
    km = expert.known_mask
    kt = expert.known_target
    (km === nothing || kt === nothing) && throw(ArgumentError("call reveal! first"))
    pool = merge_shortlists(uncertainty_list, search_list)
    chosen = Int[i for i in pool if km[i] && kt[i]]
    length(chosen) > budget && resize!(chosen, budget)
    taken = Set{Int}(chosen)

    unknown(seq) = Int[Int(i) for i in seq if !(Int(i) in taken) && !km[i]]

    srch = unknown(search_list)
    unc = unknown(uncertainty_list)
    n_left = budget - length(chosen)
    n_unc = Int(round(expert.explore_share * n_left))  # ties to even, as in Python
    n_srch = n_left - n_unc
    for (lst, n0) in ((srch, n_srch), (unc, n_unc))
        n = n0
        for i in lst
            n == 0 && break
            if !(i in taken)
                push!(chosen, i)
                push!(taken, i)
                n -= 1
            end
        end
    end
    # top up from whatever is left among unknowns, then from known non-targets
    for i in unknown(vcat(collect(Int, search_list), collect(Int, uncertainty_list)))
        length(chosen) >= budget && break
        if !(i in taken)
            push!(chosen, i)
            push!(taken, i)
        end
    end
    for i in pool
        length(chosen) >= budget && break
        if !(i in taken)
            push!(chosen, i)
            push!(taken, i)
        end
    end
    length(chosen) > budget && resize!(chosen, budget)
    return chosen
end

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

"""
    is_target(labels) -> Vector{Bool}

A target has every one of its `K` properties positive. `labels` is `N x K`.
"""
is_target(labels::AbstractMatrix{<:Real}) = Bool[all(!iszero, view(labels, i, :)) for i in axes(labels, 1)]

"""
    hit_rate(n_hits, budget_per_round, rounds) -> Float64

Final hit rate `S / (B R)`: targets found per lab test spent.
"""
hit_rate(n_hits::Real, budget_per_round::Integer, rounds::Integer) =
    Float64(n_hits) / Float64(budget_per_round * rounds)

"""
    recall_rate(n_hits, n_targets_in_pool) -> Float64

Share of all targets in the pool found so far.
"""
recall_rate(n_hits::Real, n_targets_in_pool::Real) = Float64(n_hits) / Float64(n_targets_in_pool)

# --------------------------------------------------------------------------
# A dependency-light surrogate (shared across the three languages)
# --------------------------------------------------------------------------

"""
Bootstrap ensemble of L2-regularised logistic regressions, one per property.

`n_members` members are fitted to bootstrap resamples, so their disagreement is a crude
posterior over parameters. `predict_samples` returns the `(S, N, K)` array that
`decompose_uncertainty` consumes. Weights `w` have shape `(S, K, D+1)` (bias last).
"""
mutable struct BootstrapLogisticEnsemble
    n_members::Int
    l2::Float64
    lr::Float64
    epochs::Int
    rng::Xoshiro
    w::Union{Nothing,Array{Float64,3}}
end

function BootstrapLogisticEnsemble(; n_members::Integer=10, l2::Real=1e-2, lr::Real=0.5,
                                   epochs::Integer=200, seed::Integer=0)
    return BootstrapLogisticEnsemble(Int(n_members), Float64(l2), Float64(lr), Int(epochs),
                                     Xoshiro(seed), nothing)
end

_sigmoid(z) = 1.0 / (1.0 + exp(-z))

"""
    fit!(ens, x, y) -> ens

Fit on features `x` (`N x D`) and binary labels `y` (`N x K`).
"""
function fit!(ens::BootstrapLogisticEnsemble, x::AbstractMatrix{<:Real},
              y::AbstractMatrix{<:Real})
    n, d = size(x)
    size(y, 1) == n || throw(ArgumentError("x and y must have the same number of rows"))
    k = size(y, 2)
    xb = hcat(Float64.(x), ones(n))
    yf = Float64.(y)
    w = zeros(ens.n_members, k, d + 1)
    for s in 1:ens.n_members
        idx = rand(ens.rng, 1:n, n)
        xs = xb[idx, :]
        ys = yf[idx, :]
        ws = zeros(k, d + 1)
        for _ in 1:ens.epochs
            p = _sigmoid.(xs * transpose(ws))
            grad = (transpose(p .- ys) * xs) ./ n .+ ens.l2 .* ws
            ws .-= ens.lr .* grad
        end
        w[s, :, :] = ws
    end
    ens.w = w
    return ens
end

"""
    predict_samples(ens, x) -> Array{Float64,3}

Posterior draws of P(property k is positive), shape `(S, N, K)`.
"""
function predict_samples(ens::BootstrapLogisticEnsemble, x::AbstractMatrix{<:Real})
    w = ens.w
    w === nothing && throw(ArgumentError("call fit! first"))
    xb = hcat(Float64.(x), ones(size(x, 1)))
    s, k, _ = size(w)
    out = Array{Float64,3}(undef, s, size(xb, 1), k)
    for i in 1:s
        out[i, :, :] = _sigmoid.(xb * transpose(view(w, i, :, :)))
    end
    return out
end

# --------------------------------------------------------------------------
# The sequential-experiment simulator
# --------------------------------------------------------------------------

"""The nine selection strategies accepted by `simulate`."""
const STRATEGIES = ("random", "search", "active_learning", "ucb", "random_then_ucb",
                    "al_then_ucb", "human_only", "algorithm_only", "hitl")

"""Cumulative hits per round plus the run configuration."""
struct SimResult
    strategy::String
    hits_by_round::Vector{Int}
    n_targets::Int
    budget::Int
    rounds::Int
end

"""    hit_rate(result::SimResult) -> Float64"""
hit_rate(r::SimResult) = hit_rate(r.hits_by_round[end], r.budget, r.rounds)

"""    recall_by_round(result::SimResult) -> Vector{Float64}"""
recall_by_round(r::SimResult) = [recall_rate(h, r.n_targets) for h in r.hits_by_round]

_sample_without_replacement(rng::AbstractRNG, pool::Vector{Int}, k::Int) =
    pool[randperm(rng, length(pool))[1:k]]

"""
    simulate(strategy, x, labels, fit_predict; budget=50, rounds=50, q=50, h=50, beta=1.0,
             warmup_rounds=10, knowledge=0.5, explore_share=0.3, human_view=200,
             seed=0) -> SimResult

Run one sequential experiment and return cumulative hits per round. `x` is `N x D`,
`labels` is `N x K`. `fit_predict(x_labeled, y_labeled)` must return a function mapping a
feature matrix to an `(S, N, K)` array of posterior draws of P(positive).

Strategies (see `STRATEGIES`):

* `random`: `B` random untested molecules each round.
* `search`: top `B` by posterior mean.
* `active_learning`: top `B` by epistemic uncertainty.
* `ucb`: top `B` by the search score.
* `random_then_ucb`: random for `warmup_rounds`, then `ucb`.
* `al_then_ucb`: `active_learning` for `warmup_rounds`, then `ucb`.
* `human_only`: the expert screens `human_view` random molecules and picks `B` from
  private knowledge alone.
* `algorithm_only`: half of `B` from the search list, the rest from the uncertainty list,
  no human override.
* `hitl`: the expert chooses `B` from the union of the two shortlists.
"""
function simulate(strategy::Union{AbstractString,Symbol}, x::AbstractMatrix{<:Real},
                  labels::AbstractMatrix{<:Real}, fit_predict::Function;
                  budget::Integer=50, rounds::Integer=50, q::Integer=50, h::Integer=50,
                  beta::Real=1.0, warmup_rounds::Integer=10, knowledge::Real=0.5,
                  explore_share::Real=0.3, human_view::Integer=200, seed::Integer=0)
    strategy = String(strategy)
    strategy in STRATEGIES || throw(ArgumentError("unknown strategy $(repr(strategy))"))
    rng = Xoshiro(seed)
    n = size(x, 1)
    tgt = is_target(labels)
    tested = falses(n)
    hits = Int[]
    total = 0
    expert = nothing
    if strategy in ("human_only", "hitl")
        expert = Expert(knowledge, explore_share)
        reveal!(expert, tgt, rng)
    end

    predictor = nothing
    for r in 0:(rounds - 1)
        if any(tested) && !(strategy in ("random", "human_only"))
            predictor = fit_predict(x[tested, :], labels[tested, :])
        end
        avail = findall(!, tested)
        random_pick() = _sample_without_replacement(rng, avail, min(Int(budget), length(avail)))

        local pick::Vector{Int}
        if strategy == "random" || (predictor === nothing && strategy != "human_only")
            pick = random_pick()
        elseif strategy == "human_only"
            view_idx = _sample_without_replacement(rng, avail, min(Int(human_view), length(avail)))
            km, kt = expert.known_mask, expert.known_target
            known_pos = Int[i for i in view_idx if km[i] && kt[i]]
            rest = Int[i for i in view_idx if !km[i]]
            shuffle!(rng, rest)
            both = vcat(known_pos, rest)
            pick = both[1:min(Int(budget), length(both))]
        else
            mu, _, epi = decompose_uncertainty(predictor(x))
            sigma_d = sqrt.(mu .* (1.0 .- mu))
            search_score = ucb_scores(mu, sigma_d, beta)
            unc_score = vec(sum(epi; dims=2))
            strat = strategy
            strat == "random_then_ucb" && (strat = r < warmup_rounds ? "random" : "ucb")
            strat == "al_then_ucb" && (strat = r < warmup_rounds ? "active_learning" : "ucb")
            if strat == "random"
                pick = random_pick()
            elseif strat == "search"
                pick = top_k(vec(sum(mu; dims=2)), budget, tested)
            elseif strat == "active_learning"
                pick = top_k(unc_score, budget, tested)
            elseif strat == "ucb"
                pick = top_k(search_score, budget, tested)
            else
                s_list = top_k(search_score, h, tested)
                u_list = top_k(unc_score, q, tested)
                if strat == "algorithm_only"
                    half = Int(budget) ÷ 2
                    a = s_list[1:min(half, length(s_list))]
                    aset = Set(a)
                    b = Int[i for i in u_list if !(i in aset)]
                    need = max(0, Int(budget) - length(a))
                    pick = vcat(a, b[1:min(need, length(b))])
                else  # hitl
                    pick = expert_select(expert, s_list, u_list, budget)
                end
            end
        end
        tested[pick] .= true
        total += count(i -> tgt[i], pick)
        push!(hits, total)
    end
    return SimResult(strategy, hits, count(tgt), Int(budget), Int(rounds))
end

end # module Ch526HitlDiscovery
