"""
    Ch150AssociationRules

Association rule and frequent-pattern mining from scratch (Julia).

Mirrors the Python module `aiinaction.ch150_association_rules` and the Rust module
`aiinaction::ch150_association_rules`. Implements Apriori (Agrawal and Srikant,
1994) and FP-Growth (Han, Pei and Yin, 2000) plus association-rule extraction with
support, confidence, lift, leverage and conviction.

Items are `Int`; a transaction is a collection of items. Frequent itemsets are
returned as a `Dict` mapping a sorted `Vector{Int}` to its support, so Apriori and
FP-Growth return identical results, which the shared fixtures rely on.
"""
module Ch150AssociationRules

export Rule, apriori, fpgrowth, association_rules, support

"""A single association rule `antecedent => consequent` with its metrics."""
struct Rule
    antecedent::Vector{Int}
    consequent::Vector{Int}
    support::Float64
    confidence::Float64
    lift::Float64
    leverage::Float64
    conviction::Float64  # Inf for a perfectly confident rule
end

function _normalize(transactions)
    isempty(transactions) && throw(ArgumentError("transactions must be non-empty"))
    data = Vector{Set{Int}}(undef, length(transactions))
    for (k, t) in enumerate(transactions)
        s = Set{Int}()
        for it in t
            push!(s, Int(it))
        end
        data[k] = s
    end
    return data
end

function _check_min_support(min_support::Real)
    (0.0 < min_support <= 1.0) ||
        throw(ArgumentError("min_support must be in (0, 1], got $min_support"))
    return nothing
end

"""Smallest integer count meeting the support threshold (ceil with epsilon)."""
function _min_count(min_support::Real, n::Int)
    return max(1, Int(ceil(min_support * n - 1e-9)))
end

"""Fraction of transactions that contain every item of `itemset`."""
function support(transactions, itemset)
    data = _normalize(transactions)
    target = Set{Int}(Int(i) for i in itemset)
    isempty(target) && throw(ArgumentError("itemset must be non-empty"))
    cover = count(t -> issubset(target, t), data)
    return cover / length(data)
end

function _frequent_singletons(data::Vector{Set{Int}}, mc::Int)
    counts = Dict{Int,Int}()
    for t in data, it in t
        counts[it] = get(counts, it, 0) + 1
    end
    return Dict(it => c for (it, c) in counts if c >= mc)
end

"""Join + prune step. `prev` is a sorted vector of frequent (k-1)-itemsets."""
function _apriori_gen(prev::Vector{Vector{Int}})
    isempty(prev) && return Vector{Vector{Int}}()
    prev_set = Set(prev)
    k = length(prev[1]) + 1
    candidates = Vector{Vector{Int}}()
    n = length(prev)
    for i in 1:n
        for j in (i + 1):n
            a = prev[i]
            b = prev[j]
            a_pref = @view a[1:end-1]
            b_pref = @view b[1:end-1]
            if a_pref == b_pref && a[end] < b[end]
                cand = vcat(a, b[end])
                ok = true
                for skip in 1:length(cand)
                    sub = [cand[idx] for idx in 1:length(cand) if idx != skip]
                    if length(sub) == k - 1 && !(sub in prev_set)
                        ok = false
                        break
                    end
                end
                ok && push!(candidates, cand)
            elseif a_pref != b_pref
                break
            end
        end
    end
    sort!(candidates)
    return candidates
end

"""
    apriori(transactions, min_support)

Mine all frequent itemsets with the Apriori level-wise algorithm. Returns a
`Dict{Vector{Int},Float64}` mapping each frequent (sorted) itemset to its support.
"""
function apriori(transactions, min_support::Real)
    data = _normalize(transactions)
    _check_min_support(min_support)
    n = length(data)
    mc = _min_count(min_support, n)

    frequent = Dict{Vector{Int},Int}()
    singles = _frequent_singletons(data, mc)
    level = sort([[it] for it in keys(singles)])
    for it in level
        frequent[it] = singles[it[1]]
    end

    while !isempty(level)
        candidates = _apriori_gen(level)
        isempty(candidates) && break
        counts = zeros(Int, length(candidates))
        for t in data
            for (ci, c) in enumerate(candidates)
                if all(item -> item in t, c)
                    counts[ci] += 1
                end
            end
        end
        next = Vector{Vector{Int}}()
        for (ci, c) in enumerate(candidates)
            if counts[ci] >= mc
                push!(next, c)
                frequent[c] = counts[ci]
            end
        end
        sort!(next)
        level = next
    end

    return Dict(k => v / n for (k, v) in frequent)
end

# --------------------------------------------------------------------------- #
# FP-Growth (arena-based tree)
# --------------------------------------------------------------------------- #

mutable struct _FpNode
    item::Int
    count::Int
    parent::Int
    children::Dict{Int,Int}
    link::Int  # 0 = none
end

mutable struct _FpTree
    nodes::Vector{_FpNode}
    header::Dict{Int,Int}  # item -> head node index (1-based)
end

const _ROOT_ITEM = typemin(Int)

function _new_tree()
    root = _FpNode(_ROOT_ITEM, 0, 0, Dict{Int,Int}(), 0)
    return _FpTree([root], Dict{Int,Int}())
end

function _link!(tree::_FpTree, idx::Int)
    item = tree.nodes[idx].item
    if !haskey(tree.header, item)
        tree.header[item] = idx
    else
        head = tree.header[item]
        while tree.nodes[head].link != 0
            head = tree.nodes[head].link
        end
        tree.nodes[head].link = idx
    end
    return nothing
end

function _insert!(tree::_FpTree, ordered::Vector{Int}, weight::Int)
    node = 1  # root is index 1
    for it in ordered
        children = tree.nodes[node].children
        if haskey(children, it)
            child_idx = children[it]
        else
            push!(tree.nodes, _FpNode(it, 0, node, Dict{Int,Int}(), 0))
            child_idx = length(tree.nodes)
            children[it] = child_idx
            _link!(tree, child_idx)
        end
        tree.nodes[child_idx].count += weight
        node = child_idx
    end
    return nothing
end

function _ascend(tree::_FpTree, idx::Int)
    path = Int[]
    cur = tree.nodes[idx].parent
    while cur != 0 && tree.nodes[cur].item != _ROOT_ITEM
        push!(path, tree.nodes[cur].item)
        cur = tree.nodes[cur].parent
    end
    return path
end

"""Order items by descending support, ties by ascending item id."""
function _order_items(items, counts::Dict{Int,Int})
    v = collect(items)
    sort!(v; by = it -> (-get(counts, it, 0), it))
    return v
end

function _build_tree(data, counts::Dict{Int,Int}, frequent_items::Set{Int})
    tree = _new_tree()
    for (items, weight) in data
        filtered = [it for it in items if it in frequent_items]
        ordered = _order_items(filtered, counts)
        isempty(ordered) || _insert!(tree, ordered, weight)
    end
    return tree
end

function _mine_tree!(tree::_FpTree, counts::Dict{Int,Int}, mc::Int,
                     suffix::Vector{Int}, frequent::Dict{Vector{Int},Int})
    items = sort(collect(keys(tree.header)); by = it -> (counts[it], it))
    for item in items
        new_suffix = sort(vcat(suffix, item))
        frequent[new_suffix] = counts[item]

        cond_patterns = Vector{Tuple{Vector{Int},Int}}()
        node = get(tree.header, item, 0)
        while node != 0
            prefix = _ascend(tree, node)
            isempty(prefix) || push!(cond_patterns, (prefix, tree.nodes[node].count))
            node = tree.nodes[node].link
        end

        cond_counts = Dict{Int,Int}()
        for (prefix, cnt) in cond_patterns, it in prefix
            cond_counts[it] = get(cond_counts, it, 0) + cnt
        end
        cond_frequent = Dict(it => c for (it, c) in cond_counts if c >= mc)
        isempty(cond_frequent) && continue
        cond_items = Set{Int}(keys(cond_frequent))

        cond_data = Vector{Tuple{Vector{Int},Int}}()
        for (prefix, cnt) in cond_patterns
            kept = [it for it in prefix if it in cond_items]
            isempty(kept) || push!(cond_data, (kept, cnt))
        end

        cond_tree = _build_tree(cond_data, cond_frequent, cond_items)
        if !isempty(cond_tree.header)
            _mine_tree!(cond_tree, cond_frequent, mc, new_suffix, frequent)
        end
    end
    return nothing
end

"""
    fpgrowth(transactions, min_support)

Mine all frequent itemsets with FP-Growth. Returns the same `Dict` as `apriori`.
"""
function fpgrowth(transactions, min_support::Real)
    data = _normalize(transactions)
    _check_min_support(min_support)
    n = length(data)
    mc = _min_count(min_support, n)

    counts = Dict{Int,Int}()
    for t in data, it in t
        counts[it] = get(counts, it, 0) + 1
    end
    frequent_items = Set{Int}(it for (it, c) in counts if c >= mc)

    weighted = [(collect(t), 1) for t in data]
    tree = _build_tree(weighted, counts, frequent_items)

    frequent = Dict{Vector{Int},Int}()
    _mine_tree!(tree, counts, mc, Int[], frequent)
    return Dict(k => v / n for (k, v) in frequent)
end

# --------------------------------------------------------------------------- #
# Rule generation
# --------------------------------------------------------------------------- #

function _subsets_of_size(items::Vector{Int}, r::Int)
    out = Vector{Vector{Int}}()
    n = length(items)
    (r == 0 || r > n) && return out
    idx = collect(1:r)
    while true
        push!(out, [items[i] for i in idx])
        i = r
        while true
            i == 0 && return out
            if idx[i] != i + n - r
                break
            end
            i -= 1
        end
        idx[i] += 1
        for j in (i + 1):r
            idx[j] = idx[j - 1] + 1
        end
    end
end

"""
    association_rules(frequent_itemsets, min_confidence)

Extract association rules from a `Dict` of frequent itemsets to supports. Emits
`A => Z\\A` for every frequent itemset `Z` (|Z| >= 2) and proper non-empty subset
`A` whose confidence is at least `min_confidence`. Rules are sorted by descending
confidence, then antecedent, then consequent.
"""
function association_rules(frequent_itemsets::Dict{Vector{Int},Float64},
                           min_confidence::Real)
    (0.0 <= min_confidence <= 1.0) ||
        throw(ArgumentError("min_confidence must be in [0, 1], got $min_confidence"))
    rules = Vector{Rule}()
    for (z, supp_z) in frequent_itemsets
        length(z) < 2 && continue
        zs = sort(z)
        for r in 1:(length(zs) - 1)
            for antecedent in _subsets_of_size(zs, r)
                aset = Set(antecedent)
                consequent = [it for it in zs if !(it in aset)]
                haskey(frequent_itemsets, antecedent) || continue
                haskey(frequent_itemsets, consequent) || continue
                supp_a = frequent_itemsets[antecedent]
                supp_c = frequent_itemsets[consequent]
                conf = supp_z / supp_a
                conf < min_confidence && continue
                lift = conf / supp_c
                leverage = supp_z - supp_a * supp_c
                conviction = conf >= 1.0 ? Inf : (1.0 - supp_c) / (1.0 - conf)
                push!(rules, Rule(antecedent, consequent, supp_z, conf, lift,
                                  leverage, conviction))
            end
        end
    end
    sort!(rules; by = r -> (-r.confidence, r.antecedent, r.consequent))
    return rules
end

end # module Ch150AssociationRules
