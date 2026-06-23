"""
    Ch126Diana

DIANA: divisive (top-down) hierarchical clustering.

Mirrors the Python module `aiinaction.ch126_diana` and the Rust module
`aiinaction::ch126_diana`. The shared fixtures in `test/test_ch126_diana.jl`
match the Python/Rust suites to keep the three at parity.

The algorithm is DIANA (Kaufman & Rousseeuw) with the Macnaughton-Smith
splinter-group heuristic. Distance matrices are square, symmetric, hollow,
non-negative `AbstractMatrix{<:Real}`; member indices are 1-based.
"""
module Ch126Diana

export Split, diameter, macnaughton_smith_split, diana, diana_labels

"""A single recorded division produced by DIANA (member indices are 1-based)."""
struct Split
    parent::Vector{Int}
    splinter::Vector{Int}
    remainder::Vector{Int}
    diameter::Float64
end

function _validate_matrix(dist::AbstractMatrix{<:Real})
    n, m = size(dist)
    n == m || throw(ArgumentError("distance matrix must be square 2-D; got size $(size(dist))"))
    n == 0 && throw(ArgumentError("distance matrix must be non-empty"))
    for i in 1:n
        abs(float(dist[i, i])) > 1e-12 &&
            throw(ArgumentError("distance matrix must have a zero diagonal"))
        for j in 1:n
            v = float(dist[i, j])
            isfinite(v) || throw(ArgumentError("distance matrix must contain only finite values"))
            v < 0 && throw(ArgumentError("distance matrix must be non-negative"))
            abs(v - float(dist[j, i])) > 1e-12 &&
                throw(ArgumentError("distance matrix must be symmetric"))
        end
    end
    return n
end

function _check_members(members::AbstractVector{<:Integer}, n::Int)
    isempty(members) && throw(ArgumentError("member list must be non-empty"))
    seen = falses(n)
    for i in members
        (1 <= i <= n) || throw(ArgumentError("member index $i out of range [1, $n]"))
        seen[i] && throw(ArgumentError("duplicate member index $i"))
        seen[i] = true
    end
    return nothing
end

"""
    diameter(dist, members)

Largest pairwise dissimilarity inside `members`; `0.0` for fewer than two members.
"""
function diameter(dist::AbstractMatrix{<:Real}, members::AbstractVector{<:Integer})
    n = _validate_matrix(dist)
    _check_members(members, n)
    length(members) < 2 && return 0.0
    d = 0.0
    @inbounds for a in 1:length(members), b in (a + 1):length(members)
        i, j = members[a], members[b]
        v = float(dist[i, j])
        v > d && (d = v)
    end
    return d
end

function _avg_to(dist, i::Int, group::AbstractVector{<:Integer}, exclude_self::Bool)
    s = 0.0
    c = 0
    @inbounds for j in group
        (exclude_self && j == i) && continue
        s += float(dist[i, j])
        c += 1
    end
    return s / c
end

"""
    macnaughton_smith_split(dist, members) -> (splinter, remainder)

Split one cluster into a splinter group and the remainder via Macnaughton-Smith.
Both returned vectors are sorted ascending.
"""
function macnaughton_smith_split(dist::AbstractMatrix{<:Real}, members::AbstractVector{<:Integer})
    n = _validate_matrix(dist)
    _check_members(members, n)

    if length(members) < 2
        return (Int[], sort(collect(Int, members)))
    end

    remainder = collect(Int, members)
    splinter = Int[]

    # Seed: object with the largest average dissimilarity to the rest of the cluster.
    best_seed = remainder[1]
    best_avg = -Inf
    for i in remainder
        avg = _avg_to(dist, i, remainder, true)
        if avg > best_avg
            best_avg = avg
            best_seed = i
        end
    end
    filter!(x -> x != best_seed, remainder)
    push!(splinter, best_seed)

    # Grow the splinter while some object prefers it.
    while length(remainder) > 1
        best_obj = 0
        best_d = 0.0
        for i in remainder
            avg_rem = _avg_to(dist, i, remainder, true)
            avg_spl = _avg_to(dist, i, splinter, false)
            d_i = avg_rem - avg_spl
            if d_i > best_d
                best_d = d_i
                best_obj = i
            end
        end
        best_obj == 0 && break
        filter!(x -> x != best_obj, remainder)
        push!(splinter, best_obj)
    end

    return (sort(splinter), sort(remainder))
end

"""
    diana(dist) -> Vector{Split}

Run DIANA to completion, returning splits in largest-diameter-first order
(`n - 1` entries).
"""
function diana(dist::AbstractMatrix{<:Real})
    n = _validate_matrix(dist)
    n == 1 && return Split[]

    clusters = Vector{Vector{Int}}()
    push!(clusters, collect(1:n))
    splits = Split[]

    while !isempty(clusters)
        best_k = 1
        best_diam = -1.0
        for (k, c) in enumerate(clusters)
            dm = diameter(dist, c)
            if dm > best_diam
                best_diam = dm
                best_k = k
            end
        end
        target = clusters[best_k]
        deleteat!(clusters, best_k)
        splinter, remainder = macnaughton_smith_split(dist, target)
        push!(splits, Split(sort(copy(target)), copy(splinter), copy(remainder), best_diam))
        for part in (splinter, remainder)
            length(part) > 1 && push!(clusters, part)
        end
    end

    return splits
end

"""
    diana_labels(dist, k) -> Vector{Int}

Cut the DIANA hierarchy into exactly `k` flat clusters, labelled `0..k-1` in
order of each cluster's smallest member index.
"""
function diana_labels(dist::AbstractMatrix{<:Real}, k::Integer)
    n = _validate_matrix(dist)
    (1 <= k <= n) || throw(ArgumentError("k must be in [1, $n]; got $k"))

    clusters = Vector{Vector{Int}}()
    push!(clusters, collect(1:n))
    while length(clusters) < k
        best_i = 0
        best_diam = -1.0
        for (i, c) in enumerate(clusters)
            length(c) < 2 && continue
            dm = diameter(dist, c)
            if dm > best_diam
                best_diam = dm
                best_i = i
            end
        end
        best_i == 0 && break
        target = clusters[best_i]
        deleteat!(clusters, best_i)
        splinter, remainder = macnaughton_smith_split(dist, target)
        push!(clusters, splinter)
        push!(clusters, remainder)
    end

    sort!(clusters, by = minimum)
    labels = zeros(Int, n)
    for (label, c) in enumerate(clusters)
        for obj in c
            labels[obj] = label - 1
        end
    end
    return labels
end

end # module Ch126Diana
