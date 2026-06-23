"""
    Ch071Eda

Easy Data Augmentation (EDA) for text classification (Julia).

Mirrors the Python module `aiinaction.ch071_eda` and the Rust module
`aiinaction::ch071_eda`. Implements the four EDA operations of Wei and Zou
(2019): synonym replacement, random insertion, random swap, random deletion.
Cross-language parity requires bit-identical randomness, so this uses the same
Park-Miller 32-bit LCG and the same fixed synonym table as the other two
implementations. The shared fixtures in `test/test_ch071_eda.jl` match the
Python/Rust suites.
"""
module Ch071Eda

export Lcg, next_u32!, next_float!, randint!, tokenize,
    synonym_replacement, random_insertion, random_swap, random_deletion,
    eda, SYNONYMS

const LCG_MOD = 2147483647   # 2^31 - 1
const LCG_MULT = 16807

# Fixed synonym table, identical to the Python/Rust versions.
const SYNONYMS = Dict{String,Vector{String}}(
    "quick" => ["fast", "rapid", "swift"],
    "fast" => ["quick", "rapid", "speedy"],
    "happy" => ["glad", "joyful", "cheerful"],
    "sad" => ["unhappy", "gloomy", "downcast"],
    "big" => ["large", "huge", "massive"],
    "small" => ["tiny", "little", "compact"],
    "good" => ["great", "fine", "decent"],
    "bad" => ["poor", "awful", "lousy"],
    "smart" => ["clever", "bright", "sharp"],
    "movie" => ["film", "picture", "feature"],
    "car" => ["automobile", "vehicle", "auto"],
    "house" => ["home", "dwelling", "residence"],
)

"""A 32-bit Park-Miller linear congruential generator."""
mutable struct Lcg
    state::Int64
    function Lcg(seed::Integer)
        seed < 0 && throw(ArgumentError("seed must be non-negative, got $seed"))
        s = Int64(seed) % LCG_MOD
        s == 0 && (s = 1)
        new(s)
    end
end

"""Advance the generator and return the new 31-bit state."""
function next_u32!(rng::Lcg)
    rng.state = (rng.state * LCG_MULT) % LCG_MOD
    return rng.state
end

"""Return the next value mapped to the half-open interval `[0, 1)`."""
next_float!(rng::Lcg) = (next_u32!(rng) - 1) / (LCG_MOD - 1)

"""Return an integer in `0:(n-1)`. Throws if `n <= 0`."""
function randint!(rng::Lcg, n::Integer)
    n <= 0 && throw(ArgumentError("randint bound must be positive, got $n"))
    return next_u32!(rng) % n
end

"""Split text into whitespace-delimited tokens. Throws on empty input."""
function tokenize(text::AbstractString)
    tokens = String.(split(text))
    isempty(tokens) && throw(ArgumentError("text must contain at least one token"))
    return tokens
end

_synonyms_for(word::AbstractString, table) = get(table, lowercase(word), String[])
_has_any_synonym(tokens, table) = any(!isempty(_synonyms_for(t, table)) for t in tokens)

"""Replace up to `n` words with a synonym from `table`."""
function synonym_replacement(tokens::AbstractVector{<:AbstractString}, n::Integer,
                             rng::Lcg, table=SYNONYMS)
    isempty(tokens) && throw(ArgumentError("tokens must be non-empty"))
    n < 0 && throw(ArgumentError("n must be non-negative, got $n"))
    out = String.(collect(tokens))
    candidate_idx = [i for (i, t) in enumerate(out) if !isempty(_synonyms_for(t, table))]
    isempty(candidate_idx) && return out
    for _ in 1:n
        pos = candidate_idx[randint!(rng, length(candidate_idx)) + 1]
        cands = _synonyms_for(out[pos], table)
        out[pos] = cands[randint!(rng, length(cands)) + 1]
    end
    return out
end

"""Insert `n` synonyms of random words at random positions."""
function random_insertion(tokens::AbstractVector{<:AbstractString}, n::Integer,
                          rng::Lcg, table=SYNONYMS)
    isempty(tokens) && throw(ArgumentError("tokens must be non-empty"))
    n < 0 && throw(ArgumentError("n must be non-negative, got $n"))
    out = String.(collect(tokens))
    _has_any_synonym(out, table) || return out
    for _ in 1:n
        word_synonyms = String[]
        for _attempt in 1:10
            cand = out[randint!(rng, length(out)) + 1]
            word_synonyms = _synonyms_for(cand, table)
            isempty(word_synonyms) || break
        end
        isempty(word_synonyms) && continue
        new_word = word_synonyms[randint!(rng, length(word_synonyms)) + 1]
        insert_at = randint!(rng, length(out) + 1)  # 0-based position
        insert!(out, insert_at + 1, new_word)
    end
    return out
end

"""Swap two random tokens `n` times."""
function random_swap(tokens::AbstractVector{<:AbstractString}, n::Integer, rng::Lcg)
    isempty(tokens) && throw(ArgumentError("tokens must be non-empty"))
    n < 0 && throw(ArgumentError("n must be non-negative, got $n"))
    out = String.(collect(tokens))
    length(out) < 2 && return out
    for _ in 1:n
        i = randint!(rng, length(out)) + 1
        j = randint!(rng, length(out)) + 1
        out[i], out[j] = out[j], out[i]
    end
    return out
end

"""Delete each token independently with probability `p`; never returns empty."""
function random_deletion(tokens::AbstractVector{<:AbstractString}, p::Real, rng::Lcg)
    isempty(tokens) && throw(ArgumentError("tokens must be non-empty"))
    (0.0 <= p <= 1.0) || throw(ArgumentError("p must be in [0, 1], got $p"))
    out = String.(collect(tokens))
    length(out) == 1 && return out
    kept = String[]
    for t in out
        if next_float!(rng) >= p
            push!(kept, t)
        end
    end
    isempty(kept) && (kept = [out[randint!(rng, length(out)) + 1]])
    return kept
end

_n_for(alpha, n_words) = max(1, round(Int, alpha * n_words))

"""Generate `num_aug` augmented sentences from `text`."""
function eda(text::AbstractString; alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1,
             p_rd=0.1, num_aug::Integer=4, seed::Integer=0, table=SYNONYMS)
    num_aug < 0 && throw(ArgumentError("num_aug must be non-negative, got $num_aug"))
    for (name, a) in (("alpha_sr", alpha_sr), ("alpha_ri", alpha_ri), ("alpha_rs", alpha_rs))
        (0.0 <= a <= 1.0) || throw(ArgumentError("$name must be in [0, 1], got $a"))
    end
    (0.0 <= p_rd <= 1.0) || throw(ArgumentError("p_rd must be in [0, 1], got $p_rd"))
    tokens = tokenize(text)
    n_words = length(tokens)
    rng = Lcg(seed)
    out = String[]
    for i in 0:(num_aug - 1)
        op = i % 4
        if op == 0
            aug = synonym_replacement(tokens, _n_for(alpha_sr, n_words), rng, table)
        elseif op == 1
            aug = random_insertion(tokens, _n_for(alpha_ri, n_words), rng, table)
        elseif op == 2
            aug = random_swap(tokens, _n_for(alpha_rs, n_words), rng)
        else
            aug = random_deletion(tokens, p_rd, rng)
        end
        push!(out, join(aug, " "))
    end
    return out
end

end # module Ch071Eda
