"""
    Ch089SoftmaxRegression

Softmax regression from scratch (Julia). Mirrors the Python module
`aiinaction.ch089_softmax_regression` and the Rust module
`aiinaction::ch089_softmax_regression`. Shared fixtures in
`test/test_ch089_softmax_regression.jl` match the Python/Rust suites to keep the
three implementations at parity.

Class labels in the public API are 0-based integers (as in Python and Rust).
"""
module Ch089SoftmaxRegression

export softmax, log_sum_exp, cross_entropy_from_logits, SoftmaxRegression,
    fit!, predict_proba, predict, loss

"""Numerically stable softmax with optional temperature `T > 0`."""
function softmax(z::AbstractVector{<:Real}; temperature::Real=1.0)
    isempty(z) && throw(ArgumentError("z must be non-empty"))
    temperature > 0 || throw(ArgumentError("temperature must be positive, got $temperature"))
    scaled = [v / temperature for v in z]
    m = maximum(scaled)
    exps = [exp(v - m) for v in scaled]
    total = sum(exps)
    return exps ./ total
end

"""Stable `log(sum(exp(z)))` via the max-subtraction trick."""
function log_sum_exp(z::AbstractVector{<:Real})
    isempty(z) && throw(ArgumentError("z must be non-empty"))
    m = maximum(z)
    return m + log(sum(exp(v - m) for v in z))
end

"""Fused, stable cross-entropy from logits: `log_sum_exp(z) - z[label]`.

`label` is a 0-based class index, so the loss uses `z[label + 1]` internally.
"""
function cross_entropy_from_logits(z::AbstractVector{<:Real}, label::Integer)
    isempty(z) && throw(ArgumentError("z must be non-empty"))
    (0 <= label < length(z)) ||
        throw(ArgumentError("label $label out of range for $(length(z)) classes"))
    return log_sum_exp(z) - z[label + 1]
end

"""Multinomial logistic (softmax) regression trained by batch gradient descent.

`W` is a `(K, d)` weight matrix and `b` a length-`K` bias vector, both starting
at zero so fitting is deterministic.
"""
mutable struct SoftmaxRegression
    learning_rate::Float64
    n_iter::Int
    l2::Float64
    W::Matrix{Float64}
    b::Vector{Float64}
    fitted::Bool
    function SoftmaxRegression(; learning_rate::Real=0.5, n_iter::Integer=200, l2::Real=0.0)
        learning_rate > 0 ||
            throw(ArgumentError("learning_rate must be positive, got $learning_rate"))
        n_iter >= 1 || throw(ArgumentError("n_iter must be >= 1, got $n_iter"))
        l2 >= 0 || throw(ArgumentError("l2 must be non-negative, got $l2"))
        return new(Float64(learning_rate), Int(n_iter), Float64(l2),
            zeros(0, 0), Float64[], false)
    end
end

function _softmax_row(z::AbstractVector{<:Real})
    m = maximum(z)
    e = exp.(z .- m)
    return e ./ sum(e)
end

# Logits for one feature row: z = W x + b (length K).
_logits(model::SoftmaxRegression, x::AbstractVector{<:Real}) = model.W * x .+ model.b

"""Fit on rows `X` (N×d) with 0-based integer labels `y` (length N)."""
function fit!(model::SoftmaxRegression, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Integer})
    n, d = size(X)
    n == 0 && throw(ArgumentError("X must contain at least one row"))
    length(y) == n ||
        throw(ArgumentError("X and y disagree on N: $n != $(length(y))"))
    minimum(y) >= 0 || throw(ArgumentError("class labels must be non-negative"))
    k = maximum(y) + 1
    model.W = zeros(k, d)
    model.b = zeros(k)
    nf = Float64(n)
    for _ in 1:model.n_iter
        gradW = zeros(k, d)
        gradb = zeros(k)
        for i in 1:n
            xi = @view X[i, :]
            p = _softmax_row(_logits(model, xi))
            for c in 1:k
                g = p[c] - (c == y[i] + 1 ? 1.0 : 0.0)
                gradb[c] += g
                @inbounds for j in 1:d
                    gradW[c, j] += g * xi[j]
                end
            end
        end
        model.b .-= model.learning_rate .* (gradb ./ nf)
        model.W .-= model.learning_rate .* (gradW ./ nf .+ model.l2 .* model.W)
    end
    model.fitted = true
    return model
end

"""Class-probability matrix `(N, K)` for rows of `X`."""
function predict_proba(model::SoftmaxRegression, X::AbstractMatrix{<:Real})
    model.fitted || throw(ArgumentError("model is not fitted; call fit! first"))
    size(X, 2) == size(model.W, 2) ||
        throw(ArgumentError("X has $(size(X, 2)) features but model expects $(size(model.W, 2))"))
    n = size(X, 1)
    P = Matrix{Float64}(undef, n, size(model.W, 1))
    for i in 1:n
        P[i, :] = _softmax_row(_logits(model, @view X[i, :]))
    end
    return P
end

"""Predicted 0-based class indices (argmax) for rows of `X`."""
function predict(model::SoftmaxRegression, X::AbstractMatrix{<:Real})
    P = predict_proba(model, X)
    return [argmax(@view P[i, :]) - 1 for i in 1:size(P, 1)]
end

"""Mean cross-entropy loss (excluding L2) over `(X, y)`."""
function loss(model::SoftmaxRegression, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Integer})
    model.fitted || throw(ArgumentError("model is not fitted; call fit! first"))
    length(y) == size(X, 1) ||
        throw(ArgumentError("X and y disagree on N: $(size(X, 1)) != $(length(y))"))
    total = 0.0
    for i in 1:size(X, 1)
        total += cross_entropy_from_logits(_logits(model, @view X[i, :]), y[i])
    end
    return total / size(X, 1)
end

end # module Ch089SoftmaxRegression
