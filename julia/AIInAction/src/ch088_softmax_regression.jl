"""
    Ch088SoftmaxRegression

Multiclass logistic regression (softmax regression) from scratch (Julia).
Mirrors the Python module `aiinaction.ch088_softmax_regression` and the Rust
module `aiinaction::ch088_softmax_regression`. The shared fixtures in
`test/test_ch088_softmax_regression.jl` match the Python/Rust suites to keep the
three implementations at parity (1e-9 tolerance).
"""
module Ch088SoftmaxRegression

export softmax, cross_entropy, SoftmaxRegression, fit!, predict_proba, predict

"""
    softmax(z) -> Matrix

Row-wise numerically stable softmax. `z` is an `(n, K)` matrix of logits; each
row is mapped to a probability distribution by subtracting the row maximum
before exponentiating (overflow-safe, result unchanged).
"""
function softmax(z::AbstractMatrix{<:Real})
    isempty(z) && throw(ArgumentError("z must be non-empty"))
    all(isfinite, z) || throw(ArgumentError("z must contain only finite values"))
    n, k = size(z)
    out = Matrix{Float64}(undef, n, k)
    for i in 1:n
        m = maximum(@view z[i, :])
        s = 0.0
        for c in 1:k
            out[i, c] = exp(z[i, c] - m)
            s += out[i, c]
        end
        for c in 1:k
            out[i, c] /= s
        end
    end
    return out
end

"""
    cross_entropy(probs, y) -> Float64

Mean multiclass cross-entropy. `y` holds 0-based integer labels in `[0, K)` to
match the Python/Rust API.
"""
function cross_entropy(probs::AbstractMatrix{<:Real}, y::AbstractVector{<:Integer})
    isempty(probs) && throw(ArgumentError("probs must be non-empty"))
    n, k = size(probs)
    n == length(y) ||
        throw(ArgumentError("length mismatch: probs has $n rows but y has $(length(y))"))
    eps = 1e-15
    total = 0.0
    for i in 1:n
        label = y[i]
        (0 <= label < k) ||
            throw(ArgumentError("labels must lie in [0, $k); got $label"))
        p = clamp(probs[i, label + 1], eps, 1.0)
        total += -log(p)
    end
    return total / n
end

"""
    SoftmaxRegression(; learning_rate=0.5, n_iter=500, l2=0.0)

Multinomial logistic regression trained by full-batch gradient descent. Holds
weights `W` of shape `(d, K)` and bias `b` of length `K` after `fit!`.
"""
mutable struct SoftmaxRegression
    learning_rate::Float64
    n_iter::Int
    l2::Float64
    W::Matrix{Float64}
    b::Vector{Float64}
    n_classes::Int
    fitted::Bool
end

function SoftmaxRegression(; learning_rate::Real=0.5, n_iter::Integer=500, l2::Real=0.0)
    learning_rate > 0 ||
        throw(ArgumentError("learning_rate must be positive, got $learning_rate"))
    n_iter > 0 || throw(ArgumentError("n_iter must be positive, got $n_iter"))
    l2 >= 0 || throw(ArgumentError("l2 must be non-negative, got $l2"))
    return SoftmaxRegression(Float64(learning_rate), Int(n_iter), Float64(l2),
                             zeros(0, 0), Float64[], 0, false)
end

"""
    fit!(model, X, y) -> model

Fit on features `X` (`n x d`) and 0-based integer labels `y` (`n`).
"""
function fit!(model::SoftmaxRegression, X::AbstractMatrix{<:Real},
              y::AbstractVector{<:Integer})
    isempty(X) && throw(ArgumentError("X must be non-empty"))
    n, d = size(X)
    n == length(y) ||
        throw(ArgumentError("length mismatch: X has $n rows but y has $(length(y))"))
    minimum(y) >= 0 || throw(ArgumentError("labels must be non-negative"))
    k = maximum(y) + 1
    k >= 2 || throw(ArgumentError("need at least 2 classes, got $k"))

    onehot = zeros(Float64, n, k)
    for i in 1:n
        onehot[i, y[i] + 1] = 1.0
    end

    W = zeros(Float64, d, k)
    b = zeros(Float64, k)
    for _ in 1:model.n_iter
        logits = X * W .+ b'
        probs = softmax(logits)
        diff = probs .- onehot
        grad_W = (X' * diff) ./ n .+ 2.0 * model.l2 .* W
        grad_b = vec(sum(diff; dims=1)) ./ n
        W .-= model.learning_rate .* grad_W
        b .-= model.learning_rate .* grad_b
    end

    model.W = W
    model.b = b
    model.n_classes = k
    model.fitted = true
    return model
end

"""
    predict_proba(model, X) -> Matrix

Predicted class-probability matrix (`n x K`).
"""
function predict_proba(model::SoftmaxRegression, X::AbstractMatrix{<:Real})
    model.fitted || throw(ArgumentError("model is not fitted; call fit! first"))
    isempty(X) && throw(ArgumentError("X must be non-empty"))
    d = size(model.W, 1)
    size(X, 2) == d ||
        throw(ArgumentError("X has $(size(X, 2)) features but model was fit on $d"))
    return softmax(X * model.W .+ model.b')
end

"""
    predict(model, X) -> Vector{Int}

Predicted 0-based integer class labels (argmax of the probabilities).
"""
function predict(model::SoftmaxRegression, X::AbstractMatrix{<:Real})
    probs = predict_proba(model, X)
    return [argmax(@view probs[i, :]) - 1 for i in 1:size(probs, 1)]
end

end # module Ch088SoftmaxRegression
