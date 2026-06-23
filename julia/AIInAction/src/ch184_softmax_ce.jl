"""
    Ch184SoftmaxCE

Softmax cross-entropy loss and its gradient, from scratch (Julia).

Mirrors the Python module `aiinaction.ch184_softmax_ce` and the Rust module
`aiinaction::ch184_softmax_ce`. The shared fixtures in `test/test_ch184_softmax_ce.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

Everything is computed in a numerically stable way directly from logits: the
per-row maximum is subtracted before exponentiating, and the loss uses the fused
identity `log softmax(z)_k = z_k - logsumexp(z)`. The gradient is the clean
predicted-minus-target form `(p - q') / N`.

Inputs are `N x K` matrices of logits (`N` samples of `K >= 2` class scores).
Labels are integer class indices in `1:K` (1-based, matching Julia convention).
Optional label smoothing replaces the one-hot target `q` with
`q'(k) = (1 - eps) * 1[k = y] + eps / K`.
"""
module Ch184SoftmaxCE

export softmax, log_softmax, cross_entropy_loss, cross_entropy_grad

function _check_logits(Z)
    A = Matrix{Float64}(Z)
    n, k = size(A)
    n >= 1 || throw(ArgumentError("need at least one sample (N >= 1)"))
    k >= 2 || throw(ArgumentError("need at least 2 classes (K >= 2), got K=$k"))
    all(isfinite, A) || throw(ArgumentError("logits contain non-finite values (nan or inf)"))
    return A, n, k
end

function _check_labels(labels, n, k)
    length(labels) == n ||
        throw(ArgumentError("labels has length $(length(labels)) but logits has $n rows"))
    all(y -> 1 <= y <= k, labels) ||
        throw(ArgumentError("labels must be in [1, $k]"))
    return nothing
end

function _check_smoothing(eps)
    (0.0 <= eps < 1.0) ||
        throw(ArgumentError("label_smoothing must be in [0, 1), got $eps"))
    return Float64(eps)
end

function _softmax_row(row::AbstractVector{Float64})
    m = maximum(row)
    ex = exp.(row .- m)
    return ex ./ sum(ex)
end

function _log_softmax_row(row::AbstractVector{Float64})
    m = maximum(row)
    shifted = row .- m
    lse = log(sum(exp, shifted))
    return shifted .- lse
end

"""Row-wise softmax of an `N x K` logit matrix."""
function softmax(Z)
    A, n, _ = _check_logits(Z)
    out = similar(A)
    for i in 1:n
        out[i, :] = _softmax_row(@view A[i, :])
    end
    return out
end

"""Row-wise log-softmax of an `N x K` logit matrix, without forming the softmax."""
function log_softmax(Z)
    A, n, _ = _check_logits(Z)
    out = similar(A)
    for i in 1:n
        out[i, :] = _log_softmax_row(@view A[i, :])
    end
    return out
end

"""
    cross_entropy_loss(Z, labels; label_smoothing=0.0)

Mean softmax cross-entropy loss over a batch of logits. `label_smoothing = 0.0` is
standard hard-target cross-entropy; with `eps > 0` the one-hot target is replaced
by `q'(k) = (1 - eps) 1[k = y] + eps / K`. Labels are 1-based class indices.
"""
function cross_entropy_loss(Z, labels; label_smoothing::Real=0.0)
    A, n, k = _check_logits(Z)
    _check_labels(labels, n, k)
    eps = _check_smoothing(label_smoothing)

    total = 0.0
    for i in 1:n
        logp = _log_softmax_row(@view A[i, :])
        correct = logp[labels[i]]
        if eps == 0.0
            total += -correct
        else
            uniform = sum(logp) / k
            total += -((1.0 - eps) * correct + eps * uniform)
        end
    end
    return total / n
end

"""
    cross_entropy_grad(Z, labels; label_smoothing=0.0)

Gradient of [`cross_entropy_loss`](@ref) with respect to the logits: the clean
predicted-minus-target form `(p - q') / N`, returned as an `N x K` matrix.
"""
function cross_entropy_grad(Z, labels; label_smoothing::Real=0.0)
    A, n, k = _check_logits(Z)
    _check_labels(labels, n, k)
    eps = _check_smoothing(label_smoothing)

    out = Matrix{Float64}(undef, n, k)
    for i in 1:n
        p = _softmax_row(@view A[i, :])
        for j in 1:k
            q = eps / k
            if j == labels[i]
                q += 1.0 - eps
            end
            out[i, j] = (p[j] - q) / n
        end
    end
    return out
end

end # module Ch184SoftmaxCE
