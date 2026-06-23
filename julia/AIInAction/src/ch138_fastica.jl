"""
    Ch138Fastica

FastICA for Independent Component Analysis (from scratch, Julia).

Mirrors the Python module `aiinaction.ch138_fastica` and the Rust module
`aiinaction::ch138_fastica`. The shared fixtures in `test/test_ch138_fastica.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

The estimator centers the data, whitens it via the covariance eigendecomposition,
then runs the Hyvarinen-Oja FastICA fixed-point iteration with the `logcosh`
contrast (`g(u) = tanh(u)`) and symmetric orthogonalization
`W <- (W W^T)^{-1/2} W`. For cross-language determinism the unmixing matrix is
initialized to the identity, a fixed iteration count is used (no tolerance stop),
and a self-contained cyclic Jacobi eigensolver (the same one used by the PCA
chapter) drives both whitening and orthogonalization. Components are ordered by
descending recovered-source variance and each unmixing row's largest-magnitude
entry is forced positive.
"""
module Ch138Fastica

export ICAResult, fit_ica, transform

"""The fitted state of a FastICA model. `components` are rows (one source per row)."""
struct ICAResult
    mean::Vector{Float64}
    whitening::Matrix{Float64}
    unmixing::Matrix{Float64}
    components::Matrix{Float64}
    mixing::Matrix{Float64}
    n_iter::Int
end

n_components(r::ICAResult) = size(r.components, 1)
n_features(r::ICAResult) = size(r.components, 2)

function _as_matrix(X)
    A = Matrix{Float64}(X)
    n, d = size(A)
    n >= 2 || throw(ArgumentError("need at least 2 samples, got $n"))
    d >= 1 || throw(ArgumentError("X must have at least one feature"))
    all(isfinite, A) || throw(ArgumentError("X contains non-finite values (nan or inf)"))
    return A
end

"""Symmetric eigendecomposition by the cyclic Jacobi method.

Returns `(eigenvalues, V)` sorted by descending eigenvalue, where eigenvector `k`
is column `k` of `V`."""
function _jacobi_eigen(A0::Matrix{Float64})
    n = size(A0, 1)
    A = copy(A0)
    V = Matrix{Float64}(undef, n, n)
    fill!(V, 0.0)
    @inbounds for i in 1:n
        V[i, i] = 1.0
    end
    for _ in 1:100
        off = 0.0
        @inbounds for p in 1:n, q in (p + 1):n
            off += A[p, q]^2
        end
        off < 1e-30 && break
        @inbounds for p in 1:n, q in (p + 1):n
            apq = A[p, q]
            abs(apq) < 1e-300 && continue
            theta = (A[q, q] - A[p, p]) / (2 * apq)
            t = theta == 0 ? 1.0 : sign(theta) / (abs(theta) + sqrt(theta^2 + 1))
            c = 1 / sqrt(t^2 + 1)
            s = t * c
            for k in 1:n
                akp = A[k, p]
                akq = A[k, q]
                A[k, p] = c * akp - s * akq
                A[k, q] = s * akp + c * akq
            end
            for k in 1:n
                apk = A[p, k]
                aqk = A[q, k]
                A[p, k] = c * apk - s * aqk
                A[q, k] = s * apk + c * aqk
            end
            for k in 1:n
                vkp = V[k, p]
                vkq = V[k, q]
                V[k, p] = c * vkp - s * vkq
                V[k, q] = s * vkp + c * vkq
            end
        end
    end
    eigvals = [A[i, i] for i in 1:n]
    order = sortperm(eigvals; rev=true)
    return eigvals[order], V[:, order]
end

"""Symmetric orthogonalization `W <- (W W^T)^{-1/2} W` via Jacobi eigen."""
function _symmetric_decorrelation(W::Matrix{Float64})
    eigvals, V = _jacobi_eigen(W * W')
    inv_sqrt = V * Diagonal_like(1.0 ./ sqrt.(eigvals)) * V'
    return inv_sqrt * W
end

# Build a dense diagonal matrix without depending on LinearAlgebra.Diagonal.
function Diagonal_like(v::Vector{Float64})
    n = length(v)
    M = zeros(Float64, n, n)
    @inbounds for i in 1:n
        M[i, i] = v[i]
    end
    return M
end

function _fix_signs!(W::AbstractMatrix{Float64})
    for j in 1:size(W, 1)
        row = @view W[j, :]
        k = argmax(abs.(row))
        if row[k] < 0
            row .= .-row
        end
    end
    return W
end

"""Moore-Penrose pseudoinverse of a `k x d` matrix via the normal equations.

For the square, full-rank operators produced here this is the exact inverse:
`A^+ = A^T (A A^T)^{-1}`, returning a `d x k` matrix."""
function _pseudoinverse(A::Matrix{Float64})
    k = size(A, 1)
    G = A * A'
    eigvals, V = _jacobi_eigen(G)
    Ginv = V * Diagonal_like(1.0 ./ eigvals) * V'
    return A' * Ginv
end

"""
    fit_ica(X; n_components=nothing, max_iter=200)

Fit a FastICA model to the `n x d` matrix `X` (samples in rows, signals in
columns). Defaults to `d` components. `max_iter` is a fixed iteration count (no
tolerance stop) so the result is deterministic across languages.
"""
function fit_ica(X; n_components=nothing, max_iter::Integer=200)
    A = _as_matrix(X)
    n, d = size(A)

    k = n_components === nothing ? d : Int(n_components)
    (1 <= k <= d) ||
        throw(ArgumentError("n_components must be in [1, $d] for a $(n)x$(d) matrix, got $k"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be a positive integer, got $max_iter"))

    # 1. Center.
    mu = vec(sum(A; dims=1)) ./ n
    Xc = A .- mu'

    # 2. Whiten. Cov = Xc' Xc / (n - 1), d x d.
    cov = (Xc' * Xc) ./ (n - 1)
    eigvals, E = _jacobi_eigen(Matrix(cov))
    any(eigvals[1:k] .<= 0.0) &&
        throw(ArgumentError("data is rank-deficient: a retained whitening direction has zero variance"))
    Ek = E[:, 1:k]
    # K maps centered data to whitened: z = K x_c, shape k x d.
    K = (Ek ./ sqrt.(eigvals[1:k])')'
    K = Matrix{Float64}(K)
    Z = Xc * K'  # n x k

    # 3. FastICA fixed-point iteration with symmetric orthogonalization.
    W = _symmetric_decorrelation(Matrix{Float64}(I_like(k)))
    n_iter = 0
    for _ in 1:max_iter
        n_iter += 1
        WZ = Z * W'              # n x k, column j is w_j^T z
        gwz = tanh.(WZ)
        gprime = 1.0 .- gwz .^ 2
        # w_j^+ = E[z g(w^T z)] - E[g'(w^T z)] w_j
        Wnew = (gwz' * Z) ./ n .- (vec(sum(gprime; dims=1) ./ n) .* W)
        W = _symmetric_decorrelation(Matrix{Float64}(Wnew))
    end
    _fix_signs!(W)

    components = W * K  # k x d

    # Order by descending recovered-source variance.
    S = Xc * components'  # n x k
    Smu = vec(sum(S; dims=1) ./ n)
    var = vec(sum(abs2, S .- Smu'; dims=1) ./ (n - 1))
    order = sortperm(var; rev=true)
    W = W[order, :]
    components = components[order, :]

    mixing = _pseudoinverse(Matrix{Float64}(components))

    return ICAResult(mu, K, Matrix{Float64}(W), Matrix{Float64}(components),
        Matrix{Float64}(mixing), n_iter)
end

# Identity matrix without depending on LinearAlgebra.I.
function I_like(n::Int)
    M = zeros(Float64, n, n)
    @inbounds for i in 1:n
        M[i, i] = 1.0
    end
    return M
end

"""Recover the independent sources from observed mixtures `X`, returning `n x k`."""
function transform(model::ICAResult, X)
    A = Matrix{Float64}(X)
    size(A, 2) == n_features(model) ||
        throw(ArgumentError("X has $(size(A, 2)) features but model was fit on $(n_features(model))"))
    all(isfinite, A) || throw(ArgumentError("X contains non-finite values (nan or inf)"))

    Xc = A .- model.mean'
    return Xc * model.components'
end

end # module Ch138Fastica
