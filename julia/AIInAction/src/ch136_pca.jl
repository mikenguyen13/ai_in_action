"""
    Ch136Pca

Principal Component Analysis (PCA) from scratch (Julia).

Mirrors the Python module `aiinaction.ch136_pca` and the Rust module
`aiinaction::ch136_pca`. The shared fixtures in `test/test_ch136_pca.jl` match the
Python/Rust suites, which keeps the three implementations at parity.

The estimator centers each feature (always), optionally scales to unit standard
deviation (correlation PCA), and diagonalizes the small `d x d` covariance matrix
with a self-contained cyclic Jacobi eigensolver (Base-only, no LinearAlgebra
dependency). Each loading vector's sign is fixed so its largest-magnitude entry is
positive.
"""
module Ch136Pca

export PCAResult, fit_pca, transform, inverse_transform, reconstruction_error

"""The fitted state of a PCA model. `components` are rows (one loading per row)."""
struct PCAResult
    mean::Vector{Float64}
    scale::Vector{Float64}
    components::Matrix{Float64}
    explained_variance::Vector{Float64}
    explained_variance_ratio::Vector{Float64}
    whiten::Bool
end

n_components(r::PCAResult) = size(r.components, 1)
n_features(r::PCAResult) = size(r.components, 2)

function _as_matrix(X)
    A = Matrix{Float64}(X)
    n, d = size(A)
    n >= 2 || throw(ArgumentError("need at least 2 samples to estimate variance, got $n"))
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

function _fix_signs!(C::AbstractMatrix{Float64})
    for j in 1:size(C, 1)
        row = @view C[j, :]
        k = argmax(abs.(row))
        if row[k] < 0
            row .= .-row
        end
    end
    return C
end

"""
    fit_pca(X; n_components=nothing, scale=false, whiten=false)

Fit a PCA model to the `n x d` matrix `X` by diagonalizing the covariance matrix
of the centered (and optionally scaled) data. Defaults to `min(n, d)` components.
"""
function fit_pca(X; n_components=nothing, scale::Bool=false, whiten::Bool=false)
    A = _as_matrix(X)
    n, d = size(A)
    max_components = min(n, d)

    k = n_components === nothing ? max_components : Int(n_components)
    (1 <= k <= max_components) ||
        throw(ArgumentError("n_components must be in [1, $max_components] for a $(n)x$(d) matrix, got $k"))

    mu = vec(sum(A; dims=1)) ./ n
    Xc = A .- mu'

    scale_vec = ones(Float64, d)
    if scale
        for j in 1:d
            s = sqrt(sum(abs2, @view(Xc[:, j])) / (n - 1))
            s == 0 && throw(ArgumentError("cannot scale: feature $(j - 1) has zero variance"))
            scale_vec[j] = s
        end
        Xc = Xc ./ scale_vec'
    end

    # Covariance matrix C = Xc' Xc / (n - 1), shape d x d.
    cov = (Xc' * Xc) ./ (n - 1)
    eigvals, V = _jacobi_eigen(Matrix(cov))

    total_var = sum(max.(eigvals, 0.0))
    total_var == 0 &&
        throw(ArgumentError("X has zero total variance after centering; PCA is undefined"))

    # Component j is column j of V; store as rows.
    components = Matrix(transpose(V[:, 1:k]))
    _fix_signs!(components)
    explained_variance = max.(eigvals[1:k], 0.0)
    explained_variance_ratio = explained_variance ./ total_var

    return PCAResult(mu, scale_vec, components, explained_variance,
        explained_variance_ratio, whiten)
end

"""Project `X` onto the fitted components, returning an `n x n_components` matrix."""
function transform(model::PCAResult, X)
    A = Matrix{Float64}(X)
    size(A, 2) == n_features(model) ||
        throw(ArgumentError("X has $(size(A, 2)) features but model was fit on $(n_features(model))"))
    all(isfinite, A) || throw(ArgumentError("X contains non-finite values (nan or inf)"))

    Xc = (A .- model.mean') ./ model.scale'
    scores = Xc * model.components'
    if model.whiten
        std = sqrt.(model.explained_variance)
        any(std .== 0) && throw(ArgumentError("cannot whiten: a retained component has zero variance"))
        scores = scores ./ std'
    end
    return scores
end

"""Map scores back to the original feature space (best rank-m reconstruction)."""
function inverse_transform(model::PCAResult, scores)
    S = Matrix{Float64}(scores)
    size(S, 2) == n_components(model) ||
        throw(ArgumentError("scores has $(size(S, 2)) columns but model has $(n_components(model)) components"))

    t = model.whiten ? S .* sqrt.(model.explained_variance)' : S
    Xc = t * model.components
    return Xc .* model.scale' .+ model.mean'
end

"""Mean squared reconstruction error of `X` under the truncated model."""
function reconstruction_error(model::PCAResult, X)
    A = _as_matrix(X)
    recon = inverse_transform(model, transform(model, A))
    return sum(abs2, A .- recon) / size(A, 1)
end

end # module Ch136Pca
