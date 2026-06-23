"""
    Ch131SpectralClustering

Spectral clustering from scratch (chapter 131). Mirrors the Python module
`aiinaction.ch131_spectral_clustering` and the Rust module
`aiinaction::ch131_spectral_clustering` one-to-one. The eigensolver (cyclic
Jacobi) and k-means are implemented from scratch and made deterministic so all
three language ports agree to within `1e-9` on the shared fixtures.
"""
module Ch131SpectralClustering

export rbf_affinity, normalized_laplacian, jacobi_eigh,
    spectral_embedding, kmeans, spectral_clustering

function _validate_matrix(X::AbstractMatrix{<:Real})
    size(X, 1) == 0 && throw(ArgumentError("X must contain at least one sample"))
    all(isfinite, X) || throw(ArgumentError("X must contain only finite values"))
    return nothing
end

"""Gaussian (RBF) similarity matrix with a zero diagonal."""
function rbf_affinity(X::AbstractMatrix{<:Real}, sigma::Real)
    _validate_matrix(X)
    (isfinite(sigma) && sigma > 0) ||
        throw(ArgumentError("sigma must be a positive finite number; got $sigma"))
    n = size(X, 1)
    denom = 2.0 * sigma * sigma
    W = zeros(Float64, n, n)
    for i in 1:n
        for j in (i + 1):n
            diff = @view(X[i, :]) .- @view(X[j, :])
            w = exp(-sum(abs2, diff) / denom)
            W[i, j] = w
            W[j, i] = w
        end
    end
    return W
end

"""Symmetric normalized Laplacian `L_sym = I - D^{-1/2} W D^{-1/2}`."""
function normalized_laplacian(W::AbstractMatrix{<:Real})
    n = size(W, 1)
    n == 0 && throw(ArgumentError("W must be non-empty"))
    size(W, 2) == n || throw(ArgumentError("W must be a square matrix"))
    all(isfinite, W) || throw(ArgumentError("W must contain only finite values"))
    all(>=(0), W) || throw(ArgumentError("W must have non-negative entries"))
    for i in 1:n, j in 1:n
        abs(W[i, j] - W[j, i]) > 1e-12 && throw(ArgumentError("W must be symmetric"))
    end
    deg = vec(sum(W, dims=2))
    all(>(0), deg) ||
        throw(ArgumentError("every vertex must have positive degree (no isolated vertices)"))
    dinv = 1.0 ./ sqrt.(deg)
    L = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        id = i == j ? 1.0 : 0.0
        L[i, j] = id - dinv[i] * W[i, j] * dinv[j]
    end
    for i in 1:n, j in (i + 1):n
        m = 0.5 * (L[i, j] + L[j, i])
        L[i, j] = m
        L[j, i] = m
    end
    return L
end

"""
Eigen-decomposition of a real symmetric matrix via cyclic Jacobi rotations.
Returns `(eigenvalues, eigenvectors)` with ascending eigenvalues and
sign-fixed eigenvectors as columns.
"""
function jacobi_eigh(A::AbstractMatrix{<:Real})
    n = size(A, 1)
    n == 0 && throw(ArgumentError("A must be non-empty"))
    size(A, 2) == n || throw(ArgumentError("A must be a square matrix"))
    for i in 1:n, j in 1:n
        abs(A[i, j] - A[j, i]) > 1e-12 && throw(ArgumentError("A must be symmetric"))
    end
    max_sweeps = 100
    tol = 1e-12
    M = Matrix{Float64}(A)
    V = Matrix{Float64}(I_eye(n))

    for _ in 1:max_sweeps
        off = 0.0
        for p in 1:n, q in (p + 1):n
            off += M[p, q]^2
        end
        sqrt(off) <= tol && break
        for p in 1:n
            for q in (p + 1):n
                apq = M[p, q]
                abs(apq) <= 1e-300 && continue
                app = M[p, p]
                aqq = M[q, q]
                phi = 0.5 * atan(2.0 * apq, aqq - app)
                c = cos(phi)
                s = sin(phi)
                for i in 1:n
                    mip = M[i, p]
                    miq = M[i, q]
                    M[i, p] = c * mip - s * miq
                    M[i, q] = s * mip + c * miq
                end
                for i in 1:n
                    mpi = M[p, i]
                    mqi = M[q, i]
                    M[p, i] = c * mpi - s * mqi
                    M[q, i] = s * mpi + c * mqi
                end
                for i in 1:n
                    vip = V[i, p]
                    viq = V[i, q]
                    V[i, p] = c * vip - s * viq
                    V[i, q] = s * vip + c * viq
                end
            end
        end
    end

    eig = [M[i, i] for i in 1:n]
    order = sortperm(eig)
    eigvals = eig[order]
    vecs = V[:, order]
    for j in 1:n
        for i in 1:n
            if abs(vecs[i, j]) > 1e-12
                if vecs[i, j] < 0.0
                    @views vecs[:, j] .= .-vecs[:, j]
                end
                break
            end
        end
    end
    return eigvals, vecs
end

# Small identity helper to avoid a LinearAlgebra dependency.
function I_eye(n::Int)
    M = zeros(Float64, n, n)
    for i in 1:n
        M[i, i] = 1.0
    end
    return M
end

"""Row-normalized spectral embedding: the `k` smallest eigenvectors of `L_sym`."""
function spectral_embedding(W::AbstractMatrix{<:Real}, k::Integer)
    L = normalized_laplacian(W)
    n = size(L, 1)
    (1 <= k <= n) || throw(ArgumentError("k must be an integer in [1, n]=$n; got $k"))
    _, vecs = jacobi_eigh(L)
    U = Matrix{Float64}(vecs[:, 1:k])
    for i in 1:n
        nrm = sqrt(sum(abs2, @view U[i, :]))
        if nrm > 1e-12
            @views U[i, :] ./= nrm
        end
    end
    return U
end

"""Lloyd's k-means with deterministic furthest-point seeding. Returns `(labels, centers)`."""
function kmeans(X::AbstractMatrix{<:Real}, k::Integer)
    _validate_matrix(X)
    n, d = size(X)
    (1 <= k <= n) || throw(ArgumentError("k must be an integer in [1, n]=$n; got $k"))
    max_iter = 100
    tol = 1e-10

    first = 1
    for i in 2:n
        if _lex_less(@view(X[i, :]), @view(X[first, :]))
            first = i
        end
    end
    center_idx = [first]
    while length(center_idx) < k
        best_i = -1
        best_d = -1.0
        for i in 1:n
            i in center_idx && continue
            dist = minimum(sum(abs2, @view(X[i, :]) .- @view(X[c, :])) for c in center_idx)
            if dist > best_d
                best_d = dist
                best_i = i
            end
        end
        push!(center_idx, best_i)
    end
    centers = Matrix{Float64}(X[center_idx, :])

    labels = zeros(Int, n)
    for _ in 1:max_iter
        for i in 1:n
            best_c = 1
            best_d = Inf
            for c in 1:k
                dist = sum(abs2, @view(X[i, :]) .- @view(centers[c, :]))
                if dist < best_d
                    best_d = dist
                    best_c = c
                end
            end
            labels[i] = best_c
        end
        new_centers = copy(centers)
        for c in 1:k
            members = [i for i in 1:n if labels[i] == c]
            if !isempty(members)
                for t in 1:d
                    new_centers[c, t] = sum(X[i, t] for i in members) / length(members)
                end
            end
        end
        shift = sqrt(sum(abs2, new_centers .- centers))
        centers = new_centers
        shift <= tol && break
    end
    # Return labels in 0-based form to match the Python/Rust ports.
    return labels .- 1, centers
end

function _lex_less(a, b)
    for (x, y) in zip(a, b)
        x < y && return true
        x > y && return false
    end
    return false
end

"""End-to-end spectral clustering (Ng-Jordan-Weiss). Returns 0-based labels."""
function spectral_clustering(X::AbstractMatrix{<:Real}, k::Integer, sigma::Real=1.0)
    W = rbf_affinity(X, sigma)
    embedding = spectral_embedding(W, k)
    labels, _ = kmeans(embedding, k)
    return labels
end

end # module Ch131SpectralClustering
