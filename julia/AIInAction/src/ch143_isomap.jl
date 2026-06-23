"""
    Ch143Isomap

Isomap (isometric feature mapping) from scratch (Julia).

Mirrors the Python module `aiinaction.ch143_isomap` and the Rust module
`aiinaction::ch143_isomap`. The shared fixtures in `test/test_ch143_isomap.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

The three stages are: build a symmetric k-nearest-neighbor graph weighted by
Euclidean distance; approximate geodesics with all-pairs shortest paths
(Floyd-Warshall); and embed via classical MDS, diagonalizing the double-centered
Gram matrix with a self-contained cyclic Jacobi eigensolver (Base-only, no
LinearAlgebra dependency). Each embedding coordinate's sign is fixed so its
largest-magnitude entry is positive.
"""
module Ch143Isomap

export IsomapResult, pairwise_distances, knn_graph, graph_shortest_paths,
    classical_mds, fit_isomap

"""The fitted state of an Isomap embedding. `embedding` rows are embedded samples."""
struct IsomapResult
    embedding::Matrix{Float64}
    eigenvalues::Vector{Float64}
    geodesic_distances::Matrix{Float64}
    n_neighbors::Int
end

n_samples(r::IsomapResult) = size(r.embedding, 1)
n_components(r::IsomapResult) = size(r.embedding, 2)

function _as_matrix(X)
    A = Matrix{Float64}(X)
    n, d = size(A)
    n >= 2 || throw(ArgumentError("need at least 2 samples, got $n"))
    d >= 1 || throw(ArgumentError("X must have at least one feature"))
    all(isfinite, A) || throw(ArgumentError("X contains non-finite values (nan or inf)"))
    return A
end

"""Euclidean distance matrix of the rows of `X`, shape `(n, n)`."""
function pairwise_distances(X)
    A = _as_matrix(X)
    n, d = size(A)
    D = zeros(Float64, n, n)
    @inbounds for i in 1:n, j in (i + 1):n
        ss = 0.0
        for c in 1:d
            diff = A[i, c] - A[j, c]
            ss += diff * diff
        end
        dist = sqrt(max(ss, 0.0))
        D[i, j] = dist
        D[j, i] = dist
    end
    return D
end

"""Symmetric weighted k-nearest-neighbor graph as an `(n, n)` adjacency matrix.

Non-edges are `Inf`, the diagonal is `0`."""
function knn_graph(X, n_neighbors::Integer)
    A = _as_matrix(X)
    n = size(A, 1)
    (1 <= n_neighbors <= n - 1) ||
        throw(ArgumentError("n_neighbors must be in [1, $(n - 1)] for $n samples, got $n_neighbors"))
    D = pairwise_distances(A)
    adj = fill(Inf, n, n)
    for i in 1:n
        adj[i, i] = 0.0
        others = [j for j in 1:n if j != i]
        # Stable sort by distance, tie-break on index.
        order = sort(others; by = j -> (D[i, j], j))
        for j in order[1:n_neighbors]
            adj[i, j] = D[i, j]
        end
    end
    # Symmetrize: keep an edge if it exists in either direction (min weight).
    sym = fill(Inf, n, n)
    for i in 1:n, j in 1:n
        sym[i, j] = min(adj[i, j], adj[j, i])
    end
    for i in 1:n
        sym[i, i] = 0.0
    end
    return sym
end

"""All-pairs shortest-path distances via Floyd-Warshall. Disconnected pairs stay `Inf`."""
function graph_shortest_paths(adj::AbstractMatrix)
    D = Matrix{Float64}(adj)
    n = size(D, 1)
    size(D, 2) == n || throw(ArgumentError("adjacency must be a square matrix"))
    @inbounds for k in 1:n
        for i in 1:n
            dik = D[i, k]
            isinf(dik) && continue
            for j in 1:n
                through = dik + D[k, j]
                if through < D[i, j]
                    D[i, j] = through
                end
            end
        end
    end
    return D
end

"""Symmetric eigendecomposition by the cyclic Jacobi method.

Returns `(eigenvalues, V)` sorted by descending eigenvalue, where eigenvector `k`
is column `k` of `V`."""
function _jacobi_eigen(A0::Matrix{Float64})
    n = size(A0, 1)
    A = copy(A0)
    V = zeros(Float64, n, n)
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

function _fix_signs!(Y::AbstractMatrix{Float64})
    for j in 1:size(Y, 2)
        col = @view Y[:, j]
        k = argmax(abs.(col))
        if col[k] < 0
            col .= .-col
        end
    end
    return Y
end

"""
    classical_mds(distances, n_components)

Classical (Torgerson) MDS of a square distance matrix. Returns `(embedding,
eigenvalues)`, where `embedding` is `(n, n_components)` and `eigenvalues` are the
top (descending, clamped non-negative) eigenvalues of the centered Gram matrix.
"""
function classical_mds(distances::AbstractMatrix, n_components::Integer)
    D = Matrix{Float64}(distances)
    n = size(D, 1)
    size(D, 2) == n || throw(ArgumentError("distances must be a square matrix"))
    all(isfinite, D) || throw(ArgumentError(
        "distance matrix contains non-finite values; the neighborhood graph is " *
        "likely disconnected (increase n_neighbors)"))
    (1 <= n_components <= n) ||
        throw(ArgumentError("n_components must be in [1, $n], got $n_components"))

    D2 = D .^ 2
    row_mean = vec(sum(D2; dims=2)) ./ n
    col_mean = vec(sum(D2; dims=1)) ./ n
    grand = sum(D2) / (n * n)

    B = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        B[i, j] = -0.5 * (D2[i, j] - row_mean[i] - col_mean[j] + grand)
    end
    B = (B .+ B') ./ 2

    eigvals, V = _jacobi_eigen(B)
    embedding = Matrix{Float64}(undef, n, n_components)
    out_vals = Vector{Float64}(undef, n_components)
    for c in 1:n_components
        lam = max(eigvals[c], 0.0)
        out_vals[c] = lam
        embedding[:, c] = V[:, c] .* sqrt(lam)
    end
    _fix_signs!(embedding)
    return embedding, out_vals
end

"""
    fit_isomap(X; n_components=2, n_neighbors=5)

Fit an Isomap embedding of the `n x d` matrix `X` into `n_components` dimensions
using a `n_neighbors`-nearest-neighbor graph.
"""
function fit_isomap(X; n_components::Integer=2, n_neighbors::Integer=5)
    A = _as_matrix(X)
    n = size(A, 1)
    (1 <= n_components <= n) ||
        throw(ArgumentError("n_components must be in [1, $n], got $n_components"))
    adj = knn_graph(A, n_neighbors)
    geo = graph_shortest_paths(adj)
    all(isfinite, geo) || throw(ArgumentError(
        "neighborhood graph is disconnected; some geodesic distances are " *
        "infinite. Increase n_neighbors."))
    embedding, eigenvalues = classical_mds(geo, n_components)
    return IsomapResult(embedding, eigenvalues, geo, Int(n_neighbors))
end

end # module Ch143Isomap
