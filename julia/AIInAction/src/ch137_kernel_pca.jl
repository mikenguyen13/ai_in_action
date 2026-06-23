"""
    Ch137KernelPca

Kernel PCA (kernel principal component analysis) from scratch (Julia).

Mirrors the Python module `aiinaction.ch137_kernel_pca` and the Rust module
`aiinaction::ch137_kernel_pca`. The shared fixtures in `test/test_ch137_kernel_pca.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

Linear PCA is run implicitly in a kernel-induced feature space. The `n x n` Gram
matrix is centered via `K_tilde = H K H`, diagonalized by a self-contained cyclic
Jacobi eigensolver (Base-only, no LinearAlgebra dependency), and the unit
eigenvectors are scaled by `1 / sqrt(mu_k)` so the feature-space components have
unit norm. Each coefficient vector's sign is fixed so its largest-magnitude entry
is positive; tied-magnitude components have a numerically arbitrary overall sign.

Kernels are passed as `(name, params)` where `name` is one of `"linear"`, `"poly"`,
`"rbf"` and `params` is a `Dict{String,Float64}` (or `NamedTuple`).
"""
module Ch137KernelPca

export KernelPCAResult, kernel_matrix, fit_kernel_pca, transform

const EIGENVALUE_TOL = 1e-12

"""The fitted state of a Kernel PCA model.

`alphas` holds the normalized coefficient vectors as columns (one per component);
`x_fit` retains the training inputs for out-of-sample projection."""
struct KernelPCAResult
    x_fit::Matrix{Float64}
    kernel::Tuple{String,Dict{String,Float64}}
    alphas::Matrix{Float64}
    eigenvalues::Vector{Float64}
    explained_variance_ratio::Vector{Float64}
    row_means::Vector{Float64}
    total_mean::Float64
end

n_components(r::KernelPCAResult) = size(r.alphas, 2)
n_train(r::KernelPCAResult) = size(r.x_fit, 1)

_param(params, key, default) = Float64(get(params, key, default))

function _kernel_eval(a::AbstractVector{Float64}, b::AbstractVector{Float64},
                      name::String, params)
    if name == "linear"
        return sum(a .* b)
    elseif name == "poly"
        gamma = _param(params, "gamma", 1.0)
        coef0 = _param(params, "coef0", 1.0)
        degree = _param(params, "degree", 2.0)
        return (gamma * sum(a .* b) + coef0)^degree
    elseif name == "rbf"
        gamma = _param(params, "gamma", 1.0)
        sq = sum((a .- b) .^ 2)
        return exp(-gamma * sq)
    else
        throw(ArgumentError("unknown kernel $(name)"))
    end
end

_check_kernel(kernel) = (String(kernel[1]), Dict{String,Float64}(kernel[2]))

"""
    kernel_matrix(A, B, kernel)

Cross-kernel matrix `K[i, t] = k(a_i, b_t)` for point sets `A` (`n_a x d`) and
`B` (`n_b x d`)."""
function kernel_matrix(A, B, kernel)
    Am = Matrix{Float64}(A)
    Bm = Matrix{Float64}(B)
    size(Am, 2) == size(Bm, 2) ||
        throw(ArgumentError("A has $(size(Am, 2)) features but B has $(size(Bm, 2))"))
    name, params = _check_kernel(kernel)
    name in ("linear", "poly", "rbf") || throw(ArgumentError("unknown kernel $(name)"))
    if name == "rbf"
        _param(params, "gamma", 1.0) > 0 ||
            throw(ArgumentError("rbf gamma must be positive"))
    end
    na, nb = size(Am, 1), size(Bm, 1)
    K = Matrix{Float64}(undef, na, nb)
    for i in 1:na, t in 1:nb
        K[i, t] = _kernel_eval(@view(Am[i, :]), @view(Bm[t, :]), name, params)
    end
    return K
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

function _fix_signs!(alphas::AbstractMatrix{Float64})
    for k in 1:size(alphas, 2)
        col = @view alphas[:, k]
        i = argmax(abs.(col))
        if col[i] < 0
            col .= .-col
        end
    end
    return alphas
end

"""
    fit_kernel_pca(X; n_components=nothing, kernel=("rbf", Dict("gamma" => 1.0)))

Fit a Kernel PCA model to the `n x d` matrix `X`. `n_components=nothing` retains
every strictly-positive component, capped at `n - 1`."""
function fit_kernel_pca(X; n_components=nothing,
                        kernel=("rbf", Dict("gamma" => 1.0)))
    A = Matrix{Float64}(X)
    n, d = size(A)
    n >= 2 || throw(ArgumentError("need at least 2 samples for Kernel PCA, got $n"))
    d >= 1 || throw(ArgumentError("X must have at least one feature"))
    all(isfinite, A) || throw(ArgumentError("X contains non-finite values (nan or inf)"))
    name, params = _check_kernel(kernel)
    max_components = n - 1

    K = kernel_matrix(A, A, (name, params))

    row_means = vec(sum(K; dims=2)) ./ n
    total_mean = sum(K) / (n * n)
    Kt = K .- row_means .- row_means' .+ total_mean
    Kt = (Kt .+ Kt') ./ 2

    eigvals, V = _jacobi_eigen(Matrix(Kt))

    total_var = sum(max.(eigvals, 0.0))
    total_var > 0 ||
        throw(ArgumentError("kernel matrix has no positive variance; Kernel PCA is undefined"))

    n_positive = count(e -> e > EIGENVALUE_TOL, eigvals[1:max_components])

    k = if n_components === nothing
        n_positive >= 1 ||
            throw(ArgumentError("kernel matrix has no positive variance; Kernel PCA is undefined"))
        n_positive
    else
        kk = Int(n_components)
        (1 <= kk <= max_components) ||
            throw(ArgumentError("n_components must be in [1, $max_components] for $n samples, got $kk"))
        kk
    end

    mu = eigvals[1:k]
    bad = findfirst(e -> e <= EIGENVALUE_TOL, mu)
    bad === nothing ||
        throw(ArgumentError("component $(bad - 1) has non-positive eigenvalue; request fewer components or a different kernel"))

    # alpha^k = (unit eigenvector column k) / sqrt(mu_k).
    alphas = V[:, 1:k] ./ sqrt.(mu')
    _fix_signs!(alphas)

    explained_variance_ratio = mu ./ total_var

    return KernelPCAResult(A, (name, params), alphas, mu,
        explained_variance_ratio, row_means, total_mean)
end

"""
    transform(model, Z)

Project new points `Z` (`n_z x d`) onto the fitted components, returning an
`n_z x n_components` matrix. Each test point is centered with the training row and
grand means."""
function transform(model::KernelPCAResult, Z)
    Zm = Matrix{Float64}(Z)
    d = size(model.x_fit, 2)
    size(Zm, 2) == d ||
        throw(ArgumentError("Z has $(size(Zm, 2)) features but model was fit on $d"))
    all(isfinite, Zm) || throw(ArgumentError("Z contains non-finite values (nan or inf)"))

    n = n_train(model)
    # K_z[i, t] = k(x_i, z_t).
    Kz = kernel_matrix(model.x_fit, Zm, model.kernel)
    col_means = vec(sum(Kz; dims=1)) ./ n
    Kz_centered = Kz .- model.row_means .- col_means' .+ model.total_mean
    return Kz_centered' * model.alphas
end

end # module Ch137KernelPca
