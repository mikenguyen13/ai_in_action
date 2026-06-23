"""
    Ch082RobustRegression

Linear regression extensions: robust (Huber), WLS, GLS, quantile, basis (Julia).

Mirrors the Python module `aiinaction.ch082_robust_regression` and the Rust module
`aiinaction::ch082_robust_regression`. The shared fixtures in `test/test_ch082_robust_regression.jl`
match the Python/Rust suites to keep the three at parity.

The core algorithm is Huber robust regression solved by iteratively reweighted least
squares (IRLS). The same weighted-normal-equations engine powers weighted least
squares (WLS), generalized least squares (GLS), and quantile regression, and
`vandermonde` provides a polynomial basis expansion. Linear systems are solved with a
plain Gaussian-elimination routine so the numerics match the other two languages.

Design matrices are `Vector{Vector{Float64}}` (a vector of rows); responses are
`Vector{Float64}`.
"""
module Ch082RobustRegression

export solve_linear, fit_ols, fit_wls, fit_gls, fit_huber, fit_quantile,
       vandermonde, predict, HuberResult

"""Result of a Huber IRLS fit."""
struct HuberResult
    coef::Vector{Float64}
    scale::Float64
    n_iter::Int
    converged::Bool
end

function _check_matrix(X::AbstractVector)
    isempty(X) && throw(ArgumentError("design matrix X must have at least one row"))
    ncol = length(X[1])
    ncol == 0 && throw(ArgumentError("design matrix X must have at least one column"))
    for (i, row) in enumerate(X)
        length(row) == ncol || throw(ArgumentError(
            "ragged design matrix: row 0 has $ncol columns but row $(i-1) has $(length(row))"))
    end
    return ncol
end

function _check_xy(X::AbstractVector, y::AbstractVector)
    ncol = _check_matrix(X)
    length(X) == length(y) || throw(ArgumentError(
        "length mismatch: X has $(length(X)) rows but y has $(length(y)) entries"))
    length(X) >= ncol || throw(ArgumentError(
        "underdetermined system: $(length(X)) rows < $ncol columns"))
    return ncol
end

"""Solve the square linear system `A x = b` by Gaussian elimination with partial pivoting."""
function solve_linear(A::AbstractVector, b::AbstractVector)
    n = length(A)
    n == 0 && throw(ArgumentError("system must be non-empty"))
    a = [Float64.(row) for row in A]
    rhs = Float64.(collect(b))
    for row in a
        length(row) == n || throw(ArgumentError(
            "matrix must be square: got $(length(row)) columns for $n rows"))
    end
    length(rhs) == n || throw(ArgumentError(
        "length mismatch: A is $(n)x$(n) but b has $(length(rhs)) entries"))

    for col in 1:n
        pivot = col
        for r in (col+1):n
            if abs(a[r][col]) > abs(a[pivot][col])
                pivot = r
            end
        end
        abs(a[pivot][col]) < 1e-14 && throw(ArgumentError("matrix is singular or nearly singular"))
        if pivot != col
            a[col], a[pivot] = a[pivot], a[col]
            rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        end
        inv = 1.0 / a[col][col]
        for r in (col+1):n
            factor = a[r][col] * inv
            if factor != 0.0
                for c in col:n
                    a[r][c] -= factor * a[col][c]
                end
                rhs[r] -= factor * rhs[col]
            end
        end
    end

    x = zeros(Float64, n)
    for col in n:-1:1
        s = rhs[col]
        for c in (col+1):n
            s -= a[col][c] * x[c]
        end
        x[col] = s / a[col][col]
    end
    return x
end

_matvec(X, beta) = [sum(xij * bj for (xij, bj) in zip(row, beta)) for row in X]

"""Linear prediction `X * beta`."""
function predict(X::AbstractVector, beta::AbstractVector)
    ncol = _check_matrix(X)
    ncol == length(beta) || throw(ArgumentError(
        "shape mismatch: X has $ncol columns but beta has $(length(beta)) entries"))
    return _matvec(X, beta)
end

"""Solve `(Xᵀ W X) β = Xᵀ W y` for diagonal weights `w`."""
function _weighted_normal_equations(X, y, w)
    n = length(X)
    p = length(X[1])
    xtwx = [zeros(Float64, p) for _ in 1:p]
    xtwy = zeros(Float64, p)
    for i in 1:n
        wi = w[i]
        row = X[i]
        wyi = wi * y[i]
        for a in 1:p
            xa = row[a]
            xtwy[a] += xa * wyi
            wxa = wi * xa
            for b in a:p
                xtwx[a][b] += wxa * row[b]
            end
        end
    end
    for a in 1:p
        for b in 1:(a-1)
            xtwx[a][b] = xtwx[b][a]
        end
    end
    return solve_linear(xtwx, xtwy)
end

function _median(values)
    s = sort(collect(values))
    n = length(s)
    mid = div(n, 2)
    return isodd(n) ? s[mid+1] : 0.5 * (s[mid] + s[mid+1])
end

function _mad_scale(residuals)
    med = _median(residuals)
    abs_dev = [abs(r - med) for r in residuals]
    return 1.4826 * _median(abs_dev)
end

"""Ordinary least squares: solves the normal equations `XᵀX β = Xᵀy`."""
function fit_ols(X::AbstractVector, y::AbstractVector)
    _check_xy(X, y)
    return _weighted_normal_equations(X, y, ones(Float64, length(X)))
end

"""Weighted least squares: `argmin_β Σ wᵢ (yᵢ - xᵢᵀβ)²`. Weights must be non-negative."""
function fit_wls(X::AbstractVector, y::AbstractVector, weights::AbstractVector)
    _check_xy(X, y)
    length(weights) == length(X) || throw(ArgumentError(
        "length mismatch: X has $(length(X)) rows but weights has $(length(weights)) entries"))
    any(w -> w < 0.0, weights) && throw(ArgumentError("weights must be non-negative"))
    return _weighted_normal_equations(X, y, Float64.(weights))
end

"""Generalized least squares for known error covariance `cov` (Ω):
`β = (Xᵀ Ω⁻¹ X)⁻¹ Xᵀ Ω⁻¹ y`, computed by solving systems against Ω."""
function fit_gls(X::AbstractVector, y::AbstractVector, cov::AbstractVector)
    p = _check_xy(X, y)
    n = length(X)
    (length(cov) == n && all(r -> length(r) == n, cov)) ||
        throw(ArgumentError("cov must be $(n)x$(n) to match $n observations"))
    omega = [Float64.(r) for r in cov]

    oinv_y = solve_linear(omega, Float64.(y))
    oinv_cols = [solve_linear(omega, [X[i][j] for i in 1:n]) for j in 1:p]

    xtoix = [zeros(Float64, p) for _ in 1:p]
    xtoiy = zeros(Float64, p)
    for a in 1:p
        col_a = [X[i][a] for i in 1:n]
        xtoiy[a] = sum(col_a[i] * oinv_y[i] for i in 1:n)
        for b in 1:p
            xtoix[a][b] = sum(col_a[i] * oinv_cols[b][i] for i in 1:n)
        end
    end
    return solve_linear(xtoix, xtoiy)
end

"""Robust regression with the Huber loss via IRLS, seeded with OLS. `delta` is the
tuning constant in scale units (1.345 gives ~95% Gaussian efficiency)."""
function fit_huber(X::AbstractVector, y::AbstractVector;
                   delta::Float64=1.345, max_iter::Int=100, tol::Float64=1e-10)
    _check_xy(X, y)
    delta > 0.0 || throw(ArgumentError("delta must be positive, got $delta"))
    max_iter > 0 || throw(ArgumentError("max_iter must be positive, got $max_iter"))
    tol >= 0.0 || throw(ArgumentError("tol must be non-negative, got $tol"))

    beta = _weighted_normal_equations(X, y, ones(Float64, length(X)))
    scale = 1.0
    converged = false
    n_iter = 0
    for it in 1:max_iter
        n_iter = it
        resid = [yi - ri for (yi, ri) in zip(y, _matvec(X, beta))]
        scale = _mad_scale(resid)
        if scale <= 1e-12
            converged = true
            break
        end
        thresh = delta * scale
        weights = [abs(r) > 1e-30 ? min(1.0, thresh / abs(r)) : 1.0 for r in resid]
        new_beta = _weighted_normal_equations(X, y, weights)
        change = maximum(abs(nb - b) for (nb, b) in zip(new_beta, beta))
        beta = new_beta
        if change < tol
            converged = true
            break
        end
    end
    return HuberResult(beta, scale, n_iter, converged)
end

"""Quantile regression at level `tau` via IRLS on the pinball loss. `tau` in (0, 1);
`tau = 0.5` recovers least-absolute-deviations."""
function fit_quantile(X::AbstractVector, y::AbstractVector;
                      tau::Float64=0.5, max_iter::Int=200, tol::Float64=1e-10,
                      eps::Float64=1e-6)
    _check_xy(X, y)
    (0.0 < tau < 1.0) || throw(ArgumentError(
        "tau must be in the open interval (0, 1), got $tau"))
    max_iter > 0 || throw(ArgumentError("max_iter must be positive, got $max_iter"))
    eps > 0.0 || throw(ArgumentError("eps must be positive, got $eps"))

    beta = _weighted_normal_equations(X, y, ones(Float64, length(X)))
    for _ in 1:max_iter
        resid = [yi - ri for (yi, ri) in zip(y, _matvec(X, beta))]
        weights = [(r >= 0.0 ? tau : 1.0 - tau) / max(abs(r), eps) for r in resid]
        new_beta = _weighted_normal_equations(X, y, weights)
        change = maximum(abs(nb - b) for (nb, b) in zip(new_beta, beta))
        beta = new_beta
        if change < tol
            break
        end
    end
    return beta
end

"""Polynomial basis expansion (Vandermonde design matrix). Each scalar `xᵢ` maps to
`[1, xᵢ, xᵢ², ..., xᵢ^degree]`; the leading 1 is dropped when `include_bias` is false."""
function vandermonde(x::AbstractVector, degree::Int; include_bias::Bool=true)
    degree >= 0 || throw(ArgumentError("degree must be non-negative, got $degree"))
    isempty(x) && throw(ArgumentError("input x must be non-empty"))
    (!include_bias && degree == 0) &&
        throw(ArgumentError("degree must be >= 1 when include_bias is false"))
    start = include_bias ? 0 : 1
    return [[Float64(xi)^k for k in start:degree] for xi in x]
end

end # module Ch082RobustRegression
