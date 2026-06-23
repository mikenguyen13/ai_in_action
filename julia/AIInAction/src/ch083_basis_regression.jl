"""
    Ch083BasisRegression

Polynomial and basis-function regression with least squares and ridge (Julia).

Mirrors the Python module `aiinaction.ch083_basis_regression` and the Rust module
`aiinaction::ch083_basis_regression`. The shared fixtures in `test/test_ch083_basis_regression.jl`
match the Python/Rust suites to keep the three at parity.

Design matrices are `Matrix{Float64}` with one row per observation. The ridge solve
uses the symmetric normal equations; effective degrees of freedom come from the
singular values of the design matrix.
"""
module Ch083BasisRegression

using LinearAlgebra

export polynomial_design, rbf_design, fit_ridge, predict, effective_dof, BasisRegression

function _validate_1d(v::AbstractVector{<:Real}, name::AbstractString)
    isempty(v) && throw(ArgumentError("$name must be non-empty"))
    all(isfinite, v) || throw(ArgumentError("$name must contain only finite values"))
    return nothing
end

"""Build the polynomial design matrix with columns `1, x, x^2, ..., x^degree`."""
function polynomial_design(x::AbstractVector{<:Real}, degree::Integer)
    _validate_1d(x, "x")
    degree >= 0 || throw(ArgumentError("degree must be non-negative, got $degree"))
    n = length(x)
    phi = Matrix{Float64}(undef, n, degree + 1)
    for i in 1:n
        p = 1.0
        for j in 0:degree
            phi[i, j + 1] = p
            p *= x[i]
        end
    end
    return phi
end

"""Build a Gaussian radial-basis-function design matrix.

Column `j` is `exp(-(x - c_j)^2 / (2 * width^2))`; with `include_bias` a leading
column of ones is prepended.
"""
function rbf_design(x::AbstractVector{<:Real}, centers::AbstractVector{<:Real},
                    width::Real; include_bias::Bool=true)
    _validate_1d(x, "x")
    _validate_1d(centers, "centers")
    (isfinite(width) && width > 0) ||
        throw(ArgumentError("width must be a positive finite number, got $width"))
    n = length(x)
    k = length(centers)
    ncol = k + (include_bias ? 1 : 0)
    phi = Matrix{Float64}(undef, n, ncol)
    denom = 2.0 * width^2
    for i in 1:n
        off = 0
        if include_bias
            phi[i, 1] = 1.0
            off = 1
        end
        for j in 1:k
            d = x[i] - centers[j]
            phi[i, off + j] = exp(-(d^2) / denom)
        end
    end
    return phi
end

"""Solve the ridge least-squares problem `(phi' phi + penalty I) beta = phi' y`."""
function fit_ridge(phi::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, penalty::Real=0.0)
    n, m = size(phi)
    length(y) == n ||
        throw(ArgumentError("length mismatch: phi has $n rows but y has $(length(y))"))
    (isfinite(penalty) && penalty >= 0) ||
        throw(ArgumentError("penalty must be a non-negative finite number, got $penalty"))
    gram = phi' * phi + penalty * Matrix{Float64}(I, m, m)
    rhs = phi' * collect(float.(y))
    return Symmetric(Matrix(gram)) \ rhs
end

"""Evaluate fitted coefficients on a design matrix: `phi * beta`."""
function predict(phi::AbstractMatrix{<:Real}, beta::AbstractVector{<:Real})
    size(phi, 2) == length(beta) ||
        throw(ArgumentError("shape mismatch: phi has $(size(phi, 2)) columns but beta has $(length(beta))"))
    return phi * collect(float.(beta))
end

"""Effective degrees of freedom of a ridge fit: `sum_j s_j^2 / (s_j^2 + penalty)`."""
function effective_dof(phi::AbstractMatrix{<:Real}, penalty::Real=0.0)
    (isfinite(penalty) && penalty >= 0) ||
        throw(ArgumentError("penalty must be a non-negative finite number, got $penalty"))
    s = svdvals(Matrix(float.(phi)))
    s2 = s .^ 2
    return sum(s2 ./ (s2 .+ penalty))
end

"""Basis-function regression estimator with optional ridge penalty."""
mutable struct BasisRegression
    degree::Int
    penalty::Float64
    basis::String
    centers::Union{Nothing,Vector{Float64}}
    width::Union{Nothing,Float64}
    coef::Union{Nothing,Vector{Float64}}
end

function BasisRegression(; degree::Integer=2, penalty::Real=0.0, basis::AbstractString="poly",
                         centers=nothing, width=nothing)
    c = centers === nothing ? nothing : Vector{Float64}(centers)
    w = width === nothing ? nothing : Float64(width)
    return BasisRegression(Int(degree), Float64(penalty), String(basis), c, w, nothing)
end

function _design(model::BasisRegression, x::AbstractVector{<:Real})
    if model.basis == "poly"
        return polynomial_design(x, model.degree)
    elseif model.basis == "rbf"
        (model.centers !== nothing && model.width !== nothing) ||
            throw(ArgumentError("rbf basis requires both 'centers' and 'width'"))
        return rbf_design(x, model.centers, model.width)
    else
        throw(ArgumentError("unknown basis $(repr(model.basis)); expected 'poly' or 'rbf'"))
    end
end

"""Fit coefficients to inputs `x` and targets `y`; returns the model."""
function fit!(model::BasisRegression, x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    phi = _design(model, x)
    model.coef = fit_ridge(phi, y, model.penalty)
    return model
end

"""Predict targets at new inputs `x` using the fitted coefficients."""
function predict(model::BasisRegression, x::AbstractVector{<:Real})
    model.coef === nothing && throw(ArgumentError("model is not fitted; call fit! first"))
    return predict(_design(model, x), model.coef)
end

end # module Ch083BasisRegression
