"""
    Ch086ElasticNet

Elastic Net regression via coordinate descent (Chapter 086).

Solves

    minimize (1 / (2 n)) ||y - X b||^2 + lambda (alpha ||b||_1 + ((1 - alpha) / 2) ||b||_2^2)

by cyclic coordinate descent with soft thresholding, fitting an unpenalized
intercept by centering internally. Mirrors the Python module
`aiinaction.ch086_elastic_net` and the Rust module `aiinaction::ch086_elastic_net`;
the shared fixtures in `test/test_ch086_elastic_net.jl` match the Python/Rust
suites to keep the three at parity.
"""
module Ch086ElasticNet

export soft_threshold, elastic_net_fit, elastic_net_predict

"""Soft-thresholding operator `S(z, gamma) = sign(z) * max(|z| - gamma, 0)`."""
function soft_threshold(z::Real, gamma::Real)
    gamma < 0 && throw(ArgumentError("gamma must be non-negative, got $gamma"))
    if z > gamma
        return float(z - gamma)
    elseif z < -gamma
        return float(z + gamma)
    else
        return 0.0
    end
end

function _matrix_dims(X::AbstractMatrix{<:Real})
    (size(X, 1) == 0 || size(X, 2) == 0) && throw(ArgumentError("X must be non-empty"))
    return size(X)
end

"""
    elastic_net_fit(X, y, lam; alpha=0.5, max_iter=1000, tol=1e-8) -> (coef, intercept)

Fit Elastic Net coefficients by cyclic coordinate descent. `coef` is a vector of
`n_features` coefficients on the original feature scale and `intercept` is the
unpenalized intercept.
"""
function elastic_net_fit(
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    lam::Real;
    alpha::Real = 0.5,
    max_iter::Integer = 1000,
    tol::Real = 1e-8,
)
    n, p = _matrix_dims(X)
    length(y) == n ||
        throw(ArgumentError("length mismatch: X has $n rows but y has $(length(y))"))
    lam < 0 && throw(ArgumentError("lam must be non-negative, got $lam"))
    (0 <= alpha <= 1) || throw(ArgumentError("alpha must lie in [0, 1], got $alpha"))
    max_iter <= 0 && throw(ArgumentError("max_iter must be positive, got $max_iter"))
    tol <= 0 && throw(ArgumentError("tol must be positive, got $tol"))

    nf = float(n)
    x_mean = vec(sum(X; dims = 1)) ./ nf
    y_mean = sum(y) / nf

    Xc = float.(X) .- reshape(x_mean, 1, p)
    yc = float.(y) .- y_mean

    col_sq = vec(sum(Xc .^ 2; dims = 1)) ./ nf

    beta = zeros(Float64, p)
    residual = copy(yc)  # residual = yc - Xc * beta, kept in sync
    l1 = lam * alpha
    l2 = lam * (1 - alpha)

    for _ in 1:max_iter
        max_change = 0.0
        for j in 1:p
            xj = @view Xc[:, j]
            if col_sq[j] == 0.0
                if beta[j] != 0.0
                    @. residual += xj * beta[j]
                    beta[j] = 0.0
                end
                continue
            end
            beta_j_old = beta[j]
            rho = 0.0
            @inbounds for i in 1:n
                rho += xj[i] * (residual[i] + xj[i] * beta_j_old)
            end
            rho /= nf
            beta_j_new = soft_threshold(rho, l1) / (col_sq[j] + l2)
            if beta_j_new != beta_j_old
                delta = beta_j_old - beta_j_new
                @. residual += xj * delta
                beta[j] = beta_j_new
                change = abs(beta_j_new - beta_j_old)
                change > max_change && (max_change = change)
            end
        end
        max_change < tol && break
    end

    intercept = y_mean - sum(x_mean .* beta)
    return beta, intercept
end

"""
    elastic_net_predict(X, coef, intercept) -> Vector

Predict responses `X * coef + intercept` for a fitted model.
"""
function elastic_net_predict(
    X::AbstractMatrix{<:Real},
    coef::AbstractVector{<:Real},
    intercept::Real,
)
    _, p = _matrix_dims(X)
    length(coef) == p ||
        throw(ArgumentError("length mismatch: X has $p features but coef has $(length(coef))"))
    return X * collect(float.(coef)) .+ intercept
end

end # module Ch086ElasticNet
