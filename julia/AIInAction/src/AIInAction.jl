"""
    AIInAction

Reusable, tested reference implementations accompanying the AI in Action book (Julia).
Mirrors the Python package `aiinaction` and the Rust crate `aiinaction`. The shared
fixtures in `test/runtests.jl` match the Python/Rust suites to keep the three at parity.
"""
module AIInAction

module Metrics

export rmse, mae, r2_score, accuracy

function _validate(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    length(y_true) == length(y_pred) ||
        throw(ArgumentError("length mismatch: $(length(y_true)) != $(length(y_pred))"))
    isempty(y_true) && throw(ArgumentError("inputs must be non-empty"))
    return nothing
end

"""Root mean squared error."""
function rmse(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    _validate(y_true, y_pred)
    return sqrt(sum((t - p)^2 for (t, p) in zip(y_true, y_pred)) / length(y_true))
end

"""Mean absolute error."""
function mae(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    _validate(y_true, y_pred)
    return sum(abs(t - p) for (t, p) in zip(y_true, y_pred)) / length(y_true)
end

"""Coefficient of determination R^2. Errors if the target variance is zero."""
function r2_score(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    _validate(y_true, y_pred)
    m = sum(y_true) / length(y_true)
    ss_tot = sum((t - m)^2 for t in y_true)
    ss_tot == 0 &&
        throw(ArgumentError("R^2 is undefined when all y_true values are equal (zero variance)"))
    ss_res = sum((t - p)^2 for (t, p) in zip(y_true, y_pred))
    return 1.0 - ss_res / ss_tot
end

"""Classification accuracy."""
function accuracy(y_true::AbstractVector, y_pred::AbstractVector)
    length(y_true) == length(y_pred) ||
        throw(ArgumentError("length mismatch: $(length(y_true)) != $(length(y_pred))"))
    isempty(y_true) && throw(ArgumentError("inputs must be non-empty"))
    return count(t == p for (t, p) in zip(y_true, y_pred)) / length(y_true)
end

end # module Metrics

using .Metrics

end # module AIInAction
