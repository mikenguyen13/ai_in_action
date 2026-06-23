"""
    Ch157RegressionMetrics

Regression metrics: MSE, RMSE, MAE, and the Huber loss (Julia).

Mirrors the Python module `aiinaction.ch157_regression_metrics` and the Rust
module `aiinaction::ch157_regression_metrics`. The shared fixtures in
`test/test_ch157_regression_metrics.jl` match the Python/Rust suites, which keeps
the three implementations at parity.

RMSE and MAE already live in `AIInAction.Metrics`; this module *reuses* those
(`Metrics.rmse`, `Metrics.mae`) rather than re-deriving them, and adds the
squared-error (`mse`) and robust Huber-loss (`huber_loss`, `huber_loss_mean`)
pieces the chapter introduces. `rmse` and `mae` are re-exported for convenience.
"""
module Ch157RegressionMetrics

using ..Metrics: rmse, mae

export mse, rmse, mae, huber_loss, huber_loss_mean

function _validate(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    length(y_true) == length(y_pred) ||
        throw(ArgumentError("length mismatch: $(length(y_true)) != $(length(y_pred))"))
    isempty(y_true) && throw(ArgumentError("inputs must be non-empty"))
    return nothing
end

"""Mean squared error: the average of the squared residuals."""
function mse(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    _validate(y_true, y_pred)
    return sum((t - p)^2 for (t, p) in zip(y_true, y_pred)) / length(y_true)
end

"""
    huber_loss(y_true, y_pred; delta=1.0)

Per-observation Huber loss with threshold `delta`: quadratic for `|r| <= delta`
and linear beyond. Returns one value per observation (elementwise, not averaged).
Errors if `delta <= 0`.
"""
function huber_loss(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real};
                    delta::Real=1.0)
    _validate(y_true, y_pred)
    (isfinite(delta) && delta > 0) ||
        throw(ArgumentError("delta must be a positive finite number, got $delta"))
    d = float(delta)
    return [begin
        a = abs(t - p)
        a <= d ? 0.5 * a * a : d * (a - 0.5 * d)
    end for (t, p) in zip(y_true, y_pred)]
end

"""
    huber_loss_mean(y_true, y_pred; delta=1.0)

Mean Huber loss over all observations: the scalar objective minimized by Huber
(robust) regression.
"""
function huber_loss_mean(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real};
                         delta::Real=1.0)
    losses = huber_loss(y_true, y_pred; delta=delta)
    return sum(losses) / length(losses)
end

end # module Ch157RegressionMetrics
