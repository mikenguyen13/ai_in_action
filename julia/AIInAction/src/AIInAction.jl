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

include("ch071_eda.jl")
include("ch082_robust_regression.jl")
include("ch083_basis_regression.jl")
include("ch086_elastic_net.jl")
include("ch088_softmax_regression.jl")
include("ch089_softmax_regression.jl")
include("ch118_smote.jl")
include("ch124_kmeans_variants.jl")
include("ch125_agglomerative_clustering.jl")
include("ch126_diana.jl")
include("ch130_gmm_em.jl")
include("ch131_spectral_clustering.jl")
include("ch132_clustering_validation.jl")
include("ch133_clustering_at_scale.jl")
include("ch136_pca.jl")

include("ch137_kernel_pca.jl")
include("ch138_fastica.jl")
include("ch139_nmf.jl")
include("ch143_isomap.jl")
include("ch145_anomaly.jl")
include("ch146_lof.jl")
include("ch150_association_rules.jl")
include("ch155_pr_curves.jl")
include("ch157_regression_metrics.jl")
include("ch161_calibration.jl")
include("ch162_ranking_metrics.jl")
include("ch163_ir_metrics.jl")
include("ch164_clustering_metrics.jl")
include("ch165_mcnemar.jl")
include("ch169_bootstrap.jl")
include("ch184_softmax_ce.jl")
include("ch186_forward_propagation.jl")
include("ch187_backprop.jl")
include("ch192_momentum.jl")
include("ch193_nesterov.jl")
include("ch194_adagrad.jl")
include("ch195_rmsprop.jl")
include("ch197_adamw.jl")
include("ch198_lr_schedules.jl")
include("ch200_weight_init.jl")
include("ch201_batch_norm.jl")
include("ch202_layer_norm.jl")
include("ch204_dropout.jl")

end # module AIInAction
