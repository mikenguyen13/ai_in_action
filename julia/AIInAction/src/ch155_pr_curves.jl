"""
    Ch155PrCurves

Precision-Recall curves and Average Precision from scratch (Julia).

Mirrors the Python module `aiinaction.ch155_pr_curves` and the Rust module
`aiinaction::ch155_pr_curves`. The shared fixtures in `test/test_ch155_pr_curves.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

Given binary labels `y_i in {0, 1}` and real-valued `scores` (larger means more
positive), thresholding at `tau` predicts positive iff `score >= tau`. Sweeping
`tau` over distinct score values from high to low traces the PR curve with
`precision = TP / (TP + FP)` and `recall = TP / P`. `average_precision` is the
rank-based AP estimator (the scikit-learn `AP`); `auprc_trapezoid` integrates the
curve produced here by the trapezoidal rule. Base-only, no extra dependency.
"""
module Ch155PrCurves

export PRCurve, pr_curve, average_precision, auprc_trapezoid

"""A precision-recall curve, one point per distinct threshold (decreasing order).

`precision` and `recall` are vectors in order of decreasing threshold; `recall` is
non-decreasing. `thresholds` are the distinct score values in decreasing order."""
struct PRCurve
    precision::Vector{Float64}
    recall::Vector{Float64}
    thresholds::Vector{Float64}
end

Base.length(c::PRCurve) = length(c.thresholds)

function _validate(y_true, scores)
    yt = collect(Int, y_true)
    sc = collect(Float64, scores)
    length(yt) == length(sc) ||
        throw(ArgumentError("length mismatch: len(y_true)=$(length(yt)) != len(scores)=$(length(sc))"))
    isempty(yt) && throw(ArgumentError("inputs must be non-empty"))
    for v in yt
        (v == 0 || v == 1) ||
            throw(ArgumentError("y_true must contain only 0/1 labels, found $v"))
    end
    all(isfinite, sc) ||
        throw(ArgumentError("scores contains non-finite values (nan or inf)"))
    sum(yt) == 0 &&
        throw(ArgumentError("y_true must contain at least one positive (label 1)"))
    return yt, sc
end

"""
    pr_curve(y_true, scores) -> PRCurve

Compute the precision-recall curve over distinct score thresholds. `y_true` are
binary labels (0/1) with at least one positive; `scores` are the matching scores.
"""
function pr_curve(y_true, scores)
    yt, sc = _validate(y_true, scores)
    p_total = sum(yt)
    thresholds = sort(collect(Set(sc)); rev=true)

    precision = Float64[]
    recall = Float64[]
    for tau in thresholds
        tp = 0
        fp = 0
        for (label, score) in zip(yt, sc)
            if score >= tau
                if label == 1
                    tp += 1
                else
                    fp += 1
                end
            end
        end
        predicted_pos = tp + fp
        prec = predicted_pos == 0 ? 1.0 : tp / predicted_pos
        push!(precision, prec)
        push!(recall, tp / p_total)
    end

    return PRCurve(precision, recall, thresholds)
end

"""
    average_precision(y_true, scores) -> Float64

Rank-based average precision (the scikit-learn `AP` estimator). Sorts instances by
descending score (ties broken by original index) and averages `Precision@k` at each
rank where a positive appears.
"""
function average_precision(y_true, scores)
    yt, sc = _validate(y_true, scores)
    p_total = sum(yt)

    # Stable descending sort: ties keep ascending original index order.
    order = sortperm(sc; rev=true)

    ap = 0.0
    tp = 0
    seen = 0
    for i in order
        seen += 1
        if yt[i] == 1
            tp += 1
            ap += tp / seen
        end
    end
    return ap / p_total
end

"""
    auprc_trapezoid(y_true, scores) -> Float64

Area under the PR curve by the trapezoidal rule over the [`pr_curve`] points.
Generally differs from (and is mildly optimistic relative to) [`average_precision`]
because of the curve's sawtooth structure.
"""
function auprc_trapezoid(y_true, scores)
    c = pr_curve(y_true, scores)
    rec = c.recall
    prec = c.precision
    area = 0.0
    for k in 2:length(rec)
        width = abs(rec[k] - rec[k - 1])
        area += width * (prec[k] + prec[k - 1]) / 2.0
    end
    return area
end

end # module Ch155PrCurves
