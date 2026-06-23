"""
    Ch161Calibration

Calibration metrics: reliability curves and Expected Calibration Error (Julia).

Mirrors the Python module `aiinaction.ch161_calibration` and the Rust module
`aiinaction::ch161_calibration`. The shared fixtures in `test/test_ch161_calibration.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

Confidence calibration: each example has a scalar confidence `p_i = max_k f_k(x_i)`
in `[0, 1]` and a binary correctness indicator `c_i`. Partition `[0, 1]` into `M`
equal-width bins and, within each occupied bin `B_m`, compute the empirical
accuracy `acc(B_m)` and average confidence `conf(B_m)`. Then

    ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|
    MCE = max_m |acc(B_m) - conf(B_m)|

Binning convention: example `i` lands in bin `floor(p_i * M)`, with `p_i = 1`
folded into the last bin. Empty bins contribute nothing.
"""
module Ch161Calibration

export Bin, ReliabilityCurve, reliability_curve, expected_calibration_error,
    maximum_calibration_error, brier_score, gap, occupied

"""Summary of one reliability-diagram bin. `[lower, upper)` is half-open."""
struct Bin
    lower::Float64
    upper::Float64
    count::Int
    accuracy::Float64
    confidence::Float64
end

"""Signed calibration gap `acc(B_m) - conf(B_m)`."""
gap(b::Bin) = b.accuracy - b.confidence

"""A binned reliability curve plus the sample size it was built from."""
struct ReliabilityCurve
    bins::Vector{Bin}
    n_samples::Int
end

n_bins(rc::ReliabilityCurve) = length(rc.bins)

"""The bins that contain at least one example."""
occupied(rc::ReliabilityCurve) = filter(b -> b.count > 0, rc.bins)

function _validate(confidences, correct, nbins::Integer)
    length(confidences) == length(correct) ||
        throw(ArgumentError("length mismatch: $(length(confidences)) != $(length(correct))"))
    isempty(confidences) && throw(ArgumentError("inputs must be non-empty"))
    nbins >= 1 || throw(ArgumentError("n_bins must be a positive integer, got $nbins"))
    for p in confidences
        (0.0 <= p <= 1.0) || throw(ArgumentError("confidences must lie in [0, 1], got $p"))
    end
    for c in correct
        (c == 0 || c == 1) || throw(ArgumentError("correct must be 0 or 1, got $c"))
    end
    return nothing
end

"""Map a confidence to its bin: `floor(p * M)`, with `p == 1` folded into the last bin."""
function _bin_index(p::Float64, nbins::Integer)
    b = Int(floor(p * nbins))
    return b >= nbins ? nbins - 1 : b
end

"""
    reliability_curve(confidences, correct; n_bins=10)

Build an equal-width reliability curve over `n_bins` bins (empty bins included).
"""
function reliability_curve(confidences, correct; n_bins::Integer=10)
    conf = Float64.(confidences)
    corr = Float64.(correct)
    _validate(conf, corr, n_bins)
    n = length(conf)
    counts = zeros(Int, n_bins)
    acc_sum = zeros(Float64, n_bins)
    conf_sum = zeros(Float64, n_bins)
    for (p, c) in zip(conf, corr)
        b = _bin_index(p, n_bins) + 1  # 1-based
        counts[b] += 1
        acc_sum[b] += c
        conf_sum[b] += p
    end

    bins = Vector{Bin}(undef, n_bins)
    for m in 1:n_bins
        lower = (m - 1) / n_bins
        upper = m / n_bins
        if counts[m] > 0
            acc = acc_sum[m] / counts[m]
            cf = conf_sum[m] / counts[m]
        else
            acc = 0.0
            cf = 0.0
        end
        bins[m] = Bin(lower, upper, counts[m], acc, cf)
    end
    return ReliabilityCurve(bins, n)
end

"""
    expected_calibration_error(confidences, correct; n_bins=10)

Expected Calibration Error (ECE), the occupancy-weighted mean absolute gap.
"""
function expected_calibration_error(confidences, correct; n_bins::Integer=10)
    rc = reliability_curve(confidences, correct; n_bins=n_bins)
    total = 0.0
    for b in rc.bins
        if b.count > 0
            total += (b.count / rc.n_samples) * abs(gap(b))
        end
    end
    return total
end

"""
    maximum_calibration_error(confidences, correct; n_bins=10)

Maximum Calibration Error (MCE), the largest absolute gap over occupied bins.
"""
function maximum_calibration_error(confidences, correct; n_bins::Integer=10)
    rc = reliability_curve(confidences, correct; n_bins=n_bins)
    worst = 0.0
    for b in rc.bins
        if b.count > 0
            worst = max(worst, abs(gap(b)))
        end
    end
    return worst
end

"""
    brier_score(confidences, correct)

Binary Brier score `BS = (1/n) sum_i (p_i - c_i)^2`.
"""
function brier_score(confidences, correct)
    conf = Float64.(confidences)
    corr = Float64.(correct)
    _validate(conf, corr, 1)
    n = length(conf)
    return sum((p - c)^2 for (p, c) in zip(conf, corr)) / n
end

end # module Ch161Calibration
