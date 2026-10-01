"""
    Ch525Oversight

Human-in-the-loop oversight: measuring, gating, and maintaining the overseer (Julia).

Mirrors the Python module `aiinaction.ch525_oversight` and the Rust module
`aiinaction::ch525_oversight`. The shared fixtures in `test/test_ch525_oversight.jl`
match the Python/Rust suites, which keeps the three implementations at parity. The
stateful `OversightGate` harness is Python only; every numeric function is ported.

* Measuring the overseer: signal detection theory (`sdt_measures`) treats a defective
  agent action as the signal. Sensitivity `d'` says how well a reviewer separates good
  actions from bad ones; the criterion `c` says how willing they are to approve.
* Canaries: known-defective actions planted in the review stream give an unbiased
  estimate of the catch rate (`wilson_interval`, `zero_failure_upper_bound`,
  `canaries_to_detect_decline`).
* Behavioral signatures: review time that stops scaling with complexity (`ols_slope`),
  a falling override rate (`rolling_rate`), and a downward shift in review time
  (`cusum_lower`).
* Approval design: `gate_action` routes an action to `AUTO`, `REVIEW`, or `BLOCK` by
  minimizing expected loss subject to a cap on residual harm.
* Skill maintenance: a first-order skill model (`skill_at`, `steady_state_skill`,
  `min_practice_rate`) and a half-life practice scheduler (`practice_schedule`).
* Independent panels: `majority_vote_accuracy` and `layered_residual_error`.

Indexing note: `cusum_lower` reports a 1-based `alarm_index` (Python reports 0-based),
and `rolling_rate` value `t` averages `flags[t:t+window-1]`.
"""
module Ch525Oversight

export probit, SDTResult, sdt_measures, wilson_interval, zero_failure_upper_bound,
    canaries_to_detect_decline, ols_slope, rolling_rate, CusumResult, cusum_lower,
    Route, AUTO, REVIEW, BLOCK, GateDecision, gate_action, review_threshold,
    detection_from_review_time, vigilance, steady_state_skill, skill_at,
    min_practice_rate, detection_from_skill, recall_probability, next_interval,
    update_half_life, practice_schedule, majority_vote_accuracy, layered_residual_error

"""Two-sided 95% standard normal quantile, `probit(0.975)`."""
const Z_95_TWO_SIDED = 1.959963984540054

# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

function _check_prob(name::AbstractString, value::Real; open_low::Bool=false, open_high::Bool=false)
    isfinite(value) || throw(ArgumentError("$name must be finite, got $value"))
    low_ok = open_low ? value > 0 : value >= 0
    high_ok = open_high ? value < 1 : value <= 1
    if !(low_ok && high_ok)
        lo = open_low ? "(0" : "[0"
        hi = open_high ? "1)" : "1]"
        throw(ArgumentError("$name must lie in $lo, $hi, got $value"))
    end
    return nothing
end

function _check_positive(name::AbstractString, value::Real)
    (isfinite(value) && value > 0) ||
        throw(ArgumentError("$name must be a positive finite number, got $value"))
    return nothing
end

function _check_nonneg(name::AbstractString, value::Real)
    (isfinite(value) && value >= 0) ||
        throw(ArgumentError("$name must be a non-negative finite number, got $value"))
    return nothing
end

function _check_count(name::AbstractString, value::Integer)
    (value isa Bool || value < 0) &&
        throw(ArgumentError("$name must be a non-negative integer, got $value"))
    return nothing
end

# --------------------------------------------------------------------------
# Signal detection theory for overseers
# --------------------------------------------------------------------------

# Wichura (1988), Algorithm AS241 (PPND16): the inverse standard normal CDF to about
# 1e-16 relative accuracy. Same coefficients and Horner order as Python's
# statistics.NormalDist.inv_cdf, so the two agree to the last bit or two.
function _ppnd16(p::Float64)
    q = p - 0.5
    if abs(q) <= 0.425
        r = 0.180625 - q * q
        num = (((((((2.5090809287301226727e+3 * r +
                     3.3430575583588128105e+4) * r +
                     6.7265770927008700853e+4) * r +
                     4.5921953931549871457e+4) * r +
                     1.3731693765509461125e+4) * r +
                     1.9715909503065514427e+3) * r +
                     1.3314166789178437745e+2) * r +
                     3.3871328727963666080e+0) * q
        den = (((((((5.2264952788528545610e+3 * r +
                     2.8729085735721942674e+4) * r +
                     3.9307895800092710610e+4) * r +
                     2.1213794301586595867e+4) * r +
                     5.3941960214247511077e+3) * r +
                     6.8718700749205790830e+2) * r +
                     4.2313330701600911252e+1) * r +
                     1.0)
        return num / den
    end
    r = q <= 0.0 ? p : 1.0 - p
    r = sqrt(-log(r))
    if r <= 5.0
        r = r - 1.6
        num = (((((((7.74545014278341407640e-4 * r +
                     2.27238449892691845833e-2) * r +
                     2.41780725177450611770e-1) * r +
                     1.27045825245236838258e+0) * r +
                     3.64784832476320460504e+0) * r +
                     5.76949722146069140550e+0) * r +
                     4.63033784615654529590e+0) * r +
                     1.42343711074968357734e+0)
        den = (((((((1.05075007164441684324e-9 * r +
                     5.47593808499534494600e-4) * r +
                     1.51986665636164571966e-2) * r +
                     1.48103976427480074590e-1) * r +
                     6.89767334985100004550e-1) * r +
                     1.67638483018380384940e+0) * r +
                     2.05319162663775882187e+0) * r +
                     1.0)
    else
        r = r - 5.0
        num = (((((((2.01033439929228813265e-7 * r +
                     2.71155556874348757815e-5) * r +
                     1.24266094738807843860e-3) * r +
                     2.65321895265761230930e-2) * r +
                     2.96560571828504891230e-1) * r +
                     1.78482653991729133580e+0) * r +
                     5.46378491116411436990e+0) * r +
                     6.65790464350110377720e+0)
        den = (((((((2.04426310338993978564e-15 * r +
                     1.42151175831644588870e-7) * r +
                     1.84631831751005468180e-5) * r +
                     7.86869131145613259100e-4) * r +
                     1.48753612908506148525e-2) * r +
                     1.36929880922735805310e-1) * r +
                     5.99832206555887937690e-1) * r +
                     1.0)
    end
    x = num / den
    return q < 0.0 ? -x : x
end

"""
    probit(p) -> Float64

Inverse standard normal CDF, `z(p) = Phi^{-1}(p)`, for `p` in the open interval
`(0, 1)` (Wichura AS241).
"""
function probit(p::Real)
    _check_prob("p", p; open_low=true, open_high=true)
    return _ppnd16(Float64(p))
end

"""
Equal-variance Gaussian signal detection summary of one reviewer.

The signal is a defective agent action. A hit is a rejected defective action, a false
alarm is a rejected correct action. `criterion` is `c = -(z(H) + z(F)) / 2`; positive
`c` means the reviewer leans toward approving, and a rising `c` over a shift is the
signature of rubber-stamping.
"""
struct SDTResult
    hit_rate::Float64
    false_alarm_rate::Float64
    d_prime::Float64
    criterion::Float64
end

"""
    sdt_measures(hits, misses, false_alarms, correct_rejections; correction=:loglinear) -> SDTResult

Sensitivity `d'` and criterion `c` from a reviewer's confusion counts.

`correction=:loglinear` (Hautus, 1995) adds 0.5 to every cell, which keeps `z` finite
when a reviewer catches every canary or never rejects a correct action.
`correction=:none` uses raw rates and throws if either rate is 0 or 1.
"""
function sdt_measures(hits::Integer, misses::Integer, false_alarms::Integer,
                      correct_rejections::Integer;
                      correction::Union{Symbol,AbstractString}=:loglinear)
    _check_count("hits", hits)
    _check_count("misses", misses)
    _check_count("false_alarms", false_alarms)
    _check_count("correct_rejections", correct_rejections)
    n_signal = hits + misses
    n_noise = false_alarms + correct_rejections
    (n_signal == 0 || n_noise == 0) &&
        throw(ArgumentError("need at least one defective and one correct action"))
    mode = Symbol(correction)
    if mode === :loglinear
        h = (hits + 0.5) / (n_signal + 1.0)
        f = (false_alarms + 0.5) / (n_noise + 1.0)
    elseif mode === :none
        h = hits / n_signal
        f = false_alarms / n_noise
        if h == 0.0 || h == 1.0 || f == 0.0 || f == 1.0
            throw(ArgumentError("raw hit or false-alarm rate is 0 or 1; use correction=:loglinear"))
        end
    else
        throw(ArgumentError("unknown correction $(repr(correction))"))
    end
    zh = probit(h)
    zf = probit(f)
    return SDTResult(h, f, zh - zf, -(zh + zf) / 2.0)
end

# --------------------------------------------------------------------------
# Canary statistics
# --------------------------------------------------------------------------

"""
    wilson_interval(successes, trials, z=1.959963984540054) -> Tuple{Float64,Float64}

Wilson score interval `(lower, upper)` for a binomial proportion such as a canary
catch rate.
"""
function wilson_interval(successes::Integer, trials::Integer, z::Real=Z_95_TWO_SIDED)
    _check_count("successes", successes)
    _check_count("trials", trials)
    trials == 0 && throw(ArgumentError("trials must be positive"))
    successes > trials && throw(ArgumentError("successes cannot exceed trials"))
    _check_positive("z", z)
    zf = Float64(z)
    n = Float64(trials)
    p = successes / n
    z2 = zf * zf
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = zf * sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))
end

"""
    zero_failure_upper_bound(trials, alpha=0.05) -> Float64

Exact one-sided upper bound on the miss rate after `trials` caught canaries: solves
`(1 - p)^n = alpha` for `p`. For `alpha = 0.05` this is close to the rule of three,
`3 / n`.
"""
function zero_failure_upper_bound(trials::Integer, alpha::Real=0.05)
    _check_count("trials", trials)
    trials == 0 && throw(ArgumentError("trials must be positive"))
    _check_prob("alpha", alpha; open_low=true, open_high=true)
    return 1.0 - Float64(alpha)^(1.0 / trials)
end

"""
    canaries_to_detect_decline(p0, p1, alpha=0.05, power=0.8) -> Int

Canaries needed for a one-sided test to detect a catch-rate drop from `p0` to `p1`:
`n = ((z_{1-alpha} sqrt(p0 q0) + z_{power} sqrt(p1 q1)) / (p0 - p1))^2`, rounded up.
"""
function canaries_to_detect_decline(p0::Real, p1::Real, alpha::Real=0.05, power::Real=0.8)
    _check_prob("p0", p0; open_low=true, open_high=true)
    _check_prob("p1", p1; open_low=true, open_high=true)
    _check_prob("alpha", alpha; open_low=true, open_high=true)
    _check_prob("power", power; open_low=true, open_high=true)
    p1 < p0 || throw(ArgumentError("p1 must be smaller than p0 (we detect a decline)"))
    a0 = Float64(p0)
    a1 = Float64(p1)
    za = probit(1.0 - alpha)
    zb = probit(power)
    num = za * sqrt(a0 * (1.0 - a0)) + zb * sqrt(a1 * (1.0 - a1))
    ratio = num / (a0 - a1)
    return ceil(Int, ratio * ratio - 1e-12)
end

# --------------------------------------------------------------------------
# Behavioral signatures
# --------------------------------------------------------------------------

"""
    ols_slope(x, y) -> Float64

Least-squares slope of `y` on `x`. Applied to review time against action complexity,
a slope near zero means scrutiny no longer scales with what is at stake.
"""
function ols_slope(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length"))
    length(x) < 2 && throw(ArgumentError("need at least two points"))
    xs = collect(Float64, x)
    ys = collect(Float64, y)
    n = length(xs)
    # Plain left-to-right sums (no pairwise or SIMD reordering) to mirror Python.
    sx = 0.0
    sy = 0.0
    for i in 1:n
        sx += xs[i]
        sy += ys[i]
    end
    mx = sx / n
    my = sy / n
    sxx = 0.0
    for i in 1:n
        sxx += (xs[i] - mx)^2
    end
    sxx == 0.0 && throw(ArgumentError("x has zero variance"))
    sxy = 0.0
    for i in 1:n
        sxy += (xs[i] - mx) * (ys[i] - my)
    end
    return sxy / sxx
end

"""
    rolling_rate(flags, window) -> Vector{Float64}

Trailing-window mean of 0/1 flags (for example, override decisions). Returns
`length(flags) - window + 1` values; value `t` (1-based) averages
`flags[t:t+window-1]`.
"""
function rolling_rate(flags::AbstractVector{<:Real}, window::Integer)
    (window isa Bool || window < 1) && throw(ArgumentError("window must be a positive integer"))
    window > length(flags) && throw(ArgumentError("window longer than the series"))
    v = collect(Float64, flags)
    w = Int(window)
    s = 0.0
    for i in 1:w
        s += v[i]
    end
    out = Float64[s / w]
    for t in (w + 1):length(v)
        s += v[t] - v[t - w]
        push!(out, s / w)
    end
    return out
end

"""Lower one-sided CUSUM path and the first alarm (1-based), or `nothing`."""
struct CusumResult
    statistic::Vector{Float64}
    alarm_index::Union{Int,Nothing}
end

"""
    cusum_lower(values, target_mean, slack, threshold) -> CusumResult

Page's one-sided CUSUM for a downward shift in the mean:
`S_t = max(0, S_{t-1} + (target_mean - x_t) - slack)` with `S_0 = 0`; alarm at the
first `t` (1-based) with `S_t > threshold`. On log review times this flags the moment
a reviewer starts spending less time per action.
"""
function cusum_lower(values::AbstractVector{<:Real}, target_mean::Real, slack::Real,
                     threshold::Real)
    _check_nonneg("slack", slack)
    _check_positive("threshold", threshold)
    isfinite(target_mean) || throw(ArgumentError("target_mean must be finite"))
    m = Float64(target_mean)
    k = Float64(slack)
    h = Float64(threshold)
    s = 0.0
    path = Float64[]
    alarm::Union{Int,Nothing} = nothing
    for (t, x) in enumerate(values)
        cand = s + (m - x) - k
        s = cand > 0.0 ? cand : 0.0  # Python max(0.0, cand), NaN maps to 0.0
        push!(path, s)
        if alarm === nothing && s > h
            alarm = t
        end
    end
    return CusumResult(path, alarm)
end

# --------------------------------------------------------------------------
# Approval design: expected-loss gating
# --------------------------------------------------------------------------

"""Routing outcome for one agent action: execute automatically, send to a human, or block."""
@enum Route AUTO REVIEW BLOCK

"""Routing decision plus the expected losses and residual harms behind it."""
struct GateDecision
    route::Route
    loss_auto::Float64
    loss_review::Float64
    residual_auto::Float64
    residual_review::Float64
    threshold::Float64
end

"""
    review_threshold(harm, review_cost, detection_prob) -> Float64

Error probability above which review beats automatic execution, `c / (d H)`.
Returns `Inf` when `detection_prob == 0` or `harm == 0`.
"""
function review_threshold(harm::Real, review_cost::Real, detection_prob::Real)
    _check_nonneg("harm", harm)
    _check_nonneg("review_cost", review_cost)
    _check_prob("detection_prob", detection_prob)
    (detection_prob == 0 || harm == 0) && return Inf
    return Float64(review_cost) / (Float64(detection_prob) * Float64(harm))
end

"""
    gate_action(p_error, harm, review_cost, detection_prob; max_residual_loss=Inf) -> GateDecision

Route one agent action to `AUTO`, `REVIEW`, or `BLOCK`. Expected losses are
`AUTO = p H` and `REVIEW = c + p (1 - d) H`. The gate picks the cheaper route among
those whose residual expected harm (`p H` or `p (1 - d) H`) stays within
`max_residual_loss` (ties go to `AUTO`); if neither does, the action is blocked.
"""
function gate_action(p_error::Real, harm::Real, review_cost::Real, detection_prob::Real;
                     max_residual_loss::Real=Inf)
    _check_prob("p_error", p_error)
    _check_nonneg("harm", harm)
    _check_nonneg("review_cost", review_cost)
    _check_prob("detection_prob", detection_prob)
    max_residual_loss >= 0 || throw(ArgumentError("max_residual_loss must be non-negative"))
    p = Float64(p_error)
    hm = Float64(harm)
    c = Float64(review_cost)
    d = Float64(detection_prob)
    residual_auto = p * hm
    residual_review = p * (1.0 - d) * hm
    loss_auto = residual_auto
    loss_review = c + residual_review
    auto_ok = residual_auto <= max_residual_loss
    review_ok = residual_review <= max_residual_loss
    if auto_ok && review_ok
        route = loss_review < loss_auto ? REVIEW : AUTO
    elseif auto_ok
        route = AUTO
    elseif review_ok
        route = REVIEW
    else
        route = BLOCK
    end
    return GateDecision(route, loss_auto, loss_review, residual_auto, residual_review,
                        review_threshold(hm, c, d))
end

"""
    detection_from_review_time(review_minutes, d_max, time_scale) -> Float64

Saturating detection curve `d(r) = d_max (1 - exp(-r / r0))`.
"""
function detection_from_review_time(review_minutes::Real, d_max::Real, time_scale::Real)
    _check_nonneg("review_minutes", review_minutes)
    _check_prob("d_max", d_max)
    _check_positive("time_scale", time_scale)
    return Float64(d_max) * (1.0 - exp(-Float64(review_minutes) / Float64(time_scale)))
end

"""
    vigilance(time_on_task, d_start, d_floor, time_constant) -> Float64

Vigilance decrement `d(t) = d_floor + (d_start - d_floor) exp(-t / tau)`.
"""
function vigilance(time_on_task::Real, d_start::Real, d_floor::Real, time_constant::Real)
    _check_nonneg("time_on_task", time_on_task)
    _check_prob("d_start", d_start)
    _check_prob("d_floor", d_floor)
    _check_positive("time_constant", time_constant)
    lo = Float64(d_floor)
    return lo + (Float64(d_start) - lo) * exp(-Float64(time_on_task) / Float64(time_constant))
end

# --------------------------------------------------------------------------
# Skill maintenance: the irony of automation as an ODE
# --------------------------------------------------------------------------

function _practice_intensity(automation::Real, practice::Real)
    _check_prob("automation", automation)
    _check_nonneg("practice", practice)
    return (1.0 - Float64(automation)) + Float64(practice)
end

"""
    steady_state_skill(automation, practice, learn_rate, decay_rate) -> Float64

Fixed point of `ds/dt = eta u (1 - s) - lambda s` with `u = (1 - a) + q`.
"""
function steady_state_skill(automation::Real, practice::Real, learn_rate::Real,
                            decay_rate::Real)
    _check_positive("learn_rate", learn_rate)
    _check_positive("decay_rate", decay_rate)
    gain = Float64(learn_rate) * _practice_intensity(automation, practice)
    return gain / (gain + Float64(decay_rate))
end

"""
    skill_at(t, s0, automation, practice, learn_rate, decay_rate) -> Float64

Closed-form skill `s(t) = s* + (s0 - s*) exp(-(eta u + lambda) t)`.
"""
function skill_at(t::Real, s0::Real, automation::Real, practice::Real, learn_rate::Real,
                  decay_rate::Real)
    _check_nonneg("t", t)
    _check_prob("s0", s0)
    _check_positive("learn_rate", learn_rate)
    _check_positive("decay_rate", decay_rate)
    lam = Float64(decay_rate)
    gain = Float64(learn_rate) * _practice_intensity(automation, practice)
    s_star = gain / (gain + lam)
    return s_star + (Float64(s0) - s_star) * exp(-((gain + lam) * Float64(t)))
end

"""
    min_practice_rate(target_skill, automation, learn_rate, decay_rate) -> Float64

Smallest deliberate-practice rate `q` with steady-state skill at least `target_skill`:
`q_min = max(0, lambda s / (eta (1 - s)) - (1 - a))`. `target_skill` lies in `[0, 1)`.
"""
function min_practice_rate(target_skill::Real, automation::Real, learn_rate::Real,
                           decay_rate::Real)
    _check_prob("target_skill", target_skill; open_high=true)
    _check_prob("automation", automation)
    _check_positive("learn_rate", learn_rate)
    _check_positive("decay_rate", decay_rate)
    s = Float64(target_skill)
    required = Float64(decay_rate) * s / (Float64(learn_rate) * (1.0 - s))
    return max(0.0, required - (1.0 - Float64(automation)))
end

"""
    detection_from_skill(skill, d_floor, d_ceiling) -> Float64

Linear map from skill in `[0, 1]` to detection probability in `[d_floor, d_ceiling]`.
"""
function detection_from_skill(skill::Real, d_floor::Real, d_ceiling::Real)
    _check_prob("skill", skill)
    _check_prob("d_floor", d_floor)
    _check_prob("d_ceiling", d_ceiling)
    d_ceiling < d_floor && throw(ArgumentError("d_ceiling must be >= d_floor"))
    lo = Float64(d_floor)
    return lo + (Float64(d_ceiling) - lo) * Float64(skill)
end

# --------------------------------------------------------------------------
# Spaced practice scheduling (half-life model)
# --------------------------------------------------------------------------

"""
    recall_probability(elapsed, half_life) -> Float64

Exponential forgetting `p = 2^(-elapsed / h)`.
"""
function recall_probability(elapsed::Real, half_life::Real)
    _check_nonneg("elapsed", elapsed)
    _check_positive("half_life", half_life)
    return 2.0^(-Float64(elapsed) / Float64(half_life))
end

"""
    next_interval(half_life, target_retention) -> Float64

Days until predicted recall falls to `target_retention`: `h log2(1 / R)`.
"""
function next_interval(half_life::Real, target_retention::Real)
    _check_positive("half_life", half_life)
    _check_prob("target_retention", target_retention; open_low=true, open_high=true)
    return Float64(half_life) * log2(1.0 / Float64(target_retention))
end

"""
    update_half_life(half_life, success; growth=2.0, shrink=0.5, floor=0.5) -> Float64

Grow the half-life after a successful session (`h * growth`), shrink it after a failure
(`max(floor, h * shrink)`). Requires `growth >= 1` and `shrink <= 1`.
"""
function update_half_life(half_life::Real, success::Bool; growth::Real=2.0,
                          shrink::Real=0.5, floor::Real=0.5)
    _check_positive("half_life", half_life)
    _check_positive("growth", growth)
    _check_positive("shrink", shrink)
    _check_positive("floor", floor)
    growth < 1 && throw(ArgumentError("growth must be >= 1"))
    shrink > 1 && throw(ArgumentError("shrink must be <= 1"))
    h = Float64(half_life)
    return success ? h * Float64(growth) : max(Float64(floor), h * Float64(shrink))
end

"""
    practice_schedule(outcomes; initial_half_life=1.0, target_retention=0.8,
                      growth=2.0, shrink=0.5, floor=0.5) -> Vector{Float64}

Day of every practice session given the outcomes of the sessions so far. Session 1
happens on day 0. After each session the half-life is updated with `update_half_life`
and the next session is scheduled `next_interval(h, target_retention)` days later.
Returns `length(outcomes) + 1` days (the last one is the next planned session).
"""
function practice_schedule(outcomes::AbstractVector; initial_half_life::Real=1.0,
                           target_retention::Real=0.8, growth::Real=2.0,
                           shrink::Real=0.5, floor::Real=0.5)
    _check_positive("initial_half_life", initial_half_life)
    h = Float64(initial_half_life)
    days = [0.0]
    for ok in outcomes
        h = update_half_life(h, Bool(ok); growth=growth, shrink=shrink, floor=floor)
        push!(days, days[end] + next_interval(h, target_retention))
    end
    return days
end

# --------------------------------------------------------------------------
# Independent review panels
# --------------------------------------------------------------------------

"""
    majority_vote_accuracy(n_reviewers, accuracy) -> Float64

Probability that a majority of `n_reviewers` independent reviewers, each correct with
probability `accuracy`, is correct. Ties (even panels) are split 50/50, so an even panel
is no better than the odd panel below it.
"""
function majority_vote_accuracy(n_reviewers::Integer, accuracy::Real)
    (n_reviewers isa Bool || n_reviewers < 1) &&
        throw(ArgumentError("n_reviewers must be a positive integer"))
    _check_prob("accuracy", accuracy)
    n = Int(n_reviewers)
    a = Float64(accuracy)
    total = 0.0
    for k in 0:n
        # Exact binomial coefficient (BigInt avoids Int64 overflow for large panels).
        pk = Float64(binomial(big(n), k)) * a^k * (1.0 - a)^(n - k)
        if 2 * k > n
            total += pk
        elseif 2 * k == n
            total += 0.5 * pk
        end
    end
    return total
end

"""
    layered_residual_error(base_error, catch_probs) -> Float64

Error rate left after independent QA layers: `eps * prod(1 - p_i)`.
"""
function layered_residual_error(base_error::Real, catch_probs::AbstractVector)
    _check_prob("base_error", base_error)
    out = Float64(base_error)
    for (i, p) in enumerate(catch_probs)
        _check_prob("catch_probs[$i]", p)
        out *= 1.0 - Float64(p)
    end
    return out
end

end # module Ch525Oversight
