//! Human-in-the-loop oversight: measuring, gating, and maintaining the overseer (Rust).
//!
//! Mirrors the Python module `aiinaction.ch525_oversight` and the Julia module
//! `AIInAction.Ch525Oversight`. The shared fixtures in the tests below match the
//! Python/Julia suites, which keeps the three implementations at parity. Every
//! numeric function is ported one-to-one; the stateful `OversightGate` harness
//! (with `AgentAction`, `ReviewRecord`, and `OversightReport`) is Python only.
//!
//! The module turns the remedies of Chapter 525 ("Human in the Loop") into code
//! that can run next to a real agent:
//!
//! * **Measuring the overseer.** Equal-variance signal detection theory
//!   ([`sdt_measures`]) treats a defective agent action as the signal. The
//!   sensitivity `d' = z(H) - z(F)` says how well a reviewer separates good actions
//!   from bad ones; the criterion `c = -(z(H) + z(F)) / 2` says how willing they are
//!   to approve. Rubber-stamping shows up as a rising `c`, fatigue as a falling `d'`.
//! * **Canaries.** Known-defective actions planted in the review stream give an
//!   unbiased estimate of the catch rate ([`wilson_interval`],
//!   [`zero_failure_upper_bound`], [`canaries_to_detect_decline`]).
//! * **Behavioral signatures.** Review time that stops scaling with complexity
//!   ([`ols_slope`]), a falling override rate ([`rolling_rate`]), and a downward
//!   shift in review time ([`cusum_lower`]).
//! * **Approval design.** [`gate_action`] routes an action to automatic execution,
//!   human review, or a block by minimizing expected loss subject to a cap on
//!   residual harm. Review pays only when `p * d * H > c`.
//! * **Skill maintenance.** A first-order skill model ([`skill_at`],
//!   [`steady_state_skill`], [`min_practice_rate`]) formalizes Bainbridge's irony
//!   of automation, and a half-life scheduler ([`practice_schedule`]) spaces the
//!   deliberate practice sessions.
//! * **Independent panels.** [`majority_vote_accuracy`] and
//!   [`layered_residual_error`] quantify what independent, multi-layer review buys.
//!
//! This is a std-only implementation. Errors are reported as `Err(String)` for
//! every input on which the Python module raises `ValueError`. Inputs that Python
//! must check at run time but Rust rules out by type (negative counts, an unknown
//! correction name) need no run-time check here.

/// Two-sided 95% standard normal critical value `z_{0.975}`.
pub const Z_95_TWO_SIDED: f64 = 1.959963984540054;

// --------------------------------------------------------------------------
// Validation helpers
// --------------------------------------------------------------------------

/// Checks that `value` is a finite probability in `[0, 1]`, with either end
/// optionally open.
fn check_prob(name: &str, value: f64, open_low: bool, open_high: bool) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("{} must be finite, got {}", name, value));
    }
    let low_ok = if open_low { value > 0.0 } else { value >= 0.0 };
    let high_ok = if open_high { value < 1.0 } else { value <= 1.0 };
    if !(low_ok && high_ok) {
        let lo = if open_low { "(0" } else { "[0" };
        let hi = if open_high { "1)" } else { "1]" };
        return Err(format!(
            "{} must lie in {}, {}, got {}",
            name, lo, hi, value
        ));
    }
    Ok(())
}

/// Checks that `value` is a positive finite number.
fn check_positive(name: &str, value: f64) -> Result<(), String> {
    if !(value.is_finite() && value > 0.0) {
        return Err(format!(
            "{} must be a positive finite number, got {}",
            name, value
        ));
    }
    Ok(())
}

/// Checks that `value` is a non-negative finite number.
fn check_nonneg(name: &str, value: f64) -> Result<(), String> {
    if !(value.is_finite() && value >= 0.0) {
        return Err(format!(
            "{} must be a non-negative finite number, got {}",
            name, value
        ));
    }
    Ok(())
}

// --------------------------------------------------------------------------
// Signal detection theory for overseers
// --------------------------------------------------------------------------

// AS241 (PPND16) coefficients, highest degree first, exactly as in CPython's
// `statistics._normal_dist_inv_cdf`.
#[allow(clippy::excessive_precision)]
const AS241_CENTRAL_NUM: [f64; 8] = [
    2.5090809287301226727e+3,
    3.3430575583588128105e+4,
    6.7265770927008700853e+4,
    4.5921953931549871457e+4,
    1.3731693765509461125e+4,
    1.9715909503065514427e+3,
    1.3314166789178437745e+2,
    3.3871328727963666080e+0,
];
#[allow(clippy::excessive_precision)]
const AS241_CENTRAL_DEN: [f64; 8] = [
    5.2264952788528545610e+3,
    2.8729085735721942674e+4,
    3.9307895800092710610e+4,
    2.1213794301586595867e+4,
    5.3941960214247511077e+3,
    6.8718700749205790830e+2,
    4.2313330701600911252e+1,
    1.0,
];
#[allow(clippy::excessive_precision)]
const AS241_NEAR_NUM: [f64; 8] = [
    7.7454501427834140764e-4,
    2.2723844989269184583e-2,
    2.4178072517745061177e-1,
    1.2704582524523683826e+0,
    3.6478483247632045960e+0,
    5.7694972214606914055e+0,
    4.6303378461565452959e+0,
    1.4234371107496835773e+0,
];
#[allow(clippy::excessive_precision)]
const AS241_NEAR_DEN: [f64; 8] = [
    1.0507500716444168432e-9,
    5.4759380849953449460e-4,
    1.5198666563616457197e-2,
    1.4810397642748007459e-1,
    6.8976733498510000455e-1,
    1.6763848301838038494e+0,
    2.0531916266377588219e+0,
    1.0,
];
#[allow(clippy::excessive_precision)]
const AS241_FAR_NUM: [f64; 8] = [
    2.0103343992922881327e-7,
    2.7115555687434875782e-5,
    1.2426609473880784386e-3,
    2.6532189526576123093e-2,
    2.9656057182850489123e-1,
    1.7848265399172913358e+0,
    5.4637849111641143699e+0,
    6.6579046435011037772e+0,
];
#[allow(clippy::excessive_precision)]
const AS241_FAR_DEN: [f64; 8] = [
    2.0442631033899397856e-15,
    1.4215117583164458887e-7,
    1.8463183175100546818e-5,
    7.8686913114561325910e-4,
    1.4875361290850614853e-2,
    1.3692988092273580531e-1,
    5.9983220655588793769e-1,
    1.0,
];

/// Horner evaluation `(((c0 r + c1) r + c2) ... ) r + c7`, the same operation
/// order as the nested expressions in CPython.
fn horner(coeffs: &[f64; 8], r: f64) -> f64 {
    coeffs[1..].iter().fold(coeffs[0], |acc, &c| acc * r + c)
}

/// Wichura's algorithm AS241 (PPND16), the same rational approximation and the
/// same operation order that CPython uses for `statistics.NormalDist.inv_cdf`.
///
/// Assumes `0 < p < 1`; callers validate. Accurate to about 1e-16 relative.
fn ppnd16(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return horner(&AS241_CENTRAL_NUM, r) * q / horner(&AS241_CENTRAL_DEN, r);
    }
    let r = (-(if q <= 0.0 { p } else { 1.0 - p }).ln()).sqrt();
    let x = if r <= 5.0 {
        let r = r - 1.6;
        horner(&AS241_NEAR_NUM, r) / horner(&AS241_NEAR_DEN, r)
    } else {
        let r = r - 5.0;
        horner(&AS241_FAR_NUM, r) / horner(&AS241_FAR_DEN, r)
    };
    if q < 0.0 {
        -x
    } else {
        x
    }
}

/// Inverse standard normal CDF `z(p) = Phi^{-1}(p)` (Wichura AS241).
///
/// Matches Python's `statistics.NormalDist().inv_cdf(p)`. `p` must lie strictly
/// inside `(0, 1)`.
pub fn probit(p: f64) -> Result<f64, String> {
    check_prob("p", p, true, true)?;
    Ok(ppnd16(p))
}

/// How [`sdt_measures`] turns confusion counts into rates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Correction {
    /// Hautus (1995) log-linear correction: add 0.5 to every cell, so `z` stays
    /// finite for a reviewer who catches every canary. The Python default.
    #[default]
    LogLinear,
    /// Raw rates; errors if either the hit or the false-alarm rate is 0 or 1.
    None,
}

/// Equal-variance Gaussian signal detection summary of one reviewer.
///
/// The *signal* is a defective agent action. A hit is a rejected defective action,
/// a false alarm is a rejected correct action. Positive `criterion` means the
/// reviewer needs strong evidence before rejecting, i.e. leans toward approving; a
/// rising criterion over a shift is the signature of rubber-stamping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdtResult {
    /// Hit rate `H` (after any correction).
    pub hit_rate: f64,
    /// False-alarm rate `F` (after any correction).
    pub false_alarm_rate: f64,
    /// Sensitivity `d' = z(H) - z(F)`.
    pub d_prime: f64,
    /// Criterion `c = -(z(H) + z(F)) / 2`.
    pub criterion: f64,
}

/// Sensitivity `d'` and criterion `c` from a reviewer's confusion counts.
///
/// Errors if there is no defective (`hits + misses == 0`) or no correct
/// (`false_alarms + correct_rejections == 0`) action, or if
/// `Correction::None` meets a raw rate of 0 or 1.
pub fn sdt_measures(
    hits: u64,
    misses: u64,
    false_alarms: u64,
    correct_rejections: u64,
    correction: Correction,
) -> Result<SdtResult, String> {
    let n_signal = hits + misses;
    let n_noise = false_alarms + correct_rejections;
    if n_signal == 0 || n_noise == 0 {
        return Err("need at least one defective and one correct action".to_string());
    }
    let (h, f) = match correction {
        Correction::LogLinear => (
            (hits as f64 + 0.5) / (n_signal as f64 + 1.0),
            (false_alarms as f64 + 0.5) / (n_noise as f64 + 1.0),
        ),
        Correction::None => {
            let h = hits as f64 / n_signal as f64;
            let f = false_alarms as f64 / n_noise as f64;
            if h == 0.0 || h == 1.0 || f == 0.0 || f == 1.0 {
                return Err(
                    "raw hit or false-alarm rate is 0 or 1; use Correction::LogLinear".to_string(),
                );
            }
            (h, f)
        }
    };
    let zh = probit(h)?;
    let zf = probit(f)?;
    Ok(SdtResult {
        hit_rate: h,
        false_alarm_rate: f,
        d_prime: zh - zf,
        criterion: -(zh + zf) / 2.0,
    })
}

// --------------------------------------------------------------------------
// Canary statistics
// --------------------------------------------------------------------------

/// Wilson score interval `(lower, upper)` for a binomial proportion such as a
/// canary catch rate. Pass [`Z_95_TWO_SIDED`] for the Python default `z`.
pub fn wilson_interval(successes: u64, trials: u64, z: f64) -> Result<(f64, f64), String> {
    if trials == 0 {
        return Err("trials must be positive".to_string());
    }
    if successes > trials {
        return Err("successes cannot exceed trials".to_string());
    }
    check_positive("z", z)?;
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    Ok((0.0f64.max(center - half), 1.0f64.min(center + half)))
}

/// Exact one-sided upper bound on the miss rate after `trials` caught canaries.
///
/// Solves `(1 - p)^n = alpha` for `p`, giving `1 - alpha^(1/n)`. For
/// `alpha = 0.05` (the Python default) this is close to the rule of three, `3 / n`.
pub fn zero_failure_upper_bound(trials: u64, alpha: f64) -> Result<f64, String> {
    if trials == 0 {
        return Err("trials must be positive".to_string());
    }
    check_prob("alpha", alpha, true, true)?;
    Ok(1.0 - alpha.powf(1.0 / trials as f64))
}

/// Canaries needed for a one-sided test to detect a catch-rate drop from `p0` to
/// `p1`.
///
/// Normal-approximation sample size
/// `n = ceil(((z_{1-alpha} sqrt(p0 q0) + z_{power} sqrt(p1 q1)) / (p0 - p1))^2)`.
/// The Python defaults are `alpha = 0.05` and `power = 0.8`.
pub fn canaries_to_detect_decline(p0: f64, p1: f64, alpha: f64, power: f64) -> Result<u64, String> {
    check_prob("p0", p0, true, true)?;
    check_prob("p1", p1, true, true)?;
    check_prob("alpha", alpha, true, true)?;
    check_prob("power", power, true, true)?;
    if p1 >= p0 {
        return Err("p1 must be smaller than p0 (we detect a decline)".to_string());
    }
    let za = probit(1.0 - alpha)?;
    let zb = probit(power)?;
    let num = za * (p0 * (1.0 - p0)).sqrt() + zb * (p1 * (1.0 - p1)).sqrt();
    Ok(((num / (p0 - p1)).powi(2) - 1e-12).ceil() as u64)
}

// --------------------------------------------------------------------------
// Behavioral signatures
// --------------------------------------------------------------------------

/// Least-squares slope of `y` on `x`.
///
/// Applied to review time against action complexity, a slope near zero means
/// scrutiny no longer scales with what is at stake.
pub fn ols_slope(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".to_string());
    }
    if x.len() < 2 {
        return Err("need at least two points".to_string());
    }
    let n = x.len() as f64;
    let mx = x.iter().fold(0.0, |acc, &v| acc + v) / n;
    let my = y.iter().fold(0.0, |acc, &v| acc + v) / n;
    let sxx = x.iter().fold(0.0, |acc, &xi| acc + (xi - mx).powi(2));
    if sxx == 0.0 {
        return Err("x has zero variance".to_string());
    }
    let sxy = x
        .iter()
        .zip(y)
        .fold(0.0, |acc, (&xi, &yi)| acc + (xi - mx) * (yi - my));
    Ok(sxy / sxx)
}

/// Trailing-window mean of 0/1 flags (for example, override decisions).
///
/// Returns `flags.len() - window + 1` values; value `t` averages
/// `flags[t..t + window]`.
pub fn rolling_rate(flags: &[f64], window: usize) -> Result<Vec<f64>, String> {
    if window < 1 {
        return Err("window must be a positive integer".to_string());
    }
    if window > flags.len() {
        return Err("window longer than the series".to_string());
    }
    let w = window as f64;
    let mut s = flags[..window].iter().fold(0.0, |acc, &v| acc + v);
    let mut out = Vec::with_capacity(flags.len() - window + 1);
    out.push(s / w);
    for t in window..flags.len() {
        s += flags[t] - flags[t - window];
        out.push(s / w);
    }
    Ok(out)
}

/// Lower one-sided CUSUM path and the first alarm, if any.
#[derive(Clone, Debug, PartialEq)]
pub struct CusumResult {
    /// The statistic `S_t` after each observation.
    pub statistic: Vec<f64>,
    /// 0-based index of the first `t` with `S_t > threshold` (Julia reports it
    /// 1-based), or `None` if the series never alarms.
    pub alarm_index: Option<usize>,
}

/// Page's one-sided CUSUM for a *downward* shift in the mean.
///
/// `S_t = max(0, S_{t-1} + (target_mean - x_t) - slack)` with `S_{-1} = 0`; alarm
/// at the first `t` with `S_t > threshold`. On log review times this flags the
/// moment a reviewer starts spending less time per action.
pub fn cusum_lower(
    values: &[f64],
    target_mean: f64,
    slack: f64,
    threshold: f64,
) -> Result<CusumResult, String> {
    check_nonneg("slack", slack)?;
    check_positive("threshold", threshold)?;
    if !target_mean.is_finite() {
        return Err("target_mean must be finite".to_string());
    }
    let mut s = 0.0f64;
    let mut path = Vec::with_capacity(values.len());
    let mut alarm: Option<usize> = None;
    for (t, &x) in values.iter().enumerate() {
        s = 0.0f64.max(s + (target_mean - x) - slack);
        path.push(s);
        if alarm.is_none() && s > threshold {
            alarm = Some(t);
        }
    }
    Ok(CusumResult {
        statistic: path,
        alarm_index: alarm,
    })
}

// --------------------------------------------------------------------------
// Approval design: expected-loss gating
// --------------------------------------------------------------------------

/// Where the gate sends one agent action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Route {
    /// Execute without human review.
    Auto,
    /// Send to a human reviewer.
    Review,
    /// Block and escalate.
    Block,
}

impl Route {
    /// The lowercase name used by the Python `Route` string enum.
    pub fn as_str(&self) -> &'static str {
        match self {
            Route::Auto => "auto",
            Route::Review => "review",
            Route::Block => "block",
        }
    }
}

/// The routing decision for one action plus the quantities behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateDecision {
    /// The chosen route.
    pub route: Route,
    /// Expected loss of automatic execution, `p H`.
    pub loss_auto: f64,
    /// Expected loss of review, `c + p (1 - d) H`.
    pub loss_review: f64,
    /// Residual expected harm under automatic execution, `p H`.
    pub residual_auto: f64,
    /// Residual expected harm under review, `p (1 - d) H`.
    pub residual_review: f64,
    /// Error probability above which review beats automatic execution.
    pub threshold: f64,
}

/// Error probability above which review beats automatic execution: `c / (d H)`.
///
/// Returns `f64::INFINITY` when `detection_prob == 0` or `harm == 0`, since review
/// then never pays.
pub fn review_threshold(harm: f64, review_cost: f64, detection_prob: f64) -> Result<f64, String> {
    check_nonneg("harm", harm)?;
    check_nonneg("review_cost", review_cost)?;
    check_prob("detection_prob", detection_prob, false, false)?;
    if detection_prob == 0.0 || harm == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(review_cost / (detection_prob * harm))
}

/// Route one agent action to [`Route::Auto`], [`Route::Review`], or
/// [`Route::Block`].
///
/// Expected losses: `AUTO = p H` and `REVIEW = c + p (1 - d) H`. The gate picks the
/// cheaper route (ties go to `Auto`) among those whose residual expected harm
/// (`p H` or `p (1 - d) H`) stays within `max_residual_loss`; if neither does, the
/// action is blocked and escalated. Pass `f64::INFINITY` for the Python default
/// (no residual cap).
pub fn gate_action(
    p_error: f64,
    harm: f64,
    review_cost: f64,
    detection_prob: f64,
    max_residual_loss: f64,
) -> Result<GateDecision, String> {
    check_prob("p_error", p_error, false, false)?;
    check_nonneg("harm", harm)?;
    check_nonneg("review_cost", review_cost)?;
    check_prob("detection_prob", detection_prob, false, false)?;
    // NaN fails this check too; +infinity (no cap) passes.
    if max_residual_loss.is_nan() || max_residual_loss < 0.0 {
        return Err("max_residual_loss must be non-negative".to_string());
    }
    let residual_auto = p_error * harm;
    let residual_review = p_error * (1.0 - detection_prob) * harm;
    let loss_auto = residual_auto;
    let loss_review = review_cost + residual_review;
    let auto_ok = residual_auto <= max_residual_loss;
    let review_ok = residual_review <= max_residual_loss;
    let route = match (auto_ok, review_ok) {
        (true, true) => {
            if loss_review < loss_auto {
                Route::Review
            } else {
                Route::Auto
            }
        }
        (true, false) => Route::Auto,
        (false, true) => Route::Review,
        (false, false) => Route::Block,
    };
    Ok(GateDecision {
        route,
        loss_auto,
        loss_review,
        residual_auto,
        residual_review,
        threshold: review_threshold(harm, review_cost, detection_prob)?,
    })
}

/// Saturating detection curve `d(r) = d_max (1 - exp(-r / r0))`.
pub fn detection_from_review_time(
    review_minutes: f64,
    d_max: f64,
    time_scale: f64,
) -> Result<f64, String> {
    check_nonneg("review_minutes", review_minutes)?;
    check_prob("d_max", d_max, false, false)?;
    check_positive("time_scale", time_scale)?;
    Ok(d_max * (1.0 - (-review_minutes / time_scale).exp()))
}

/// Vigilance decrement `d(t) = d_floor + (d_start - d_floor) exp(-t / tau)`.
pub fn vigilance(
    time_on_task: f64,
    d_start: f64,
    d_floor: f64,
    time_constant: f64,
) -> Result<f64, String> {
    check_nonneg("time_on_task", time_on_task)?;
    check_prob("d_start", d_start, false, false)?;
    check_prob("d_floor", d_floor, false, false)?;
    check_positive("time_constant", time_constant)?;
    Ok(d_floor + (d_start - d_floor) * (-time_on_task / time_constant).exp())
}

// --------------------------------------------------------------------------
// Skill maintenance: the irony of automation as an ODE
// --------------------------------------------------------------------------

/// Practice intensity `u = (1 - a) + q`.
fn practice_intensity(automation: f64, practice: f64) -> Result<f64, String> {
    check_prob("automation", automation, false, false)?;
    check_nonneg("practice", practice)?;
    Ok((1.0 - automation) + practice)
}

/// Fixed point of `ds/dt = eta u (1 - s) - lambda s` with `u = (1 - a) + q`:
/// `s* = eta u / (eta u + lambda)`.
pub fn steady_state_skill(
    automation: f64,
    practice: f64,
    learn_rate: f64,
    decay_rate: f64,
) -> Result<f64, String> {
    check_positive("learn_rate", learn_rate)?;
    check_positive("decay_rate", decay_rate)?;
    let gain = learn_rate * practice_intensity(automation, practice)?;
    Ok(gain / (gain + decay_rate))
}

/// Closed-form skill `s(t) = s* + (s0 - s*) exp(-(eta u + lambda) t)`.
pub fn skill_at(
    t: f64,
    s0: f64,
    automation: f64,
    practice: f64,
    learn_rate: f64,
    decay_rate: f64,
) -> Result<f64, String> {
    check_nonneg("t", t)?;
    check_prob("s0", s0, false, false)?;
    check_positive("learn_rate", learn_rate)?;
    check_positive("decay_rate", decay_rate)?;
    let gain = learn_rate * practice_intensity(automation, practice)?;
    let s_star = gain / (gain + decay_rate);
    Ok(s_star + (s0 - s_star) * (-(gain + decay_rate) * t).exp())
}

/// Smallest deliberate-practice rate `q` with steady-state skill at least
/// `target_skill`: `q_min = max(0, lambda s / (eta (1 - s)) - (1 - a))`.
pub fn min_practice_rate(
    target_skill: f64,
    automation: f64,
    learn_rate: f64,
    decay_rate: f64,
) -> Result<f64, String> {
    check_prob("target_skill", target_skill, false, true)?;
    check_prob("automation", automation, false, false)?;
    check_positive("learn_rate", learn_rate)?;
    check_positive("decay_rate", decay_rate)?;
    let required = decay_rate * target_skill / (learn_rate * (1.0 - target_skill));
    Ok(0.0f64.max(required - (1.0 - automation)))
}

/// Linear map from skill in `[0, 1]` to detection probability in
/// `[d_floor, d_ceiling]`.
pub fn detection_from_skill(skill: f64, d_floor: f64, d_ceiling: f64) -> Result<f64, String> {
    check_prob("skill", skill, false, false)?;
    check_prob("d_floor", d_floor, false, false)?;
    check_prob("d_ceiling", d_ceiling, false, false)?;
    if d_ceiling < d_floor {
        return Err("d_ceiling must be >= d_floor".to_string());
    }
    Ok(d_floor + (d_ceiling - d_floor) * skill)
}

// --------------------------------------------------------------------------
// Spaced practice scheduling (half-life model)
// --------------------------------------------------------------------------

/// Exponential forgetting `p = 2^(-elapsed / h)`.
pub fn recall_probability(elapsed: f64, half_life: f64) -> Result<f64, String> {
    check_nonneg("elapsed", elapsed)?;
    check_positive("half_life", half_life)?;
    Ok(2.0f64.powf(-elapsed / half_life))
}

/// Days until predicted recall falls to `target_retention`: `h log2(1 / R)`.
pub fn next_interval(half_life: f64, target_retention: f64) -> Result<f64, String> {
    check_positive("half_life", half_life)?;
    check_prob("target_retention", target_retention, true, true)?;
    Ok(half_life * (1.0 / target_retention).log2())
}

/// Grow the half-life after a successful session (`h * growth`), shrink it after a
/// failure (`max(floor, h * shrink)`). The Python defaults are `growth = 2.0`,
/// `shrink = 0.5`, `floor = 0.5`.
pub fn update_half_life(
    half_life: f64,
    success: bool,
    growth: f64,
    shrink: f64,
    floor: f64,
) -> Result<f64, String> {
    check_positive("half_life", half_life)?;
    check_positive("growth", growth)?;
    check_positive("shrink", shrink)?;
    check_positive("floor", floor)?;
    if growth < 1.0 {
        return Err("growth must be >= 1".to_string());
    }
    if shrink > 1.0 {
        return Err("shrink must be <= 1".to_string());
    }
    Ok(if success {
        half_life * growth
    } else {
        floor.max(half_life * shrink)
    })
}

/// Day of every practice session given the outcomes of the sessions so far.
///
/// Session 0 happens on day 0. After session `i` (outcome `outcomes[i]`) the
/// half-life is updated and session `i + 1` is scheduled
/// `next_interval(h, target_retention)` days later. Returns `outcomes.len() + 1`
/// days (the last one is the next planned session). The Python defaults are
/// `initial_half_life = 1.0`, `target_retention = 0.8`, `growth = 2.0`,
/// `shrink = 0.5`, `floor = 0.5`.
pub fn practice_schedule(
    outcomes: &[bool],
    initial_half_life: f64,
    target_retention: f64,
    growth: f64,
    shrink: f64,
    floor: f64,
) -> Result<Vec<f64>, String> {
    check_positive("initial_half_life", initial_half_life)?;
    let mut h = initial_half_life;
    let mut days = Vec::with_capacity(outcomes.len() + 1);
    let mut day = 0.0f64;
    days.push(day);
    for &ok in outcomes {
        h = update_half_life(h, ok, growth, shrink, floor)?;
        day += next_interval(h, target_retention)?;
        days.push(day);
    }
    Ok(days)
}

// --------------------------------------------------------------------------
// Independent review panels
// --------------------------------------------------------------------------

/// Probability that a majority of `n_reviewers` independent reviewers, each correct
/// with probability `accuracy`, is correct (exact binomial; ties split 50/50).
///
/// With ties split by a coin, an even panel is no better than the odd panel below
/// it.
pub fn majority_vote_accuracy(n_reviewers: u64, accuracy: f64) -> Result<f64, String> {
    if n_reviewers < 1 {
        return Err("n_reviewers must be a positive integer".to_string());
    }
    check_prob("accuracy", accuracy, false, false)?;
    let n = n_reviewers;
    let mut total = 0.0f64;
    // Binomial coefficient C(n, k), updated multiplicatively; exact while it stays
    // below 2^53.
    let mut comb = 1.0f64;
    for k in 0..=n {
        if k > 0 {
            comb = comb * (n - k + 1) as f64 / k as f64;
        }
        let pk = comb * accuracy.powf(k as f64) * (1.0 - accuracy).powf((n - k) as f64);
        if 2 * k > n {
            total += pk;
        } else if 2 * k == n {
            total += 0.5 * pk;
        }
    }
    Ok(total)
}

/// Error rate left after independent QA layers: `eps * prod_i (1 - p_i)`.
pub fn layered_residual_error(base_error: f64, catch_probs: &[f64]) -> Result<f64, String> {
    check_prob("base_error", base_error, false, false)?;
    let mut out = base_error;
    for (i, &p) in catch_probs.iter().enumerate() {
        check_prob(&format!("catch_probs[{}]", i), p, false, false)?;
        out *= 1.0 - p;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const SDT_COUNTS: (u64, u64, u64, u64) = (18, 2, 5, 75); // hits, misses, FAs, CRs
    const SDT_LOGLINEAR_DPRIME: f64 = 2.6713670993913583;
    const SDT_LOGLINEAR_CRITERION: f64 = 0.15592243208381895;
    const SDT_RAW_DPRIME: f64 = 2.8156721098971467;
    const SDT_RAW_CRITERION: f64 = 0.12628448940397252;
    const SDT_PERFECT_DPRIME: f64 = 4.482858869990343; // (20, 0, 0, 80), log-linear
    const SDT_PERFECT_CRITERION: f64 = 0.26067703834789313;

    const WILSON_18_OF_20: (f64, f64) = (0.6989663547715128, 0.9721335187862319);
    const ZERO_FAILURE_30: f64 = 0.09503385285530419;
    const CANARIES_09_07: u64 = 20;
    const CANARIES_095_085: u64 = 44;
    const CANARIES_09_08_POWER09: u64 = 102;

    const SLOPE_X: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    const SLOPE_Y: [f64; 5] = [2.0, 4.1, 5.9, 8.2, 9.8];
    const SLOPE: f64 = 1.97;

    const ROLLING_FLAGS: [f64; 8] = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    const ROLLING_3: [f64; 6] = [
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
        2.0 / 3.0,
        2.0 / 3.0,
        1.0 / 3.0,
    ];

    const CUSUM_VALUES: [f64; 8] = [5.0, 5.1, 4.9, 4.2, 4.0, 3.9, 4.1, 3.8];
    const CUSUM_PATH: [f64; 8] = [0.0, 0.0, 0.0, 0.55, 1.3, 2.15, 2.8, 3.75];
    const CUSUM_ALARM: usize = 5; // 0-based (Julia: 6)

    const SKILL_AT_3: f64 = 0.34869187738724783; // t=3, s0=0.8, a=0.9, q=0, eta=2, lambda=0.5
    const MIN_PRACTICE: f64 = 0.275; // target 0.6, a=0.9, eta=2, lambda=0.5
    const STEADY_A09_Q015: f64 = 0.5;

    const SCHEDULE_TTFT: [f64; 5] = [
        0.0,
        0.6438561897747247,
        1.931568569324174,
        2.575424759098899,
        3.863137138648348,
    ];

    const MAJORITY_3_085: f64 = 0.93925;
    const MAJORITY_5_07: f64 = 0.83692;
    const VIGILANCE_30: f64 = 0.666939048044529; // t=30, d0=0.9, floor=0.6, tau=20
    const DETECT_REVIEW_2: f64 = 0.6995827187900596; // r=2, d_max=0.95, r0=1.5

    const TOL: f64 = 1e-12;

    /// Absolute-tolerance comparison (pytest `approx(..., abs=tol)`).
    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {} within {} of {}",
            actual,
            tol,
            expected
        );
    }

    /// pytest's default `approx`: relative 1e-6 or absolute 1e-12.
    fn assert_approx(actual: f64, expected: f64) {
        let tol = (1e-6 * expected.abs()).max(1e-12);
        assert!(
            (actual - expected).abs() <= tol,
            "expected {} approx {}",
            actual,
            expected
        );
    }

    fn sdt(counts: (u64, u64, u64, u64), correction: Correction) -> Result<SdtResult, String> {
        sdt_measures(counts.0, counts.1, counts.2, counts.3, correction)
    }

    // ----------------------------------------------------------------------
    // Signal detection
    // ----------------------------------------------------------------------

    #[test]
    fn probit_known_values() {
        assert_close(probit(0.975).unwrap(), 1.959963984540054, TOL);
        assert_close(probit(0.5).unwrap(), 0.0, 1e-15);
        assert_close(probit(0.02).unwrap(), -2.053748910631822, TOL);
        assert!(probit(0.0).is_err());
        assert!(probit(1.0).is_err());
        assert!(probit(f64::NAN).is_err());
    }

    #[test]
    fn probit_matches_python_inv_cdf_on_every_branch() {
        // Reference values from Python's statistics.NormalDist().inv_cdf(p),
        // covering the central region and both tail branches of AS241.
        let cases: [(f64, f64); 14] = [
            (1e-300, -37.0470962993612),
            (1e-12, -7.034483825301132),
            (1e-10, -6.361340902404056),
            (0.001, -3.090232306167813),
            (0.02, -2.0537489106318225),
            (0.07, -1.4757910281791702),
            (0.075, -1.4395314709384557),
            (0.3, -0.5244005127080407),
            (0.6, 0.2533471031357998),
            (0.925, 1.439531470938456),
            (0.93, 1.4757910281791708),
            (0.975, 1.9599639845400536),
            (0.999, 3.090232306167813),
            (0.999999999999, 7.0344869100478356),
        ];
        for (p, z) in cases {
            let got = probit(p).unwrap();
            assert!(
                (got - z).abs() <= 1e-15 * z.abs().max(1.0),
                "probit({}) = {} but Python gives {}",
                p,
                got,
                z
            );
        }
        assert_eq!(probit(0.5).unwrap(), 0.0);
    }

    #[test]
    fn sdt_loglinear_fixture() {
        let r = sdt(SDT_COUNTS, Correction::LogLinear).unwrap();
        assert_close(r.d_prime, SDT_LOGLINEAR_DPRIME, TOL);
        assert_close(r.criterion, SDT_LOGLINEAR_CRITERION, TOL);
        assert_approx(r.hit_rate, 18.5 / 21.0);
        assert_approx(r.false_alarm_rate, 5.5 / 81.0);
    }

    #[test]
    fn correction_default_is_loglinear() {
        assert_eq!(Correction::default(), Correction::LogLinear);
        let a = sdt(SDT_COUNTS, Correction::default()).unwrap();
        let b = sdt(SDT_COUNTS, Correction::LogLinear).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn sdt_raw_fixture() {
        let r = sdt(SDT_COUNTS, Correction::None).unwrap();
        assert_close(r.d_prime, SDT_RAW_DPRIME, TOL);
        assert_close(r.criterion, SDT_RAW_CRITERION, TOL);
    }

    #[test]
    fn sdt_perfect_reviewer_stays_finite() {
        let r = sdt_measures(20, 0, 0, 80, Correction::LogLinear).unwrap();
        assert!(r.d_prime.is_finite());
        assert_close(r.d_prime, SDT_PERFECT_DPRIME, TOL);
        assert_close(r.criterion, SDT_PERFECT_CRITERION, TOL);
        assert!(sdt_measures(20, 0, 0, 80, Correction::None).is_err());
    }

    #[test]
    fn sdt_rubber_stamp_raises_criterion() {
        let careful = sdt_measures(18, 2, 8, 72, Correction::LogLinear).unwrap();
        let stamping = sdt_measures(10, 10, 1, 79, Correction::LogLinear).unwrap();
        assert!(stamping.criterion > careful.criterion);
    }

    #[test]
    fn sdt_validation() {
        // No defective actions, then no correct actions.
        assert!(sdt_measures(0, 0, 1, 1, Correction::LogLinear).is_err());
        assert!(sdt_measures(1, 1, 0, 0, Correction::LogLinear).is_err());
        // Python's negative-count and unknown-correction errors are ruled out by
        // the u64 count type and the Correction enum.
    }

    // ----------------------------------------------------------------------
    // Canaries
    // ----------------------------------------------------------------------

    #[test]
    fn wilson_fixture() {
        let (lo, hi) = wilson_interval(18, 20, Z_95_TWO_SIDED).unwrap();
        assert_close(lo, WILSON_18_OF_20.0, TOL);
        assert_close(hi, WILSON_18_OF_20.1, TOL);
    }

    #[test]
    fn wilson_edges_and_validation() {
        let (lo, hi) = wilson_interval(0, 10, Z_95_TWO_SIDED).unwrap();
        assert!(lo == 0.0 && 0.0 < hi && hi < 0.35);
        let (lo, hi) = wilson_interval(10, 10, Z_95_TWO_SIDED).unwrap();
        assert_approx(hi, 1.0);
        assert!(lo > 0.65);
        assert!(wilson_interval(3, 2, Z_95_TWO_SIDED).is_err());
        assert!(wilson_interval(0, 0, Z_95_TWO_SIDED).is_err());
        assert!(wilson_interval(1, 2, 0.0).is_err());
        assert!(wilson_interval(1, 2, f64::INFINITY).is_err());
    }

    #[test]
    fn zero_failure_bound_matches_rule_of_three() {
        assert_close(
            zero_failure_upper_bound(30, 0.05).unwrap(),
            ZERO_FAILURE_30,
            TOL,
        );
        for n in [50u64, 100, 1000] {
            let b = zero_failure_upper_bound(n, 0.05).unwrap();
            let three_over_n = 3.0 / n as f64;
            assert!((b - three_over_n).abs() <= 0.05 * three_over_n);
        }
        assert!(zero_failure_upper_bound(0, 0.05).is_err());
        assert!(zero_failure_upper_bound(10, 0.0).is_err());
        assert!(zero_failure_upper_bound(10, 1.0).is_err());
    }

    #[test]
    fn canary_sample_sizes() {
        assert_eq!(
            canaries_to_detect_decline(0.9, 0.7, 0.05, 0.8).unwrap(),
            CANARIES_09_07
        );
        assert_eq!(
            canaries_to_detect_decline(0.95, 0.85, 0.05, 0.8).unwrap(),
            CANARIES_095_085
        );
        assert_eq!(
            canaries_to_detect_decline(0.9, 0.8, 0.05, 0.9).unwrap(),
            CANARIES_09_08_POWER09
        );
        assert!(canaries_to_detect_decline(0.7, 0.9, 0.05, 0.8).is_err());
        assert!(canaries_to_detect_decline(0.8, 0.8, 0.05, 0.8).is_err());
        assert!(canaries_to_detect_decline(0.9, 0.7, 0.0, 0.8).is_err());
    }

    // ----------------------------------------------------------------------
    // Behavioral signatures
    // ----------------------------------------------------------------------

    #[test]
    fn ols_slope_fixture() {
        assert_close(ols_slope(&SLOPE_X, &SLOPE_Y).unwrap(), SLOPE, TOL);
        assert!(ols_slope(&[1.0, 1.0], &[2.0, 3.0]).is_err());
        assert!(ols_slope(&[1.0], &[2.0]).is_err());
        assert!(ols_slope(&[1.0, 2.0], &[2.0]).is_err());
    }

    #[test]
    fn rolling_rate_fixture() {
        let r = rolling_rate(&ROLLING_FLAGS, 3).unwrap();
        assert_eq!(r.len(), ROLLING_3.len());
        for (a, e) in r.iter().zip(ROLLING_3.iter()) {
            assert_approx(*a, *e);
        }
        assert!(rolling_rate(&ROLLING_FLAGS, 0).is_err());
        assert!(rolling_rate(&ROLLING_FLAGS, 9).is_err());
    }

    #[test]
    fn cusum_fixture() {
        let res = cusum_lower(&CUSUM_VALUES, 5.0, 0.25, 1.5).unwrap();
        assert_eq!(res.statistic.len(), CUSUM_PATH.len());
        for (a, e) in res.statistic.iter().zip(CUSUM_PATH.iter()) {
            assert_close(*a, *e, TOL);
        }
        assert_eq!(res.alarm_index, Some(CUSUM_ALARM));
    }

    #[test]
    fn cusum_no_alarm_on_stable_series() {
        let res = cusum_lower(&[5.0, 5.2, 4.9, 5.1, 4.95], 5.0, 0.25, 1.5).unwrap();
        assert_eq!(res.alarm_index, None);
        let max = res
            .statistic
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(max, 0.0);
    }

    #[test]
    fn cusum_validation() {
        assert!(cusum_lower(&CUSUM_VALUES, 5.0, -0.1, 1.5).is_err());
        assert!(cusum_lower(&CUSUM_VALUES, 5.0, 0.25, 0.0).is_err());
        assert!(cusum_lower(&CUSUM_VALUES, f64::NAN, 0.25, 1.5).is_err());
    }

    // ----------------------------------------------------------------------
    // Gating
    // ----------------------------------------------------------------------

    #[test]
    fn gate_routes() {
        let inf = f64::INFINITY;
        assert_eq!(
            gate_action(0.005, 100.0, 1.0, 0.9, inf).unwrap().route,
            Route::Auto
        );
        assert_eq!(
            gate_action(0.02, 100.0, 1.0, 0.9, inf).unwrap().route,
            Route::Review
        );
        assert_eq!(
            gate_action(0.2, 100.0, 1.0, 0.9, 5.0).unwrap().route,
            Route::Review
        );
        assert_eq!(
            gate_action(0.2, 100.0, 1.0, 0.2, 5.0).unwrap().route,
            Route::Block
        );
        assert_eq!(Route::Review.as_str(), "review");
    }

    #[test]
    fn gate_losses_and_threshold() {
        let d = gate_action(0.02, 100.0, 1.0, 0.9, f64::INFINITY).unwrap();
        assert_approx(d.loss_auto, 2.0);
        assert_approx(d.loss_review, 1.2);
        assert_approx(d.threshold, 1.0 / 90.0);
        assert_eq!(review_threshold(100.0, 1.0, 0.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn gate_threshold_is_the_indifference_point() {
        let (harm, cost, det) = (250.0, 3.0, 0.8);
        let p_star = review_threshold(harm, cost, det).unwrap();
        let d = gate_action(p_star, harm, cost, det, f64::INFINITY).unwrap();
        assert_approx(d.loss_auto, d.loss_review);
    }

    #[test]
    fn gate_validation() {
        assert!(gate_action(1.5, 100.0, 1.0, 0.9, f64::INFINITY).is_err());
        assert!(gate_action(0.1, -1.0, 1.0, 0.9, f64::INFINITY).is_err());
        assert!(gate_action(0.1, 100.0, f64::NAN, 0.9, f64::INFINITY).is_err());
        assert!(gate_action(0.1, 100.0, 1.0, 1.1, f64::INFINITY).is_err());
        assert!(gate_action(0.1, 100.0, 1.0, 0.9, -1.0).is_err());
        assert!(gate_action(0.1, 100.0, 1.0, 0.9, f64::NAN).is_err());
        assert!(review_threshold(f64::INFINITY, 1.0, 0.5).is_err());
    }

    #[test]
    fn vigilance_and_review_time_fixtures() {
        assert_close(vigilance(30.0, 0.9, 0.6, 20.0).unwrap(), VIGILANCE_30, TOL);
        assert_approx(vigilance(0.0, 0.9, 0.6, 20.0).unwrap(), 0.9);
        assert_close(
            detection_from_review_time(2.0, 0.95, 1.5).unwrap(),
            DETECT_REVIEW_2,
            TOL,
        );
        assert!(vigilance(-1.0, 0.9, 0.6, 20.0).is_err());
        assert!(vigilance(1.0, 0.9, 0.6, 0.0).is_err());
        assert!(detection_from_review_time(2.0, 1.2, 1.5).is_err());
    }

    // ----------------------------------------------------------------------
    // Skill model
    // ----------------------------------------------------------------------

    #[test]
    fn skill_fixtures() {
        assert_close(
            skill_at(3.0, 0.8, 0.9, 0.0, 2.0, 0.5).unwrap(),
            SKILL_AT_3,
            TOL,
        );
        assert_close(
            min_practice_rate(0.6, 0.9, 2.0, 0.5).unwrap(),
            MIN_PRACTICE,
            TOL,
        );
        assert_close(
            steady_state_skill(0.9, 0.15, 2.0, 0.5).unwrap(),
            STEADY_A09_Q015,
            TOL,
        );
    }

    #[test]
    fn skill_converges_to_steady_state() {
        let s_star = steady_state_skill(0.7, 0.05, 1.5, 0.3).unwrap();
        assert_close(
            skill_at(200.0, 0.1, 0.7, 0.05, 1.5, 0.3).unwrap(),
            s_star,
            1e-10,
        );
    }

    #[test]
    fn min_practice_rate_achieves_target() {
        let q = min_practice_rate(0.75, 0.95, 2.0, 0.5).unwrap();
        assert_approx(steady_state_skill(0.95, q, 2.0, 0.5).unwrap(), 0.75);
        assert_eq!(min_practice_rate(0.1, 0.0, 2.0, 0.5).unwrap(), 0.0);
        assert!(min_practice_rate(1.0, 0.9, 2.0, 0.5).is_err());
    }

    #[test]
    fn full_automation_without_practice_erodes_skill_to_zero() {
        assert_eq!(steady_state_skill(1.0, 0.0, 2.0, 0.5).unwrap(), 0.0);
    }

    #[test]
    fn skill_validation() {
        assert!(steady_state_skill(1.1, 0.0, 2.0, 0.5).is_err());
        assert!(steady_state_skill(0.9, -0.1, 2.0, 0.5).is_err());
        assert!(steady_state_skill(0.9, 0.0, 0.0, 0.5).is_err());
        assert!(skill_at(-1.0, 0.8, 0.9, 0.0, 2.0, 0.5).is_err());
        assert!(skill_at(1.0, 1.5, 0.9, 0.0, 2.0, 0.5).is_err());
    }

    #[test]
    fn detection_from_skill_values_and_validation() {
        assert_approx(detection_from_skill(0.5, 0.3, 0.95).unwrap(), 0.625);
        assert!(detection_from_skill(0.5, 0.9, 0.3).is_err());
    }

    // ----------------------------------------------------------------------
    // Spaced practice
    // ----------------------------------------------------------------------

    #[test]
    fn recall_and_interval() {
        assert_approx(recall_probability(2.0, 2.0).unwrap(), 0.5);
        let (h, r) = (4.0, 0.8);
        assert_approx(
            recall_probability(next_interval(h, r).unwrap(), h).unwrap(),
            r,
        );
        assert!(next_interval(4.0, 1.0).is_err());
        assert!(recall_probability(-1.0, 2.0).is_err());
    }

    #[test]
    fn schedule_fixture() {
        let days = practice_schedule(&[true, true, false, true], 1.0, 0.8, 2.0, 0.5, 0.5).unwrap();
        assert_eq!(days.len(), SCHEDULE_TTFT.len());
        for (a, e) in days.iter().zip(SCHEDULE_TTFT.iter()) {
            assert_close(*a, *e, TOL);
        }
        assert!(practice_schedule(&[true], 0.0, 0.8, 2.0, 0.5, 0.5).is_err());
    }

    #[test]
    fn update_half_life_floor() {
        assert_eq!(update_half_life(0.6, false, 2.0, 0.5, 0.5).unwrap(), 0.5);
        assert!(update_half_life(1.0, true, 0.5, 0.5, 0.5).is_err());
        assert!(update_half_life(1.0, true, 2.0, 1.5, 0.5).is_err());
    }

    // ----------------------------------------------------------------------
    // Panels
    // ----------------------------------------------------------------------

    #[test]
    fn majority_vote_fixtures() {
        assert_close(
            majority_vote_accuracy(3, 0.85).unwrap(),
            MAJORITY_3_085,
            TOL,
        );
        assert_close(majority_vote_accuracy(5, 0.7).unwrap(), MAJORITY_5_07, TOL);
        // With ties split by a coin, an even panel is no better than the odd panel below it.
        assert_approx(
            majority_vote_accuracy(4, 0.85).unwrap(),
            majority_vote_accuracy(3, 0.85).unwrap(),
        );
        assert_approx(majority_vote_accuracy(1, 0.85).unwrap(), 0.85);
        assert!(majority_vote_accuracy(0, 0.85).is_err());
        assert!(majority_vote_accuracy(3, 1.5).is_err());
    }

    #[test]
    fn layered_residual_error_values() {
        assert_approx(
            layered_residual_error(0.1, &[0.7, 0.6, 0.5]).unwrap(),
            0.006,
        );
        assert_approx(layered_residual_error(0.1, &[]).unwrap(), 0.1);
        assert!(layered_residual_error(0.1, &[0.5, 1.2]).is_err());
        assert!(layered_residual_error(-0.1, &[0.5]).is_err());
    }
}
