//! Numeric core of DRDL for dynamic financial risk prediction (Rust).
//!
//! Mirrors the Python module `aiinaction.ch1058_drdl` and the Julia module
//! `AIInAction.Ch1058Drdl`. The shared fixtures in the tests below match the
//! Python/Julia suites, which keeps the three implementations at parity.
//!
//! Reference implementations of the deterministic building blocks of the discrete
//! and regularized deep learning (DRDL) method of Wang, Che, Jiang, and Zhao (2026),
//! "Leveraging Multiview Data Through Discrete and Regularized Deep Learning for
//! Dynamic Financial Risk Prediction", Information Systems Research 37(3), together
//! with the discrete-time survival metrics used to evaluate it.
//!
//! **Notation.** There are `m` prediction horizons `t_1 < ... < t_m`. Each company
//! `i` has an observed horizon index `A_i` in `{1, ..., m}` (`times[i]`, 1-based)
//! and an event flag `delta_i` (`events[i]`): `1` means the first risk event
//! happened in horizon `A_i`; `0` means no event was seen through `A_i`
//! (censored). The model outputs a cumulative risk matrix `F` with
//! `F[i][u] = P(T_i <= t_{u+1})` (0-based column `u`).
//!
//! * [`risk_accumulation`] and [`risk_accumulation_matrix`]: DRDL's risk
//!   accumulation function (paper Eq. 8), `p_u = 1 - exp(-sum_{j<=u} exp(o_j))`.
//!   It is the cumulative incidence of a discrete-time hazard model with a
//!   complementary log-log link, so it is monotone in `u` by construction.
//! * [`nearest_codes`] and [`first_occurrence_mask`]: the discrete multiview
//!   recoder's nearest-codebook assignment (Eq. 2) and the mask-based view fusion
//!   decoder's redundancy mask (Eq. 3).
//! * [`orthogonal_loss`]: the codebook orthogonality regularizer (Eq. 7).
//! * [`focal_sigma`] and [`focal_survival_loss`]: the focal survival loss
//!   (Eqs. 9-10), which penalizes only wrongly ranked comparable pairs.
//! * [`multiperiod_bce`]: the per-horizon binary cross-entropy on cumulative
//!   labels ([`cumulative_labels`]).
//! * [`project_conflicting`]: adaptive gradient balancing (Eq. 11).
//! * [`td_concordance`], [`td_auc`], [`brier_scores`]: Antolini time-dependent
//!   C-index, cumulative/dynamic AUC at a horizon, and per-horizon Brier scores.
//!
//! This is a std-only implementation. Matrices are `&[Vec<f64>]` (one `Vec` per
//! row), horizon indices are `usize` (1-based), and event flags are `u8` (0 or 1).
//! Errors are reported as `Err(String)` for every input on which the Python module
//! raises `ValueError`. A matrix must have at least one row and equal-length rows
//! (Python turns an empty list into a 1-D array and rejects it, and NumPy rejects
//! ragged lists). Inputs that Python must check at run time but Rust rules out by
//! type (non-integer horizon indices, negative codes) need no run-time check here.
//! The Python `risk_accumulation` accepts a vector or a matrix; Rust splits it into
//! [`risk_accumulation`] (one row) and [`risk_accumulation_matrix`] (row-wise).

use std::collections::HashSet;

// --------------------------------------------------------------------------
// Validation helpers
// --------------------------------------------------------------------------

/// Checks that `x` is a non-empty rectangular matrix of finite values and returns
/// its shape `(rows, cols)`.
fn check_matrix(name: &str, x: &[Vec<f64>]) -> Result<(usize, usize), String> {
    if x.is_empty() {
        return Err(format!("{} must be a 2-D array with at least one row", name));
    }
    let cols = x[0].len();
    for (i, row) in x.iter().enumerate() {
        if row.len() != cols {
            return Err(format!(
                "{} must be rectangular: row {} has length {}, expected {}",
                name,
                i,
                row.len(),
                cols
            ));
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err(format!("{} contains non-finite values", name));
        }
    }
    Ok((x.len(), cols))
}

/// Checks survival targets for `n` companies and `m` horizons: both slices have
/// length `n`, every horizon index lies in `1..=m`, and every event flag is 0 or 1.
fn check_survival_targets(times: &[usize], events: &[u8], n: usize, m: usize) -> Result<(), String> {
    if times.len() != n || events.len() != n {
        return Err(format!(
            "times and events must have length {}, got {} and {}",
            n,
            times.len(),
            events.len()
        ));
    }
    if let Some(&t) = times.iter().find(|&&t| !(1..=m).contains(&t)) {
        return Err(format!("times must lie in 1..{}, got {}", m, t));
    }
    if let Some(&e) = events.iter().find(|&&e| e > 1) {
        return Err(format!("events must be 0 or 1, got {}", e));
    }
    Ok(())
}

/// Dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |acc, (&x, &y)| acc + x * y)
}

// --------------------------------------------------------------------------
// Risk accumulation function (Eq. 8)
// --------------------------------------------------------------------------

/// Map one row of unconstrained outputs to monotone cumulative risk probabilities.
///
/// `p_u = 1 - exp(-sum_{j=1}^{u} exp(o_j))`, computed as `-expm1(-cum)` for
/// accuracy when the cumulative hazard is small. Errors if `outputs` is empty or
/// contains a non-finite value.
pub fn risk_accumulation(outputs: &[f64]) -> Result<Vec<f64>, String> {
    if outputs.is_empty() {
        return Err("outputs must be a non-empty 1-D or 2-D array".to_string());
    }
    if outputs.iter().any(|v| !v.is_finite()) {
        return Err("outputs contain non-finite values".to_string());
    }
    let mut cum = 0.0f64;
    Ok(outputs
        .iter()
        .map(|&o| {
            cum += o.exp();
            -(-cum).exp_m1()
        })
        .collect())
}

/// Row-wise [`risk_accumulation`] on an `n x m` matrix of outputs.
///
/// Errors if the matrix has no rows, ragged rows, zero columns, or a non-finite
/// value.
pub fn risk_accumulation_matrix(outputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let (_, m) = check_matrix("outputs", outputs)?;
    if m == 0 {
        return Err("outputs must be a non-empty 1-D or 2-D array".to_string());
    }
    outputs.iter().map(|row| risk_accumulation(row)).collect()
}

// --------------------------------------------------------------------------
// Discrete multiview recoder and mask-based fusion (Eqs. 2-4)
// --------------------------------------------------------------------------

/// Index of the nearest codebook row (squared L2) for every embedding row.
///
/// Ties go to the lowest index, matching NumPy's `argmin`. Errors if either
/// matrix is invalid or the two have different column counts.
pub fn nearest_codes(embeddings: &[Vec<f64>], codebook: &[Vec<f64>]) -> Result<Vec<usize>, String> {
    let (_, d) = check_matrix("embeddings", embeddings)?;
    let (_, dc) = check_matrix("codebook", codebook)?;
    if d != dc {
        return Err(format!(
            "embeddings and codebook must have the same dimension, got {} and {}",
            d, dc
        ));
    }
    Ok(embeddings
        .iter()
        .map(|e| {
            let mut best = 0usize;
            let mut best_d2 = f64::INFINITY;
            for (k, c) in codebook.iter().enumerate() {
                let d2 = e
                    .iter()
                    .zip(c)
                    .fold(0.0, |acc, (&x, &y)| acc + (x - y) * (x - y));
                // Strict `<` keeps the lowest index on ties; `k == 0` also covers
                // the case where every distance overflows to infinity.
                if k == 0 || d2 < best_d2 {
                    best = k;
                    best_d2 = d2;
                }
            }
            best
        })
        .collect())
}

/// Redundancy mask: `m_i = 0` if code `z_i` already appeared among
/// `z_1..z_{i-1}`, else `1`.
///
/// Python rejects negative codes at run time; the `usize` type rules them out here.
pub fn first_occurrence_mask(codes: &[usize]) -> Vec<u8> {
    let mut seen: HashSet<usize> = HashSet::with_capacity(codes.len());
    codes.iter().map(|&z| u8::from(seen.insert(z))).collect()
}

/// Frobenius norm of the strict upper triangle of the normalized codebook Gram
/// matrix.
///
/// For unit-normalized codes `c_k`, this is `sqrt(sum_{k<l} (c_k . c_l)^2)`, zero
/// exactly when the codes are mutually orthogonal. Errors if any codebook row is
/// zero.
pub fn orthogonal_loss(codebook: &[Vec<f64>]) -> Result<f64, String> {
    check_matrix("codebook", codebook)?;
    let mut units: Vec<Vec<f64>> = Vec::with_capacity(codebook.len());
    for (k, row) in codebook.iter().enumerate() {
        let norm = dot(row, row).sqrt();
        if norm == 0.0 {
            return Err(format!("codebook rows must be non-zero (row {} is zero)", k));
        }
        units.push(row.iter().map(|v| v / norm).collect());
    }
    let mut total = 0.0f64;
    for (k, uk) in units.iter().enumerate() {
        for ul in &units[k + 1..] {
            let g = dot(uk, ul);
            total += g * g;
        }
    }
    Ok(total.sqrt())
}

// --------------------------------------------------------------------------
// Focal survival loss (Eqs. 9-10)
// --------------------------------------------------------------------------

/// Focal weight `sigma(x) = 2 / (1 + exp(-x^gamma)) - 1` for `x >= 0`.
///
/// It is 0 at 0 and stays below 1. Errors unless `gamma` is a positive finite
/// number and `x` is finite and non-negative. (The Python function also accepts
/// arrays; map this one over a slice for the same effect.)
pub fn focal_sigma(x: f64, gamma: f64) -> Result<f64, String> {
    if !(gamma.is_finite() && gamma > 0.0) {
        return Err(format!("gamma must be a positive finite number, got {}", gamma));
    }
    if !(x.is_finite() && x >= 0.0) {
        return Err(format!("x must be finite and non-negative, got {}", x));
    }
    Ok(2.0 / (1.0 + (-(x.powf(gamma))).exp()) - 1.0)
}

/// Ordered pairs `(i, j)` (0-based) where company `i` is known to fail before `j`.
///
/// `events[i] == 1` and either `times[i] < times[j]`, or `times[i] == times[j]`
/// with `events[j] == 0` (`j` survived the horizon in which `i` failed). Pairs are
/// listed with `i` in the outer loop and `j` in the inner loop, as in Python. Like
/// the Python function this does not validate the targets beyond requiring equal
/// lengths; the losses and metrics that call it do.
pub fn comparable_pairs(times: &[usize], events: &[u8]) -> Result<Vec<(usize, usize)>, String> {
    if times.len() != events.len() {
        return Err(format!(
            "times and events must have the same length, got {} and {}",
            times.len(),
            events.len()
        ));
    }
    let mut pairs = Vec::new();
    for (i, (&ti, &ei)) in times.iter().zip(events).enumerate() {
        if ei != 1 {
            continue;
        }
        for (j, (&tj, &ej)) in times.iter().zip(events).enumerate() {
            if i != j && (ti < tj || (ti == tj && ej == 0)) {
                pairs.push((i, j));
            }
        }
    }
    Ok(pairs)
}

/// Mean focal penalty over comparable pairs (Eq. 9).
///
/// For a comparable pair `(i, j)` both companies are compared at the horizon
/// `A_i` where `i` failed. The pair is wrongly ranked when `F[j][A_i] > F[i][A_i]`;
/// its penalty is `sigma(max(0, F[j][A_i] - F[i][A_i]))`, so correctly ranked pairs
/// contribute exactly zero. Returns 0 if no pair is comparable (in which case,
/// as in Python, `gamma` is never checked). The Python default is `gamma = 1.0`.
pub fn focal_survival_loss(
    cum_probs: &[Vec<f64>],
    times: &[usize],
    events: &[u8],
    gamma: f64,
) -> Result<f64, String> {
    let (n, m) = check_matrix("cum_probs", cum_probs)?;
    check_survival_targets(times, events, n, m)?;
    let pairs = comparable_pairs(times, events)?;
    if pairs.is_empty() {
        return Ok(0.0);
    }
    let mut total = 0.0f64;
    for &(i, j) in &pairs {
        let u = times[i] - 1;
        total += focal_sigma(0.0f64.max(cum_probs[j][u] - cum_probs[i][u]), gamma)?;
    }
    Ok(total / pairs.len() as f64)
}

// --------------------------------------------------------------------------
// Multiperiod BCE
// --------------------------------------------------------------------------

/// Cumulative labels: `y[i][u] = 1` if company `i` had its first event at or
/// before horizon `u + 1`, else `0`. Returns a `times.len() x m` matrix.
///
/// Errors if `m == 0` or the targets are invalid for `m` horizons.
pub fn cumulative_labels(times: &[usize], events: &[u8], m: usize) -> Result<Vec<Vec<f64>>, String> {
    if m < 1 {
        return Err("m must be a positive integer".to_string());
    }
    check_survival_targets(times, events, times.len(), m)?;
    Ok(times
        .iter()
        .zip(events)
        .map(|(&t, &e)| {
            (1..=m)
                .map(|u| if e == 1 && t <= u { 1.0 } else { 0.0 })
                .collect()
        })
        .collect())
}

/// Mean over companies of the summed per-horizon binary cross-entropy.
///
/// Probabilities are clipped to `[eps, 1 - eps]` before taking logs (the Python
/// default is `eps = 1e-12`). Assumes censoring only at the end of the window
/// (every company is observed through `t_m` unless it fails earlier), which is the
/// setting of the paper's yearly, quarterly, and monthly windows. Errors if any
/// probability lies outside `[0, 1]` or the targets are invalid.
pub fn multiperiod_bce(
    cum_probs: &[Vec<f64>],
    times: &[usize],
    events: &[u8],
    eps: f64,
) -> Result<f64, String> {
    let (n, m) = check_matrix("cum_probs", cum_probs)?;
    if cum_probs.iter().flatten().any(|&v| !(0.0..=1.0).contains(&v)) {
        return Err("cum_probs must lie in [0, 1]".to_string());
    }
    check_survival_targets(times, events, n, m)?;
    let y = cumulative_labels(times, events, m)?;
    let mut total = 0.0f64;
    for (row, labels) in cum_probs.iter().zip(&y) {
        let mut s = 0.0f64;
        for (&f, &yv) in row.iter().zip(labels) {
            // Same order as NumPy's clip: maximum with eps, then minimum with 1 - eps.
            let p = f.max(eps).min(1.0 - eps);
            s += -(yv * p.ln() + (1.0 - yv) * (1.0 - p).ln());
        }
        total += s;
    }
    Ok(total / n as f64)
}

// --------------------------------------------------------------------------
// Adaptive gradient balancing (Eq. 11)
// --------------------------------------------------------------------------

/// Remove the component of `g_aux` that conflicts with `g_main`.
///
/// If `cos(g_main, g_aux) < 0` return `g_aux - (g_aux . g_main / ||g_main||^2) g_main`
/// (orthogonal to `g_main`); otherwise (including a zero `g_main`) return `g_aux`
/// unchanged. Errors on a length mismatch or a non-finite entry.
pub fn project_conflicting(g_main: &[f64], g_aux: &[f64]) -> Result<Vec<f64>, String> {
    if g_main.len() != g_aux.len() {
        return Err(format!(
            "gradients must have the same number of elements, got {} and {}",
            g_main.len(),
            g_aux.len()
        ));
    }
    if g_main.iter().chain(g_aux).any(|v| !v.is_finite()) {
        return Err("gradients must be finite".to_string());
    }
    let nn = dot(g_main, g_main);
    let d = dot(g_main, g_aux);
    if nn == 0.0 || d >= 0.0 {
        return Ok(g_aux.to_vec());
    }
    let scale = d / nn;
    Ok(g_aux
        .iter()
        .zip(g_main)
        .map(|(&b, &a)| b - scale * a)
        .collect())
}

// --------------------------------------------------------------------------
// Evaluation metrics
// --------------------------------------------------------------------------

/// Antolini's time-dependent concordance index.
///
/// Over comparable pairs `(i, j)`, the fraction with `F[i][A_i] > F[j][A_i]`,
/// counting ties as one half. Errors if no pair is comparable.
pub fn td_concordance(cum_probs: &[Vec<f64>], times: &[usize], events: &[u8]) -> Result<f64, String> {
    let (n, m) = check_matrix("cum_probs", cum_probs)?;
    check_survival_targets(times, events, n, m)?;
    let pairs = comparable_pairs(times, events)?;
    if pairs.is_empty() {
        return Err("no comparable pairs".to_string());
    }
    let mut score = 0.0f64;
    for &(i, j) in &pairs {
        let u = times[i] - 1;
        let (fi, fj) = (cum_probs[i][u], cum_probs[j][u]);
        if fi > fj {
            score += 1.0;
        } else if fi == fj {
            score += 0.5;
        }
    }
    Ok(score / pairs.len() as f64)
}

/// Cumulative/dynamic AUC at horizon `horizon` (1-based).
///
/// Cases had their event at or before `horizon` (`events[i] == 1` and
/// `times[i] <= horizon`); controls are known to be event-free through `horizon`
/// (`times[i] > horizon`, or `times[i] == horizon` with `events[i] == 0`). Ties
/// count one half. Errors if `horizon` is outside `1..=m` or there is no case or
/// no control.
pub fn td_auc(
    cum_probs: &[Vec<f64>],
    times: &[usize],
    events: &[u8],
    horizon: usize,
) -> Result<f64, String> {
    let (n, m) = check_matrix("cum_probs", cum_probs)?;
    check_survival_targets(times, events, n, m)?;
    if !(1..=m).contains(&horizon) {
        return Err(format!("horizon out of range: {} not in 1..{}", horizon, m));
    }
    let u = horizon - 1;
    let mut cases = Vec::new();
    let mut controls = Vec::new();
    for ((row, &t), &e) in cum_probs.iter().zip(times).zip(events) {
        if e == 1 && t <= horizon {
            cases.push(row[u]);
        }
        if t > horizon || (t == horizon && e == 0) {
            controls.push(row[u]);
        }
    }
    if cases.is_empty() || controls.is_empty() {
        return Err("need at least one case and one control".to_string());
    }
    let mut gt = 0usize;
    let mut eq = 0usize;
    for &c in &cases {
        for &k in &controls {
            if c > k {
                gt += 1;
            } else if c == k {
                eq += 1;
            }
        }
    }
    Ok((gt as f64 + 0.5 * eq as f64) / (cases.len() * controls.len()) as f64)
}

/// Per-horizon Brier score `mean_i (y[i][u] - F[i][u])^2` (end-of-window
/// censoring). Returns one score per column of `cum_probs`.
pub fn brier_scores(cum_probs: &[Vec<f64>], times: &[usize], events: &[u8]) -> Result<Vec<f64>, String> {
    let (n, m) = check_matrix("cum_probs", cum_probs)?;
    check_survival_targets(times, events, n, m)?;
    let y = cumulative_labels(times, events, m)?;
    let mut out = vec![0.0f64; m];
    for (row, labels) in cum_probs.iter().zip(&y) {
        for ((acc, &f), &yv) in out.iter_mut().zip(row).zip(labels) {
            *acc += (yv - f) * (yv - f);
        }
    }
    for v in &mut out {
        *v /= n as f64;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const OUTPUTS: [[f64; 3]; 4] = [
        [-1.0, 0.5, -0.3],
        [0.2, -2.0, 0.1],
        [-3.0, -1.0, 1.0],
        [0.0, 0.0, 0.0],
    ];
    const CUM_PROBS: [[f64; 3]; 4] = [
        [0.3077993724446536, 0.8668928334755382, 0.9365446603138965],
        [0.7051836792708419, 0.7425006392477038, 0.9147279853227372],
        [0.048568007099546597, 0.3414181774380723, 0.9565414790857123],
        [0.6321205588285577, 0.8646647167633873, 0.950212931632136],
    ];
    const TIMES: [usize; 4] = [1, 2, 3, 3];
    const EVENTS: [u8; 4] = [1, 1, 1, 0];
    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    const FSL_G1: f64 = 0.0696463795589742;
    const FSL_G05: f64 = 0.1259078886595625;
    const FSL_G2: f64 = 0.023133174339533064;
    const BCE: f64 = 2.376713158199545;
    const CINDEX: f64 = 0.5;
    const AUC: [f64; 3] = [1.0 / 3.0, 0.75, 1.0 / 3.0];
    const BRIER: [f64; 3] = [0.34459024562633167, 0.23705872071716394, 0.22902278877569854];

    const EMB: [[f64; 2]; 4] = [[0.1, 0.9], [1.0, 0.0], [0.6, 0.6], [-0.9, 0.1]];
    const CODEBOOK: [[f64; 2]; 3] = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
    const CODES: [usize; 4] = [1, 0, 0, 2];
    const MASK_CODES: [usize; 6] = [2, 0, 2, 1, 0, 3];
    const MASK: [u8; 6] = [1, 1, 0, 1, 0, 1];
    const ORTH_CODEBOOK: [[f64; 2]; 3] = [[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]];
    const ORTH: f64 = 1.0;
    const SIGMA_03: [(f64, f64); 3] = [
        (0.5, 0.2672140914204164),
        (1.0, 0.14888503362331806),
        (2.0, 0.04496964958360028),
    ];
    const PROJ_IN: ([f64; 3], [f64; 3]) = ([1.0, 0.0, 1.0], [-1.0, 2.0, 0.0]);
    const PROJ_OUT: [f64; 3] = [-0.5, 2.0, 0.5];

    const TOL: f64 = 1e-12;

    /// Ranked-perfectly fixture used by the loss and concordance property tests.
    const RANKED: [[f64; 3]; 3] = [[0.9, 0.95, 0.99], [0.1, 0.8, 0.9], [0.05, 0.1, 0.2]];

    /// Fixed-width array rows to a `Vec<Vec<f64>>` matrix.
    fn mat<const N: usize>(rows: &[[f64; N]]) -> Vec<Vec<f64>> {
        rows.iter().map(|r| r.to_vec()).collect()
    }

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

    fn assert_all_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert_close(*a, *e, tol);
        }
    }

    /// Deterministic 64-bit LCG (Knuth's MMIX constants) for the property tests.
    struct Lcg(u64);

    impl Lcg {
        /// Uniform draw in `[-1, 1)` from the top 53 bits of the state.
        fn next_signed(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
            2.0 * unit - 1.0
        }

        fn vector(&mut self, n: usize) -> Vec<f64> {
            (0..n).map(|_| self.next_signed()).collect()
        }
    }

    // ----------------------------------------------------------------------
    // Risk accumulation
    // ----------------------------------------------------------------------

    #[test]
    fn risk_accumulation_fixture_and_monotone() {
        let f = risk_accumulation_matrix(&mat(&OUTPUTS)).unwrap();
        assert_eq!(f.len(), CUM_PROBS.len());
        for (row, expected) in f.iter().zip(CUM_PROBS.iter()) {
            assert_all_close(row, expected, TOL);
            assert!(row.windows(2).all(|w| w[1] > w[0]));
            assert!(row.iter().all(|&p| p > 0.0 && p < 1.0));
        }
        assert_all_close(&risk_accumulation(&OUTPUTS[0]).unwrap(), &CUM_PROBS[0], TOL);
    }

    #[test]
    fn risk_accumulation_is_cloglog_hazard() {
        // F(t_u) = 1 - prod_j (1 - h_j) with h_j = 1 - exp(-exp(o_j)).
        let f = risk_accumulation_matrix(&mat(&OUTPUTS)).unwrap();
        for (row, o) in f.iter().zip(OUTPUTS.iter()) {
            let mut surv = 1.0f64;
            for (&p, &oj) in row.iter().zip(o) {
                let h = 1.0 - (-oj.exp()).exp();
                surv *= 1.0 - h;
                assert_close(p, 1.0 - surv, TOL);
            }
        }
    }

    #[test]
    fn risk_accumulation_validation() {
        assert!(risk_accumulation(&[]).is_err());
        assert!(risk_accumulation(&[1.0, f64::NAN]).is_err());
        assert!(risk_accumulation(&[1.0, f64::INFINITY]).is_err());
        assert!(risk_accumulation_matrix(&[]).is_err());
        assert!(risk_accumulation_matrix(&[vec![], vec![]]).is_err());
        assert!(risk_accumulation_matrix(&[vec![1.0, 2.0], vec![1.0]]).is_err());
        assert!(risk_accumulation_matrix(&[vec![1.0, f64::NAN]]).is_err());
    }

    // ----------------------------------------------------------------------
    // Recoder, mask, orthogonality
    // ----------------------------------------------------------------------

    #[test]
    fn nearest_codes_and_mask() {
        assert_eq!(nearest_codes(&mat(&EMB), &mat(&CODEBOOK)).unwrap(), CODES.to_vec());
        assert_eq!(first_occurrence_mask(&MASK_CODES), MASK.to_vec());
        assert_eq!(first_occurrence_mask(&[]), Vec::<u8>::new());
        assert!(nearest_codes(&[vec![1.0, 2.0, 3.0]], &mat(&CODEBOOK)).is_err());
        assert!(nearest_codes(&mat(&EMB), &[]).is_err());
        assert!(nearest_codes(&[vec![f64::NAN, 0.0]], &mat(&CODEBOOK)).is_err());
        // Python's negative-code error is ruled out by the usize code type.
    }

    #[test]
    fn nearest_codes_tie_goes_to_lowest_index() {
        let codebook = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];
        assert_eq!(nearest_codes(&[vec![0.0, 0.0]], &codebook).unwrap(), vec![0]);
    }

    #[test]
    fn orthogonal_loss_values() {
        assert_close(orthogonal_loss(&mat(&ORTH_CODEBOOK)).unwrap(), ORTH, TOL);
        let eye: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..4).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        assert_close(orthogonal_loss(&eye).unwrap(), 0.0, 1e-15);
        assert_approx(
            orthogonal_loss(&[vec![2.0, 0.0], vec![5.0, 0.0]]).unwrap(),
            1.0,
        );
        assert!(orthogonal_loss(&[vec![0.0, 0.0], vec![1.0, 0.0]]).is_err());
    }

    // ----------------------------------------------------------------------
    // Focal survival loss
    // ----------------------------------------------------------------------

    #[test]
    fn focal_sigma_values() {
        for (g, v) in SIGMA_03 {
            assert_close(focal_sigma(0.3, g).unwrap(), v, TOL);
        }
        assert_eq!(focal_sigma(0.0, 0.7).unwrap(), 0.0);
        assert_approx(focal_sigma(50.0, 1.0).unwrap(), 1.0);
        assert!(focal_sigma(-0.1, 1.0).is_err());
        assert!(focal_sigma(0.1, 0.0).is_err());
        assert!(focal_sigma(0.1, -1.0).is_err());
        assert!(focal_sigma(0.1, f64::INFINITY).is_err());
        assert!(focal_sigma(f64::NAN, 1.0).is_err());
    }

    #[test]
    fn comparable_pairs_fixture() {
        assert_eq!(comparable_pairs(&TIMES, &EVENTS).unwrap(), PAIRS.to_vec());
        assert!(comparable_pairs(&TIMES, &EVENTS[..3]).is_err());
    }

    #[test]
    fn focal_survival_loss_fixture() {
        let f = mat(&CUM_PROBS);
        assert_close(focal_survival_loss(&f, &TIMES, &EVENTS, 1.0).unwrap(), FSL_G1, TOL);
        assert_close(focal_survival_loss(&f, &TIMES, &EVENTS, 0.5).unwrap(), FSL_G05, TOL);
        assert_close(focal_survival_loss(&f, &TIMES, &EVENTS, 2.0).unwrap(), FSL_G2, TOL);
        assert!(focal_survival_loss(&f, &TIMES, &EVENTS, 0.0).is_err());
    }

    #[test]
    fn focal_survival_loss_zero_when_perfectly_ranked() {
        let f = mat(&RANKED);
        assert_eq!(focal_survival_loss(&f, &[1, 2, 3], &[1, 1, 0], 1.0).unwrap(), 0.0);
        assert_eq!(focal_survival_loss(&f, &[3, 3, 3], &[0, 0, 0], 1.0).unwrap(), 0.0);
    }

    // ----------------------------------------------------------------------
    // Multiperiod BCE
    // ----------------------------------------------------------------------

    #[test]
    fn multiperiod_bce_fixture() {
        assert_close(
            multiperiod_bce(&mat(&CUM_PROBS), &TIMES, &EVENTS, 1e-12).unwrap(),
            BCE,
            TOL,
        );
        let y = cumulative_labels(&TIMES, &EVENTS, 3).unwrap();
        let expected = vec![
            vec![1.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ];
        assert_eq!(y, expected);
    }

    #[test]
    fn survival_target_validation() {
        let f = mat(&CUM_PROBS);
        assert!(multiperiod_bce(&f, &[0, 1, 2, 3], &EVENTS, 1e-12).is_err());
        assert!(multiperiod_bce(&f, &[1, 2, 3, 4], &EVENTS, 1e-12).is_err());
        assert!(multiperiod_bce(&f, &TIMES, &[1, 2, 0, 0], 1e-12).is_err());
        assert!(multiperiod_bce(&f, &TIMES[..3], &EVENTS[..3], 1e-12).is_err());
        assert!(multiperiod_bce(&[vec![0.5, 1.5, 0.5]], &[1], &[1], 1e-12).is_err());
        assert!(multiperiod_bce(&[vec![0.5, f64::NAN, 0.5]], &[1], &[1], 1e-12).is_err());
        assert!(cumulative_labels(&TIMES, &EVENTS, 0).is_err());
        assert!(cumulative_labels(&TIMES, &EVENTS[..3], 3).is_err());
        assert!(focal_survival_loss(&f, &[1, 2, 3, 4], &EVENTS, 1.0).is_err());
        assert!(brier_scores(&f, &TIMES, &[1, 1, 1, 2]).is_err());
    }

    // ----------------------------------------------------------------------
    // Gradient balancing
    // ----------------------------------------------------------------------

    #[test]
    fn project_conflicting_values() {
        let out = project_conflicting(&PROJ_IN.0, &PROJ_IN.1).unwrap();
        assert_all_close(&out, &PROJ_OUT, TOL);
        assert_close(dot(&out, &PROJ_IN.0), 0.0, TOL);
        assert_all_close(
            &project_conflicting(&[1.0, 0.0], &[0.5, 1.0]).unwrap(),
            &[0.5, 1.0],
            TOL,
        );
        assert_all_close(
            &project_conflicting(&[0.0, 0.0], &[0.5, 1.0]).unwrap(),
            &[0.5, 1.0],
            TOL,
        );
        assert!(project_conflicting(&[1.0, 0.0], &[1.0]).is_err());
        assert!(project_conflicting(&[1.0, f64::NAN], &[1.0, 0.0]).is_err());
        assert!(project_conflicting(&[1.0, 0.0], &[f64::INFINITY, 0.0]).is_err());
    }

    #[test]
    fn combined_step_never_conflicts_with_main_loss() {
        let mut rng = Lcg(0);
        let mut projected = 0usize;
        for _ in 0..200 {
            let a = rng.vector(6);
            let b = rng.vector(6);
            if dot(&a, &b) < 0.0 {
                projected += 1;
            }
            let pb = project_conflicting(&a, &b).unwrap();
            let total: Vec<f64> = a.iter().zip(&pb).map(|(x, y)| x + y).collect();
            assert!(dot(&total, &a) >= dot(&a, &a) - 1e-9);
        }
        // The conflicting branch must actually be exercised.
        assert!(projected > 0);
    }

    // ----------------------------------------------------------------------
    // Metrics
    // ----------------------------------------------------------------------

    #[test]
    fn metrics_fixture() {
        let f = mat(&CUM_PROBS);
        assert_close(td_concordance(&f, &TIMES, &EVENTS).unwrap(), CINDEX, TOL);
        for (h, &v) in AUC.iter().enumerate() {
            assert_close(td_auc(&f, &TIMES, &EVENTS, h + 1).unwrap(), v, TOL);
        }
        assert_all_close(&brier_scores(&f, &TIMES, &EVENTS).unwrap(), &BRIER, TOL);
    }

    #[test]
    fn concordance_perfect_and_reversed() {
        let f = mat(&RANKED);
        assert_eq!(td_concordance(&f, &[1, 2, 3], &[1, 1, 0]).unwrap(), 1.0);
        let reversed: Vec<Vec<f64>> = f.iter().rev().cloned().collect();
        assert_eq!(td_concordance(&reversed, &[1, 2, 3], &[1, 1, 0]).unwrap(), 0.0);
        assert!(td_concordance(&f, &[3, 3, 3], &[0, 0, 0]).is_err());
    }

    #[test]
    fn td_auc_validation() {
        let f = mat(&CUM_PROBS);
        assert!(td_auc(&f, &TIMES, &EVENTS, 0).is_err());
        assert!(td_auc(&f, &TIMES, &EVENTS, 4).is_err());
        // No case: nobody has an event.
        assert!(td_auc(&f, &TIMES, &[0, 0, 0, 0], 2).is_err());
        // No control: everyone fails in the first horizon.
        assert!(td_auc(&f, &[1, 1, 1, 1], &[1, 1, 1, 1], 1).is_err());
        assert!(td_auc(&[vec![0.5, 0.5], vec![0.5]], &[1, 1], &[1, 0], 1).is_err());
    }
}
