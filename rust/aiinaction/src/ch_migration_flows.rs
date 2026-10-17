//! Stock-flow accounting, scaling, balancing, loss and calibration pieces of Gaskin and
//! Abel (2026), "Deep learning four decades of human migration" (Rust).
//!
//! Mirrors the Python module `aiinaction.ch_migration_flows` and the Julia module
//! `AIInAction.ChMigrationFlows`. The shared fixtures in the tests below match the
//! Python/Julia suites. The synthetic world generator, the torch recurrent model and
//! the training loop are Python only; every numeric function is ported.
//!
//! * **Accounting.** [`stock_step`], [`od_flows`] and [`net_migration`] implement
//!   `S_ij(t+1) = S_ij + delta_ij B_j - mu_j S_ij + sum_k (T_ikj - T_ijk)` for the
//!   flow tensor `T[i, j, k]` (born in `i`, moving from `j` to `k`).
//! * **Scaling.** [`yj_sym`], [`yj_sym_inv`], [`yj_sym_deriv`], [`yj_profile_loglik`],
//!   [`fit_yj_lambda`] and [`elasticity_chain`].
//! * **Balancing.** [`ipf`] and [`demographic_balance`], then [`uncertainty_weights`].
//! * **Loss and calibration.** [`weighted_l2`], [`survival_fraction`],
//!   [`calibrate_offset`].
//! * **Baseline and metrics.** [`stock_difference_baseline`], [`gravity_log_rate`],
//!   [`log_rmse`], [`pearson`], [`coverage`], [`ensemble_mean_std`].
//!
//! Storage: an `n x n` matrix is a flat row-major `&[f64]` (`m[i * n + j]`); the
//! tensor `T[i, j, k]` is flat row-major `t[(i * n + j) * n + k]`. This is a std-only
//! implementation. Errors are reported as `Err(String)` for every input on which the
//! Python module raises `ValueError`.

/// Result of [`demographic_balance`]: balanced tables and relative adjustments.
#[derive(Debug, Clone, PartialEq)]
pub struct Balanced {
    pub bal1: Vec<f64>,
    pub bal2: Vec<f64>,
    pub rel1: Vec<f64>,
    pub rel2: Vec<f64>,
}

// --------------------------------------------------------------------------
// Validation helpers
// --------------------------------------------------------------------------

fn check_finite(name: &str, x: &[f64]) -> Result<(), String> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(format!("{} must be finite", name))
    }
}

fn check_lambda(lam: f64) -> Result<(), String> {
    if lam.is_finite() {
        Ok(())
    } else {
        Err(format!("lambda must be finite, got {}", lam))
    }
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// --------------------------------------------------------------------------
// Symmetrised Yeo-Johnson transform
// --------------------------------------------------------------------------

/// `sgn(x) ((|x|+1)^lam - 1)/lam`, or `sgn(x) log(|x|+1)` when `lam` is near zero.
pub fn yj_sym(x: f64, lam: f64) -> Result<f64, String> {
    check_lambda(lam)?;
    if !x.is_finite() {
        return Err("x must be finite".to_string());
    }
    let mag = x.abs();
    let body = if lam.abs() < 1e-12 {
        mag.ln_1p()
    } else {
        (lam * mag.ln_1p()).exp_m1() / lam
    };
    Ok(sign(x) * body)
}

/// Inverse of [`yj_sym`]; errors if `y` is outside the transform's range.
pub fn yj_sym_inv(y: f64, lam: f64) -> Result<f64, String> {
    check_lambda(lam)?;
    if !y.is_finite() {
        return Err("y must be finite".to_string());
    }
    let mag = y.abs();
    let body = if lam.abs() < 1e-12 {
        mag.exp_m1()
    } else {
        let inner = 1.0 + lam * mag;
        if inner <= 0.0 {
            return Err("y is outside the range of the transform for this lambda".to_string());
        }
        (inner.ln() / lam).exp_m1()
    };
    Ok(sign(y) * body)
}

/// `d phi / d x = (|x|+1)^(lam-1)`.
pub fn yj_sym_deriv(x: f64, lam: f64) -> Result<f64, String> {
    check_lambda(lam)?;
    if !x.is_finite() {
        return Err("x must be finite".to_string());
    }
    Ok(((lam - 1.0) * x.abs().ln_1p()).exp())
}

/// Profile Gaussian log-likelihood `-n/2 log var(phi(x)) + (lam - 1) sum log(|x|+1)`.
pub fn yj_profile_loglik(x: &[f64], lam: f64) -> Result<f64, String> {
    if x.len() < 2 {
        return Err("need at least two observations".to_string());
    }
    check_finite("x", x)?;
    let mut phi = Vec::with_capacity(x.len());
    for &v in x {
        phi.push(yj_sym(v, lam)?);
    }
    let m = mean(&phi);
    let var = phi.iter().map(|p| (p - m) * (p - m)).sum::<f64>() / phi.len() as f64;
    if var <= 0.0 {
        return Err("transformed sample has zero variance".to_string());
    }
    let jac = (lam - 1.0) * x.iter().map(|v| v.abs().ln_1p()).sum::<f64>();
    Ok(-0.5 * x.len() as f64 * var.ln() + jac)
}

/// Grid-search maximum likelihood lambda over `-1.00, -0.99, ..., 2.00`.
pub fn fit_yj_lambda(x: &[f64]) -> Result<f64, String> {
    let mut best: Option<f64> = None;
    let mut best_ll = f64::NEG_INFINITY;
    for k in -100..=200 {
        let lam = k as f64 / 100.0;
        if let Ok(ll) = yj_profile_loglik(x, lam) {
            if ll > best_ll {
                best = Some(lam);
                best_ll = ll;
            }
        }
    }
    best.ok_or_else(|| "no admissible lambda on the grid".to_string())
}

/// `eps = (du/dphi) phi'(x) |x|`: elasticity of `u = log T` with respect to raw `x`.
pub fn elasticity_chain(du_dphi: f64, x: f64, lam: f64) -> Result<f64, String> {
    if !du_dphi.is_finite() {
        return Err("du_dphi must be finite".to_string());
    }
    Ok(du_dphi * yj_sym_deriv(x, lam)? * x.abs())
}

// --------------------------------------------------------------------------
// Stock-flow accounting
// --------------------------------------------------------------------------

/// One annual step: `S_ij + delta_ij B_j - mu_j S_ij + sum_k T[i,k,j] - sum_k T[i,j,k]`.
pub fn stock_step(
    stock: &[f64],
    n: usize,
    births: &[f64],
    mortality: &[f64],
    flows: &[f64],
) -> Result<Vec<f64>, String> {
    if stock.len() != n * n || births.len() != n || mortality.len() != n || flows.len() != n * n * n
    {
        return Err("births, mortality, flows must match the stock dimension".to_string());
    }
    check_finite("stock", stock)?;
    check_finite("births", births)?;
    check_finite("mortality", mortality)?;
    check_finite("flows", flows)?;
    if births.iter().any(|&b| b < 0.0) || mortality.iter().any(|&m| !(0.0..=1.0).contains(&m)) {
        return Err("births must be non-negative and mortality in [0, 1]".to_string());
    }
    if flows.iter().any(|&f| f < 0.0) {
        return Err("flows must be non-negative".to_string());
    }
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut arrivals = 0.0;
            let mut departures = 0.0;
            for k in 0..n {
                arrivals += flows[(i * n + k) * n + j];
                departures += flows[(i * n + j) * n + k];
            }
            let birth = if i == j { births[j] } else { 0.0 };
            let s = stock[i * n + j];
            out[i * n + j] = s + birth - mortality[j] * s + arrivals - departures;
        }
    }
    Ok(out)
}

/// Total origin-destination flow `F[j, k] = sum_i T[i, j, k]` (flat `n x n`).
pub fn od_flows(flows: &[f64], n: usize) -> Result<Vec<f64>, String> {
    if flows.len() != n * n * n {
        return Err("flows must have n^3 entries".to_string());
    }
    check_finite("flows", flows)?;
    let mut f = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                f[j * n + k] += flows[(i * n + j) * n + k];
            }
        }
    }
    Ok(f)
}

/// `mu_j = sum_k F[k, j] - sum_k F[j, k]`; the entries sum to zero.
pub fn net_migration(f: &[f64], n: usize) -> Result<Vec<f64>, String> {
    if f.len() != n * n {
        return Err("F must be square".to_string());
    }
    check_finite("F", f)?;
    let mut out = vec![0.0; n];
    for j in 0..n {
        let inflow: f64 = (0..n).map(|k| f[k * n + j]).sum();
        let outflow: f64 = (0..n).map(|k| f[j * n + k]).sum();
        out[j] = inflow - outflow;
    }
    Ok(out)
}

/// Classical stock differencing: `max(S2 - S1, 0)/interval` with a zero diagonal.
pub fn stock_difference_baseline(
    s_prev: &[f64],
    s_next: &[f64],
    n: usize,
    interval: f64,
) -> Result<Vec<f64>, String> {
    if interval <= 0.0 || !interval.is_finite() {
        return Err("interval must be positive".to_string());
    }
    if s_prev.len() != n * n || s_next.len() != n * n {
        return Err("stock tables must have the same shape".to_string());
    }
    check_finite("s_prev", s_prev)?;
    check_finite("s_next", s_next)?;
    let mut out: Vec<f64> = s_prev
        .iter()
        .zip(s_next.iter())
        .map(|(a, b)| (b - a).max(0.0) / interval)
        .collect();
    for i in 0..n {
        out[i * n + i] = 0.0;
    }
    Ok(out)
}

// --------------------------------------------------------------------------
// Iterative proportional fitting and demographic balancing
// --------------------------------------------------------------------------

/// Iterative proportional fitting (Deming and Stephan, 1940). Returns the fitted
/// flat `r x c` table and the iteration count.
pub fn ipf(
    table: &[f64],
    row_targets: &[f64],
    col_targets: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<(Vec<f64>, usize), String> {
    let nr = row_targets.len();
    let nc = col_targets.len();
    if table.len() != nr * nc {
        return Err("target lengths must match the table shape".to_string());
    }
    check_finite("table", table)?;
    check_finite("row_targets", row_targets)?;
    check_finite("col_targets", col_targets)?;
    if table.iter().any(|&v| v < 0.0)
        || row_targets.iter().any(|&v| v < 0.0)
        || col_targets.iter().any(|&v| v < 0.0)
    {
        return Err("table and targets must be non-negative".to_string());
    }
    let (rsum, csum) = (row_targets.iter().sum::<f64>(), col_targets.iter().sum::<f64>());
    if (rsum - csum).abs() > 1e-9 * rsum.max(1.0) {
        return Err("row and column targets must have the same total".to_string());
    }
    let mut x = table.to_vec();
    let row_sum = |x: &[f64], i: usize| -> f64 { (0..nc).map(|j| x[i * nc + j]).sum() };
    let col_sum = |x: &[f64], j: usize| -> f64 { (0..nr).map(|i| x[i * nc + j]).sum() };
    for i in 0..nr {
        if row_sum(&x, i) == 0.0 && row_targets[i] > 0.0 {
            return Err("a target is positive but the table row or column is empty".to_string());
        }
    }
    for j in 0..nc {
        if col_sum(&x, j) == 0.0 && col_targets[j] > 0.0 {
            return Err("a target is positive but the table row or column is empty".to_string());
        }
    }
    let scale = rsum.max(1.0);
    for it in 1..=max_iter {
        for i in 0..nr {
            let rs = row_sum(&x, i);
            let f = if rs > 0.0 { row_targets[i] / rs } else { 0.0 };
            for j in 0..nc {
                x[i * nc + j] *= f;
            }
        }
        for j in 0..nc {
            let cs = col_sum(&x, j);
            let f = if cs > 0.0 { col_targets[j] / cs } else { 0.0 };
            for i in 0..nr {
                x[i * nc + j] *= f;
            }
        }
        let mut err = 0.0_f64;
        for i in 0..nr {
            err = err.max((row_sum(&x, i) - row_targets[i]).abs());
        }
        for j in 0..nc {
            err = err.max((col_sum(&x, j) - col_targets[j]).abs());
        }
        if err <= tol * scale {
            return Ok((x, it));
        }
    }
    Ok((x, max_iter))
}

/// Add births (diagonal) to `s1` and deaths to `s2`, fit both to the average of their
/// marginals with [`ipf`], then subtract births and deaths again.
pub fn demographic_balance(
    s1: &[f64],
    s2: &[f64],
    n: usize,
    births: &[f64],
    deaths: &[f64],
) -> Result<Balanced, String> {
    if s1.len() != n * n || s2.len() != n * n || deaths.len() != n * n || births.len() != n {
        return Err("shape mismatch".to_string());
    }
    let mut a1 = s1.to_vec();
    for i in 0..n {
        a1[i * n + i] += births[i];
    }
    let a2: Vec<f64> = s2.iter().zip(deaths.iter()).map(|(a, b)| a + b).collect();
    let rows: Vec<f64> = (0..n)
        .map(|i| {
            0.5 * ((0..n).map(|j| a1[i * n + j]).sum::<f64>()
                + (0..n).map(|j| a2[i * n + j]).sum::<f64>())
        })
        .collect();
    let cols: Vec<f64> = (0..n)
        .map(|j| {
            0.5 * ((0..n).map(|i| a1[i * n + j]).sum::<f64>()
                + (0..n).map(|i| a2[i * n + j]).sum::<f64>())
        })
        .collect();
    let (f1, _) = ipf(&a1, &rows, &cols, 1e-10, 10_000)?;
    let (f2, _) = ipf(&a2, &rows, &cols, 1e-10, 10_000)?;
    let mut bal1 = f1;
    for i in 0..n {
        bal1[i * n + i] -= births[i];
    }
    let bal2: Vec<f64> = f2.iter().zip(deaths.iter()).map(|(a, b)| a - b).collect();
    let eps = 1e-12;
    let rel1 = bal1
        .iter()
        .zip(s1.iter())
        .map(|(b, a)| (b - a).abs() / a.abs().max(eps))
        .collect();
    let rel2 = bal2
        .iter()
        .zip(s2.iter())
        .map(|(b, a)| (b - a).abs() / a.abs().max(eps))
        .collect();
    Ok(Balanced { bal1, bal2, rel1, rel2 })
}

// --------------------------------------------------------------------------
// Loss weights, loss, calibration
// --------------------------------------------------------------------------

/// Inverse relative uncertainty, normalised to mean 1, clamped to `[lo, hi]`.
pub fn uncertainty_weights(rel_unc: &[f64], lo: f64, hi: f64, floor: f64) -> Result<Vec<f64>, String> {
    check_finite("rel_unc", rel_unc)?;
    if rel_unc.iter().any(|&u| u < 0.0) {
        return Err("relative uncertainty must be non-negative".to_string());
    }
    if !(0.0 < lo && lo <= hi) {
        return Err("need 0 < lo <= hi".to_string());
    }
    let raw: Vec<f64> = rel_unc.iter().map(|&u| 1.0 / u.max(floor)).collect();
    let m = mean(&raw);
    Ok(raw.iter().map(|&r| (r / m).max(lo).min(hi)).collect())
}

/// `mean_k w_k (pred_k - target_k)^2`.
pub fn weighted_l2(pred: &[f64], target: &[f64], weights: &[f64]) -> Result<f64, String> {
    if pred.len() != target.len() || pred.len() != weights.len() {
        return Err("pred, target, weights must have the same shape".to_string());
    }
    check_finite("pred", pred)?;
    check_finite("target", target)?;
    check_finite("weights", weights)?;
    if weights.iter().any(|&w| w < 0.0) {
        return Err("weights must be non-negative".to_string());
    }
    let s: f64 = pred
        .iter()
        .zip(target.iter())
        .zip(weights.iter())
        .map(|((p, t), w)| w * (p - t) * (p - t))
        .sum();
    Ok(s / pred.len() as f64)
}

/// `s(t0) = 1`, `s(t) = prod (1 - mu(tau))`; the result has `len + 1` entries.
pub fn survival_fraction(mortality: &[f64]) -> Result<Vec<f64>, String> {
    check_finite("mortality", mortality)?;
    if mortality.iter().any(|&m| !(0.0..=1.0).contains(&m)) {
        return Err("mortality must lie in [0, 1]".to_string());
    }
    let mut out = Vec::with_capacity(mortality.len() + 1);
    out.push(1.0);
    for &m in mortality {
        let last = *out.last().unwrap();
        out.push(last * (1.0 - m));
    }
    Ok(out)
}

/// Closed-form initial-stock offset `sum s w (S - S_hat) / sum w s^2`, raised to the
/// smallest value keeping every shifted stock non-negative.
pub fn calibrate_offset(
    observed: &[f64],
    predicted: &[f64],
    weights: &[f64],
    survival: &[f64],
) -> Result<f64, String> {
    let l = observed.len();
    if predicted.len() != l || weights.len() != l || survival.len() != l {
        return Err("all series must have the same length".to_string());
    }
    check_finite("observed", observed)?;
    check_finite("predicted", predicted)?;
    check_finite("weights", weights)?;
    check_finite("survival", survival)?;
    if weights.iter().any(|&w| w < 0.0) || survival.iter().any(|&s| s <= 0.0) {
        return Err("weights must be non-negative and survival positive".to_string());
    }
    let den: f64 = weights.iter().zip(survival.iter()).map(|(w, s)| w * s * s).sum();
    if den <= 0.0 {
        return Err("weights have no mass".to_string());
    }
    let num: f64 = (0..l).map(|t| survival[t] * weights[t] * (observed[t] - predicted[t])).sum();
    let b = num / den;
    let lower = (0..l).map(|t| -predicted[t] / survival[t]).fold(f64::NEG_INFINITY, f64::max);
    Ok(b.max(lower))
}

// --------------------------------------------------------------------------
// Gravity rate and metrics
// --------------------------------------------------------------------------

/// `b0 + bg (log gdp_d - log gdp_o) + bd log dist + bl lang + bp log pop_d`.
pub fn gravity_log_rate(
    beta: &[f64],
    log_gdp_o: f64,
    log_gdp_d: f64,
    log_dist: f64,
    lang: f64,
    log_pop_d: f64,
) -> Result<f64, String> {
    if beta.len() != 5 {
        return Err("beta must have five entries".to_string());
    }
    Ok(beta[0]
        + beta[1] * (log_gdp_d - log_gdp_o)
        + beta[2] * log_dist
        + beta[3] * lang
        + beta[4] * log_pop_d)
}

/// Root mean squared error on the natural-log scale.
pub fn log_rmse(truth: &[f64], estimate: &[f64], floor: f64) -> Result<f64, String> {
    if truth.len() != estimate.len() || truth.is_empty() {
        return Err("shape mismatch".to_string());
    }
    let s: f64 = truth
        .iter()
        .zip(estimate.iter())
        .map(|(t, e)| {
            let d = e.max(floor).ln() - t.max(floor).ln();
            d * d
        })
        .sum();
    Ok((s / truth.len() as f64).sqrt())
}

/// Pearson correlation (`NaN` when either sample has zero variance).
pub fn pearson(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() != y.len() || x.len() < 2 {
        return Err("need two samples of equal length >= 2".to_string());
    }
    let (mx, my) = (mean(x), mean(y));
    let sxy: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let syy: f64 = y.iter().map(|b| (b - my) * (b - my)).sum();
    let den = (sxx * syy).sqrt();
    Ok(if den > 0.0 { sxy / den } else { f64::NAN })
}

/// Share of truths inside `mean +/- z std`.
pub fn coverage(truth: &[f64], m: &[f64], s: &[f64], z: f64) -> Result<f64, String> {
    if truth.len() != m.len() || truth.len() != s.len() || truth.is_empty() {
        return Err("shape mismatch".to_string());
    }
    let inside = (0..truth.len()).filter(|&i| (truth[i] - m[i]).abs() <= z * s[i]).count();
    Ok(inside as f64 / truth.len() as f64)
}

/// Mean and sample standard deviation (n - 1) of each column of `samples`
/// (`samples[d]` is draw `d`).
pub fn ensemble_mean_std(samples: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<f64>), String> {
    let d = samples.len();
    if d < 2 {
        return Err("need at least two samples".to_string());
    }
    let w = samples[0].len();
    if samples.iter().any(|r| r.len() != w) {
        return Err("samples must have equal length".to_string());
    }
    for r in samples {
        check_finite("samples", r)?;
    }
    let mut m = vec![0.0; w];
    let mut s = vec![0.0; w];
    for c in 0..w {
        let col: Vec<f64> = samples.iter().map(|r| r[c]).collect();
        m[c] = mean(&col);
        s[c] = (col.iter().map(|v| (v - m[c]) * (v - m[c])).sum::<f64>() / (d as f64 - 1.0)).sqrt();
    }
    Ok((m, s))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites. Matrices are flat
    // row-major; the tensor is flat (i * n + j) * n + k.
    const YJ_X: [f64; 5] = [-5.0, -0.3, 0.0, 2.0, 10.0];
    const YJ_07: [f64; 5] = [-3.57877726629599, -0.28800168756136063, 0.0, 1.6538132571065618, 6.225223813548735];
    const YJ_00: [f64; 5] = [-1.791759469228055, -0.262364264467491, 0.0, 1.0986122886681098, 2.3978952727983707];
    const YJ_M05: [f64; 5] = [-1.183503419072274, -0.24588396138594154, 0.0, 0.8452994616207485, 1.3969773108444727];
    const YJ_DERIV_07: [f64; 5] = [0.5841906810678655, 0.9243086009945788, 1.0, 0.7192230933248643, 0.4870596972258284];
    const YJ_SAMPLE: [f64; 13] = [
        0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 40.0, 100.0, -1.0, -3.0, -7.0,
    ];
    const YJ_LOGLIK_05: f64 = -33.338467549439706;
    const YJ_FIT: f64 = -0.01;
    const ELASTICITY: f64 = 0.8307692307692308;

    const STOCK: [f64; 9] = [100.0, 5.0, 2.0, 3.0, 80.0, 4.0, 1.0, 6.0, 60.0];
    const BIRTHS: [f64; 3] = [1.5, 1.0, 0.8];
    const MORT: [f64; 3] = [0.01, 0.02, 0.015];
    const STEP: [f64; 9] = [100.2, 4.9, 2.27, 2.37, 79.4, 4.54, 0.0900000000000003, 5.88, 60.800000000000004];
    const OD: [f64; 9] = [0.0, 2.4000000000000004, 3.6000000000000005, 1.8000000000000003, 0.0, 5.4, 2.4000000000000004, 4.800000000000001, 0.0];
    const NET: [f64; 3] = [-1.7999999999999998, 0.0, 1.799999999999999];

    const IPF_TABLE: [f64; 9] = [10.0, 20.0, 30.0, 15.0, 25.0, 5.0, 8.0, 2.0, 40.0];
    const IPF_ROWS: [f64; 3] = [60.0, 50.0, 50.0];
    const IPF_COLS: [f64; 3] = [45.0, 55.0, 60.0];
    const IPF: [f64; 9] = [13.584306505781386, 24.27835009115688, 22.137343404288874, 18.723560813157828, 27.886171700008326, 3.3902674980586913, 12.692132681060787, 2.835478208834802, 34.47238909765244];

    const BAL_S1: [f64; 9] = [50.0, 5.0, 2.0, 3.0, 40.0, 4.0, 1.0, 6.0, 30.0];
    const BAL_S2: [f64; 9] = [48.0, 7.0, 3.0, 4.0, 42.0, 3.0, 2.0, 5.0, 33.0];
    const BAL_BIRTHS: [f64; 3] = [2.0, 1.5, 1.0];
    const BAL1: [f64; 9] = [49.68615791114227, 5.279992903186189, 2.1138491735349096, 2.877097428817112, 40.78378334376581, 4.079119232039964, 0.9767446600406219, 6.226223753048012, 31.19703159442512];
    const BAL2: [f64; 9] = [48.32400928274534, 6.712813076986998, 2.8831776522759487, 4.118486084752095, 41.19277564057946, 2.9487382738222503, 2.0175046325025683, 4.80441128243354, 31.7780840739018];
    const REL1: [f64; 9] = [0.006276841777154658, 0.055998580637237796, 0.05692458676745482, 0.04096752372762932, 0.019594583594145298, 0.019779808009990907, 0.02325533995937812, 0.03770395884133526, 0.039901053147503994];
    const REL2: [f64; 9] = [0.006750193390527848, 0.04102670328757174, 0.03894078257468377, 0.029621521188023747, 0.019219627605250894, 0.017087242059249903, 0.008752316251284142, 0.03911774351329207, 0.03702775533630914];

    const WEIGHT_INPUT: [f64; 6] = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6];
    const WEIGHTS: [f64; 6] = [2.0, 1.5238095238095237, 0.7619047619047619, 0.5, 0.5, 0.5];
    const WL2: f64 = 1.21875;
    const SURVIVAL: [f64; 4] = [1.0, 0.99, 0.9702, 0.9410939999999999];
    const CALIBRATE: f64 = 1.22456821045488;
    const STOCK_NEXT: [f64; 9] = [110.0, 3.0, 9.0, 2.0, 90.0, 1.0, 4.0, 6.0, 55.0];
    const BASELINE: [f64; 9] = [0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0];
    const GRAVITY: f64 = -6.6899999999999995;
    const LOG_RMSE: f64 = 0.18451760994274952;
    const PEARSON: f64 = 0.9986517555689657;
    const ENSEMBLE_MEAN: [f64; 2] = [3.0, 5.0];
    const ENSEMBLE_STD: [f64; 2] = [1.8257418583505538, 2.943920288775949];

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{} vs {}", a, b);
    }

    fn close_all(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            close(*x, *y, tol);
        }
    }

    fn flows() -> Vec<f64> {
        let mut t = vec![0.0; 27];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    if j != k {
                        t[(i * 3 + j) * 3 + k] =
                            0.1 * (i as f64 + 1.0) * (j as f64 + 2.0) * (k as f64 + 1.0);
                    }
                }
            }
        }
        t
    }

    #[test]
    fn yeo_johnson_fixtures() {
        let f = |lam: f64| -> Vec<f64> { YJ_X.iter().map(|&x| yj_sym(x, lam).unwrap()).collect() };
        close_all(&f(0.7), &YJ_07, 1e-12);
        close_all(&f(0.0), &YJ_00, 1e-12);
        close_all(&f(-0.5), &YJ_M05, 1e-12);
        close_all(&f(1.0), &YJ_X, 1e-12);
        let d: Vec<f64> = YJ_X.iter().map(|&x| yj_sym_deriv(x, 0.7).unwrap()).collect();
        close_all(&d, &YJ_DERIV_07, 1e-12);
        for lam in [0.7, 0.0, 1.0, 1.6] {
            for &x in YJ_X.iter() {
                let y = yj_sym(x, lam).unwrap();
                close(yj_sym_inv(y, lam).unwrap(), x, 1e-10);
                let h = 1e-6;
                let num = (yj_sym(x + h, lam).unwrap() - yj_sym(x - h, lam).unwrap()) / (2.0 * h);
                close(num, yj_sym_deriv(x, lam).unwrap(), 1e-6);
            }
        }
        assert!(yj_sym_inv(5.0, -0.5).is_err());
        assert!(yj_sym(f64::NAN, 0.5).is_err());
        assert!(yj_profile_loglik(&[1.0], 0.5).is_err());
        close(yj_profile_loglik(&YJ_SAMPLE, 0.5).unwrap(), YJ_LOGLIK_05, 1e-10);
        close(fit_yj_lambda(&YJ_SAMPLE).unwrap(), YJ_FIT, 1e-12);
        close(elasticity_chain(0.9, 12.0, 0.0).unwrap(), ELASTICITY, 1e-12);
        close(elasticity_chain(1.0, 1e6, 0.0).unwrap(), 1e6 / (1e6 + 1.0), 1e-12);
    }

    #[test]
    fn accounting_fixtures() {
        let t = flows();
        let nxt = stock_step(&STOCK, 3, &BIRTHS, &MORT, &t).unwrap();
        close_all(&nxt, &STEP, 1e-12);
        let births: f64 = BIRTHS.iter().sum();
        let mut deaths = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                deaths += MORT[j] * STOCK[i * 3 + j];
            }
        }
        let change = nxt.iter().sum::<f64>() - STOCK.iter().sum::<f64>();
        close(change, births - deaths, 1e-12);
        let f = od_flows(&t, 3).unwrap();
        close_all(&f, &OD, 1e-12);
        let net = net_migration(&f, 3).unwrap();
        close_all(&net, &NET, 1e-12);
        assert!(net.iter().sum::<f64>().abs() < 1e-12);
        assert!(stock_step(&STOCK, 3, &BIRTHS, &[0.1, 0.2, 1.5], &t).is_err());
        let mut bad = t.clone();
        bad[1] = -1.0;
        assert!(stock_step(&STOCK, 3, &BIRTHS, &MORT, &bad).is_err());
        assert!(stock_step(&STOCK, 3, &BIRTHS[..2], &MORT, &t).is_err());
    }

    #[test]
    fn ipf_and_balancing_fixtures() {
        let (fit, iters) = ipf(&IPF_TABLE, &IPF_ROWS, &IPF_COLS, 1e-10, 10_000).unwrap();
        close_all(&fit, &IPF, 1e-8);
        assert!(iters > 1 && iters < 100);
        for i in 0..3 {
            close((0..3).map(|j| fit[i * 3 + j]).sum::<f64>(), IPF_ROWS[i], 1e-6);
            close((0..3).map(|j| fit[j * 3 + i]).sum::<f64>(), IPF_COLS[i], 1e-6);
        }
        assert!(ipf(&IPF_TABLE, &IPF_ROWS, &[45.0, 55.0, 70.0], 1e-10, 100).is_err());
        assert!(ipf(&[0.0, 0.0, 1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0], 1e-10, 100).is_err());
        let deaths: Vec<f64> = BAL_S2.iter().map(|v| 0.02 * v).collect();
        let b = demographic_balance(&BAL_S1, &BAL_S2, 3, &BAL_BIRTHS, &deaths).unwrap();
        close_all(&b.bal1, &BAL1, 1e-8);
        close_all(&b.bal2, &BAL2, 1e-8);
        close_all(&b.rel1, &REL1, 1e-8);
        close_all(&b.rel2, &REL2, 1e-8);
    }

    #[test]
    fn loss_and_calibration_fixtures() {
        let w = uncertainty_weights(&WEIGHT_INPUT, 0.5, 2.0, 1e-6).unwrap();
        close_all(&w, &WEIGHTS, 1e-12);
        assert!(w.iter().all(|&v| (0.5..=2.0).contains(&v)));
        assert!(w.windows(2).all(|p| p[0] >= p[1]));
        close(
            weighted_l2(&[1.0, 2.0, 3.0, 4.0], &[1.5, 1.5, 3.5, 2.0], &[1.0, 0.5, 2.0, 1.0]).unwrap(),
            WL2,
            1e-12,
        );
        assert!(weighted_l2(&[1.0], &[1.0, 2.0], &[1.0]).is_err());
        assert!(uncertainty_weights(&[-0.1, 0.2], 0.5, 2.0, 1e-6).is_err());
        let s = survival_fraction(&[0.01, 0.02, 0.03]).unwrap();
        close_all(&s, &SURVIVAL, 1e-12);
        close(
            calibrate_offset(&[10.0, 12.0, 14.0, 15.0], &[9.0, 10.5, 13.0, 13.2], &[1.0, 1.0, 2.0, 0.5], &s)
                .unwrap(),
            CALIBRATE,
            1e-12,
        );
        close(
            calibrate_offset(&[-1.0, -1.0, -1.0], &[2.0, 1.0, 3.0], &[1.0, 0.0, 0.0], &[1.0, 1.0, 1.0])
                .unwrap(),
            -1.0,
            1e-12,
        );
        let pred = [5.0, 6.0, 7.0, 8.0];
        let obs: Vec<f64> = pred.iter().zip(s.iter()).map(|(p, sv)| p + 2.5 * sv).collect();
        close(calibrate_offset(&obs, &pred, &[1.0; 4], &s).unwrap(), 2.5, 1e-12);
        assert!(calibrate_offset(&[1.0], &[1.0, 2.0], &[1.0], &[1.0]).is_err());
    }

    #[test]
    fn baseline_and_metric_fixtures() {
        close_all(&stock_difference_baseline(&STOCK, &STOCK_NEXT, 3, 5.0).unwrap(), &BASELINE, 1e-12);
        assert!(stock_difference_baseline(&STOCK, &STOCK_NEXT, 3, 0.0).is_err());
        close(
            gravity_log_rate(&[-8.7, 0.8, -0.7, 0.8, 0.5], 1.0, 2.0, -0.3, 1.0, 0.4).unwrap(),
            GRAVITY,
            1e-12,
        );
        close(
            log_rmse(&[1.0, 2.0, 3.0, 4.0], &[1.1, 1.8, 3.6, 3.0], 1e-9).unwrap(),
            LOG_RMSE,
            1e-12,
        );
        close(
            pearson(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.1, 3.9, 6.2, 7.8, 10.1]).unwrap(),
            PEARSON,
            1e-12,
        );
        assert!(pearson(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).unwrap().is_nan());
        close(
            coverage(&[1.0, 2.0, 3.0, 4.0], &[1.1, 2.5, 2.9, 3.0], &[0.2; 4], 1.96).unwrap(),
            0.5,
            1e-12,
        );
        let samples = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![4.0, 5.0], vec![5.0, 9.0]];
        let (m, s) = ensemble_mean_std(&samples).unwrap();
        close_all(&m, &ENSEMBLE_MEAN, 1e-12);
        close_all(&s, &ENSEMBLE_STD, 1e-12);
        assert!(ensemble_mean_std(&samples[..1]).is_err());
    }
}
