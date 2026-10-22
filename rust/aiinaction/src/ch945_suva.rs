//! SUVA: auditing LLM decisions through state, understanding, value, and action (Rust).
//!
//! Mirrors the numeric core of the Python module `aiinaction.ch945_suva` and the
//! Julia module `AIInAction.Ch945Suva`. The shared fixtures in the tests below
//! match the Python/Julia suites, which keeps the three implementations at
//! parity. Every numeric function is ported one-to-one; the audit harness
//! (`State`, `Codebook`, `LexiconCoder`, `SimulatedLLM`, the analyses of the
//! paper, the DPO adapter, and `AuditSpec`) is Python only.
//!
//! The module follows Leng and Yuan (2026), *SUVA: A Probabilistic Framework for
//! Auditing LLMs with an Application to Social Preferences*, Information Systems
//! Research 37(3), 1805-1830. SUVA treats the prompt as the **state** `S`, codes
//! the model's chain of thought into **understanding** `U` and stated **values**
//! `V`, extracts the final **action** `A`, and estimates how `S`, `U`, and `V`
//! predict `A`. The numeric core covers:
//!
//! * **Game design.** [`payoff_grid`] enumerates dictator-game configurations and
//!   [`motive_features`] gives the sign regressors of Eq. (3).
//! * **Estimation.** [`ols_cluster`] (OLS with CR1 cluster-robust standard
//!   errors), [`demean_by_group`] (the within transform behind prompt fixed
//!   effects), [`diff_proportions`] (the reciprocity statistic), and
//!   [`normal_cdf`].
//! * **Measurement quality.** [`cohen_kappa`] (coder agreement) and [`auc_score`]
//!   (CoT predictability).
//! * **Audit decisions.** [`cohens_d`], [`tost_p_value`] (equivalence tests for
//!   "this cue must not matter" requirements), [`design_effect`] and
//!   [`samples_per_arm`] (power planning), and [`dpo_smoothed_margin`].
//! * **Synthetic respondents.** [`threshold_choice`] (a semiorder respondent),
//!   [`pairwise_transitive`] (weak-order and strict-part tests of three pairwise
//!   answers), and [`cell_agreement`] (correlation and calibration slope of
//!   synthetic on human cell means, optionally within group).
//! * **Reasoning trees.** [`collapse_repeats`], [`ReasoningTree`], and
//!   [`path_probability`] (Algorithm 1 of the paper).
//!
//! This is a std-only implementation. Errors are reported as `Err(String)` for
//! every input on which the Python module raises `ValueError` (or `KeyError`, for
//! [`ReasoningTree::node`]). Inputs that Python must check at run time but Rust
//! rules out by type (negative or fractional counts, an unknown game name) need
//! no run-time check here.

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

/// Two-sided 5 percent critical value of the standard normal, `z_{0.975}`.
pub const Z_975: f64 = 1.959963984540054;
/// Standard normal quantile for 80 percent power, `z_{0.80}`.
pub const Z_80: f64 = 0.8416212335729143;

/// Regressors of Eq. (3) in Leng and Yuan, in the order [`motive_features`]
/// returns them.
pub const MOTIVES: [&str; 4] = [
    "self_interest",
    "competition",
    "difference_aversion",
    "social_welfare",
];

/// The Python default payoff levels of [`payoff_grid`].
pub const DEFAULT_LEVELS: [i64; 4] = [0, 200, 400, 600];

// --------------------------------------------------------------------------
// Validation helpers
// --------------------------------------------------------------------------

/// Checks that `value` is finite.
fn check_finite(name: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("{} must be finite, got {}", name, value));
    }
    Ok(())
}

/// Checks that `value` is a finite probability in `[0, 1]`, with either end
/// optionally open.
fn check_prob(name: &str, value: f64, open_low: bool, open_high: bool) -> Result<(), String> {
    check_finite(name, value)?;
    if !(0.0..=1.0).contains(&value)
        || (open_low && value == 0.0)
        || (open_high && value == 1.0)
    {
        return Err(format!(
            "{} must lie in the unit interval, got {}",
            name, value
        ));
    }
    Ok(())
}

/// Sign of `x` as `+1`, `-1`, or `0` (also `0` for NaN, as in Python).
fn sgn(x: f64) -> i32 {
    (x > 0.0) as i32 - (x < 0.0) as i32
}

// --------------------------------------------------------------------------
// Normal CDF (Hart 1968 / West 2005), identical in Python, Julia, and Rust
// --------------------------------------------------------------------------

/// The Hart/West rational approximation without input validation.
fn normal_cdf_unchecked(x: f64) -> f64 {
    let xa = x.abs();
    let c = if xa > 37.0 {
        0.0
    } else {
        let e = (-xa * xa / 2.0).exp();
        if xa < 7.07106781186547 {
            let mut num = 3.52624965998911e-02 * xa + 0.700383064443688;
            num = num * xa + 6.37396220353165;
            num = num * xa + 33.912866078383;
            num = num * xa + 112.079291497871;
            num = num * xa + 221.213596169931;
            num = num * xa + 220.206867912376;
            let mut den = 8.83883476483184e-02 * xa + 1.75566716318264;
            den = den * xa + 16.064177579207;
            den = den * xa + 86.7807322029461;
            den = den * xa + 296.564248779674;
            den = den * xa + 637.333633378831;
            den = den * xa + 793.826512519948;
            den = den * xa + 440.413735824752;
            e * num / den
        } else {
            let mut b = xa + 0.65;
            b = xa + 4.0 / b;
            b = xa + 3.0 / b;
            b = xa + 2.0 / b;
            b = xa + 1.0 / b;
            e / b / 2.506628274631
        }
    };
    if x > 0.0 {
        1.0 - c
    } else {
        c
    }
}

/// Standard normal CDF, accurate to about 1e-14 everywhere.
///
/// Uses the double-precision rational approximation of Hart (1968) as given by
/// West (2005), with the same constants and the same operation order as the
/// Python and Julia ports, so p-values agree across languages without a
/// special-function dependency. Errors if `x` is not finite.
pub fn normal_cdf(x: f64) -> Result<f64, String> {
    check_finite("x", x)?;
    Ok(normal_cdf_unchecked(x))
}

// --------------------------------------------------------------------------
// Game design
// --------------------------------------------------------------------------

/// Which dictator game [`payoff_grid`] enumerates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Game {
    /// Every configuration except identical options. The Python default.
    #[default]
    Distributional,
    /// Only configurations in which one option is better for A and no better
    /// for B, so helping A is costly or free but never also self-serving.
    Reciprocity,
}

impl Game {
    /// The lowercase name used by the Python `game` string argument.
    pub fn as_str(&self) -> &'static str {
        match self {
            Game::Distributional => "distributional",
            Game::Reciprocity => "reciprocity",
        }
    }
}

/// Enumerate dictator-game payoff configurations `(a1, b1, a2, b2)`.
///
/// `a1, b1` are the payoffs to person A (the simulated match) and person B (the
/// audited model) under option B1; `a2, b2` under option B2. The order is
/// lexicographic in `(a1, b1, a2, b2)`, following the order of `levels`.
///
/// * [`Game::Distributional`]: every configuration except identical options
///   (`L^4 - L^2` of them for `L` levels, 240 for four levels).
/// * [`Game::Reciprocity`]: configurations with `a1 > a2 && b1 <= b2` or
///   `a1 < a2 && b1 >= b2` (Leng and Yuan, Section 3.2).
///
/// Pass [`DEFAULT_LEVELS`] for the Python default levels. Errors unless `levels`
/// holds at least two distinct values.
pub fn payoff_grid(levels: &[i64], game: Game) -> Result<Vec<(i64, i64, i64, i64)>, String> {
    let mut seen: Vec<i64> = levels.to_vec();
    seen.sort_unstable();
    seen.dedup();
    if levels.len() < 2 || seen.len() != levels.len() {
        return Err("levels must contain at least two distinct values".to_string());
    }
    let mut out = Vec::new();
    for &a1 in levels {
        for &b1 in levels {
            for &a2 in levels {
                for &b2 in levels {
                    let keep = match game {
                        Game::Distributional => (a1, b1) != (a2, b2),
                        Game::Reciprocity => (a1 > a2 && b1 <= b2) || (a1 < a2 && b1 >= b2),
                    };
                    if keep {
                        out.push((a1, b1, a2, b2));
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Sign regressors of Eq. (3): self interest, competition, difference aversion,
/// and social welfare (the order of [`MOTIVES`]).
///
/// Each is `+1` if the motive favors B1, `-1` if it favors B2, and `0` if it is
/// indifferent:
///
/// * self interest: `sgn(b1 - b2)`;
/// * competition: `sgn((b1 - a1) - (b2 - a2))`;
/// * difference aversion: `sgn(|b2 - a2| - |b1 - a1|)`;
/// * social welfare: `sgn((a1 + b1) - (a2 + b2))`.
///
/// Errors if any payoff is not finite.
pub fn motive_features(a1: f64, b1: f64, a2: f64, b2: f64) -> Result<(i32, i32, i32, i32), String> {
    check_finite("a1", a1)?;
    check_finite("b1", b1)?;
    check_finite("a2", a2)?;
    check_finite("b2", b2)?;
    Ok((
        sgn(b1 - b2),
        sgn((b1 - a1) - (b2 - a2)),
        sgn((b2 - a2).abs() - (b1 - a1).abs()),
        sgn((a1 + b1) - (a2 + b2)),
    ))
}

// --------------------------------------------------------------------------
// Estimation
// --------------------------------------------------------------------------

/// OLS fit with cluster-robust (CR1) standard errors.
#[derive(Clone, Debug, PartialEq)]
pub struct OlsResult {
    /// Coefficients, one per column of `x`.
    pub coef: Vec<f64>,
    /// CR1 cluster-robust standard errors.
    pub se: Vec<f64>,
    /// Number of observations `N`.
    pub n: usize,
    /// Number of clusters `G`.
    pub n_clusters: usize,
}

impl OlsResult {
    /// Wald statistics `coef / se` (NaN where `se` is not positive).
    pub fn z(&self) -> Vec<f64> {
        self.coef
            .iter()
            .zip(&self.se)
            .map(|(&c, &s)| if s > 0.0 { c / s } else { f64::NAN })
            .collect()
    }

    /// Two-sided p-values from the normal reference distribution (NaN where the
    /// Wald statistic is not finite).
    pub fn p_values(&self) -> Vec<f64> {
        self.z()
            .into_iter()
            .map(|z| {
                if z.is_finite() {
                    2.0 * normal_cdf_unchecked(-z.abs())
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Normal confidence intervals `coef -/+ z se`. Pass [`Z_975`] for the Python
    /// default (95 percent).
    pub fn ci(&self, z: f64) -> Vec<(f64, f64)> {
        self.coef
            .iter()
            .zip(&self.se)
            .map(|(&c, &s)| (c - z * s, c + z * s))
            .collect()
    }
}

/// Gauss-Jordan inverse with partial pivoting (same algorithm in all ports).
///
/// The pivot is the first row with the largest absolute value in the column,
/// and a pivot no larger than `1e-12 * max(max |m_ij|, 1)` is treated as
/// singular.
fn invert(m: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let k = m.len();
    let mut a: Vec<Vec<f64>> = m
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..k).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    let scale = m
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0f64, |acc, &v| if v.abs() > acc { v.abs() } else { acc });
    let tol = 1e-12 * scale.max(1.0);
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() <= tol {
            return Err(
                "design matrix is singular (collinear or constant regressors)".to_string(),
            );
        }
        a.swap(col, piv);
        let p = a[col][col];
        for v in a[col].iter_mut() {
            *v /= p;
        }
        let pivot_row = a[col].clone();
        for (r, row) in a.iter_mut().enumerate() {
            if r != col && row[col] != 0.0 {
                let f = row[col];
                for (vr, &vc) in row.iter_mut().zip(&pivot_row) {
                    *vr -= f * vc;
                }
            }
        }
    }
    Ok(a.into_iter().map(|row| row[k..].to_vec()).collect())
}

/// Least squares with CR1 cluster-robust standard errors.
///
/// `x` holds one row per observation (include a column of ones for an
/// intercept). The covariance is the sandwich
///
/// `V = c (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1`, with
/// `c = G/(G-1) * (N-1)/(N-K)`,
///
/// the default of Stata's `vce(cluster)` and statsmodels' `cov_type="cluster"`.
/// Clusters are visited in order of first appearance. Replicates of one prompt
/// share its sampling noise, so the prompt (or the payoff configuration) is the
/// natural cluster.
///
/// Errors on empty or mismatched inputs, ragged or empty rows, `N <= K`, a
/// non-finite `y`, a singular `X'X`, or fewer than two clusters.
pub fn ols_cluster<T: Eq + Hash + Clone>(
    x: &[Vec<f64>],
    y: &[f64],
    clusters: &[T],
) -> Result<OlsResult, String> {
    let n = y.len();
    if n == 0 || x.len() != n || clusters.len() != n {
        return Err("x, y, and clusters must be non-empty and of equal length".to_string());
    }
    let k = x[0].len();
    if k == 0 || x.iter().any(|row| row.len() != k) {
        return Err("every row of x must have the same positive length".to_string());
    }
    if n <= k {
        return Err("need more observations than regressors".to_string());
    }
    for &v in y {
        check_finite("y", v)?;
    }
    let mut xtx = vec![vec![0.0f64; k]; k];
    let mut xty = vec![0.0f64; k];
    for (row, &yi) in x.iter().zip(y) {
        for i in 0..k {
            xty[i] += row[i] * yi;
            for j in 0..k {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    let inv = invert(&xtx)?;
    let beta: Vec<f64> = (0..k)
        .map(|i| (0..k).fold(0.0, |acc, j| acc + inv[i][j] * xty[j]))
        .collect();
    let mut index: HashMap<T, usize> = HashMap::new();
    let mut scores: Vec<Vec<f64>> = Vec::new();
    for ((row, &yi), g) in x.iter().zip(y).zip(clusters) {
        let fit = beta
            .iter()
            .zip(row)
            .fold(0.0, |acc, (&b, &v)| acc + b * v);
        let u = yi - fit;
        let gi = *index.entry(g.clone()).or_insert_with(|| {
            scores.push(vec![0.0; k]);
            scores.len() - 1
        });
        let s = &mut scores[gi];
        for i in 0..k {
            s[i] += row[i] * u;
        }
    }
    let n_g = scores.len();
    if n_g < 2 {
        return Err("need at least two clusters".to_string());
    }
    let mut meat = vec![vec![0.0f64; k]; k];
    for s in &scores {
        for i in 0..k {
            for j in 0..k {
                meat[i][j] += s[i] * s[j];
            }
        }
    }
    let c = n_g as f64 / (n_g - 1) as f64 * (n - 1) as f64 / (n - k) as f64;
    let tmp: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| (0..k).fold(0.0, |acc, m| acc + inv[i][m] * meat[m][j]))
                .collect()
        })
        .collect();
    let se: Vec<f64> = (0..k)
        .map(|i| {
            let var = c * (0..k).fold(0.0, |acc, m| acc + tmp[i][m] * inv[m][i]);
            var.max(0.0).sqrt()
        })
        .collect();
    Ok(OlsResult {
        coef: beta,
        se,
        n,
        n_clusters: n_g,
    })
}

/// Subtract each group's mean: the within transform behind fixed effects.
///
/// By the Frisch-Waugh-Lovell theorem, regressing demeaned `y` on demeaned `W`
/// (no intercept) gives the same slopes as including one dummy per group, which
/// is how the Python `dependency_analysis` absorbs prompt fixed effects.
///
/// Errors on mismatched lengths or a non-finite value.
pub fn demean_by_group<T: Eq + Hash + Clone>(values: &[f64], groups: &[T]) -> Result<Vec<f64>, String> {
    if values.len() != groups.len() {
        return Err("values and groups must have equal length".to_string());
    }
    let mut stats: HashMap<T, (f64, u64)> = HashMap::new();
    for (&v, g) in values.iter().zip(groups) {
        check_finite("values", v)?;
        let e = stats.entry(g.clone()).or_insert((0.0, 0));
        e.0 += v;
        e.1 += 1;
    }
    Ok(values
        .iter()
        .zip(groups)
        .map(|(&v, g)| {
            let (total, count) = stats[g];
            v - total / count as f64
        })
        .collect())
}

/// Difference of proportions `k1/n1 - k0/n0` and its unpooled Wald SE.
///
/// With `k1` prosocial choices out of `n1` after the match helped and `k0` of
/// `n0` after it misbehaved, the difference is the reciprocity statistic of
/// Eq. (5). Errors unless `0 <= k <= n` and `n > 0` in both arms.
pub fn diff_proportions(k1: u64, n1: u64, k0: u64, n0: u64) -> Result<(f64, f64), String> {
    if n1 == 0 || n0 == 0 || k1 > n1 || k0 > n0 {
        return Err("need 0 <= k <= n and n > 0 in both arms".to_string());
    }
    let p1 = k1 as f64 / n1 as f64;
    let p0 = k0 as f64 / n0 as f64;
    let se = (p1 * (1.0 - p1) / n1 as f64 + p0 * (1.0 - p0) / n0 as f64).sqrt();
    Ok((p1 - p0, se))
}

// --------------------------------------------------------------------------
// Measurement quality
// --------------------------------------------------------------------------

/// Cohen's kappa between two coders: `(p_o - p_e) / (1 - p_e)`.
///
/// When `p_e == 1` (both coders use a single label) kappa is defined as `1` if
/// the coders agree everywhere and `0` otherwise. Errors on empty or mismatched
/// label sequences.
pub fn cohen_kappa<T: Eq + Hash + Clone>(a: &[T], b: &[T]) -> Result<f64, String> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return Err("label sequences must be non-empty and of equal length".to_string());
    }
    let agree = a.iter().zip(b).filter(|(x, y)| x == y).count();
    let po = agree as f64 / n as f64;
    let mut ca: HashMap<T, u64> = HashMap::new();
    let mut cb: HashMap<T, u64> = HashMap::new();
    for v in a {
        *ca.entry(v.clone()).or_insert(0) += 1;
    }
    for v in b {
        *cb.entry(v.clone()).or_insert(0) += 1;
    }
    let chance: u64 = ca
        .iter()
        .map(|(c, &na)| na * cb.get(c).copied().unwrap_or(0))
        .sum();
    let pe = chance as f64 / (n as u64 * n as u64) as f64;
    if pe == 1.0 {
        return Ok(if po == 1.0 { 1.0 } else { 0.0 });
    }
    Ok((po - pe) / (1.0 - pe))
}

/// Area under the ROC curve via the Mann-Whitney statistic with midranks.
///
/// Scores are ranked with a stable sort (ties keep input order) and tied scores
/// share their average rank. Errors on empty or mismatched inputs, a label other
/// than 0 or 1, or a single class.
pub fn auc_score(scores: &[f64], labels: &[u8]) -> Result<f64, String> {
    let n = scores.len();
    if n == 0 || labels.len() != n {
        return Err("scores and labels must be non-empty and of equal length".to_string());
    }
    if labels.iter().any(|&l| l > 1) {
        return Err("labels must be 0 or 1".to_string());
    }
    let n_pos = labels.iter().filter(|&&l| l == 1).count() as u64;
    let n_neg = n as u64 - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return Err("need both classes".to_string());
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        scores[i]
            .partial_cmp(&scores[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[order[j + 1]] == scores[order[i]] {
            j += 1;
        }
        let mid = (i + j) as f64 / 2.0 + 1.0;
        for &o in &order[i..=j] {
            ranks[o] = mid;
        }
        i = j + 1;
    }
    let r_pos = ranks
        .iter()
        .zip(labels)
        .filter(|(_, &l)| l == 1)
        .fold(0.0, |acc, (&r, _)| acc + r);
    Ok((r_pos - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos * n_neg) as f64)
}

// --------------------------------------------------------------------------
// Audit decisions
// --------------------------------------------------------------------------

/// Standardized mean difference `(mean(x) - mean(y)) / s_pooled`.
///
/// Errors if either sample has fewer than two values or the pooled standard
/// deviation is zero.
pub fn cohens_d(x: &[f64], y: &[f64]) -> Result<f64, String> {
    let (nx, ny) = (x.len(), y.len());
    if nx < 2 || ny < 2 {
        return Err("each sample needs at least two values".to_string());
    }
    let mx = x.iter().fold(0.0, |acc, &v| acc + v) / nx as f64;
    let my = y.iter().fold(0.0, |acc, &v| acc + v) / ny as f64;
    let vx = x.iter().fold(0.0, |acc, &v| acc + (v - mx).powi(2)) / (nx - 1) as f64;
    let vy = y.iter().fold(0.0, |acc, &v| acc + (v - my).powi(2)) / (ny - 1) as f64;
    let sp = (((nx - 1) as f64 * vx + (ny - 1) as f64 * vy) / (nx + ny - 2) as f64).sqrt();
    if sp == 0.0 {
        return Err("pooled standard deviation is zero".to_string());
    }
    Ok((mx - my) / sp)
}

/// Two one-sided tests (TOST) p-value for `|true effect| < margin`.
///
/// Equivalence is declared at level alpha when the returned p-value is below
/// alpha, i.e. when the `1 - 2 alpha` confidence interval lies inside
/// `(-margin, margin)`. Normal reference distribution. Errors unless `estimate`
/// is finite and `se` and `margin` are positive and finite.
pub fn tost_p_value(estimate: f64, se: f64, margin: f64) -> Result<f64, String> {
    check_finite("estimate", estimate)?;
    if se <= 0.0 || !se.is_finite() {
        return Err("se must be positive and finite".to_string());
    }
    if margin <= 0.0 || !margin.is_finite() {
        return Err("margin must be positive and finite".to_string());
    }
    let p_lower = 1.0 - normal_cdf((estimate + margin) / se)?;
    let p_upper = normal_cdf((estimate - margin) / se)?;
    Ok(p_lower.max(p_upper))
}

/// Kish design effect `1 + (m - 1) rho` for `m` replicates per prompt.
///
/// Errors unless `cluster_size >= 1` is finite and `icc` lies in `[0, 1]`.
pub fn design_effect(cluster_size: f64, icc: f64) -> Result<f64, String> {
    if cluster_size < 1.0 || !cluster_size.is_finite() {
        return Err("cluster_size must be at least 1".to_string());
    }
    check_prob("icc", icc, false, false)?;
    Ok(1.0 + (cluster_size - 1.0) * icc)
}

/// Responses per arm to detect `p1 - p0` with a two-sample test.
///
/// `n = deff (z_alpha + z_beta)^2 (p1 (1 - p1) + p0 (1 - p0)) / (p1 - p0)^2`,
/// rounded up (after subtracting `1e-12`, so an exact integer is not bumped by
/// rounding noise). `deff` inflates for replicates clustered within prompts. The
/// Python defaults are `z_alpha = Z_975`, `z_beta = Z_80`, `deff = 1.0`.
///
/// Errors if a probability is outside `[0, 1]`, `p1 == p0`, `deff < 1`,
/// `z_alpha <= 0`, `z_beta < 0`, or the result is not finite (Python raises on
/// `ceil` of NaN or infinity).
pub fn samples_per_arm(p1: f64, p0: f64, z_alpha: f64, z_beta: f64, deff: f64) -> Result<u64, String> {
    check_prob("p1", p1, false, false)?;
    check_prob("p0", p0, false, false)?;
    if p1 == p0 {
        return Err("p1 and p0 must differ".to_string());
    }
    if deff < 1.0 || !deff.is_finite() {
        return Err("deff must be at least 1".to_string());
    }
    if z_alpha <= 0.0 || z_beta < 0.0 {
        return Err("z_alpha must be positive and z_beta non-negative".to_string());
    }
    let var = p1 * (1.0 - p1) + p0 * (1.0 - p0);
    let n = deff * (z_alpha + z_beta).powi(2) * var / (p1 - p0).powi(2) - 1e-12;
    if !n.is_finite() {
        return Err("sample size is not finite".to_string());
    }
    Ok(n.ceil() as u64)
}

/// Optimal implicit-reward margin of label-smoothed DPO on one pair.
///
/// Minimizing `-(1 - eps) log sigma(beta m) - eps log sigma(-beta m)` over the
/// log-ratio margin `m` gives `m* = log((1 - eps) / eps) / beta`. Without
/// smoothing (`eps -> 0`) the optimum diverges, which is why hard-preference DPO
/// keeps pushing the policy toward determinism. The Python default is
/// `beta = 1.0`. Errors unless `0 < epsilon < 0.5` and `beta` is positive and
/// finite.
pub fn dpo_smoothed_margin(epsilon: f64, beta: f64) -> Result<f64, String> {
    check_prob("epsilon", epsilon, true, false)?;
    if epsilon >= 0.5 {
        return Err("epsilon must be below 0.5".to_string());
    }
    if beta <= 0.0 || !beta.is_finite() {
        return Err("beta must be positive".to_string());
    }
    Ok(((1.0 - epsilon) / epsilon).ln() / beta)
}

// --------------------------------------------------------------------------
// Synthetic respondents
// --------------------------------------------------------------------------

/// Pairwise answer of a respondent with a just-noticeable difference `delta`.
///
/// Returns `1` (first option) when `u_first - u_second > delta`, `-1` (second
/// option) when `u_second - u_first > delta`, and `0` (indifferent) otherwise.
/// With `delta > 0` this is Luce's (1956) semiorder: the strict part is
/// transitive but indifference need not be. Errors on a non-finite utility or a
/// negative or non-finite `delta`.
pub fn threshold_choice(u_first: f64, u_second: f64, delta: f64) -> Result<i8, String> {
    check_finite("u_first", u_first)?;
    check_finite("u_second", u_second)?;
    if delta < 0.0 || !delta.is_finite() {
        return Err("delta must be non-negative and finite".to_string());
    }
    let d = u_first - u_second;
    Ok(if d > delta {
        1
    } else if d < -delta {
        -1
    } else {
        0
    })
}

/// Whether three pairwise answers over items `x, y, z` are transitive.
///
/// Each answer is `1` (first item of the pair preferred), `-1` (second item
/// preferred), or `0` (indifferent). With `strict_only = false` (the Python
/// default) the weak preference "at least as good as" must be transitive, so the
/// answers must form a weak order: 13 of the 27 possible profiles pass, and 6 of
/// the 8 profiles without any indifference. With `strict_only = true` only the
/// strict part must be transitive, the semiorder reading under which
/// indifference may be intransitive: 19 of 27 profiles pass. Errors on an answer
/// outside `{-1, 0, 1}`.
pub fn pairwise_transitive(xy: i8, xz: i8, yz: i8, strict_only: bool) -> Result<bool, String> {
    if [xy, xz, yz].iter().any(|a| !(-1..=1).contains(a)) {
        return Err("answers must be 1, -1, or 0".to_string());
    }
    let mut rel = [[false; 3]; 3];
    for &((i, j), a) in &[((0, 1), xy), ((0, 2), xz), ((1, 2), yz)] {
        rel[i][j] = if strict_only { a == 1 } else { a >= 0 };
        rel[j][i] = if strict_only { a == -1 } else { a <= 0 };
    }
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                if i != j && j != k && i != k && rel[i][j] && rel[j][k] && !rel[i][k] {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// Pearson correlation and calibration slope of synthetic on human cell means.
///
/// The slope is the OLS coefficient of `llm` on `human`,
/// `Cov(llm, human) / Var(human)`: 1 when the synthetic sample reproduces the
/// size of each demographic gap, above 1 when it exaggerates them. The
/// correlation is unchanged by any positive rescaling of `llm`, so it cannot
/// detect exaggeration. Passing `Some(groups)` (for example the asset each cell
/// rates) demeans both vectors within group first, so that differences between
/// groups that every respondent agrees on do not count as agreement.
///
/// Errors on fewer than two cells, mismatched lengths, a non-finite value, or
/// zero variance on either side.
pub fn cell_agreement<T: Eq + Hash + Clone>(
    llm: &[f64],
    human: &[f64],
    groups: Option<&[T]>,
) -> Result<(f64, f64), String> {
    let n = llm.len();
    if n < 2 || human.len() != n {
        return Err("need at least two cells and equal lengths".to_string());
    }
    let (x, y) = match groups {
        Some(g) => (demean_by_group(human, g)?, demean_by_group(llm, g)?),
        None => (human.to_vec(), llm.to_vec()),
    };
    for &v in x.iter().chain(&y) {
        check_finite("cell means", v)?;
    }
    let mx = x.iter().sum::<f64>() / n as f64;
    let my = y.iter().sum::<f64>() / n as f64;
    let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
    let syy: f64 = y.iter().map(|v| (v - my).powi(2)).sum();
    let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
    if sxx == 0.0 || syy == 0.0 {
        return Err("cell means have zero variance".to_string());
    }
    Ok((sxy / (sxx * syy).sqrt(), sxy / sxx))
}

// --------------------------------------------------------------------------
// Reasoning trees (Algorithm 1)
// --------------------------------------------------------------------------

/// Merge consecutive duplicate codes, as Algorithm 1 does before building the
/// tree.
pub fn collapse_repeats<S: AsRef<str>>(labels: &[S]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for lab in labels {
        let lab = lab.as_ref();
        if out.last().map(|l| l.as_str()) != Some(lab) {
            out.push(lab.to_string());
        }
    }
    out
}

/// One node of a [`ReasoningTree`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeNode {
    /// The code (or, at a leaf, the action) this node stands for; `"root"` at
    /// the root.
    pub label: String,
    /// Number of responses that pass through this node.
    pub count: u64,
    /// How those responses end: action -> count.
    pub actions: BTreeMap<String, u64>,
    /// Child label -> index of the child in the tree's arena.
    pub children: BTreeMap<String, usize>,
}

impl TreeNode {
    fn new(label: &str) -> Self {
        TreeNode {
            label: label.to_string(),
            count: 0,
            actions: BTreeMap::new(),
            children: BTreeMap::new(),
        }
    }
}

/// Probabilistic decision tree over coded reasoning paths (Algorithm 1).
///
/// Each path is the collapsed sequence of codes followed by the action. A node
/// stores how many responses pass through it and how those responses end, so
/// `P(next | prefix, S)` is a ratio of child to parent counts and
/// `P(action | prefix, S)` is the node's action share.
///
/// Nodes live in an arena (`Vec<TreeNode>`); the root is at index
/// [`ReasoningTree::ROOT`] and children are referenced by index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningTree {
    nodes: Vec<TreeNode>,
}

impl Default for ReasoningTree {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningTree {
    /// Arena index of the root node.
    pub const ROOT: usize = 0;

    /// An empty tree holding only the root.
    pub fn new() -> Self {
        ReasoningTree {
            nodes: vec![TreeNode::new("root")],
        }
    }

    /// Build a tree from `(codes, action)` paths.
    pub fn from_paths<S: AsRef<str>, A: AsRef<str>>(paths: &[(Vec<S>, A)]) -> Self {
        let mut tree = Self::new();
        for (codes, action) in paths {
            tree.add(codes, action.as_ref());
        }
        tree
    }

    /// Add one response: its codes (collapsed first) followed by its action.
    pub fn add<S: AsRef<str>>(&mut self, codes: &[S], action: &str) {
        let mut path = collapse_repeats(codes);
        path.push(action.to_string());
        let mut idx = Self::ROOT;
        self.bump(idx, action);
        for lab in path {
            idx = match self.nodes[idx].children.get(&lab) {
                Some(&c) => c,
                None => {
                    let c = self.nodes.len();
                    self.nodes.push(TreeNode::new(&lab));
                    self.nodes[idx].children.insert(lab, c);
                    c
                }
            };
            self.bump(idx, action);
        }
    }

    fn bump(&mut self, idx: usize, action: &str) {
        let node = &mut self.nodes[idx];
        node.count += 1;
        *node.actions.entry(action.to_string()).or_insert(0) += 1;
    }

    /// The root node.
    pub fn root(&self) -> &TreeNode {
        &self.nodes[Self::ROOT]
    }

    /// All nodes, root first, in order of creation.
    pub fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    /// The child of node `idx` labeled `label`, if any.
    pub fn child(&self, idx: usize, label: &str) -> Option<&TreeNode> {
        self.nodes
            .get(idx)
            .and_then(|n| n.children.get(label))
            .map(|&c| &self.nodes[c])
    }

    /// The node reached by following `prefix` from the root.
    ///
    /// Errors (Python raises `KeyError`) if the prefix was never observed.
    pub fn node<S: AsRef<str>>(&self, prefix: &[S]) -> Result<&TreeNode, String> {
        let mut idx = Self::ROOT;
        for lab in prefix {
            match self.nodes[idx].children.get(lab.as_ref()) {
                Some(&c) => idx = c,
                None => {
                    let shown: Vec<&str> = prefix.iter().map(|s| s.as_ref()).collect();
                    return Err(format!("no path {:?}", shown));
                }
            }
        }
        Ok(&self.nodes[idx])
    }
}

/// Product of conditional transition probabilities along `path`.
///
/// `P(C1, ..., Cn, A | S) = P(C1|S) prod_j P(Cj | C1:j-1, S) P(A | C1:n, S)`,
/// estimated by child-to-parent count ratios. Returns `Ok(0.0)` for unseen
/// paths and errors on an empty tree.
pub fn path_probability<S: AsRef<str>>(tree: &ReasoningTree, path: &[S]) -> Result<f64, String> {
    let mut node = tree.root();
    if node.count == 0 {
        return Err("empty tree".to_string());
    }
    let mut prob = 1.0f64;
    for lab in path {
        let child = match node.children.get(lab.as_ref()) {
            Some(&c) => &tree.nodes[c],
            None => return Ok(0.0),
        };
        prob *= child.count as f64 / node.count as f64;
        node = child;
    }
    Ok(prob)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const CDF_POINTS: [(f64, f64); 6] = [
        (-1.96, 0.02499789514822045),
        (0.0, 0.5),
        (0.5, 0.691462461274013),
        (2.5, 0.9937903346742238),
        (-8.0, 6.220960520142783e-16),
        (10.0, 1.0),
    ];

    const GRID_COUNT: usize = 240;
    const GRID_FIRST: (i64, i64, i64, i64) = (0, 0, 0, 200);
    const GRID_LAST: (i64, i64, i64, i64) = (600, 600, 600, 400);
    const RECIP_COUNT: usize = 120;
    const RECIP_FIRST: (i64, i64, i64, i64) = (0, 0, 200, 0);
    const RECIP_LAST: (i64, i64, i64, i64) = (600, 600, 400, 600);
    const RECIP_01: [(i64, i64, i64, i64); 6] = [
        (0, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 1),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 1, 0, 1),
    ];

    #[allow(clippy::type_complexity)]
    const MOTIVE_CASES: [((f64, f64, f64, f64), (i32, i32, i32, i32)); 3] = [
        ((200.0, 400.0, 600.0, 200.0), (1, 1, 1, -1)),
        ((0.0, 600.0, 600.0, 0.0), (1, 1, 0, 0)),
        ((400.0, 400.0, 200.0, 600.0), (-1, -1, 1, 0)),
    ];

    const OLS_Y: [f64; 8] = [1.0, 2.5, 2.0, 4.5, 4.0, 6.5, 5.5, 8.0];
    const OLS_CLUSTERS: [usize; 8] = [0, 0, 1, 1, 2, 2, 3, 3];
    const OLS_COEF: [f64; 2] = [1.041666666666666, 0.916666666666667];
    const OLS_SE: [f64; 2] = [0.25027848001436165, 0.05787698374909847];

    const DEMEAN_VALUES: [f64; 5] = [1.0, 3.0, 2.0, 6.0, 10.0];
    const DEMEAN_GROUPS: [&str; 5] = ["a", "a", "b", "b", "b"];
    const DEMEAN_OUT: [f64; 5] = [-1.0, 1.0, -4.0, 0.0, 4.0];

    const DIFF_PROP_COUNTS: (u64, u64, u64, u64) = (30, 50, 18, 50);
    const DIFF_PROP_D: f64 = 0.24;
    const DIFF_PROP_SE: f64 = 0.09699484522385712;

    const KAPPA_A: [usize; 8] = [0, 0, 1, 1, 2, 2, 0, 1];
    const KAPPA_B: [usize; 8] = [0, 1, 1, 1, 2, 0, 0, 1];
    const KAPPA: f64 = 0.6097560975609756;

    const AUC_SCORES: [f64; 6] = [0.1, 0.4, 0.35, 0.8, 0.4, 0.2];
    const AUC_LABELS: [u8; 6] = [0, 0, 1, 1, 1, 0];
    const AUC: f64 = 0.8333333333333334;

    const D_X: [f64; 4] = [5.0, 6.0, 7.0, 8.0];
    const D_Y: [f64; 3] = [3.0, 4.0, 5.0];
    const COHENS_D: f64 = 2.1128856368212916;

    const TOST_PASS: ((f64, f64, f64), f64) = ((0.02, 0.04, 0.1), 0.02275013194817923);
    const TOST_FAIL: ((f64, f64, f64), f64) = ((0.12, 0.04, 0.1), 0.691462461274013);

    const DEFF_5_02: f64 = 1.8;
    const N_045_030: u64 = 160;
    const N_045_030_DEFF: u64 = 288;
    const N_05_04_POWER90: u64 = 515;
    const Z_90: f64 = 1.2815515655446004;

    const DPO_01: f64 = 2.1972245773362196;
    const DPO_025_BETA2: f64 = 0.5493061443340549;

    #[allow(clippy::type_complexity)]
    const THRESHOLD_CASES: [((f64, f64, f64), i8); 4] = [
        ((0.5, 0.2, 0.25), 1),
        ((0.2, 0.5, 0.25), -1),
        ((0.5, 0.4, 0.25), 0),
        ((0.3, 0.3, 0.0), 0),
    ];
    const TRANSITIVE_WEAK: (usize, usize, usize, usize) = (13, 27, 6, 8);
    const TRANSITIVE_STRICT_PART: usize = 19;
    const TRANSITIVE_CASES: [((i8, i8, i8), bool, bool); 5] = [
        ((1, 1, 1), true, true),
        ((1, -1, 1), false, false),
        ((0, 1, 0), false, true),
        ((0, 0, 1), false, true),
        ((0, 0, 0), true, true),
    ];
    const CELL_HUMAN: [f64; 4] = [1.0, 2.0, 5.0, 6.0];
    const CELL_LLM: [f64; 4] = [2.2, 1.8, 6.2, 5.8];
    const CELL_GROUPS: [&str; 4] = ["a", "a", "b", "b"];
    const CELL_POOLED: (f64, f64) = (0.9411946707617073, 0.9176470588235291);
    const CELL_WITHIN: (f64, f64) = (-1.0, -0.4);

    const COLLAPSE_IN: [&str; 6] = ["u", "u", "s", "s", "f", "s"];
    const COLLAPSE_OUT: [&str; 4] = ["u", "s", "f", "s"];

    const TREE_ROOT_COUNT: u64 = 4;
    const TREE_UNDERSTANDING_COUNT: u64 = 3;
    const TREE_UNDERSTANDING_ACTIONS: [(&str, u64); 2] = [("B1", 1), ("B2", 2)];
    const TREE_PATH_PROB: f64 = 0.25; // understanding -> self_interest -> B1 = 3/4 * 2/3 * 1/2

    fn ols_x() -> Vec<Vec<f64>> {
        (0..8).map(|i| vec![1.0, i as f64]).collect()
    }

    fn tree_paths() -> Vec<(Vec<&'static str>, &'static str)> {
        vec![
            (vec!["understanding", "self_interest", "self_interest"], "B1"),
            (vec!["understanding", "self_interest"], "B2"),
            (vec!["understanding", "fairness"], "B2"),
            (vec!["self_interest"], "B1"),
        ]
    }

    /// pytest `approx(expected, rel=rel, abs=abs)`: within the larger tolerance.
    fn assert_rel(actual: f64, expected: f64, rel: f64, abs: f64) {
        let tol = (rel * expected.abs()).max(abs);
        assert!(
            (actual - expected).abs() <= tol,
            "expected {} within rel {} / abs {} of {}",
            actual,
            rel,
            abs,
            expected
        );
    }

    /// pytest's default `approx`: relative 1e-6 or absolute 1e-12.
    fn assert_approx(actual: f64, expected: f64) {
        assert_rel(actual, expected, 1e-6, 1e-12);
    }

    // ----------------------------------------------------------------------
    // Normal CDF
    // ----------------------------------------------------------------------

    #[test]
    fn normal_cdf_fixture() {
        for (x, expected) in CDF_POINTS {
            assert_rel(normal_cdf(x).unwrap(), expected, 1e-12, 1e-15);
        }
    }

    #[test]
    fn normal_cdf_is_symmetric() {
        for i in -900..=900 {
            let x = i as f64 / 100.0;
            let s = normal_cdf(x).unwrap() + normal_cdf(-x).unwrap();
            assert!((s - 1.0).abs() <= 1e-15, "cdf({}) + cdf(-{}) = {}", x, x, s);
        }
    }

    #[test]
    fn normal_cdf_tails_and_validation() {
        assert_eq!(normal_cdf(40.0).unwrap(), 1.0);
        assert_eq!(normal_cdf(-40.0).unwrap(), 0.0);
        assert!(normal_cdf(f64::NAN).is_err());
        assert!(normal_cdf(f64::INFINITY).is_err());
    }

    // ----------------------------------------------------------------------
    // Game design
    // ----------------------------------------------------------------------

    #[test]
    fn payoff_grid_fixtures() {
        let g = payoff_grid(&DEFAULT_LEVELS, Game::Distributional).unwrap();
        assert_eq!(g.len(), GRID_COUNT);
        assert_eq!(g[0], GRID_FIRST);
        assert_eq!(*g.last().unwrap(), GRID_LAST);
        let r = payoff_grid(&DEFAULT_LEVELS, Game::Reciprocity).unwrap();
        assert_eq!(r.len(), RECIP_COUNT);
        assert_eq!(r[0], RECIP_FIRST);
        assert_eq!(*r.last().unwrap(), RECIP_LAST);
        assert_eq!(payoff_grid(&[0, 1], Game::Reciprocity).unwrap(), RECIP_01.to_vec());
        assert_eq!(payoff_grid(&[0, 1], Game::default()).unwrap().len(), 12);
        assert_eq!(Game::Reciprocity.as_str(), "reciprocity");
    }

    #[test]
    fn payoff_grid_reciprocity_has_no_selfserving_help() {
        for (a1, b1, a2, b2) in payoff_grid(&DEFAULT_LEVELS, Game::Reciprocity).unwrap() {
            // The option better for A is never strictly better for B.
            if a1 > a2 {
                assert!(b1 <= b2);
            } else {
                assert!(a1 < a2 && b1 >= b2);
            }
        }
    }

    #[test]
    fn payoff_grid_validation() {
        assert!(payoff_grid(&[1, 1], Game::Distributional).is_err());
        assert!(payoff_grid(&[1], Game::Distributional).is_err());
        assert!(payoff_grid(&[], Game::Reciprocity).is_err());
        // Python's unknown-game error is ruled out by the Game enum.
    }

    #[test]
    fn motive_features_fixture() {
        for ((a1, b1, a2, b2), expected) in MOTIVE_CASES {
            assert_eq!(motive_features(a1, b1, a2, b2).unwrap(), expected);
        }
        assert!(motive_features(f64::NAN, 0.0, 0.0, 0.0).is_err());
        assert!(motive_features(0.0, 0.0, 0.0, f64::INFINITY).is_err());
        assert_eq!(MOTIVES[3], "social_welfare");
    }

    // ----------------------------------------------------------------------
    // Estimation
    // ----------------------------------------------------------------------

    #[test]
    fn ols_cluster_fixture() {
        let r = ols_cluster(&ols_x(), &OLS_Y, &OLS_CLUSTERS).unwrap();
        for (a, e) in r.coef.iter().zip(OLS_COEF) {
            assert_rel(*a, e, 1e-12, 0.0);
        }
        for (a, e) in r.se.iter().zip(OLS_SE) {
            assert_rel(*a, e, 1e-10, 0.0);
        }
        assert_eq!(r.n, 8);
        assert_eq!(r.n_clusters, 4);
    }

    #[test]
    fn ols_result_z_p_and_ci() {
        let r = ols_cluster(&ols_x(), &OLS_Y, &OLS_CLUSTERS).unwrap();
        let z = r.z();
        let p = r.p_values();
        let ci = r.ci(Z_975);
        for i in 0..2 {
            assert_rel(z[i], r.coef[i] / r.se[i], 1e-15, 0.0);
            assert_rel(p[i], 2.0 * normal_cdf(-z[i].abs()).unwrap(), 1e-15, 0.0);
            assert_rel(ci[i].0, r.coef[i] - Z_975 * r.se[i], 1e-15, 0.0);
            assert_rel(ci[i].1, r.coef[i] + Z_975 * r.se[i], 1e-15, 0.0);
            assert!(ci[i].0 < r.coef[i] && r.coef[i] < ci[i].1);
        }
        let degenerate = OlsResult {
            coef: vec![1.0],
            se: vec![0.0],
            n: 3,
            n_clusters: 2,
        };
        assert!(degenerate.z()[0].is_nan());
        assert!(degenerate.p_values()[0].is_nan());
    }

    #[test]
    fn ols_cluster_recovers_known_coefficients() {
        // Deterministic design with bounded pseudo-noise.
        let n = 200;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for t in 0..n {
            let x1 = ((t * 37) % 101) as f64 / 50.0 - 1.0;
            let x2 = ((t * 53) % 97) as f64 / 48.0 - 1.0;
            let noise = 0.01 * (((t * 71) % 13) as f64 - 6.0) / 6.0;
            x.push(vec![1.0, x1, x2]);
            y.push(0.5 + 2.0 * x1 - 1.0 * x2 + noise);
        }
        let groups: Vec<usize> = (0..n).map(|t| t / 5).collect();
        let r = ols_cluster(&x, &y, &groups).unwrap();
        for (a, e) in r.coef.iter().zip([0.5, 2.0, -1.0]) {
            assert!((a - e).abs() < 0.03);
        }
        let (lo, hi) = r.ci(Z_975)[1];
        assert!(lo < r.coef[1] && r.coef[1] < hi);
        assert_eq!(r.n_clusters, 40);
    }

    #[test]
    fn ols_cluster_rejects_singular_and_bad_input() {
        let ones2 = vec![vec![1.0, 1.0]; 5];
        assert!(ols_cluster(&ones2, &[1.0, 2.0, 3.0, 4.0, 5.0], &[0, 1, 0, 1, 0]).is_err());
        let ones1 = vec![vec![1.0]; 3];
        assert!(ols_cluster(&ones1, &[1.0, 2.0, 3.0], &[0, 0, 0]).is_err());
        assert!(ols_cluster(&[vec![1.0]], &[1.0], &[0]).is_err());
        let empty: [usize; 0] = [];
        assert!(ols_cluster(&[], &[], &empty).is_err());
        assert!(ols_cluster(&ols_x(), &OLS_Y[..7], &OLS_CLUSTERS[..7]).is_err());
        let mut ragged = ols_x();
        ragged[3].push(2.0);
        assert!(ols_cluster(&ragged, &OLS_Y, &OLS_CLUSTERS).is_err());
        let mut bad_y = OLS_Y;
        bad_y[2] = f64::NAN;
        assert!(ols_cluster(&ols_x(), &bad_y, &OLS_CLUSTERS).is_err());
    }

    #[test]
    fn demean_by_group_fixture() {
        let out = demean_by_group(&DEMEAN_VALUES, &DEMEAN_GROUPS).unwrap();
        assert_eq!(out.len(), DEMEAN_OUT.len());
        for (a, e) in out.iter().zip(DEMEAN_OUT) {
            assert_approx(*a, e);
        }
        assert!(demean_by_group(&DEMEAN_VALUES, &DEMEAN_GROUPS[..4]).is_err());
        assert!(demean_by_group(&[1.0, f64::NAN], &["a", "a"]).is_err());
    }

    #[test]
    fn within_estimator_equals_dummy_regression() {
        // Frisch-Waugh-Lovell: demeaned slopes equal slopes with group dummies.
        let n_groups = 6;
        let per = 7;
        let mut g = Vec::new();
        let mut w = Vec::new();
        let mut y = Vec::new();
        for t in 0..n_groups * per {
            let gi = t / per;
            let wi = ((t * 5 + 3) % 4 < 2) as u8 as f64;
            let effect = (gi as f64 * 0.7).sin();
            let noise = 0.05 * (((t * 31) % 11) as f64 - 5.0);
            g.push(gi);
            w.push(wi);
            y.push(0.3 * wi + effect + noise);
        }
        let wd = demean_by_group(&w, &g).unwrap();
        let yd = demean_by_group(&y, &g).unwrap();
        let within_x: Vec<Vec<f64>> = wd.iter().map(|&v| vec![v]).collect();
        let within = ols_cluster(&within_x, &yd, &g).unwrap();
        let full_x: Vec<Vec<f64>> = (0..w.len())
            .map(|t| {
                let mut row = vec![w[t]];
                row.extend((0..n_groups).map(|k| if g[t] == k { 1.0 } else { 0.0 }));
                row
            })
            .collect();
        let full = ols_cluster(&full_x, &y, &g).unwrap();
        assert_rel(within.coef[0], full.coef[0], 1e-10, 0.0);
    }

    #[test]
    fn diff_proportions_fixture() {
        let (k1, n1, k0, n0) = DIFF_PROP_COUNTS;
        let (d, se) = diff_proportions(k1, n1, k0, n0).unwrap();
        assert_rel(d, DIFF_PROP_D, 1e-12, 0.0);
        assert_rel(se, DIFF_PROP_SE, 1e-12, 0.0);
        assert!(diff_proportions(5, 4, 1, 4).is_err());
        assert!(diff_proportions(1, 4, 5, 4).is_err());
        assert!(diff_proportions(0, 0, 1, 4).is_err());
        assert!(diff_proportions(1, 4, 0, 0).is_err());
        // Python's negative and fractional count errors are ruled out by u64.
    }

    // ----------------------------------------------------------------------
    // Measurement quality
    // ----------------------------------------------------------------------

    #[test]
    fn cohen_kappa_fixture() {
        assert_rel(cohen_kappa(&KAPPA_A, &KAPPA_B).unwrap(), KAPPA, 1e-12, 0.0);
        assert_eq!(cohen_kappa(&["x", "y"], &["x", "y"]).unwrap(), 1.0);
        assert_eq!(cohen_kappa(&["x", "x"], &["x", "x"]).unwrap(), 1.0);
        let empty: [usize; 0] = [];
        assert!(cohen_kappa(&empty, &empty).is_err());
        assert!(cohen_kappa(&KAPPA_A, &KAPPA_B[..7]).is_err());
    }

    #[test]
    fn auc_score_fixture_with_ties() {
        assert_rel(auc_score(&AUC_SCORES, &AUC_LABELS).unwrap(), AUC, 1e-12, 0.0);
        assert!(auc_score(&[0.1, 0.2], &[1, 1]).is_err());
        assert!(auc_score(&[0.1, 0.2], &[0, 2]).is_err());
        assert!(auc_score(&[], &[]).is_err());
        assert!(auc_score(&[0.1, 0.2], &[0]).is_err());
        // Perfect separation and perfect inversion.
        assert_eq!(auc_score(&[0.1, 0.2, 0.8, 0.9], &[0, 0, 1, 1]).unwrap(), 1.0);
        assert_eq!(auc_score(&[0.1, 0.2, 0.8, 0.9], &[1, 1, 0, 0]).unwrap(), 0.0);
        // All scores tied gives 0.5.
        assert_eq!(auc_score(&[0.3; 4], &[0, 1, 0, 1]).unwrap(), 0.5);
    }

    // ----------------------------------------------------------------------
    // Audit decisions
    // ----------------------------------------------------------------------

    #[test]
    fn cohens_d_fixture() {
        assert_rel(cohens_d(&D_X, &D_Y).unwrap(), COHENS_D, 1e-12, 0.0);
        assert!(cohens_d(&[1.0], &D_Y).is_err());
        assert!(cohens_d(&D_X, &[1.0]).is_err());
        assert!(cohens_d(&[2.0, 2.0], &[2.0, 2.0]).is_err());
    }

    #[test]
    fn tost_fixture() {
        let ((e, s, m), p) = TOST_PASS;
        assert_rel(tost_p_value(e, s, m).unwrap(), p, 1e-12, 0.0);
        assert!(p < 0.05);
        let ((e, s, m), p) = TOST_FAIL;
        assert_rel(tost_p_value(e, s, m).unwrap(), p, 1e-12, 0.0);
        assert!(p > 0.05);
        assert!(tost_p_value(0.0, 0.0, 0.1).is_err());
        assert!(tost_p_value(0.0, 0.1, 0.0).is_err());
        assert!(tost_p_value(0.0, f64::INFINITY, 0.1).is_err());
        assert!(tost_p_value(f64::NAN, 0.1, 0.1).is_err());
    }

    #[test]
    fn power_planning_fixture() {
        assert_approx(design_effect(5.0, 0.2).unwrap(), DEFF_5_02);
        assert_eq!(samples_per_arm(0.45, 0.30, Z_975, Z_80, 1.0).unwrap(), N_045_030);
        assert_eq!(
            samples_per_arm(0.45, 0.30, Z_975, Z_80, DEFF_5_02).unwrap(),
            N_045_030_DEFF
        );
        assert_eq!(
            samples_per_arm(0.5, 0.4, Z_975, Z_90, 1.0).unwrap(),
            N_05_04_POWER90
        );
    }

    #[test]
    fn power_planning_validation() {
        assert!(samples_per_arm(0.3, 0.3, Z_975, Z_80, 1.0).is_err());
        assert!(samples_per_arm(1.5, 0.3, Z_975, Z_80, 1.0).is_err());
        assert!(samples_per_arm(0.4, 0.3, Z_975, Z_80, 0.5).is_err());
        assert!(samples_per_arm(0.4, 0.3, 0.0, Z_80, 1.0).is_err());
        assert!(samples_per_arm(0.4, 0.3, Z_975, -0.1, 1.0).is_err());
        assert!(samples_per_arm(0.4, 0.3, f64::NAN, Z_80, 1.0).is_err());
        assert!(design_effect(0.5, 0.2).is_err());
        assert!(design_effect(f64::INFINITY, 0.2).is_err());
        assert!(design_effect(5.0, 1.5).is_err());
        assert!(design_effect(5.0, f64::NAN).is_err());
    }

    #[test]
    fn dpo_smoothed_margin_fixture() {
        assert_rel(dpo_smoothed_margin(0.1, 1.0).unwrap(), DPO_01, 1e-14, 0.0);
        assert_rel(
            dpo_smoothed_margin(0.25, 2.0).unwrap(),
            DPO_025_BETA2,
            1e-14,
            0.0,
        );
        assert!(dpo_smoothed_margin(0.0, 1.0).is_err());
        assert!(dpo_smoothed_margin(0.5, 1.0).is_err());
        assert!(dpo_smoothed_margin(0.1, 0.0).is_err());
        assert!(dpo_smoothed_margin(0.1, f64::INFINITY).is_err());
    }

    #[test]
    fn threshold_choice_fixture() {
        for ((u1, u2, d), e) in THRESHOLD_CASES {
            assert_eq!(threshold_choice(u1, u2, d).unwrap(), e);
        }
        assert!(threshold_choice(0.0, 1.0, -0.1).is_err());
        assert!(threshold_choice(f64::NAN, 1.0, 0.1).is_err());
    }

    #[test]
    fn pairwise_transitive_fixture() {
        for ((a, b, c), weak, strict) in TRANSITIVE_CASES {
            assert_eq!(pairwise_transitive(a, b, c, false).unwrap(), weak);
            assert_eq!(pairwise_transitive(a, b, c, true).unwrap(), strict);
        }
        let mut profiles = Vec::new();
        for a in -1..=1 {
            for b in -1..=1 {
                for c in -1..=1 {
                    profiles.push((a, b, c));
                }
            }
        }
        let no_ties = |p: &&(i8, i8, i8)| p.0 != 0 && p.1 != 0 && p.2 != 0;
        let count = |strict_only: bool, strict_profiles: bool| {
            profiles
                .iter()
                .filter(|p| !strict_profiles || no_ties(p))
                .filter(|p| pairwise_transitive(p.0, p.1, p.2, strict_only).unwrap())
                .count()
        };
        let (n_t, n, n_ts, n_s) = TRANSITIVE_WEAK;
        assert_eq!((count(false, false), profiles.len()), (n_t, n));
        assert_eq!(
            (count(false, true), profiles.iter().filter(no_ties).count()),
            (n_ts, n_s)
        );
        assert_eq!(count(true, false), TRANSITIVE_STRICT_PART);
        assert!(pairwise_transitive(2, 0, 0, false).is_err());
    }

    #[test]
    fn cell_agreement_fixture() {
        let (r, b) = cell_agreement::<&str>(&CELL_LLM, &CELL_HUMAN, None).unwrap();
        assert_rel(r, CELL_POOLED.0, 1e-12, 0.0);
        assert_rel(b, CELL_POOLED.1, 1e-12, 0.0);
        let (rw, bw) = cell_agreement(&CELL_LLM, &CELL_HUMAN, Some(&CELL_GROUPS[..])).unwrap();
        assert_rel(rw, CELL_WITHIN.0, 1e-12, 0.0);
        assert_rel(bw, CELL_WITHIN.1, 1e-12, 0.0);
        let scaled: Vec<f64> = CELL_HUMAN.iter().map(|h| 1.0 + 3.0 * h).collect();
        let (rs, bs) = cell_agreement::<&str>(&scaled, &CELL_HUMAN, None).unwrap();
        assert_approx(rs, 1.0);
        assert_approx(bs, 3.0);
        assert!(cell_agreement::<&str>(&[1.0, 1.0], &[1.0, 2.0], None).is_err());
        assert!(cell_agreement::<&str>(&[1.0], &[1.0], None).is_err());
    }

    #[test]
    fn dpo_margin_is_a_stationary_point() {
        // d/dm [-(1-e) log sig(m) - e log sig(-m)] = -(1-e)(1 - sig(m)) + e sig(m).
        for eps in [0.05, 0.1, 0.25, 0.4] {
            let m = dpo_smoothed_margin(eps, 1.0).unwrap();
            let sig = 1.0 / (1.0 + (-m).exp());
            let grad = -(1.0 - eps) * (1.0 - sig) + eps * sig;
            assert!(grad.abs() < 1e-14, "gradient {} at eps {}", grad, eps);
        }
    }

    // ----------------------------------------------------------------------
    // Reasoning trees
    // ----------------------------------------------------------------------

    #[test]
    fn collapse_repeats_fixture() {
        assert_eq!(collapse_repeats(&COLLAPSE_IN), COLLAPSE_OUT.to_vec());
        let empty: [&str; 0] = [];
        assert!(collapse_repeats(&empty).is_empty());
    }

    #[test]
    fn reasoning_tree_fixture() {
        let t = ReasoningTree::from_paths(&tree_paths());
        assert_eq!(t.root().count, TREE_ROOT_COUNT);
        let node = t.node(&["understanding"]).unwrap();
        assert_eq!(node.count, TREE_UNDERSTANDING_COUNT);
        let expected: BTreeMap<String, u64> = TREE_UNDERSTANDING_ACTIONS
            .iter()
            .map(|&(a, c)| (a.to_string(), c))
            .collect();
        assert_eq!(node.actions, expected);
        assert_approx(
            path_probability(&t, &["understanding", "self_interest", "B1"]).unwrap(),
            TREE_PATH_PROB,
        );
        assert_eq!(path_probability(&t, &["understanding", "altruism"]).unwrap(), 0.0);
        assert!(t.node(&["understanding", "altruism"]).is_err());
        assert_eq!(
            t.child(ReasoningTree::ROOT, "self_interest").unwrap().count,
            1
        );
    }

    #[test]
    fn complete_path_probabilities_sum_to_one() {
        let paths = tree_paths();
        let t = ReasoningTree::from_paths(&paths);
        let mut unique: Vec<Vec<String>> = paths
            .iter()
            .map(|(c, a)| {
                let mut p = collapse_repeats(c);
                p.push(a.to_string());
                p
            })
            .collect();
        unique.sort();
        unique.dedup();
        let total: f64 = unique
            .iter()
            .map(|p| path_probability(&t, p).unwrap())
            .sum();
        assert_approx(total, 1.0);
    }

    #[test]
    fn empty_tree_errors() {
        let t = ReasoningTree::default();
        assert!(path_probability(&t, &["understanding"]).is_err());
        let empty: [&str; 0] = [];
        assert_eq!(t.node(&empty).unwrap().label, "root");
        assert_eq!(t.nodes().len(), 1);
    }
}
