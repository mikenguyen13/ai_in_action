//! Statistical anomaly detection from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch145_anomaly` and the Julia module
//! `AIInAction.Ch145Anomaly`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! Four classical detectors: the univariate z-score, the multivariate
//! Mahalanobis distance, a Gaussian-kernel density estimate, and the Grubbs
//! single-outlier test. The chi-square and Student-t critical values used by the
//! calibrated detectors are computed by inverting their CDFs with a
//! self-contained incomplete-gamma / incomplete-beta routine so the three
//! languages produce identical numbers. std-only.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Special functions (self-contained) for calibrated thresholds.
// ---------------------------------------------------------------------------

/// Natural log of the gamma function (Lanczos approximation, g=7, n=9).
fn ln_gamma(x: f64) -> f64 {
    const COEF: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let g = 7.0;
    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = COEF[0];
    let t = x + g + 0.5;
    for (i, &c) in COEF.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized lower incomplete gamma P(s, x).
fn reg_lower_gamma(s: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < s + 1.0 {
        let mut ap = s;
        let mut total = 1.0 / s;
        let mut term = total;
        for _ in 0..1000 {
            ap += 1.0;
            term *= x / ap;
            total += term;
            if term.abs() < total.abs() * 1e-16 {
                break;
            }
        }
        total * (-x + s * x.ln() - ln_gamma(s)).exp()
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - s;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - s);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        let q = (-x + s * x.ln() - ln_gamma(s)).exp() * h;
        1.0 - q
    }
}

/// Continued fraction for the incomplete beta function (Lentz's method).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..1000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Regularized incomplete beta I_x(a, b).
fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - lbeta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// Inverse CDF of the chi-square distribution with `df` degrees of freedom.
pub fn chi2_ppf(p: f64, df: usize) -> Result<f64, String> {
    if !(0.0 < p && p < 1.0) {
        return Err(format!("p must be in (0, 1), got {}", p));
    }
    if df < 1 {
        return Err(format!("df must be >= 1, got {}", df));
    }
    let s = df as f64 / 2.0;
    let cdf = |q: f64| reg_lower_gamma(s, q / 2.0);
    let mut lo = 0.0;
    let mut hi = 1.0;
    while cdf(hi) < p {
        hi *= 2.0;
        if hi > 1e12 {
            break;
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Inverse CDF of the Student-t distribution with `df` degrees of freedom.
pub fn student_t_ppf(p: f64, df: usize) -> Result<f64, String> {
    if !(0.0 < p && p < 1.0) {
        return Err(format!("p must be in (0, 1), got {}", p));
    }
    if df < 1 {
        return Err(format!("df must be >= 1, got {}", df));
    }
    let dff = df as f64;
    let cdf = |t: f64| {
        let x = dff / (dff + t * t);
        let ib = reg_inc_beta(dff / 2.0, 0.5, x);
        if t > 0.0 {
            1.0 - 0.5 * ib
        } else {
            0.5 * ib
        }
    };
    let mut lo = -1.0;
    let mut hi = 1.0;
    while cdf(lo) > p {
        lo *= 2.0;
        if lo < -1e12 {
            break;
        }
    }
    while cdf(hi) < p {
        hi *= 2.0;
        if hi > 1e12 {
            break;
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

// ---------------------------------------------------------------------------
// Validation helpers.
// ---------------------------------------------------------------------------
fn check_vector(x: &[f64]) -> Result<(), String> {
    if x.len() < 2 {
        return Err(format!("need at least 2 observations, got {}", x.len()));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err("input contains non-finite values (nan or inf)".to_string());
    }
    Ok(())
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn std_ddof1(x: &[f64]) -> f64 {
    let m = mean(x);
    let n = x.len() as f64;
    let ss: f64 = x.iter().map(|v| (v - m) * (v - m)).sum();
    (ss / (n - 1.0)).sqrt()
}

// ---------------------------------------------------------------------------
// 1. z-score.
// ---------------------------------------------------------------------------

/// Standardized scores `z_i = (x_i - mean) / std` (ddof = 1).
pub fn zscores(x: &[f64]) -> Result<Vec<f64>, String> {
    check_vector(x)?;
    let m = mean(x);
    let s = std_ddof1(x);
    if s == 0.0 {
        return Err("standard deviation is zero; z-scores are undefined".to_string());
    }
    Ok(x.iter().map(|v| (v - m) / s).collect())
}

/// Boolean mask flagging points with `|z_i| > threshold`.
pub fn zscore_flags(x: &[f64], threshold: f64) -> Result<Vec<bool>, String> {
    if threshold <= 0.0 {
        return Err(format!("threshold must be positive, got {}", threshold));
    }
    let z = zscores(x)?;
    Ok(z.iter().map(|v| v.abs() > threshold).collect())
}

// ---------------------------------------------------------------------------
// 2. Mahalanobis distance.
// ---------------------------------------------------------------------------

/// Inverts a small square matrix by Gauss-Jordan elimination with partial
/// pivoting. Returns `(inverse, determinant)`.
fn invert(a_in: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, f64), String> {
    let n = a_in.len();
    let mut a: Vec<Vec<f64>> = a_in.to_vec();
    let mut inv = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        inv[i][i] = 1.0;
    }
    let mut det = 1.0f64;
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if best < 1e-300 {
            return Err("covariance matrix is singular; cannot invert".to_string());
        }
        if piv != col {
            a.swap(piv, col);
            inv.swap(piv, col);
            det = -det;
        }
        let pivot = a[col][col];
        det *= pivot;
        for j in 0..n {
            a[col][j] /= pivot;
            inv[col][j] /= pivot;
        }
        for r in 0..n {
            if r != col {
                let factor = a[r][col];
                for j in 0..n {
                    a[r][j] -= factor * a[col][j];
                    inv[r][j] -= factor * inv[col][j];
                }
            }
        }
    }
    Ok((inv, det))
}

/// Squared Mahalanobis distances of `points` to the Gaussian fit on `x`
/// (`n x d` rows). When `points` is empty the training rows are scored.
pub fn mahalanobis_sq(x: &[Vec<f64>], points: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    if x.len() < 2 {
        return Err(format!("need at least 2 samples, got {}", x.len()));
    }
    let d = x[0].len();
    if d < 1 {
        return Err("matrix must have at least one feature".to_string());
    }
    for row in x {
        if row.len() != d {
            return Err("all rows must have the same length".to_string());
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err("input contains non-finite values (nan or inf)".to_string());
        }
    }
    let n = x.len() as f64;
    let mut mu = vec![0.0f64; d];
    for row in x {
        for j in 0..d {
            mu[j] += row[j];
        }
    }
    for m in mu.iter_mut() {
        *m /= n;
    }
    // Sample covariance, ddof = 1.
    let mut cov = vec![vec![0.0f64; d]; d];
    for a in 0..d {
        for b in a..d {
            let mut acc = 0.0;
            for row in x {
                acc += (row[a] - mu[a]) * (row[b] - mu[b]);
            }
            let val = acc / (n - 1.0);
            cov[a][b] = val;
            cov[b][a] = val;
        }
    }
    let (inv, det) = invert(&cov)?;
    if !det.is_finite() || det.abs() < 1e-300 {
        return Err("covariance matrix is singular; cannot invert".to_string());
    }

    let pts: &[Vec<f64>] = if points.is_empty() { x } else { points };
    let mut out = Vec::with_capacity(pts.len());
    for row in pts {
        if row.len() != d {
            return Err(format!(
                "points have {} features but model was fit on {}",
                row.len(),
                d
            ));
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err("points contains non-finite values (nan or inf)".to_string());
        }
        let diff: Vec<f64> = (0..d).map(|j| row[j] - mu[j]).collect();
        let mut acc = 0.0;
        for a in 0..d {
            for b in 0..d {
                acc += diff[a] * inv[a][b] * diff[b];
            }
        }
        out.push(acc);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 3. Kernel density estimation.
// ---------------------------------------------------------------------------

/// Silverman's rule-of-thumb bandwidth `1.06 * std * n^{-1/5}` (ddof = 1).
pub fn silverman_bandwidth(x: &[f64]) -> Result<f64, String> {
    check_vector(x)?;
    let s = std_ddof1(x);
    if s == 0.0 {
        return Err("standard deviation is zero; bandwidth is undefined".to_string());
    }
    Ok(1.06 * s * (x.len() as f64).powf(-1.0 / 5.0))
}

/// Gaussian-kernel density estimate of `x` evaluated at `query`. When
/// `bandwidth` is `None` Silverman's rule is used.
pub fn gaussian_kde(x: &[f64], query: &[f64], bandwidth: Option<f64>) -> Result<Vec<f64>, String> {
    check_vector(x)?;
    if query.iter().any(|v| !v.is_finite()) {
        return Err("query contains non-finite values (nan or inf)".to_string());
    }
    let h = match bandwidth {
        Some(b) => b,
        None => silverman_bandwidth(x)?,
    };
    if h <= 0.0 {
        return Err(format!("bandwidth must be positive, got {}", h));
    }
    let n = x.len() as f64;
    let coef = 1.0 / (n * h * (2.0 * PI).sqrt());
    Ok(query
        .iter()
        .map(|&q| {
            let s: f64 = x
                .iter()
                .map(|&xi| {
                    let u = (q - xi) / h;
                    (-0.5 * u * u).exp()
                })
                .sum();
            coef * s
        })
        .collect())
}

/// Anomaly scores `-log p_hat(q)` from a Gaussian KDE (higher means rarer).
pub fn kde_scores(x: &[f64], query: &[f64], bandwidth: Option<f64>) -> Result<Vec<f64>, String> {
    let dens = gaussian_kde(x, query, bandwidth)?;
    let floor = 1e-300;
    Ok(dens.iter().map(|&d| -(d.max(floor)).ln()).collect())
}

// ---------------------------------------------------------------------------
// 4. Grubbs test.
// ---------------------------------------------------------------------------

/// Outcome of a two-sided Grubbs test for one outlier.
#[derive(Clone, Debug)]
pub struct GrubbsResult {
    pub statistic: f64,
    pub critical_value: f64,
    pub index: usize,
    pub is_outlier: bool,
    pub alpha: f64,
}

/// Two-sided Grubbs critical value for `n` observations at level `alpha`.
pub fn grubbs_critical_value(n: usize, alpha: f64) -> Result<f64, String> {
    if n < 3 {
        return Err(format!("Grubbs test needs at least 3 observations, got {}", n));
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(format!("alpha must be in (0, 1), got {}", alpha));
    }
    let t = student_t_ppf(1.0 - alpha / (2.0 * n as f64), n - 2)?;
    let t2 = t * t;
    Ok((n as f64 - 1.0) / (n as f64).sqrt() * (t2 / (n as f64 - 2.0 + t2)).sqrt())
}

/// Two-sided Grubbs test for a single outlier in approximately normal data.
pub fn grubbs_test(x: &[f64], alpha: f64) -> Result<GrubbsResult, String> {
    check_vector(x)?;
    let n = x.len();
    if n < 3 {
        return Err(format!("Grubbs test needs at least 3 observations, got {}", n));
    }
    let m = mean(x);
    let s = std_ddof1(x);
    if s == 0.0 {
        return Err("standard deviation is zero; Grubbs statistic is undefined".to_string());
    }
    let mut idx = 0usize;
    let mut best = 0.0f64;
    for (i, &v) in x.iter().enumerate() {
        let dev = (v - m).abs();
        if dev > best {
            best = dev;
            idx = i;
        }
    }
    let g = best / s;
    let crit = grubbs_critical_value(n, alpha)?;
    Ok(GrubbsResult {
        statistic: g,
        critical_value: crit,
        index: idx,
        is_outlier: g > crit,
        alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn x_z() -> [f64; 7] {
        [10.0, 12.0, 11.0, 13.0, 9.0, 11.5, 40.0]
    }

    fn x_mahal() -> Vec<Vec<f64>> {
        vec![
            vec![2.0, 1.0],
            vec![3.0, 2.0],
            vec![4.0, 2.5],
            vec![5.0, 4.0],
            vec![6.0, 5.0],
            vec![2.5, 6.0],
        ]
    }

    fn x_kde() -> [f64; 7] {
        [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]
    }

    fn x_grubbs() -> [f64; 6] {
        [1.0, 2.0, 1.5, 1.8, 2.2, 5.0]
    }

    #[test]
    fn zscores_match_fixture() {
        let z = zscores(&x_z()).unwrap();
        assert!((z[0] - (-0.4737231192137665)).abs() < TOL);
        assert!((z[6] - 2.251807155714753).abs() < TOL);
    }

    #[test]
    fn zscore_flags_match() {
        let f = zscore_flags(&x_z(), 2.0).unwrap();
        assert_eq!(f, vec![false, false, false, false, false, false, true]);
    }

    #[test]
    fn zscore_zero_std_errors() {
        let xs: [f64; 3] = [5.0, 5.0, 5.0];
        assert!(zscores(&xs).is_err());
    }

    #[test]
    fn mahalanobis_match_fixture() {
        let d2 = mahalanobis_sq(&x_mahal(), &[]).unwrap();
        assert!((d2[0] - 2.051080282894225).abs() < TOL);
        assert!((d2[5] - 4.120035750369162).abs() < TOL);
    }

    #[test]
    fn mahalanobis_external_point() {
        let pt: Vec<Vec<f64>> = vec![vec![10.0, 1.0]];
        let d2 = mahalanobis_sq(&x_mahal(), &pt).unwrap();
        assert!((d2[0] - 27.01727286857853).abs() < TOL);
    }

    #[test]
    fn mahalanobis_singular_errors() {
        let x: Vec<Vec<f64>> = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        assert!(mahalanobis_sq(&x, &[]).is_err());
    }

    #[test]
    fn kde_match_fixture() {
        let dens = gaussian_kde(&x_kde(), &[2.5, 10.0], Some(1.0)).unwrap();
        assert!((dens[0] - 0.22915412139731456).abs() < TOL);
        assert!((dens[1] - 0.056991755250522066).abs() < TOL);
    }

    #[test]
    fn kde_scores_match_fixture() {
        let s = kde_scores(&x_kde(), &[2.5, 10.0], Some(1.0)).unwrap();
        assert!((s[1] - 2.8648486663373274).abs() < TOL);
        assert!(s[1] > s[0]);
    }

    #[test]
    fn grubbs_match_fixture() {
        let r = grubbs_test(&x_grubbs(), 0.05).unwrap();
        assert!((r.statistic - 1.9489336934427666).abs() < TOL);
        assert!((r.critical_value - 1.887145117783933).abs() < TOL);
        assert_eq!(r.index, 5);
        assert!(r.is_outlier);
    }

    #[test]
    fn grubbs_no_outlier() {
        let xs: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = grubbs_test(&xs, 0.05).unwrap();
        assert!(!r.is_outlier);
    }

    #[test]
    fn grubbs_too_few_errors() {
        let xs: [f64; 2] = [1.0, 2.0];
        assert!(grubbs_test(&xs, 0.05).is_err());
    }

    #[test]
    fn chi2_ppf_match_fixture() {
        assert!((chi2_ppf(0.95, 2).unwrap() - 5.991464547107979).abs() < TOL);
        assert!((chi2_ppf(0.99, 3).unwrap() - 11.344866730144357).abs() < TOL);
    }

    #[test]
    fn student_t_ppf_match_fixture() {
        assert!((student_t_ppf(0.975, 10).unwrap() - 2.228138851986274).abs() < TOL);
    }
}
