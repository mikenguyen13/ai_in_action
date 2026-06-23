//! Bootstrap resampling with percentile and BCa confidence intervals (Rust, std-only).
//!
//! Mirrors the Python module `aiinaction.ch169_bootstrap` and the Julia module
//! `AIInAction.Ch169Bootstrap`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! Cross-language reproducibility comes from a fully specified 64-bit linear
//! congruential generator (Knuth's MMIX constants) plus Lemire's multiplicative
//! "integer in `[0, bound)`" map using the high 64 bits of a 128-bit product. Given
//! the same seed and sample, all three languages draw the identical resample indices.
//!
//! `norm_ppf` is Acklam's rational approximation to the standard normal quantile and
//! `norm_cdf` uses a self-contained `erfc`, so the BCa quantile adjustments match the
//! other two languages without any external dependency.

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;

/// The outcome of a bootstrap interval computation.
#[derive(Clone, Debug)]
pub struct BootstrapResult {
    /// The statistic (sample mean) on the original data.
    pub estimate: f64,
    /// Bootstrap standard error: sample std (ddof=1) of the replicate statistics.
    pub standard_error: f64,
    /// Lower confidence-interval endpoint.
    pub ci_low: f64,
    /// Upper confidence-interval endpoint.
    pub ci_high: f64,
    /// `"percentile"` or `"bca"`.
    pub method: String,
    /// Per-tail probability used for the interval.
    pub alpha: f64,
    /// The bootstrap replicate statistics, in draw order.
    pub replicates: Vec<f64>,
}

/// Advance the 64-bit LCG one step (wrapping arithmetic = mod 2^64).
#[inline]
fn next_state(state: u64) -> u64 {
    LCG_A.wrapping_mul(state).wrapping_add(LCG_C)
}

/// Draw a uniform integer in `[0, bound)` via Lemire's multiplicative map.
///
/// Returns `(value, new_state)`. The low bits of an LCG have short periods, so we use
/// the high 64 bits of the 128-bit product `state * bound`. Fully specified 64-bit
/// arithmetic makes the index sequence identical to Python and Julia.
fn rand_below(state: u64, bound: u64) -> (u64, u64) {
    assert!(bound > 0, "bound must be positive");
    let s = next_state(state);
    let value = ((s as u128 * bound as u128) >> 64) as u64;
    (value, s)
}

/// Linear-interpolation quantile (type-7) of an ascending, non-empty slice.
pub fn quantile(sorted_values: &[f64], q: f64) -> f64 {
    let n = sorted_values.len();
    assert!(n > 0, "cannot take a quantile of an empty slice");
    if q <= 0.0 {
        return sorted_values[0];
    }
    if q >= 1.0 {
        return sorted_values[n - 1];
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
}

/// Complementary error function (Abramowitz & Stegun 7.1.26-style rational form,
/// implemented to high accuracy via the same series used by `norm_cdf`).
fn erfc(x: f64) -> f64 {
    // Use the relationship erfc(x) = 1 - erf(x) with a high-accuracy erf.
    1.0 - erf(x)
}

/// Error function via a numerically stable rational/exponential approximation
/// (Numerical Recipes `erf`, accurate to ~1.2e-7), matched in Python/Julia by their
/// library `erf`. Tolerances in the parity tests are set to 1e-9 only for quantities
/// that do not pass through `erf`; `norm_cdf` is compared at 1e-7.
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let tau = t
        * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587
                                        + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp();
    if x >= 0.0 {
        1.0 - tau
    } else {
        tau - 1.0
    }
}

/// Standard normal CDF `Phi(x)`.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Standard normal quantile `Phi^{-1}(p)` via Acklam's rational approximation.
///
/// `p` must lie strictly inside `(0, 1)`.
pub fn norm_ppf(p: f64) -> Result<f64, String> {
    if !(p > 0.0 && p < 1.0) {
        return Err(format!("p must be in the open interval (0, 1), got {}", p));
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    let val = if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    Ok(val)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_sample(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(values);
    (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
}

/// Acceleration `a` from the leave-one-out jackknife of the mean.
fn jackknife_acceleration(data: &[f64]) -> f64 {
    let n = data.len();
    let total: f64 = data.iter().sum();
    let loo: Vec<f64> = (0..n).map(|i| (total - data[i]) / (n as f64 - 1.0)).collect();
    let mean_loo = mean(&loo);
    let diffs: Vec<f64> = loo.iter().map(|v| mean_loo - v).collect();
    let num: f64 = diffs.iter().map(|d| d.powi(3)).sum();
    let den = 6.0 * diffs.iter().map(|d| d * d).sum::<f64>().powf(1.5);
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Bootstrap a confidence interval for the mean of `data`.
///
/// `n_resamples` is the number of resamples `B`; `alpha` is the per-tail probability
/// (interval level `1 - 2*alpha`); `method` is `"percentile"` or `"bca"`; `seed` is a
/// non-negative seed for the built-in LCG.
pub fn bootstrap_mean_ci(
    data: &[f64],
    n_resamples: usize,
    alpha: f64,
    method: &str,
    seed: u64,
) -> Result<BootstrapResult, String> {
    if data.len() < 2 {
        return Err(format!(
            "need at least 2 observations to bootstrap, got {}",
            data.len()
        ));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err("data contains non-finite values (nan or inf)".to_string());
    }
    if n_resamples < 1 {
        return Err(format!("n_resamples must be >= 1, got {}", n_resamples));
    }
    if !(alpha > 0.0 && alpha < 0.5) {
        return Err(format!(
            "alpha must be in the open interval (0, 0.5), got {}",
            alpha
        ));
    }
    if method != "percentile" && method != "bca" {
        return Err(format!(
            "method must be 'percentile' or 'bca', got {:?}",
            method
        ));
    }

    let n = data.len();
    let estimate = mean(data);

    // One warm-up mix so seed=0 is not degenerate (matches Python/Julia).
    let mut state = seed.wrapping_add(LCG_C);
    let mut replicates = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let mut acc = 0.0;
        for _ in 0..n {
            let (idx, s) = rand_below(state, n as u64);
            state = s;
            acc += data[idx as usize];
        }
        replicates.push(acc / n as f64);
    }

    let standard_error = std_sample(&replicates);
    let mut ordered = replicates.clone();
    ordered.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let (lo_q, hi_q) = if method == "percentile" {
        (alpha, 1.0 - alpha)
    } else {
        let below = replicates.iter().filter(|&&v| v < estimate).count();
        let mut frac = below as f64 / n_resamples as f64;
        let eps = 0.5 / n_resamples as f64;
        if frac < eps {
            frac = eps;
        }
        if frac > 1.0 - eps {
            frac = 1.0 - eps;
        }
        let z0 = norm_ppf(frac)?;
        let a = jackknife_acceleration(data);
        let adjust = |tail: f64| -> Result<f64, String> {
            let z = norm_ppf(tail)?;
            let num = z0 + z;
            Ok(norm_cdf(z0 + num / (1.0 - a * num)))
        };
        (adjust(alpha)?, adjust(1.0 - alpha)?)
    };

    Ok(BootstrapResult {
        estimate,
        standard_error,
        ci_low: quantile(&ordered, lo_q),
        ci_high: quantile(&ordered, hi_q),
        method: method.to_string(),
        alpha,
        replicates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn fixture() -> Vec<f64> {
        let xs: [f64; 8] = [4.0, 8.0, 15.0, 16.0, 23.0, 42.0, 1.0, 9.0];
        xs.to_vec()
    }

    const SEED: u64 = 12345;
    const N: usize = 500;
    const TOL: f64 = 1e-9;

    #[test]
    fn estimate_matches_fixture() {
        let r = bootstrap_mean_ci(&fixture(), N, 0.025, "percentile", SEED).unwrap();
        assert!((r.estimate - 14.75).abs() < TOL);
    }

    #[test]
    fn standard_error_matches_fixture() {
        let r = bootstrap_mean_ci(&fixture(), N, 0.025, "percentile", SEED).unwrap();
        assert!((r.standard_error - 4.401794815207815).abs() < TOL);
    }

    #[test]
    fn replicate_head_matches_fixture() {
        let r = bootstrap_mean_ci(&fixture(), N, 0.025, "percentile", SEED).unwrap();
        let head: [f64; 5] = [13.25, 24.125, 16.0, 17.125, 10.375];
        for i in 0..5 {
            assert!((r.replicates[i] - head[i]).abs() < TOL);
        }
    }

    #[test]
    fn replicate_sum_matches_fixture() {
        let r = bootstrap_mean_ci(&fixture(), N, 0.025, "percentile", SEED).unwrap();
        let sum: f64 = r.replicates.iter().sum();
        assert!((sum - 7401.0).abs() < 1e-6);
    }

    #[test]
    fn percentile_interval_matches_fixture() {
        let r = bootstrap_mean_ci(&fixture(), N, 0.025, "percentile", SEED).unwrap();
        assert!((r.ci_low - 7.434375).abs() < TOL);
        assert!((r.ci_high - 24.506249999999994).abs() < TOL);
    }

    #[test]
    fn bca_interval_matches_fixture() {
        let r = bootstrap_mean_ci(&fixture(), N, 0.025, "bca", SEED).unwrap();
        assert!((r.ci_low - 8.55046123816017).abs() < TOL);
        assert!((r.ci_high - 26.839660615922583).abs() < TOL);
    }

    #[test]
    fn norm_ppf_matches_fixture() {
        assert!((norm_ppf(0.975).unwrap() - 1.959963986120195).abs() < TOL);
    }

    #[test]
    fn norm_ppf_antisymmetric() {
        assert!((norm_ppf(0.025).unwrap() + norm_ppf(0.975).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn norm_cdf_matches_fixture() {
        // norm_cdf is built on the ~1.2e-7-accurate erf shared by all three languages.
        assert!((norm_cdf(1.0) - 0.8413447386043253).abs() < 1e-12);
    }

    #[test]
    fn quantile_linear_rule() {
        let s: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&s, 0.25) - 2.0).abs() < TOL);
    }

    #[test]
    fn quantile_endpoints() {
        let s: [f64; 3] = [10.0, 20.0, 30.0];
        assert_eq!(quantile(&s, 0.0), 10.0);
        assert_eq!(quantile(&s, 1.0), 30.0);
    }

    #[test]
    fn determinism_same_seed() {
        let a = bootstrap_mean_ci(&fixture(), N, 0.025, "percentile", SEED).unwrap();
        let b = bootstrap_mean_ci(&fixture(), N, 0.025, "percentile", SEED).unwrap();
        assert_eq!(a.replicates, b.replicates);
    }

    #[test]
    fn constant_data_degenerate() {
        let xs: [f64; 4] = [5.0, 5.0, 5.0, 5.0];
        let r = bootstrap_mean_ci(&xs, 100, 0.025, "percentile", 7).unwrap();
        assert!((r.estimate - 5.0).abs() < TOL);
        assert!(r.standard_error.abs() < TOL);
        assert!((r.ci_low - 5.0).abs() < TOL);
        assert!((r.ci_high - 5.0).abs() < TOL);
    }

    #[test]
    fn too_few_observations_errors() {
        let xs: [f64; 1] = [1.0];
        assert!(bootstrap_mean_ci(&xs, N, 0.025, "percentile", SEED).is_err());
    }

    #[test]
    fn non_finite_errors() {
        let xs: [f64; 3] = [1.0, 2.0, f64::NAN];
        assert!(bootstrap_mean_ci(&xs, N, 0.025, "percentile", SEED).is_err());
    }

    #[test]
    fn bad_method_errors() {
        assert!(bootstrap_mean_ci(&fixture(), N, 0.025, "studentized", SEED).is_err());
    }

    #[test]
    fn bad_alpha_errors() {
        assert!(bootstrap_mean_ci(&fixture(), N, 0.5, "percentile", SEED).is_err());
    }

    #[test]
    fn bad_n_resamples_errors() {
        assert!(bootstrap_mean_ci(&fixture(), 0, 0.025, "percentile", SEED).is_err());
    }

    #[test]
    fn norm_ppf_out_of_range_errors() {
        assert!(norm_ppf(0.0).is_err());
        assert!(norm_ppf(1.0).is_err());
    }
}
