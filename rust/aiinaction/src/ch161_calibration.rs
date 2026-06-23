//! Calibration metrics: reliability curves and Expected Calibration Error (Rust).
//!
//! Mirrors the Python module `aiinaction.ch161_calibration` and the Julia module
//! `AIInAction.Ch161Calibration`. The shared fixtures in the tests below match the
//! Python/Julia suites, which keeps the three implementations at parity.
//!
//! This is a std-only implementation of confidence calibration. Each example has a
//! scalar confidence `p_i = max_k f_k(x_i)` in `[0, 1]` and a binary correctness
//! indicator `c_i = 1[argmax_k f_k(x_i) == y_i]`. We partition `[0, 1]` into `M`
//! equal-width bins and, within each occupied bin `B_m`, compute the empirical
//! accuracy `acc(B_m)` and the average confidence `conf(B_m)`. Then
//!
//! ```text
//! ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|
//! MCE = max_m |acc(B_m) - conf(B_m)|
//! ```
//!
//! Binning convention: example `i` lands in bin `floor(p_i * M)`, with `p_i = 1`
//! folded into the last bin. Empty bins contribute nothing.

/// Summary of one reliability-diagram bin.
#[derive(Clone, Debug)]
pub struct Bin {
    /// Half-open confidence interval `[lower, upper)` (final bin closed on right).
    pub lower: f64,
    pub upper: f64,
    /// Number of examples whose confidence fell in this bin.
    pub count: usize,
    /// Empirical accuracy `acc(B_m)`, or 0.0 if empty.
    pub accuracy: f64,
    /// Average confidence `conf(B_m)`, or 0.0 if empty.
    pub confidence: f64,
}

impl Bin {
    /// Signed calibration gap `acc(B_m) - conf(B_m)`.
    pub fn gap(&self) -> f64 {
        self.accuracy - self.confidence
    }
}

/// A binned reliability curve plus the sample size it was built from.
#[derive(Clone, Debug)]
pub struct ReliabilityCurve {
    pub bins: Vec<Bin>,
    pub n_samples: usize,
}

impl ReliabilityCurve {
    pub fn n_bins(&self) -> usize {
        self.bins.len()
    }

    /// The bins that contain at least one example.
    pub fn occupied(&self) -> Vec<&Bin> {
        self.bins.iter().filter(|b| b.count > 0).collect()
    }
}

/// Validates lengths, range of confidences, and the binary correctness labels.
fn validate(confidences: &[f64], correct: &[f64], n_bins: usize) -> Result<(), String> {
    if confidences.len() != correct.len() {
        return Err(format!(
            "length mismatch: {} != {}",
            confidences.len(),
            correct.len()
        ));
    }
    if confidences.is_empty() {
        return Err("inputs must be non-empty".to_string());
    }
    if n_bins < 1 {
        return Err(format!("n_bins must be a positive integer, got {}", n_bins));
    }
    for &p in confidences {
        if !(0.0..=1.0).contains(&p) {
            return Err(format!("confidences must lie in [0, 1], got {}", p));
        }
    }
    for &c in correct {
        if c != 0.0 && c != 1.0 {
            return Err(format!("correct must be 0 or 1, got {}", c));
        }
    }
    Ok(())
}

/// Maps a confidence to its bin: `floor(p * M)`, with `p == 1` folded into the last bin.
fn bin_index(p: f64, n_bins: usize) -> usize {
    let b = (p * n_bins as f64) as usize;
    if b >= n_bins {
        n_bins - 1
    } else {
        b
    }
}

/// Builds an equal-width reliability curve over `n_bins` bins.
pub fn reliability_curve(
    confidences: &[f64],
    correct: &[f64],
    n_bins: usize,
) -> Result<ReliabilityCurve, String> {
    validate(confidences, correct, n_bins)?;
    let n = confidences.len();
    let mut counts = vec![0usize; n_bins];
    let mut acc_sum = vec![0.0f64; n_bins];
    let mut conf_sum = vec![0.0f64; n_bins];
    for (&p, &c) in confidences.iter().zip(correct) {
        let b = bin_index(p, n_bins);
        counts[b] += 1;
        acc_sum[b] += c;
        conf_sum[b] += p;
    }

    let mut bins = Vec::with_capacity(n_bins);
    for m in 0..n_bins {
        let lower = m as f64 / n_bins as f64;
        let upper = (m + 1) as f64 / n_bins as f64;
        let (accuracy, confidence) = if counts[m] > 0 {
            (acc_sum[m] / counts[m] as f64, conf_sum[m] / counts[m] as f64)
        } else {
            (0.0, 0.0)
        };
        bins.push(Bin {
            lower,
            upper,
            count: counts[m],
            accuracy,
            confidence,
        });
    }
    Ok(ReliabilityCurve {
        bins,
        n_samples: n,
    })
}

/// Expected Calibration Error (ECE), the occupancy-weighted mean absolute gap.
pub fn expected_calibration_error(
    confidences: &[f64],
    correct: &[f64],
    n_bins: usize,
) -> Result<f64, String> {
    let rc = reliability_curve(confidences, correct, n_bins)?;
    let mut total = 0.0;
    for b in &rc.bins {
        if b.count > 0 {
            total += (b.count as f64 / rc.n_samples as f64) * b.gap().abs();
        }
    }
    Ok(total)
}

/// Maximum Calibration Error (MCE), the largest absolute gap over occupied bins.
pub fn maximum_calibration_error(
    confidences: &[f64],
    correct: &[f64],
    n_bins: usize,
) -> Result<f64, String> {
    let rc = reliability_curve(confidences, correct, n_bins)?;
    let mut worst = 0.0f64;
    for b in &rc.bins {
        if b.count > 0 {
            worst = worst.max(b.gap().abs());
        }
    }
    Ok(worst)
}

/// Binary Brier score `BS = (1/n) sum_i (p_i - c_i)^2`.
pub fn brier_score(confidences: &[f64], correct: &[f64]) -> Result<f64, String> {
    validate(confidences, correct, 1)?;
    let n = confidences.len() as f64;
    let s: f64 = confidences
        .iter()
        .zip(correct)
        .map(|(&p, &c)| (p - c).powi(2))
        .sum();
    Ok(s / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const CONF: [f64; 10] = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3];
    const CORRECT: [f64; 10] = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    const TOL: f64 = 1e-9;

    #[test]
    fn ece_5_bins_matches_fixture() {
        let e = expected_calibration_error(&CONF, &CORRECT, 5).unwrap();
        assert!((e - 0.195).abs() < TOL);
    }

    #[test]
    fn mce_5_bins_matches_fixture() {
        let m = maximum_calibration_error(&CONF, &CORRECT, 5).unwrap();
        assert!((m - 0.7).abs() < TOL);
    }

    #[test]
    fn ece_10_bins_matches_fixture() {
        let e = expected_calibration_error(&CONF, &CORRECT, 10).unwrap();
        assert!((e - 0.285).abs() < TOL);
    }

    #[test]
    fn brier_matches_fixture() {
        let b = brier_score(&CONF, &CORRECT).unwrap();
        assert!((b - 0.23274999999999996).abs() < TOL);
    }

    #[test]
    fn reliability_curve_occupied_bins() {
        let rc = reliability_curve(&CONF, &CORRECT, 5).unwrap();
        assert_eq!(rc.n_bins(), 5);
        assert_eq!(rc.n_samples, 10);
        let occ = rc.occupied();
        let counts: Vec<usize> = occ.iter().map(|b| b.count).collect();
        assert_eq!(counts, vec![1, 3, 2, 4]);
        assert!((occ[0].accuracy - 1.0).abs() < TOL);
        assert!((occ[0].confidence - 0.3).abs() < TOL);
        assert!((occ[0].gap() - 0.7).abs() < TOL);
        assert_eq!(occ[3].count, 4);
        assert!((occ[3].accuracy - 0.75).abs() < TOL);
        assert!((occ[3].confidence - 0.875).abs() < TOL);
    }

    #[test]
    fn empty_bin_present_but_ignored() {
        let rc = reliability_curve(&CONF, &CORRECT, 5).unwrap();
        assert_eq!(rc.bins[0].count, 0);
        assert_eq!(rc.bins[0].accuracy, 0.0);
        assert_eq!(rc.bins[0].confidence, 0.0);
    }

    #[test]
    fn perfectly_calibrated_is_zero() {
        let conf: [f64; 8] = [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75];
        let correct: [f64; 8] = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        assert!(expected_calibration_error(&conf, &correct, 2).unwrap().abs() < TOL);
        assert!(maximum_calibration_error(&conf, &correct, 2).unwrap().abs() < TOL);
    }

    #[test]
    fn confidence_one_folds_into_last_bin() {
        let conf: [f64; 2] = [1.0, 1.0];
        let correct: [f64; 2] = [1.0, 0.0];
        let rc = reliability_curve(&conf, &correct, 4).unwrap();
        let last = rc.bins.last().unwrap();
        assert_eq!(last.count, 2);
        assert!((last.accuracy - 0.5).abs() < TOL);
        assert!((last.confidence - 1.0).abs() < TOL);
    }

    #[test]
    fn length_mismatch_errors() {
        assert!(expected_calibration_error(&[0.5, 0.5], &[1.0], 5).is_err());
    }

    #[test]
    fn empty_errors() {
        assert!(expected_calibration_error(&[], &[], 5).is_err());
    }

    #[test]
    fn bad_n_bins_errors() {
        assert!(expected_calibration_error(&CONF, &CORRECT, 0).is_err());
    }

    #[test]
    fn confidence_out_of_range_errors() {
        assert!(reliability_curve(&[1.5], &[1.0], 5).is_err());
    }

    #[test]
    fn correct_not_binary_errors() {
        assert!(reliability_curve(&[0.5], &[2.0], 5).is_err());
    }
}
