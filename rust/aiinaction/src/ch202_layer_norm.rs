//! Layer Normalization and RMSNorm from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch202_layer_norm` and the Julia module
//! `AIInAction.Ch202LayerNorm`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! Both operators act on the feature axis of a single example. std-only.
//!
//! LayerNorm:  `y = gamma * (x - mean) / sqrt(var + eps) + beta`   (var is ddof=0)
//! RMSNorm:    `y = gamma * x / sqrt(mean(x^2) + eps)`             (no mean, no beta)
//!
//! The `eps` is added inside the square root so the denominator stays bounded away
//! from zero even when the activation vector is itself near zero.

fn check_inputs(x: &[f64], eps: f64, name: &str) -> Result<(), String> {
    if x.is_empty() {
        return Err(format!("{} must have at least one feature", name));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(format!("{} contains non-finite values (nan or inf)", name));
    }
    if eps < 0.0 {
        return Err(format!("eps must be non-negative, got {}", eps));
    }
    Ok(())
}

fn resolve_gamma(d: usize, gamma: Option<&[f64]>) -> Result<Vec<f64>, String> {
    match gamma {
        None => Ok(vec![1.0f64; d]),
        Some(g) => {
            if g.len() != d {
                return Err(format!("gamma must have length {}, got {}", d, g.len()));
            }
            if g.iter().any(|v| !v.is_finite()) {
                return Err("gamma contains non-finite values (nan or inf)".to_string());
            }
            Ok(g.to_vec())
        }
    }
}

fn resolve_beta(d: usize, beta: Option<&[f64]>) -> Result<Vec<f64>, String> {
    match beta {
        None => Ok(vec![0.0f64; d]),
        Some(b) => {
            if b.len() != d {
                return Err(format!("beta must have length {}, got {}", d, b.len()));
            }
            if b.iter().any(|v| !v.is_finite()) {
                return Err("beta contains non-finite values (nan or inf)".to_string());
            }
            Ok(b.to_vec())
        }
    }
}

/// Layer-normalizes a single feature vector.
///
/// `gamma = None` defaults to all ones, `beta = None` to all zeros.
pub fn layer_norm(
    x: &[f64],
    gamma: Option<&[f64]>,
    beta: Option<&[f64]>,
    eps: f64,
) -> Result<Vec<f64>, String> {
    check_inputs(x, eps, "x")?;
    let d = x.len();
    let g = resolve_gamma(d, gamma)?;
    let b = resolve_beta(d, beta)?;

    let n = d as f64;
    let mu: f64 = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / n;
    let denom = (var + eps).sqrt();

    let mut out = vec![0.0f64; d];
    for i in 0..d {
        let x_hat = (x[i] - mu) / denom;
        out[i] = g[i] * x_hat + b[i];
    }
    Ok(out)
}

/// RMS-normalizes a single feature vector.
///
/// `gamma = None` defaults to all ones. There is no additive bias.
pub fn rms_norm(x: &[f64], gamma: Option<&[f64]>, eps: f64) -> Result<Vec<f64>, String> {
    check_inputs(x, eps, "x")?;
    let d = x.len();
    let g = resolve_gamma(d, gamma)?;

    let n = d as f64;
    let ms: f64 = x.iter().map(|v| v * v).sum::<f64>() / n;
    let rms = (ms + eps).sqrt();

    let mut out = vec![0.0f64; d];
    for i in 0..d {
        out[i] = (x[i] / rms) * g[i];
    }
    Ok(out)
}

/// Layer-normalizes every row of a matrix (slice of equal-length rows) independently.
pub fn apply_layer_norm(
    rows: &[Vec<f64>],
    gamma: Option<&[f64]>,
    beta: Option<&[f64]>,
    eps: f64,
) -> Result<Vec<Vec<f64>>, String> {
    if rows.is_empty() {
        return Err("X must have at least one row".to_string());
    }
    rows.iter().map(|r| layer_norm(r, gamma, beta, eps)).collect()
}

/// RMS-normalizes every row of a matrix (slice of equal-length rows) independently.
pub fn apply_rms_norm(
    rows: &[Vec<f64>],
    gamma: Option<&[f64]>,
    eps: f64,
) -> Result<Vec<Vec<f64>>, String> {
    if rows.is_empty() {
        return Err("X must have at least one row".to_string());
    }
    rows.iter().map(|r| rms_norm(r, gamma, eps)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const X: [f64; 4] = [2.0, 4.0, 6.0, 8.0];
    const GAMMA: [f64; 4] = [1.5, 0.5, 1.0, 2.0];
    const BETA: [f64; 4] = [0.1, -0.2, 0.0, 0.3];
    const EPS: f64 = 1e-5;
    const TOL: f64 = 1e-9;

    const EXPECTED_LN_PLAIN: [f64; 4] = [
        -1.3416394448610998,
        -0.4472131482870333,
        0.4472131482870333,
        1.3416394448610998,
    ];
    const EXPECTED_LN_AFFINE: [f64; 4] = [
        -1.9124591672916496,
        -0.42360657414351666,
        0.4472131482870333,
        2.9832788897221993,
    ];
    const EXPECTED_RMS_PLAIN: [f64; 4] = [
        0.36514831081206406,
        0.7302966216241281,
        1.0954449324361921,
        1.4605932432482562,
    ];
    const EXPECTED_RMS_GAMMA: [f64; 4] = [
        0.5477224662180961,
        0.36514831081206406,
        1.0954449324361921,
        2.9211864864965125,
    ];

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() < TOL, "index {}: {} vs {}", i, a[i], b[i]);
        }
    }

    #[test]
    fn layer_norm_plain_matches_fixture() {
        let y = layer_norm(&X, None, None, EPS).unwrap();
        close(&y, &EXPECTED_LN_PLAIN);
    }

    #[test]
    fn layer_norm_affine_matches_fixture() {
        let y = layer_norm(&X, Some(&GAMMA), Some(&BETA), EPS).unwrap();
        close(&y, &EXPECTED_LN_AFFINE);
    }

    #[test]
    fn rms_norm_plain_matches_fixture() {
        let y = rms_norm(&X, None, EPS).unwrap();
        close(&y, &EXPECTED_RMS_PLAIN);
    }

    #[test]
    fn rms_norm_gamma_matches_fixture() {
        let y = rms_norm(&X, Some(&GAMMA), EPS).unwrap();
        close(&y, &EXPECTED_RMS_GAMMA);
    }

    #[test]
    fn layer_norm_zero_mean_unit_variance() {
        let y = layer_norm(&X, None, None, 0.0).unwrap();
        let n = y.len() as f64;
        let mean: f64 = y.iter().sum::<f64>() / n;
        let var: f64 = y.iter().map(|v| v * v).sum::<f64>() / n;
        assert!(mean.abs() < 1e-12);
        assert!((var - 1.0).abs() < TOL);
    }

    #[test]
    fn layer_norm_shift_scale_invariance() {
        let (a, b) = (3.0f64, 5.0f64);
        let base = layer_norm(&X, None, None, 0.0).unwrap();
        let transformed: Vec<f64> = X.iter().map(|v| a * v + b).collect();
        let got = layer_norm(&transformed, None, None, 0.0).unwrap();
        close(&got, &base);
    }

    #[test]
    fn rms_norm_scale_invariance() {
        let a = 4.0f64;
        let base = rms_norm(&X, None, 0.0).unwrap();
        let scaled: Vec<f64> = X.iter().map(|v| a * v).collect();
        let got = rms_norm(&scaled, None, 0.0).unwrap();
        close(&got, &base);
    }

    #[test]
    fn constant_vector_layer_norm_is_all_beta() {
        let y = layer_norm(&[3.0, 3.0, 3.0], None, None, 1e-5).unwrap();
        for v in &y {
            assert!(v.is_finite());
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn apply_layer_norm_rows_independent() {
        let rows = vec![vec![2.0, 4.0, 6.0, 8.0], vec![1.0, 1.0, 1.0, 1.0]];
        let out = apply_layer_norm(&rows, None, None, EPS).unwrap();
        assert_eq!(out.len(), 2);
        close(&out[0], &EXPECTED_LN_PLAIN);
        for v in &out[1] {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn apply_rms_norm_rows_independent() {
        let rows = vec![vec![2.0, 4.0, 6.0, 8.0], vec![2.0, 4.0, 6.0, 8.0]];
        let out = apply_rms_norm(&rows, None, EPS).unwrap();
        close(&out[0], &EXPECTED_RMS_PLAIN);
        close(&out[1], &EXPECTED_RMS_PLAIN);
    }

    #[test]
    fn empty_vector_errors() {
        assert!(layer_norm(&[], None, None, EPS).is_err());
    }

    #[test]
    fn non_finite_errors() {
        assert!(rms_norm(&[1.0, f64::NAN, 3.0], None, EPS).is_err());
    }

    #[test]
    fn gamma_wrong_length_errors() {
        let g: [f64; 2] = [1.0, 2.0];
        assert!(layer_norm(&X, Some(&g), None, EPS).is_err());
    }

    #[test]
    fn beta_wrong_length_errors() {
        let b: [f64; 2] = [1.0, 2.0];
        assert!(layer_norm(&X, Some(&GAMMA), Some(&b), EPS).is_err());
    }

    #[test]
    fn negative_eps_errors() {
        assert!(rms_norm(&X, None, -1e-6).is_err());
    }
}
