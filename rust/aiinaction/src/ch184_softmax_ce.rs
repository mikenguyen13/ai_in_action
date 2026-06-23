//! Softmax cross-entropy loss and its gradient, from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch184_softmax_ce` and the Julia module
//! `AIInAction.Ch184SoftmaxCE`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! Everything is computed in a numerically stable way directly from logits: the
//! per-row maximum is subtracted before exponentiating, and the loss uses the
//! fused identity `log softmax(z)_k = z_k - logsumexp(z)`. The gradient is the
//! clean predicted-minus-target form `(p - q') / N`.
//!
//! Inputs are row-major matrices of logits with shape `(N, K)`: `N` samples of
//! `K >= 2` class scores. Labels are integer class indices in `[0, K - 1]`.
//! Optional label smoothing replaces the one-hot target `q` with
//! `q'(k) = (1 - eps) * 1[k = y] + eps / K`.

/// Validates a logit matrix `(n x k)` and returns `(n, k)` on success.
fn check_logits(z: &[Vec<f64>]) -> Result<(usize, usize), String> {
    if z.is_empty() {
        return Err("need at least one sample (N >= 1)".to_string());
    }
    let k = z[0].len();
    if k < 2 {
        return Err(format!("need at least 2 classes (K >= 2), got K={}", k));
    }
    for row in z {
        if row.len() != k {
            return Err("all rows must have the same number of classes".to_string());
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err("logits contain non-finite values (nan or inf)".to_string());
        }
    }
    Ok((z.len(), k))
}

fn check_labels(labels: &[usize], n: usize, k: usize) -> Result<(), String> {
    if labels.len() != n {
        return Err(format!(
            "labels has length {} but logits has {} rows",
            labels.len(),
            n
        ));
    }
    if labels.iter().any(|&y| y >= k) {
        return Err(format!("labels must be in [0, {}]", k - 1));
    }
    Ok(())
}

fn check_smoothing(eps: f64) -> Result<(), String> {
    if !(0.0..1.0).contains(&eps) {
        return Err(format!("label_smoothing must be in [0, 1), got {}", eps));
    }
    Ok(())
}

/// Stable softmax of one logit row.
fn softmax_row(row: &[f64]) -> Vec<f64> {
    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = row.iter().map(|&v| (v - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Stable log-softmax of one logit row, computed without forming the softmax.
fn log_softmax_row(row: &[f64]) -> Vec<f64> {
    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse: f64 = row.iter().map(|&v| (v - m).exp()).sum::<f64>().ln();
    row.iter().map(|&v| (v - m) - lse).collect()
}

/// Row-wise softmax of a logit matrix, shape `(N, K)`.
pub fn softmax(z: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    check_logits(z)?;
    Ok(z.iter().map(|row| softmax_row(row)).collect())
}

/// Row-wise log-softmax of a logit matrix, shape `(N, K)`.
pub fn log_softmax(z: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    check_logits(z)?;
    Ok(z.iter().map(|row| log_softmax_row(row)).collect())
}

/// Mean softmax cross-entropy loss over a batch of logits.
///
/// `label_smoothing = 0.0` is standard hard-target cross-entropy; with `eps > 0`
/// the one-hot target is replaced by `q'(k) = (1 - eps) 1[k = y] + eps / K`.
pub fn cross_entropy_loss(
    z: &[Vec<f64>],
    labels: &[usize],
    label_smoothing: f64,
) -> Result<f64, String> {
    let (n, k) = check_logits(z)?;
    check_labels(labels, n, k)?;
    check_smoothing(label_smoothing)?;

    let eps = label_smoothing;
    let mut total = 0.0;
    for (i, row) in z.iter().enumerate() {
        let logp = log_softmax_row(row);
        let correct = logp[labels[i]];
        if eps == 0.0 {
            total += -correct;
        } else {
            let uniform: f64 = logp.iter().sum::<f64>() / k as f64;
            total += -((1.0 - eps) * correct + eps * uniform);
        }
    }
    Ok(total / n as f64)
}

/// Gradient of [`cross_entropy_loss`] with respect to the logits.
///
/// Returns `(p - q') / N`, the clean predicted-minus-target gradient, with shape
/// `(N, K)`.
pub fn cross_entropy_grad(
    z: &[Vec<f64>],
    labels: &[usize],
    label_smoothing: f64,
) -> Result<Vec<Vec<f64>>, String> {
    let (n, k) = check_logits(z)?;
    check_labels(labels, n, k)?;
    check_smoothing(label_smoothing)?;

    let eps = label_smoothing;
    let inv_n = 1.0 / n as f64;
    let mut out = vec![vec![0.0f64; k]; n];
    for (i, row) in z.iter().enumerate() {
        let p = softmax_row(row);
        for j in 0..k {
            let mut q = eps / k as f64;
            if j == labels[i] {
                q += 1.0 - eps;
            }
            out[i][j] = (p[j] - q) * inv_n;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn fixture() -> Vec<Vec<f64>> {
        vec![
            vec![2.0, 1.0, 0.1],
            vec![0.5, 2.5, 0.3],
            vec![1.0, 1.0, 1.0],
        ]
    }
    const Y184: [usize; 3] = [0, 1, 2];
    const TOL: f64 = 1e-9;

    #[test]
    fn softmax_matches_fixture() {
        let p = softmax(&fixture()).unwrap();
        let expected: [f64; 3] = [
            0.6590011388859679,
            0.24243297070471392,
            0.09856589040931818,
        ];
        for j in 0..3 {
            assert!((p[0][j] - expected[j]).abs() < TOL);
        }
        for row in &p {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn log_softmax_matches_fixture() {
        let ls = log_softmax(&fixture()).unwrap();
        let expected: [f64; 3] = [
            -0.41703001627783354,
            -1.4170300162778335,
            -2.3170300162778332,
        ];
        for j in 0..3 {
            assert!((ls[0][j] - expected[j]).abs() < TOL);
        }
    }

    #[test]
    fn loss_matches_fixture() {
        let l = cross_entropy_loss(&fixture(), &Y184, 0.0).unwrap();
        assert!((l - 0.5785639426554937).abs() < TOL);
    }

    #[test]
    fn loss_smoothed_matches_fixture() {
        let l = cross_entropy_loss(&fixture(), &Y184, 0.1).unwrap();
        assert!((l - 0.6574528315443826).abs() < TOL);
    }

    #[test]
    fn grad_matches_fixture() {
        let g = cross_entropy_grad(&fixture(), &Y184, 0.0).unwrap();
        let expected: [[f64; 3]; 3] = [
            [-0.1136662870380107, 0.08081099023490464, 0.03285529680310606],
            [0.03620124343567079, -0.06584031473611691, 0.029639071300446088],
            [0.1111111111111111, 0.1111111111111111, -0.22222222222222224],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!((g[i][j] - expected[i][j]).abs() < TOL);
            }
        }
    }

    #[test]
    fn grad_smoothed_matches_fixture() {
        let g = cross_entropy_grad(&fixture(), &Y184, 0.1).unwrap();
        let expected: [[f64; 3]; 3] = [
            [-0.09144406481578848, 0.06969987912379354, 0.02174418569199495],
            [0.025090132324559682, -0.0436180925138947, 0.018527960189334978],
            [0.09999999999999999, 0.09999999999999999, -0.20000000000000004],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!((g[i][j] - expected[i][j]).abs() < TOL);
            }
        }
    }

    #[test]
    fn grad_rows_sum_to_zero() {
        let g = cross_entropy_grad(&fixture(), &Y184, 0.0).unwrap();
        for row in &g {
            assert!(row.iter().sum::<f64>().abs() < 1e-12);
        }
    }

    #[test]
    fn uniform_logits_loss_is_log_k() {
        let z: Vec<Vec<f64>> = vec![vec![0.0, 0.0, 0.0]];
        let l = cross_entropy_loss(&z, &[1usize], 0.0).unwrap();
        assert!((l - 3.0f64.ln()).abs() < TOL);
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let z = fixture();
        let g = cross_entropy_grad(&z, &Y184, 0.0).unwrap();
        let h = 1e-6;
        for i in 0..z.len() {
            for j in 0..z[0].len() {
                let mut zp = z.clone();
                zp[i][j] += h;
                let mut zm = z.clone();
                zm[i][j] -= h;
                let num = (cross_entropy_loss(&zp, &Y184, 0.0).unwrap()
                    - cross_entropy_loss(&zm, &Y184, 0.0).unwrap())
                    / (2.0 * h);
                assert!((g[i][j] - num).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn stable_under_large_logits() {
        let z: Vec<Vec<f64>> = vec![vec![1000.0, 0.0], vec![0.0, 1000.0]];
        let p = softmax(&z).unwrap();
        assert!(p.iter().all(|r| r.iter().all(|v| v.is_finite())));
        assert!((p[0][0] - 1.0).abs() < TOL);
        let l = cross_entropy_loss(&z, &[0usize, 1usize], 0.0).unwrap();
        assert!(l.is_finite() && l.abs() < 1e-9);
    }

    #[test]
    fn too_few_classes_errors() {
        let z: Vec<Vec<f64>> = vec![vec![1.0]];
        assert!(cross_entropy_loss(&z, &[0usize], 0.0).is_err());
    }

    #[test]
    fn non_finite_errors() {
        let z: Vec<Vec<f64>> = vec![vec![1.0, f64::INFINITY]];
        assert!(softmax(&z).is_err());
    }

    #[test]
    fn label_out_of_range_errors() {
        assert!(cross_entropy_loss(&fixture(), &[0usize, 1usize, 3usize], 0.0).is_err());
    }

    #[test]
    fn label_length_mismatch_errors() {
        assert!(cross_entropy_loss(&fixture(), &[0usize, 1usize], 0.0).is_err());
    }

    #[test]
    fn bad_smoothing_errors() {
        assert!(cross_entropy_loss(&fixture(), &Y184, 1.0).is_err());
    }
}
