//! Elastic Net regression via coordinate descent (Chapter 086).
//!
//! Solves
//!
//! ```text
//! minimize (1 / (2 n)) ||y - X b||^2 + lambda (alpha ||b||_1 + ((1 - alpha) / 2) ||b||_2^2)
//! ```
//!
//! by cyclic coordinate descent with soft thresholding, fitting an unpenalized
//! intercept by centering internally. Mirrors the Python module
//! `aiinaction.ch086_elastic_net` and the Julia module
//! `AIInAction.Ch086ElasticNet`; the shared fixtures in the tests below match
//! the Python/Julia suites, which keeps the three at parity.

/// Soft-thresholding operator `S(z, gamma) = sign(z) * max(|z| - gamma, 0)`.
pub fn soft_threshold(z: f64, gamma: f64) -> Result<f64, String> {
    if gamma < 0.0 {
        return Err(format!("gamma must be non-negative, got {gamma}"));
    }
    if z > gamma {
        Ok(z - gamma)
    } else if z < -gamma {
        Ok(z + gamma)
    } else {
        Ok(0.0)
    }
}

/// Validates the design matrix and returns `(n_rows, n_cols)`.
fn matrix_dims(x: &[Vec<f64>]) -> Result<(usize, usize), String> {
    if x.is_empty() {
        return Err("X must be non-empty".to_string());
    }
    let cols = x[0].len();
    if cols == 0 {
        return Err("X must be non-empty".to_string());
    }
    for row in x {
        if row.len() != cols {
            return Err(format!(
                "X must be rectangular: expected {} columns, got {}",
                cols,
                row.len()
            ));
        }
    }
    Ok((x.len(), cols))
}

/// Fits Elastic Net coefficients by cyclic coordinate descent.
///
/// Returns `(coef, intercept)`, with `coef` of length `n_features` on the
/// original feature scale and `intercept` the unpenalized intercept.
pub fn elastic_net_fit(
    x: &[Vec<f64>],
    y: &[f64],
    lam: f64,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, f64), String> {
    let (n, p) = matrix_dims(x)?;
    if y.len() != n {
        return Err(format!(
            "length mismatch: X has {} rows but y has {}",
            n,
            y.len()
        ));
    }
    if lam < 0.0 {
        return Err(format!("lam must be non-negative, got {lam}"));
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(format!("alpha must lie in [0, 1], got {alpha}"));
    }
    if max_iter == 0 {
        return Err("max_iter must be positive".to_string());
    }
    if tol <= 0.0 {
        return Err(format!("tol must be positive, got {tol}"));
    }

    let nf = n as f64;

    // Column means and response mean for centering (unpenalized intercept).
    let mut x_mean = vec![0.0_f64; p];
    for row in x {
        for (j, &v) in row.iter().enumerate() {
            x_mean[j] += v;
        }
    }
    for m in x_mean.iter_mut() {
        *m /= nf;
    }
    let y_mean: f64 = y.iter().sum::<f64>() / nf;

    // Centered design stored column-major for cache-friendly coordinate sweeps.
    let mut xc: Vec<Vec<f64>> = vec![vec![0.0; n]; p];
    for (i, row) in x.iter().enumerate() {
        for j in 0..p {
            xc[j][i] = row[j] - x_mean[j];
        }
    }
    let yc: Vec<f64> = y.iter().map(|&v| v - y_mean).collect();

    // Per-feature curvature (||x_j||^2 / n).
    let col_sq: Vec<f64> = (0..p)
        .map(|j| xc[j].iter().map(|&v| v * v).sum::<f64>() / nf)
        .collect();

    let mut beta = vec![0.0_f64; p];
    let mut residual = yc.clone(); // residual = yc - Xc * beta, kept in sync
    let l1 = lam * alpha;
    let l2 = lam * (1.0 - alpha);

    for _ in 0..max_iter {
        let mut max_change = 0.0_f64;
        for j in 0..p {
            let xj = &xc[j];
            if col_sq[j] == 0.0 {
                if beta[j] != 0.0 {
                    for i in 0..n {
                        residual[i] += xj[i] * beta[j];
                    }
                    beta[j] = 0.0;
                }
                continue;
            }
            let beta_j_old = beta[j];
            // rho = (1/n) x_j^T (residual + x_j * beta_j_old)
            let mut rho = 0.0_f64;
            for i in 0..n {
                rho += xj[i] * (residual[i] + xj[i] * beta_j_old);
            }
            rho /= nf;
            let beta_j_new = soft_threshold(rho, l1)? / (col_sq[j] + l2);
            if beta_j_new != beta_j_old {
                let delta = beta_j_old - beta_j_new;
                for i in 0..n {
                    residual[i] += xj[i] * delta;
                }
                beta[j] = beta_j_new;
                let change = (beta_j_new - beta_j_old).abs();
                if change > max_change {
                    max_change = change;
                }
            }
        }
        if max_change < tol {
            break;
        }
    }

    let intercept = y_mean - x_mean.iter().zip(&beta).map(|(m, b)| m * b).sum::<f64>();
    Ok((beta, intercept))
}

/// Predicts responses `X * coef + intercept` for a fitted model.
pub fn elastic_net_predict(
    x: &[Vec<f64>],
    coef: &[f64],
    intercept: f64,
) -> Result<Vec<f64>, String> {
    let (_, p) = matrix_dims(x)?;
    if coef.len() != p {
        return Err(format!(
            "length mismatch: X has {} features but coef has {}",
            p,
            coef.len()
        ));
    }
    Ok(x
        .iter()
        .map(|row| row.iter().zip(coef).map(|(v, c)| v * c).sum::<f64>() + intercept)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn x() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 4.0],
            vec![4.0, 3.0],
            vec![5.0, 6.0],
        ]
    }
    fn y() -> Vec<f64> {
        vec![2.0, 3.0, 5.0, 7.0, 8.0]
    }

    const EN_COEF: [f64; 2] = [1.1076803723827306, 0.22885958106994972];
    const EN_INTERCEPT: f64 = 0.9446082234279691;
    const EN_PRED: [f64; 5] = [
        2.510007757950599,
        3.3888285492633803,
        5.18308766485596,
        6.061908456168741,
        7.856167571761321,
    ];
    const LASSO_COEF: [f64; 2] = [1.1, 0.0];
    const LASSO_INTERCEPT: f64 = 1.7;
    const RIDGE_COEF: [f64; 2] = [0.795939086294827, 0.4060913705581682];
    const RIDGE_INTERCEPT: f64 = 1.312690355329381;
    const OLS_COEF: [f64; 2] = [1.6, 0.0];
    const OLS_INTERCEPT: f64 = 0.2;
    const TOL: f64 = 1e-9;

    #[test]
    fn soft_threshold_basic() {
        assert!((soft_threshold(3.0, 1.0).unwrap() - 2.0).abs() < TOL);
        assert!((soft_threshold(-3.0, 1.0).unwrap() + 2.0).abs() < TOL);
        assert_eq!(soft_threshold(0.5, 1.0).unwrap(), 0.0);
        assert!(soft_threshold(1.0, -0.5).is_err());
    }

    #[test]
    fn elastic_net_matches_fixture() {
        let (coef, b0) = elastic_net_fit(&x(), &y(), 0.5, 0.5, 10000, 1e-12).unwrap();
        assert!((coef[0] - EN_COEF[0]).abs() < TOL);
        assert!((coef[1] - EN_COEF[1]).abs() < TOL);
        assert!((b0 - EN_INTERCEPT).abs() < TOL);
    }

    #[test]
    fn predict_matches_fixture() {
        let preds = elastic_net_predict(&x(), &EN_COEF, EN_INTERCEPT).unwrap();
        for (p, e) in preds.iter().zip(EN_PRED.iter()) {
            assert!((p - e).abs() < TOL);
        }
    }

    #[test]
    fn lasso_limit_zeros_a_coefficient() {
        let (coef, b0) = elastic_net_fit(&x(), &y(), 1.0, 1.0, 10000, 1e-12).unwrap();
        assert!((coef[0] - LASSO_COEF[0]).abs() < TOL);
        assert!((coef[1] - LASSO_COEF[1]).abs() < TOL);
        assert_eq!(coef[1], 0.0);
        assert!((b0 - LASSO_INTERCEPT).abs() < TOL);
    }

    #[test]
    fn ridge_limit_shrinks_without_zeroing() {
        let (coef, b0) = elastic_net_fit(&x(), &y(), 1.0, 0.0, 10000, 1e-12).unwrap();
        assert!((coef[0] - RIDGE_COEF[0]).abs() < TOL);
        assert!((coef[1] - RIDGE_COEF[1]).abs() < TOL);
        assert!((b0 - RIDGE_INTERCEPT).abs() < TOL);
    }

    #[test]
    fn zero_lambda_recovers_ols() {
        let (coef, b0) = elastic_net_fit(&x(), &y(), 0.0, 0.5, 10000, 1e-12).unwrap();
        assert!((coef[0] - OLS_COEF[0]).abs() < 1e-7);
        assert!((coef[1] - OLS_COEF[1]).abs() < 1e-7);
        assert!((b0 - OLS_INTERCEPT).abs() < 1e-7);
    }

    #[test]
    fn length_mismatch_errors() {
        assert!(elastic_net_fit(&[vec![1.0], vec![2.0]], &[1.0], 0.1, 0.5, 100, 1e-8).is_err());
    }

    #[test]
    fn alpha_out_of_range_errors() {
        assert!(elastic_net_fit(&x(), &y(), 0.1, 1.5, 100, 1e-8).is_err());
    }

    #[test]
    fn negative_lambda_errors() {
        assert!(elastic_net_fit(&x(), &y(), -1.0, 0.5, 100, 1e-8).is_err());
    }
}
