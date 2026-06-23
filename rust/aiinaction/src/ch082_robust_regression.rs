//! Linear regression extensions: robust (Huber), WLS, GLS, quantile, basis (Rust).
//!
//! Mirrors the Python module `aiinaction.ch082_robust_regression` and the Julia
//! module `AIInAction.Ch082RobustRegression`. The shared fixtures in the tests
//! below match the Python/Julia suites, which is what keeps the three at parity.
//!
//! The core algorithm is Huber robust regression solved by iteratively reweighted
//! least squares (IRLS). The same weighted-normal-equations engine powers weighted
//! least squares (WLS), generalized least squares (GLS), and quantile regression,
//! and [`vandermonde`] provides a polynomial basis expansion. Linear systems are
//! solved with a plain Gaussian-elimination routine (std-only) so the numerics
//! match the other two languages.

/// Result of a Huber IRLS fit.
#[derive(Debug, Clone, PartialEq)]
pub struct HuberResult {
    /// Estimated coefficients.
    pub coef: Vec<f64>,
    /// Final robust scale estimate (MAD of residuals).
    pub scale: f64,
    /// Number of IRLS iterations performed.
    pub n_iter: usize,
    /// Whether the coefficient change fell below `tol` before `max_iter`.
    pub converged: bool,
}

fn check_matrix(x: &[Vec<f64>]) -> Result<usize, String> {
    if x.is_empty() {
        return Err("design matrix X must have at least one row".to_string());
    }
    let ncol = x[0].len();
    if ncol == 0 {
        return Err("design matrix X must have at least one column".to_string());
    }
    for (i, row) in x.iter().enumerate() {
        if row.len() != ncol {
            return Err(format!(
                "ragged design matrix: row 0 has {} columns but row {} has {}",
                ncol,
                i,
                row.len()
            ));
        }
    }
    Ok(ncol)
}

fn check_xy(x: &[Vec<f64>], y: &[f64]) -> Result<usize, String> {
    let ncol = check_matrix(x)?;
    if x.len() != y.len() {
        return Err(format!(
            "length mismatch: X has {} rows but y has {} entries",
            x.len(),
            y.len()
        ));
    }
    if x.len() < ncol {
        return Err(format!(
            "underdetermined system: {} rows < {} columns",
            x.len(),
            ncol
        ));
    }
    Ok(ncol)
}

/// Solve the square linear system `A x = b` by Gaussian elimination with partial
/// pivoting. Errors if `A` is not square, dimensions are inconsistent, or `A` is
/// singular.
pub fn solve(a_in: &[Vec<f64>], b_in: &[f64]) -> Result<Vec<f64>, String> {
    let n = a_in.len();
    if n == 0 {
        return Err("system must be non-empty".to_string());
    }
    let mut a: Vec<Vec<f64>> = a_in.to_vec();
    let mut rhs: Vec<f64> = b_in.to_vec();
    for row in &a {
        if row.len() != n {
            return Err(format!(
                "matrix must be square: got {} columns for {} rows",
                row.len(),
                n
            ));
        }
    }
    if rhs.len() != n {
        return Err(format!(
            "length mismatch: A is {}x{} but b has {} entries",
            n,
            n,
            rhs.len()
        ));
    }

    for col in 0..n {
        let mut pivot = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-14 {
            return Err("matrix is singular or nearly singular".to_string());
        }
        if pivot != col {
            a.swap(col, pivot);
            rhs.swap(col, pivot);
        }
        let inv = 1.0 / a[col][col];
        for r in (col + 1)..n {
            let factor = a[r][col] * inv;
            if factor != 0.0 {
                for c in col..n {
                    a[r][c] -= factor * a[col][c];
                }
                rhs[r] -= factor * rhs[col];
            }
        }
    }

    let mut x = vec![0.0_f64; n];
    for col in (0..n).rev() {
        let mut s = rhs[col];
        for c in (col + 1)..n {
            s -= a[col][c] * x[c];
        }
        x[col] = s / a[col][col];
    }
    Ok(x)
}

fn matvec(x: &[Vec<f64>], beta: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|row| row.iter().zip(beta).map(|(xij, bj)| xij * bj).sum())
        .collect()
}

/// Linear prediction `X @ beta`.
pub fn predict(x: &[Vec<f64>], beta: &[f64]) -> Result<Vec<f64>, String> {
    let ncol = check_matrix(x)?;
    if ncol != beta.len() {
        return Err(format!(
            "shape mismatch: X has {} columns but beta has {} entries",
            ncol,
            beta.len()
        ));
    }
    Ok(matvec(x, beta))
}

/// Solve `(Xᵀ W X) β = Xᵀ W y` for diagonal weights `w`.
fn weighted_normal_equations(
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
) -> Result<Vec<f64>, String> {
    let n = x.len();
    let p = x[0].len();
    let mut xtwx = vec![vec![0.0_f64; p]; p];
    let mut xtwy = vec![0.0_f64; p];
    for i in 0..n {
        let wi = w[i];
        let row = &x[i];
        let wyi = wi * y[i];
        for a in 0..p {
            let xa = row[a];
            xtwy[a] += xa * wyi;
            let wxa = wi * xa;
            for b in a..p {
                xtwx[a][b] += wxa * row[b];
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            xtwx[a][b] = xtwx[b][a];
        }
    }
    solve(&xtwx, &xtwy)
}

fn median(values: &[f64]) -> f64 {
    let mut s: Vec<f64> = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let mid = n / 2;
    if n % 2 == 1 {
        s[mid]
    } else {
        0.5 * (s[mid - 1] + s[mid])
    }
}

fn mad_scale(residuals: &[f64]) -> f64 {
    let med = median(residuals);
    let abs_dev: Vec<f64> = residuals.iter().map(|r| (r - med).abs()).collect();
    1.4826 * median(&abs_dev)
}

/// Ordinary least squares: solves the normal equations `XᵀX β = Xᵀy`.
pub fn fit_ols(x: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>, String> {
    check_xy(x, y)?;
    let w = vec![1.0_f64; x.len()];
    weighted_normal_equations(x, y, &w)
}

/// Weighted least squares: `argmin_β Σ wᵢ (yᵢ - xᵢᵀβ)²`. Weights must be
/// non-negative and the same length as `y`.
pub fn fit_wls(x: &[Vec<f64>], y: &[f64], weights: &[f64]) -> Result<Vec<f64>, String> {
    check_xy(x, y)?;
    if weights.len() != x.len() {
        return Err(format!(
            "length mismatch: X has {} rows but weights has {} entries",
            x.len(),
            weights.len()
        ));
    }
    if weights.iter().any(|&wi| wi < 0.0) {
        return Err("weights must be non-negative".to_string());
    }
    weighted_normal_equations(x, y, weights)
}

/// Generalized least squares for a known error covariance `cov` (Ω). Computes
/// `β = (Xᵀ Ω⁻¹ X)⁻¹ Xᵀ Ω⁻¹ y` by solving systems against Ω (no explicit inverse).
pub fn fit_gls(
    x: &[Vec<f64>],
    y: &[f64],
    cov: &[Vec<f64>],
) -> Result<Vec<f64>, String> {
    let p = check_xy(x, y)?;
    let n = x.len();
    if cov.len() != n || cov.iter().any(|r| r.len() != n) {
        return Err(format!("cov must be {n}x{n} to match {n} observations"));
    }
    let omega: Vec<Vec<f64>> = cov.to_vec();

    let oinv_y = solve(&omega, y)?;
    let mut oinv_cols: Vec<Vec<f64>> = Vec::with_capacity(p);
    for j in 0..p {
        let col: Vec<f64> = (0..n).map(|i| x[i][j]).collect();
        oinv_cols.push(solve(&omega, &col)?);
    }

    let mut xtoix = vec![vec![0.0_f64; p]; p];
    let mut xtoiy = vec![0.0_f64; p];
    for a in 0..p {
        let col_a: Vec<f64> = (0..n).map(|i| x[i][a]).collect();
        xtoiy[a] = (0..n).map(|i| col_a[i] * oinv_y[i]).sum();
        for b in 0..p {
            xtoix[a][b] = (0..n).map(|i| col_a[i] * oinv_cols[b][i]).sum();
        }
    }
    solve(&xtoix, &xtoiy)
}

/// Robust regression with the Huber loss via iteratively reweighted least squares,
/// seeded with OLS. `delta` is the tuning constant in scale units (1.345 gives
/// ~95% Gaussian efficiency).
pub fn fit_huber(
    x: &[Vec<f64>],
    y: &[f64],
    delta: f64,
    max_iter: usize,
    tol: f64,
) -> Result<HuberResult, String> {
    check_xy(x, y)?;
    if delta <= 0.0 {
        return Err(format!("delta must be positive, got {delta}"));
    }
    if max_iter == 0 {
        return Err("max_iter must be positive, got 0".to_string());
    }
    if tol < 0.0 {
        return Err(format!("tol must be non-negative, got {tol}"));
    }

    let ones = vec![1.0_f64; x.len()];
    let mut beta = weighted_normal_equations(x, y, &ones)?;
    let mut scale = 1.0_f64;
    let mut converged = false;
    let mut n_iter = 0;
    for it in 1..=max_iter {
        n_iter = it;
        let pred = matvec(x, &beta);
        let resid: Vec<f64> = y.iter().zip(&pred).map(|(yi, ri)| yi - ri).collect();
        scale = mad_scale(&resid);
        if scale <= 1e-12 {
            converged = true;
            break;
        }
        let thresh = delta * scale;
        let weights: Vec<f64> = resid
            .iter()
            .map(|&r| {
                if r.abs() > 1e-30 {
                    (thresh / r.abs()).min(1.0)
                } else {
                    1.0
                }
            })
            .collect();
        let new_beta = weighted_normal_equations(x, y, &weights)?;
        let change = new_beta
            .iter()
            .zip(&beta)
            .map(|(nb, b)| (nb - b).abs())
            .fold(0.0_f64, f64::max);
        beta = new_beta;
        if change < tol {
            converged = true;
            break;
        }
    }
    Ok(HuberResult {
        coef: beta,
        scale,
        n_iter,
        converged,
    })
}

/// Quantile regression at level `tau` via IRLS on the pinball loss. `tau` must be
/// in the open interval (0, 1); `tau = 0.5` recovers least-absolute-deviations.
pub fn fit_quantile(
    x: &[Vec<f64>],
    y: &[f64],
    tau: f64,
    max_iter: usize,
    tol: f64,
    eps: f64,
) -> Result<Vec<f64>, String> {
    check_xy(x, y)?;
    if !(tau > 0.0 && tau < 1.0) {
        return Err(format!(
            "tau must be in the open interval (0, 1), got {tau}"
        ));
    }
    if max_iter == 0 {
        return Err("max_iter must be positive, got 0".to_string());
    }
    if eps <= 0.0 {
        return Err(format!("eps must be positive, got {eps}"));
    }

    let ones = vec![1.0_f64; x.len()];
    let mut beta = weighted_normal_equations(x, y, &ones)?;
    for _ in 0..max_iter {
        let pred = matvec(x, &beta);
        let resid: Vec<f64> = y.iter().zip(&pred).map(|(yi, ri)| yi - ri).collect();
        let weights: Vec<f64> = resid
            .iter()
            .map(|&r| {
                let num = if r >= 0.0 { tau } else { 1.0 - tau };
                num / r.abs().max(eps)
            })
            .collect();
        let new_beta = weighted_normal_equations(x, y, &weights)?;
        let change = new_beta
            .iter()
            .zip(&beta)
            .map(|(nb, b)| (nb - b).abs())
            .fold(0.0_f64, f64::max);
        beta = new_beta;
        if change < tol {
            break;
        }
    }
    Ok(beta)
}

/// Polynomial basis expansion (Vandermonde design matrix). Each scalar `xᵢ` maps
/// to `[1, xᵢ, xᵢ², ..., xᵢ^degree]`; the leading `1` is dropped when
/// `include_bias` is false.
pub fn vandermonde(
    x: &[f64],
    degree: usize,
    include_bias: bool,
) -> Result<Vec<Vec<f64>>, String> {
    if x.is_empty() {
        return Err("input x must be non-empty".to_string());
    }
    if !include_bias && degree == 0 {
        return Err("degree must be >= 1 when include_bias is false".to_string());
    }
    let start = if include_bias { 0 } else { 1 };
    Ok(x
        .iter()
        .map(|&xi| (start..=degree).map(|k| xi.powi(k as i32)).collect())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn x_huber() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
            vec![1.0, 4.0],
            vec![1.0, 5.0],
        ]
    }
    const Y_HUBER: [f64; 6] = [1.0, 3.0, 5.0, 20.0, 9.0, 11.0];
    const OLS_HUBER_COEF: [f64; 2] = [2.238_095_238_095_238_1, 2.371_428_571_428_571_4];

    fn x_wls() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
        ]
    }
    const Y_WLS: [f64; 4] = [1.0, 3.0, 5.0, 7.0];
    const W_WLS: [f64; 4] = [1.0, 2.0, 3.0, 4.0];

    fn x_gls() -> Vec<Vec<f64>> {
        vec![vec![1.0, 0.0], vec![1.0, 1.0], vec![1.0, 2.0]]
    }
    const Y_GLS: [f64; 3] = [1.0, 2.0, 2.5];
    fn cov_gls() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.5, 0.25],
            vec![0.5, 1.0, 0.5],
            vec![0.25, 0.5, 1.0],
        ]
    }
    const GLS_COEF: [f64; 2] = [1.05, 0.75];

    fn x_q() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
            vec![1.0, 4.0],
        ]
    }
    const Y_Q: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 50.0];

    const TOL: f64 = 1e-9;

    #[test]
    fn solve_diagonal() {
        let x = solve(&[vec![2.0, 0.0], vec![0.0, 4.0]], &[2.0, 8.0]).unwrap();
        assert!((x[0] - 1.0).abs() < TOL && (x[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn solve_singular_errors() {
        assert!(solve(&[vec![1.0, 2.0], vec![2.0, 4.0]], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn predict_matches() {
        let p = predict(&[vec![1.0, 2.0], vec![1.0, 3.0]], &[1.0, 1.0]).unwrap();
        assert!((p[0] - 3.0).abs() < TOL && (p[1] - 4.0).abs() < TOL);
    }

    #[test]
    fn vandermonde_basis() {
        let v = vandermonde(&[0.0, 2.0], 2, true).unwrap();
        assert_eq!(v, vec![vec![1.0, 0.0, 0.0], vec![1.0, 2.0, 4.0]]);
    }

    #[test]
    fn ols_pulled_by_outlier() {
        let b = fit_ols(&x_huber(), &Y_HUBER).unwrap();
        assert!((b[0] - OLS_HUBER_COEF[0]).abs() < TOL);
        assert!((b[1] - OLS_HUBER_COEF[1]).abs() < TOL);
    }

    #[test]
    fn huber_recovers_clean_line() {
        let res = fit_huber(&x_huber(), &Y_HUBER, 1.345, 100, 1e-10).unwrap();
        assert!((res.coef[0] - 1.0).abs() < 1e-7);
        assert!((res.coef[1] - 2.0).abs() < 1e-7);
        assert!(res.converged);
    }

    #[test]
    fn wls_exact() {
        let b = fit_wls(&x_wls(), &Y_WLS, &W_WLS).unwrap();
        assert!((b[0] - 1.0).abs() < TOL && (b[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn gls_correlated() {
        let b = fit_gls(&x_gls(), &Y_GLS, &cov_gls()).unwrap();
        assert!((b[0] - GLS_COEF[0]).abs() < TOL);
        assert!((b[1] - GLS_COEF[1]).abs() < TOL);
    }

    #[test]
    fn quantile_median_robust() {
        let b = fit_quantile(&x_q(), &Y_Q, 0.5, 200, 1e-10, 1e-6).unwrap();
        assert!((b[0] - 1.0).abs() < 1e-5);
        assert!((b[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn basis_expansion_fits_parabola() {
        let xs = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 - x + 3.0 * x * x).collect();
        let phi = vandermonde(&xs, 2, true).unwrap();
        let beta = fit_ols(&phi, &ys).unwrap();
        assert!((beta[0] - 2.0).abs() < TOL);
        assert!((beta[1] + 1.0).abs() < TOL);
        assert!((beta[2] - 3.0).abs() < TOL);
    }

    #[test]
    fn length_mismatch_errors() {
        assert!(fit_ols(&[vec![1.0], vec![1.0]], &[1.0]).is_err());
    }

    #[test]
    fn negative_weights_error() {
        assert!(fit_wls(&x_wls(), &Y_WLS, &[1.0, -1.0, 1.0, 1.0]).is_err());
    }

    #[test]
    fn bad_tau_errors() {
        assert!(fit_quantile(&x_wls(), &Y_WLS, 1.5, 200, 1e-10, 1e-6).is_err());
    }

    #[test]
    fn gls_wrong_cov_shape_errors() {
        assert!(fit_gls(&x_gls(), &Y_GLS, &[vec![1.0, 0.0], vec![0.0, 1.0]]).is_err());
    }
}
