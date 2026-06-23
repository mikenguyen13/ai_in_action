//! Polynomial and basis-function regression with least squares and ridge (Rust).
//!
//! Mirrors the Python module `aiinaction.ch083_basis_regression` and the Julia
//! module `AIInAction.Ch083BasisRegression`. The shared fixtures in the tests
//! below match the Python/Julia suites, which is what keeps the three libraries
//! at parity.
//!
//! Design matrices are `Vec<Vec<f64>>` in row-major order. The ridge solve uses
//! the normal equations with Gaussian elimination (partial pivoting); effective
//! degrees of freedom are computed from the eigenvalues of the Gram matrix via a
//! cyclic Jacobi sweep. Everything is `std`-only.

/// Build the polynomial design matrix with columns `1, x, x^2, ..., x^degree`.
pub fn polynomial_design(x: &[f64], degree: usize) -> Result<Vec<Vec<f64>>, String> {
    validate_1d(x, "x")?;
    let mut phi = Vec::with_capacity(x.len());
    for &xi in x {
        let mut row = Vec::with_capacity(degree + 1);
        let mut power = 1.0;
        for _ in 0..=degree {
            row.push(power);
            power *= xi;
        }
        phi.push(row);
    }
    Ok(phi)
}

/// Build a Gaussian radial-basis-function design matrix.
///
/// Column `j` is `exp(-(x - c_j)^2 / (2 * width^2))`; with `include_bias` a
/// leading column of ones is prepended.
pub fn rbf_design(
    x: &[f64],
    centers: &[f64],
    width: f64,
    include_bias: bool,
) -> Result<Vec<Vec<f64>>, String> {
    validate_1d(x, "x")?;
    validate_1d(centers, "centers")?;
    if !width.is_finite() || width <= 0.0 {
        return Err(format!("width must be a positive finite number, got {width}"));
    }
    let denom = 2.0 * width * width;
    let mut phi = Vec::with_capacity(x.len());
    for &xi in x {
        let mut row = Vec::with_capacity(centers.len() + include_bias as usize);
        if include_bias {
            row.push(1.0);
        }
        for &c in centers {
            let d = xi - c;
            row.push((-(d * d) / denom).exp());
        }
        phi.push(row);
    }
    Ok(phi)
}

/// Solve the ridge least-squares problem `(phi^T phi + penalty I) beta = phi^T y`.
pub fn fit_ridge(phi: &[Vec<f64>], y: &[f64], penalty: f64) -> Result<Vec<f64>, String> {
    let (n, m) = dims(phi)?;
    if y.len() != n {
        return Err(format!(
            "length mismatch: phi has {} rows but y has {}",
            n,
            y.len()
        ));
    }
    if !penalty.is_finite() || penalty < 0.0 {
        return Err(format!(
            "penalty must be a non-negative finite number, got {penalty}"
        ));
    }
    // Gram = phi^T phi + penalty I, rhs = phi^T y.
    let mut gram = vec![vec![0.0; m]; m];
    let mut rhs = vec![0.0; m];
    for a in 0..m {
        for (i, row) in phi.iter().enumerate() {
            rhs[a] += row[a] * y[i];
        }
        for b in 0..m {
            let mut s = 0.0;
            for row in phi.iter() {
                s += row[a] * row[b];
            }
            gram[a][b] = s;
        }
        gram[a][a] += penalty;
    }
    solve_spd(&mut gram, &mut rhs)
}

/// Evaluate fitted coefficients on a design matrix: `phi @ beta`.
pub fn predict(phi: &[Vec<f64>], beta: &[f64]) -> Result<Vec<f64>, String> {
    let (_n, m) = dims(phi)?;
    if beta.len() != m {
        return Err(format!(
            "shape mismatch: phi has {} columns but beta has {}",
            m,
            beta.len()
        ));
    }
    Ok(phi
        .iter()
        .map(|row| row.iter().zip(beta).map(|(p, b)| p * b).sum())
        .collect())
}

/// Effective degrees of freedom of a ridge fit:
/// `sum_j lambda_j / (lambda_j + penalty)` over eigenvalues of the Gram matrix.
pub fn effective_dof(phi: &[Vec<f64>], penalty: f64) -> Result<f64, String> {
    let (_n, m) = dims(phi)?;
    if !penalty.is_finite() || penalty < 0.0 {
        return Err(format!(
            "penalty must be a non-negative finite number, got {penalty}"
        ));
    }
    let mut gram = vec![vec![0.0; m]; m];
    for a in 0..m {
        for b in 0..m {
            let mut s = 0.0;
            for row in phi.iter() {
                s += row[a] * row[b];
            }
            gram[a][b] = s;
        }
    }
    let eig = jacobi_eigenvalues(&gram);
    Ok(eig.iter().map(|&l| l / (l + penalty)).sum())
}

// --- internal linear algebra (std only) ------------------------------------

fn validate_1d(v: &[f64], name: &str) -> Result<(), String> {
    if v.is_empty() {
        return Err(format!("{name} must be non-empty"));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn dims(phi: &[Vec<f64>]) -> Result<(usize, usize), String> {
    if phi.is_empty() {
        return Err("phi must be non-empty".to_string());
    }
    let m = phi[0].len();
    if m == 0 {
        return Err("phi must have at least one column".to_string());
    }
    if phi.iter().any(|r| r.len() != m) {
        return Err("phi rows must all have the same length".to_string());
    }
    Ok((phi.len(), m))
}

/// Solve a symmetric positive-definite system `a x = b` by Gaussian elimination
/// with partial pivoting. `a` and `b` are overwritten; returns the solution.
fn solve_spd(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<Vec<f64>, String> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot.
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                pivot = r;
            }
        }
        if best < 1e-300 {
            return Err(
                "normal equations are singular; increase penalty or reduce basis size".to_string(),
            );
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for r in (col + 1)..n {
            let factor = a[r][col] / a[col][col];
            if factor != 0.0 {
                for c in col..n {
                    a[r][c] -= factor * a[col][c];
                }
                b[r] -= factor * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Ok(x)
}

/// Eigenvalues of a symmetric matrix via the cyclic Jacobi method.
fn jacobi_eigenvalues(input: &[Vec<f64>]) -> Vec<f64> {
    let n = input.len();
    let mut a: Vec<Vec<f64>> = input.to_vec();
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..n).map(|i| a[i][i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const X: [f64; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
    const Y: [f64; 5] = [-7.0, -2.0, 1.0, 2.0, 1.0];
    const TOL: f64 = 1e-9;

    const EXPECTED_OLS: [f64; 3] = [1.0, 2.0, -1.0];
    const EXPECTED_RIDGE: [f64; 3] = [
        0.59090909090909061,
        1.8181818181818181,
        -0.8545454545454545,
    ];
    const EXPECTED_DOF_OLS: f64 = 3.0;
    const EXPECTED_DOF_RIDGE: f64 = 2.5363636363636362;

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < TOL, "{x} != {y}");
        }
    }

    #[test]
    fn polynomial_design_values() {
        let phi = polynomial_design(&X, 2).unwrap();
        assert_eq!(phi.len(), 5);
        assert_eq!(phi[0], vec![1.0, -2.0, 4.0]);
        assert_eq!(phi[2], vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn ols_recovers_exact_polynomial() {
        let phi = polynomial_design(&X, 2).unwrap();
        let beta = fit_ridge(&phi, &Y, 0.0).unwrap();
        close(&beta, &EXPECTED_OLS);
    }

    #[test]
    fn predict_reproduces_targets() {
        let phi = polynomial_design(&X, 2).unwrap();
        let beta = fit_ridge(&phi, &Y, 0.0).unwrap();
        close(&predict(&phi, &beta).unwrap(), &Y);
    }

    #[test]
    fn ridge_shrinks_coefficients() {
        let phi = polynomial_design(&X, 2).unwrap();
        let beta = fit_ridge(&phi, &Y, 1.0).unwrap();
        close(&beta, &EXPECTED_RIDGE);
    }

    #[test]
    fn effective_dof_matches() {
        let phi = polynomial_design(&X, 2).unwrap();
        assert!((effective_dof(&phi, 0.0).unwrap() - EXPECTED_DOF_OLS).abs() < TOL);
        assert!((effective_dof(&phi, 1.0).unwrap() - EXPECTED_DOF_RIDGE).abs() < TOL);
    }

    #[test]
    fn rbf_design_values() {
        let phi = rbf_design(&[-1.0, 0.0, 1.0], &[-1.0, 1.0], 1.0, true).unwrap();
        assert!((phi[0][0] - 1.0).abs() < TOL);
        assert!((phi[0][1] - 1.0).abs() < TOL);
        assert!((phi[0][2] - 0.1353352832366127).abs() < TOL);
        assert!((phi[1][1] - 0.6065306597126334).abs() < TOL);
    }

    #[test]
    fn negative_degree_is_usize_safe() {
        // degree is usize; the error path is the empty-input one.
        assert!(polynomial_design(&[], 2).is_err());
    }

    #[test]
    fn length_mismatch_errors() {
        let phi = polynomial_design(&X, 2).unwrap();
        assert!(fit_ridge(&phi, &[1.0, 2.0], 0.0).is_err());
    }

    #[test]
    fn negative_penalty_errors() {
        let phi = polynomial_design(&X, 2).unwrap();
        assert!(fit_ridge(&phi, &Y, -0.5).is_err());
    }

    #[test]
    fn nonpositive_width_errors() {
        assert!(rbf_design(&X, &[-1.0, 1.0], 0.0, true).is_err());
    }

    #[test]
    fn non_finite_input_errors() {
        assert!(polynomial_design(&[1.0, f64::NAN, 3.0], 2).is_err());
    }
}
