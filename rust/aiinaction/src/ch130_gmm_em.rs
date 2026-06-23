//! Expectation-Maximization for Gaussian mixture models.
//!
//! Mirrors the Python module `aiinaction.ch130_gmm_em` and the Julia module
//! `AIInAction.Ch130GmmEm`. The shared fixtures in the tests below match the
//! Python/Julia suites, which keeps the three libraries at parity.
//!
//! Data is represented as `&[Vec<f64>]` (a slice of length-`d` points); means as
//! `Vec<Vec<f64>>` (K points); covariances as `Vec<Vec<Vec<f64>>>` (K matrices,
//! each `d` by `d`). The implementation is `std`-only: it includes a small
//! Gauss-Jordan routine for the covariance determinant and inverse.

/// Parameters of a K-component, d-dimensional Gaussian mixture.
#[derive(Clone, Debug)]
pub struct GmmParams {
    pub weights: Vec<f64>,
    pub means: Vec<Vec<f64>>,
    pub covariances: Vec<Vec<Vec<f64>>>,
}

/// Outcome of [`fit_gmm`].
#[derive(Clone, Debug)]
pub struct GmmResult {
    pub params: GmmParams,
    pub responsibilities: Vec<Vec<f64>>,
    pub log_likelihood: f64,
    pub n_iter: usize,
    pub converged: bool,
    pub history: Vec<f64>,
}

/// Determinant and inverse of a square matrix via Gauss-Jordan elimination with
/// partial pivoting. Returns `Err` if the matrix is singular.
fn det_inv(m: &[Vec<f64>]) -> Result<(f64, Vec<Vec<f64>>), String> {
    let d = m.len();
    for row in m {
        if row.len() != d {
            return Err(format!("matrix must be square, got {}x{}", d, row.len()));
        }
    }
    // Augmented [m | I].
    let mut a: Vec<Vec<f64>> = (0..d)
        .map(|i| {
            let mut row = m[i].clone();
            for j in 0..d {
                row.push(if i == j { 1.0 } else { 0.0 });
            }
            row
        })
        .collect();
    let mut det = 1.0_f64;
    for col in 0..d {
        // Partial pivot.
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..d {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                pivot = r;
            }
        }
        if a[pivot][col] == 0.0 {
            return Err("matrix is singular".to_string());
        }
        if pivot != col {
            a.swap(pivot, col);
            det = -det;
        }
        let p = a[col][col];
        det *= p;
        for j in 0..(2 * d) {
            a[col][j] /= p;
        }
        for r in 0..d {
            if r != col {
                let factor = a[r][col];
                if factor != 0.0 {
                    for j in 0..(2 * d) {
                        a[r][j] -= factor * a[col][j];
                    }
                }
            }
        }
    }
    let inv: Vec<Vec<f64>> = (0..d).map(|i| a[i][d..(2 * d)].to_vec()).collect();
    Ok((det, inv))
}

fn validate_params(params: &GmmParams, d: usize) -> Result<(), String> {
    let k = params.weights.len();
    if params.weights.iter().any(|&w| w < 0.0) {
        return Err("weights must be nonnegative".to_string());
    }
    let total: f64 = params.weights.iter().sum();
    if (total - 1.0).abs() > 1e-8 {
        return Err(format!("weights must sum to 1, got {}", total));
    }
    if params.means.len() != k || params.means.iter().any(|m| m.len() != d) {
        return Err(format!("means must have shape ({}, {})", k, d));
    }
    if params.covariances.len() != k
        || params
            .covariances
            .iter()
            .any(|c| c.len() != d || c.iter().any(|r| r.len() != d))
    {
        return Err(format!("covariances must have shape ({}, {}, {})", k, d, d));
    }
    Ok(())
}

/// Density of a multivariate normal at the point `x`.
pub fn gaussian_pdf(x: &[f64], mean: &[f64], cov: &[Vec<f64>]) -> Result<f64, String> {
    let d = x.len();
    if mean.len() != d {
        return Err(format!("mean length {} != x length {}", mean.len(), d));
    }
    if cov.len() != d || cov.iter().any(|r| r.len() != d) {
        return Err(format!("cov must have shape ({}, {})", d, d));
    }
    let (det, inv) = det_inv(cov)?;
    if det <= 0.0 {
        return Err(format!("covariance must be positive definite, det={}", det));
    }
    let diff: Vec<f64> = (0..d).map(|i| x[i] - mean[i]).collect();
    // quad = diff^T inv diff.
    let mut quad = 0.0;
    for i in 0..d {
        let mut s = 0.0;
        for j in 0..d {
            s += inv[i][j] * diff[j];
        }
        quad += diff[i] * s;
    }
    let norm = (((2.0 * std::f64::consts::PI).powi(d as i32)) * det).sqrt();
    Ok((-0.5 * quad).exp() / norm)
}

/// Compute responsibilities `gamma[n][k]` given current parameters.
pub fn e_step(x: &[Vec<f64>], params: &GmmParams) -> Result<Vec<Vec<f64>>, String> {
    if x.is_empty() {
        return Err("x must be non-empty".to_string());
    }
    let d = x[0].len();
    validate_params(params, d)?;
    let k = params.weights.len();
    let n = x.len();
    let mut gamma = vec![vec![0.0; k]; n];
    for ni in 0..n {
        let mut row_sum = 0.0;
        for ki in 0..k {
            let p = params.weights[ki]
                * gaussian_pdf(&x[ni], &params.means[ki], &params.covariances[ki])?;
            gamma[ni][ki] = p;
            row_sum += p;
        }
        if row_sum <= 0.0 {
            return Err(format!(
                "data point {} has zero density under all components",
                ni
            ));
        }
        for ki in 0..k {
            gamma[ni][ki] /= row_sum;
        }
    }
    Ok(gamma)
}

/// Weighted re-estimation of mixture parameters from responsibilities.
pub fn m_step(
    x: &[Vec<f64>],
    responsibilities: &[Vec<f64>],
    reg_covar: f64,
) -> Result<GmmParams, String> {
    if x.is_empty() {
        return Err("x must be non-empty".to_string());
    }
    if reg_covar < 0.0 {
        return Err(format!("reg_covar must be nonnegative, got {}", reg_covar));
    }
    let n = x.len();
    let d = x[0].len();
    if responsibilities.len() != n {
        return Err(format!(
            "responsibilities has {} rows but x has {} points",
            responsibilities.len(),
            n
        ));
    }
    let k = responsibilities[0].len();
    let mut nk = vec![0.0; k];
    for row in responsibilities {
        for ki in 0..k {
            nk[ki] += row[ki];
        }
    }
    if nk.iter().any(|&v| v <= 0.0) {
        return Err("a component has zero effective count; cannot re-estimate".to_string());
    }
    let weights: Vec<f64> = nk.iter().map(|&v| v / n as f64).collect();
    let mut means = vec![vec![0.0; d]; k];
    let mut covs = vec![vec![vec![0.0; d]; d]; k];
    for ki in 0..k {
        for ni in 0..n {
            for j in 0..d {
                means[ki][j] += responsibilities[ni][ki] * x[ni][j];
            }
        }
        for j in 0..d {
            means[ki][j] /= nk[ki];
        }
        for ni in 0..n {
            let diff: Vec<f64> = (0..d).map(|j| x[ni][j] - means[ki][j]).collect();
            let g = responsibilities[ni][ki];
            for a in 0..d {
                for b in 0..d {
                    covs[ki][a][b] += g * diff[a] * diff[b];
                }
            }
        }
        for a in 0..d {
            for b in 0..d {
                covs[ki][a][b] /= nk[ki];
            }
            covs[ki][a][a] += reg_covar;
        }
    }
    Ok(GmmParams {
        weights,
        means,
        covariances: covs,
    })
}

/// Incomplete-data log-likelihood of the data under the mixture.
pub fn log_likelihood(x: &[Vec<f64>], params: &GmmParams) -> Result<f64, String> {
    if x.is_empty() {
        return Err("x must be non-empty".to_string());
    }
    let d = x[0].len();
    validate_params(params, d)?;
    let k = params.weights.len();
    let mut total = 0.0;
    for (ni, xi) in x.iter().enumerate() {
        let mut mix = 0.0;
        for ki in 0..k {
            mix += params.weights[ki]
                * gaussian_pdf(xi, &params.means[ki], &params.covariances[ki])?;
        }
        if mix <= 0.0 {
            return Err(format!("data point {} has zero mixture density", ni));
        }
        total += mix.ln();
    }
    Ok(total)
}

/// Run EM to convergence from explicit initial parameters.
pub fn fit_gmm(
    x: &[Vec<f64>],
    init: &GmmParams,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
) -> Result<GmmResult, String> {
    if x.is_empty() {
        return Err("x must be non-empty".to_string());
    }
    if max_iter == 0 {
        return Err("max_iter must be positive".to_string());
    }
    if tol < 0.0 {
        return Err(format!("tol must be nonnegative, got {}", tol));
    }
    validate_params(init, x[0].len())?;

    let mut params = init.clone();
    let mut gamma = e_step(x, &params)?;
    let mut prev_ll = log_likelihood(x, &params)?;
    let mut history = vec![prev_ll];
    let mut converged = false;
    let mut n_iter = 0;
    for _ in 0..max_iter {
        params = m_step(x, &gamma, reg_covar)?;
        gamma = e_step(x, &params)?;
        let ll = log_likelihood(x, &params)?;
        history.push(ll);
        n_iter += 1;
        if (ll - prev_ll).abs() < tol {
            converged = true;
            prev_ll = ll;
            break;
        }
        prev_ll = ll;
    }
    Ok(GmmResult {
        params,
        responsibilities: gamma,
        log_likelihood: prev_ll,
        n_iter,
        converged,
        history,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn data() -> Vec<Vec<f64>> {
        vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![1.0],
            vec![9.0],
            vec![10.0],
            vec![11.0],
            vec![10.0],
        ]
    }

    fn init() -> GmmParams {
        GmmParams {
            weights: vec![0.5, 0.5],
            means: vec![vec![0.0], vec![8.0]],
            covariances: vec![vec![vec![1.0]], vec![vec![1.0]]],
        }
    }

    #[test]
    fn gaussian_pdf_standard_normal() {
        let v = gaussian_pdf(&[0.0], &[0.0], &[vec![1.0]]).unwrap();
        assert!((v - 1.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < TOL);
    }

    #[test]
    fn gaussian_pdf_2d() {
        let v = gaussian_pdf(
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .unwrap();
        assert!((v - 1.0 / (2.0 * std::f64::consts::PI)).abs() < TOL);
    }

    #[test]
    fn e_step_rows_sum_to_one() {
        let g = e_step(&data(), &init()).unwrap();
        assert_eq!(g.len(), 8);
        for row in &g {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn e_step_fixture_first_row() {
        let g = e_step(&data(), &init()).unwrap();
        assert!((g[0][0] - 0.9999999999999873).abs() < TOL);
        assert!((g[0][1] - 1.2664165549094016e-14).abs() < TOL);
    }

    #[test]
    fn log_likelihood_fixture_initial() {
        let ll = log_likelihood(&data(), &init()).unwrap();
        assert!((ll - (-24.89668559750626)).abs() < TOL);
    }

    #[test]
    fn m_step_recovers_two_clusters() {
        let g = e_step(&data(), &init()).unwrap();
        let u = m_step(&data(), &g, 0.0).unwrap();
        assert!((u.weights[0] - 0.5).abs() < 1e-6);
        assert!((u.weights[1] - 0.5).abs() < 1e-6);
        assert!((u.means[0][0] - 1.0).abs() < 1e-6);
        assert!((u.means[1][0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn fit_gmm_converges_to_fixture() {
        let r = fit_gmm(&data(), &init(), 200, 1e-10, 0.0).unwrap();
        assert!(r.converged);
        // Monotonic nondecreasing log-likelihood.
        for w in r.history.windows(2) {
            assert!(w[1] >= w[0] - 1e-12);
        }
        assert!((r.log_likelihood - (-14.124096987877165)).abs() < 1e-7);
        let mut means: Vec<f64> = r.params.means.iter().map(|m| m[0]).collect();
        means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((means[0] - 1.0).abs() < 1e-6);
        assert!((means[1] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn gaussian_pdf_bad_cov_errors() {
        assert!(gaussian_pdf(&[0.0], &[0.0], &[vec![0.0]]).is_err());
    }

    #[test]
    fn e_step_weights_not_normalized_errors() {
        let bad = GmmParams {
            weights: vec![0.3, 0.3],
            means: vec![vec![0.0], vec![8.0]],
            covariances: vec![vec![vec![1.0]], vec![vec![1.0]]],
        };
        assert!(e_step(&data(), &bad).is_err());
    }

    #[test]
    fn m_step_negative_reg_errors() {
        let g = e_step(&data(), &init()).unwrap();
        assert!(m_step(&data(), &g, -1.0).is_err());
    }

    #[test]
    fn fit_gmm_bad_max_iter_errors() {
        assert!(fit_gmm(&data(), &init(), 0, 1e-6, 0.0).is_err());
    }
}
