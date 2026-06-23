//! Batch Normalization forward and backward pass from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch201_batch_norm` and the Julia module
//! `AIInAction.Ch201BatchNorm`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! For a mini-batch `x` of shape `(m, d)`, the forward transform standardizes each
//! feature over the batch (population variance, `ddof = 0`) and then applies a
//! learnable scale `gamma` and shift `beta`:
//!
//! ```text
//! mu_j     = mean_i x_ij
//! var_j    = mean_i (x_ij - mu_j)^2
//! xhat_ij  = (x_ij - mu_j) / sqrt(var_j + eps)
//! y_ij     = gamma_j * xhat_ij + beta_j
//! ```
//!
//! This is a std-only implementation. Matrices are dense row-major `Vec<f64>`.

/// A dense row-major matrix of `f64`.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Builds a matrix from a slice of equal-length rows.
    pub fn from_rows(rows: &[Vec<f64>]) -> Result<Matrix, String> {
        if rows.is_empty() {
            return Err("X must have at least one row".to_string());
        }
        let cols = rows[0].len();
        if cols == 0 {
            return Err("X must have at least one feature".to_string());
        }
        let mut data = Vec::with_capacity(rows.len() * cols);
        for r in rows {
            if r.len() != cols {
                return Err("all rows must have the same length".to_string());
            }
            data.extend_from_slice(r);
        }
        Ok(Matrix {
            rows: rows.len(),
            cols,
            data,
        })
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    /// Returns the row-major data of this matrix as a slice of rows (for tests).
    pub fn rows_vec(&self) -> Vec<Vec<f64>> {
        (0..self.rows)
            .map(|i| self.data[i * self.cols..(i + 1) * self.cols].to_vec())
            .collect()
    }
}

/// Intermediate quantities saved by the forward pass for the backward pass.
#[derive(Clone, Debug)]
pub struct BatchNormCache {
    /// Standardized activations, shape `(m, d)`, row-major.
    pub x_hat: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    /// Per-feature `1 / sqrt(var + eps)`, length `d`.
    pub inv_std: Vec<f64>,
    /// The scale vector used in the forward pass, length `d`.
    pub gamma: Vec<f64>,
    /// Per-feature batch means, length `d`.
    pub mean: Vec<f64>,
    /// Per-feature batch variances (population, `ddof = 0`), length `d`.
    pub var: Vec<f64>,
}

fn check_finite_matrix(x: &Matrix, name: &str) -> Result<(), String> {
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err(format!("{} contains non-finite values (nan or inf)", name));
    }
    Ok(())
}

fn check_vector(v: &[f64], d: usize, name: &str) -> Result<(), String> {
    if v.len() != d {
        return Err(format!("{} has length {} but X has {} features", name, v.len(), d));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(format!("{} contains non-finite values (nan or inf)", name));
    }
    Ok(())
}

/// Training-time Batch Normalization forward pass.
///
/// Returns `(y, cache)` where `y` is the row-major `(m, d)` output and `cache`
/// holds the quantities needed by [`batch_norm_backward`].
pub fn batch_norm_forward(
    x: &Matrix,
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
) -> Result<(Vec<f64>, BatchNormCache), String> {
    if !(eps > 0.0) {
        return Err(format!("eps must be positive, got {}", eps));
    }
    check_finite_matrix(x, "X")?;
    let (m, d) = (x.rows, x.cols);
    check_vector(gamma, d, "gamma")?;
    check_vector(beta, d, "beta")?;

    let mut mean = vec![0.0f64; d];
    for i in 0..m {
        for j in 0..d {
            mean[j] += x.get(i, j);
        }
    }
    for mj in mean.iter_mut() {
        *mj /= m as f64;
    }

    let mut var = vec![0.0f64; d];
    for i in 0..m {
        for j in 0..d {
            let c = x.get(i, j) - mean[j];
            var[j] += c * c;
        }
    }
    for vj in var.iter_mut() {
        *vj /= m as f64;
    }

    let inv_std: Vec<f64> = var.iter().map(|&v| 1.0 / (v + eps).sqrt()).collect();

    let mut x_hat = vec![0.0f64; m * d];
    let mut y = vec![0.0f64; m * d];
    for i in 0..m {
        for j in 0..d {
            let xh = (x.get(i, j) - mean[j]) * inv_std[j];
            x_hat[i * d + j] = xh;
            y[i * d + j] = gamma[j] * xh + beta[j];
        }
    }

    let cache = BatchNormCache {
        x_hat,
        rows: m,
        cols: d,
        inv_std,
        gamma: gamma.to_vec(),
        mean,
        var,
    };
    Ok((y, cache))
}

/// Backward pass: gradients with respect to `X`, `gamma` and `beta`.
///
/// `dy` is the row-major upstream gradient of shape `(m, d)`. Returns
/// `(dx, dgamma, dbeta)` with `dx` row-major `(m, d)` and the parameter gradients
/// length `d`.
pub fn batch_norm_backward(
    dy: &[f64],
    cache: &BatchNormCache,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let (m, d) = (cache.rows, cache.cols);
    if dy.len() != m * d {
        return Err(format!(
            "dy has {} elements but cache was built for {}x{}",
            dy.len(),
            m,
            d
        ));
    }
    if dy.iter().any(|v| !v.is_finite()) {
        return Err("dy contains non-finite values (nan or inf)".to_string());
    }

    let mut dgamma = vec![0.0f64; d];
    let mut dbeta = vec![0.0f64; d];
    for i in 0..m {
        for j in 0..d {
            dgamma[j] += dy[i * d + j] * cache.x_hat[i * d + j];
            dbeta[j] += dy[i * d + j];
        }
    }

    // g = dy * gamma; then per-feature means of g and g * x_hat.
    let mut g_mean = vec![0.0f64; d];
    let mut gxhat_mean = vec![0.0f64; d];
    for i in 0..m {
        for j in 0..d {
            let g = dy[i * d + j] * cache.gamma[j];
            g_mean[j] += g;
            gxhat_mean[j] += g * cache.x_hat[i * d + j];
        }
    }
    for j in 0..d {
        g_mean[j] /= m as f64;
        gxhat_mean[j] /= m as f64;
    }

    let mut dx = vec![0.0f64; m * d];
    for i in 0..m {
        for j in 0..d {
            let g = dy[i * d + j] * cache.gamma[j];
            dx[i * d + j] =
                cache.inv_std[j] * (g - g_mean[j] - cache.x_hat[i * d + j] * gxhat_mean[j]);
        }
    }

    Ok((dx, dgamma, dbeta))
}

/// Inference-time Batch Normalization using frozen population statistics.
///
/// Applies `y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta` per
/// feature, returning a row-major `(m, d)` matrix.
pub fn batch_norm_inference(
    x: &Matrix,
    gamma: &[f64],
    beta: &[f64],
    running_mean: &[f64],
    running_var: &[f64],
    eps: f64,
) -> Result<Vec<f64>, String> {
    if !(eps > 0.0) {
        return Err(format!("eps must be positive, got {}", eps));
    }
    check_finite_matrix(x, "X")?;
    let (m, d) = (x.rows, x.cols);
    check_vector(gamma, d, "gamma")?;
    check_vector(beta, d, "beta")?;
    check_vector(running_mean, d, "running_mean")?;
    check_vector(running_var, d, "running_var")?;
    if running_var.iter().any(|&v| v < 0.0) {
        return Err("running_var must be non-negative".to_string());
    }

    let mut y = vec![0.0f64; m * d];
    for i in 0..m {
        for j in 0..d {
            y[i * d + j] =
                gamma[j] * (x.get(i, j) - running_mean[j]) / (running_var[j] + eps).sqrt()
                    + beta[j];
        }
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn fixture() -> Matrix {
        Matrix::from_rows(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![2.0, 0.0, 1.0],
        ])
        .unwrap()
    }

    const GAMMA: [f64; 3] = [1.0, 2.0, 0.5];
    const BETA: [f64; 3] = [0.0, 1.0, -1.0];
    const EPS: f64 = 1e-5;
    const TOL: f64 = 1e-9;

    fn dy() -> Vec<f64> {
        let rows: [[f64; 3]; 4] = [
            [0.1, -0.2, 0.3],
            [0.4, 0.5, -0.6],
            [-0.7, 0.8, 0.9],
            [1.0, -1.1, 1.2],
        ];
        rows.iter().flatten().copied().collect()
    }

    #[test]
    fn forward_mean_var_match_fixture() {
        let (_y, cache) = batch_norm_forward(&fixture(), &GAMMA, &BETA, EPS).unwrap();
        let exp_mean: [f64; 3] = [3.5, 3.75, 4.75];
        let exp_var: [f64; 3] = [5.25, 9.1875, 9.1875];
        for j in 0..3 {
            assert!((cache.mean[j] - exp_mean[j]).abs() < TOL);
            assert!((cache.var[j] - exp_var[j]).abs() < TOL);
        }
    }

    #[test]
    fn forward_xhat_matches_fixture() {
        let (_y, cache) = batch_norm_forward(&fixture(), &GAMMA, &BETA, EPS).unwrap();
        let exp: [f64; 3] = [-1.0910884120486357, -0.5773499549856541, -0.5773499549856541];
        for j in 0..3 {
            assert!((cache.x_hat[j] - exp[j]).abs() < TOL);
        }
    }

    #[test]
    fn forward_output_matches_fixture() {
        let (y, _cache) = batch_norm_forward(&fixture(), &GAMMA, &BETA, EPS).unwrap();
        let exp_row0: [f64; 3] = [-1.0910884120486357, -0.15469990997130822, -1.288674977492827];
        let exp_row3: [f64; 3] = [-0.6546530472291814, -1.4743569499385174, -1.6185892374846294];
        for j in 0..3 {
            assert!((y[j] - exp_row0[j]).abs() < TOL);
            assert!((y[3 * 3 + j] - exp_row3[j]).abs() < TOL);
        }
    }

    #[test]
    fn backward_param_grads_match_fixture() {
        let (_y, cache) = batch_norm_forward(&fixture(), &GAMMA, &BETA, EPS).unwrap();
        let (_dx, dgamma, dbeta) = batch_norm_backward(&dy(), &cache).unwrap();
        let exp_dgamma: [f64; 3] = [-1.745741459277817, 2.80427120993032, -0.6433328069840143];
        let exp_dbeta: [f64; 3] = [0.8, 0.0, 1.8];
        for j in 0..3 {
            assert!((dgamma[j] - exp_dgamma[j]).abs() < TOL);
            assert!((dbeta[j] - exp_dbeta[j]).abs() < TOL);
        }
    }

    #[test]
    fn backward_input_grad_matches_fixture() {
        let (_y, cache) = batch_norm_forward(&fixture(), &GAMMA, &BETA, EPS).unwrap();
        let (dx, _dgamma, _dbeta) = batch_norm_backward(&dy(), &cache).unwrap();
        let exp_row0: [f64; 3] =
            [-0.2514695048226982, 0.1351074538761989, -0.040061000612684985];
        let exp_row3: [f64; 3] =
            [0.2244527108511118, -0.1535117479685656, 0.09089478082556916];
        for j in 0..3 {
            assert!((dx[j] - exp_row0[j]).abs() < TOL);
            assert!((dx[3 * 3 + j] - exp_row3[j]).abs() < TOL);
        }
    }

    #[test]
    fn inference_matches_fixture() {
        let running_mean: [f64; 3] = [3.0, 4.0, 5.0];
        let running_var: [f64; 3] = [2.0, 3.0, 4.0];
        let y = batch_norm_inference(&fixture(), &GAMMA, &BETA, &running_mean, &running_var, EPS)
            .unwrap();
        let exp_row0: [f64; 3] =
            [-1.4142100268524473, -1.3093972277663308, -1.4999993750011718];
        for j in 0..3 {
            assert!((y[j] - exp_row0[j]).abs() < TOL);
        }
    }

    #[test]
    fn gamma_beta_recover_identity() {
        let eps = 1e-5;
        let (_y, cache) = batch_norm_forward(&fixture(), &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], eps)
            .unwrap();
        let g: Vec<f64> = cache.var.iter().map(|&v| (v + eps).sqrt()).collect();
        let b = cache.mean.clone();
        let (y, _c) = batch_norm_forward(&fixture(), &g, &b, eps).unwrap();
        let x = fixture();
        for i in 0..x.rows {
            for j in 0..x.cols {
                assert!((y[i * x.cols + j] - x.get(i, j)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn eps_must_be_positive() {
        assert!(batch_norm_forward(&fixture(), &GAMMA, &BETA, 0.0).is_err());
    }

    #[test]
    fn gamma_length_mismatch_errors() {
        assert!(batch_norm_forward(&fixture(), &[1.0, 2.0], &BETA, EPS).is_err());
    }

    #[test]
    fn backward_shape_mismatch_errors() {
        let (_y, cache) = batch_norm_forward(&fixture(), &GAMMA, &BETA, EPS).unwrap();
        let bad = vec![1.0, 2.0, 3.0];
        assert!(batch_norm_backward(&bad, &cache).is_err());
    }

    #[test]
    fn inference_negative_var_errors() {
        let running_mean: [f64; 3] = [3.0, 4.0, 5.0];
        let running_var: [f64; 3] = [-1.0, 1.0, 1.0];
        assert!(
            batch_norm_inference(&fixture(), &GAMMA, &BETA, &running_mean, &running_var, EPS)
                .is_err()
        );
    }
}
