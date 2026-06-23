//! Kernel PCA (kernel principal component analysis) from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch137_kernel_pca` and the Julia module
//! `AIInAction.Ch137KernelPca`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! This is a std-only implementation. The centered `n x n` kernel matrix is
//! diagonalized by a cyclic Jacobi eigensolver, which is robust for the modest
//! sample counts used in the book and avoids any external linear-algebra
//! dependency.
//!
//! Supported kernels: `Kernel::Linear`, `Kernel::Poly { gamma, coef0, degree }`,
//! and `Kernel::Rbf { gamma }`. Sign convention: each coefficient vector is flipped
//! so that its largest-magnitude entry is positive; tied-magnitude components have
//! a numerically arbitrary overall sign (compared up to sign in the tests).

/// A kernel function specification.
#[derive(Clone, Copy, Debug)]
pub enum Kernel {
    /// `k(x, y) = x . y`.
    Linear,
    /// `k(x, y) = (gamma * x.y + coef0)^degree`.
    Poly { gamma: f64, coef0: f64, degree: f64 },
    /// `k(x, y) = exp(-gamma * ||x - y||^2)`.
    Rbf { gamma: f64 },
}

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
}

/// The fitted state of a Kernel PCA model.
#[derive(Clone, Debug)]
pub struct KernelPcaResult {
    /// Training inputs (needed for out-of-sample projection), `n x d` row-major.
    pub x_fit: Vec<Vec<f64>>,
    /// The kernel used for fitting.
    pub kernel: Kernel,
    /// Normalized coefficient vectors as columns: `n` rows of length `n_components`.
    pub alphas: Vec<Vec<f64>>,
    /// Kernel-matrix eigenvalues `mu_k` of retained components, descending.
    pub eigenvalues: Vec<f64>,
    /// Fraction of total feature-space variance per component.
    pub explained_variance_ratio: Vec<f64>,
    /// Per-row means of the training Gram matrix, length `n`.
    pub row_means: Vec<f64>,
    /// Grand mean of the training Gram matrix.
    pub total_mean: f64,
}

impl KernelPcaResult {
    pub fn n_components(&self) -> usize {
        self.eigenvalues.len()
    }
    pub fn n_train(&self) -> usize {
        self.x_fit.len()
    }
}

/// Evaluates the kernel between two feature vectors of equal length.
fn kernel_eval(a: &[f64], b: &[f64], kernel: Kernel) -> f64 {
    match kernel {
        Kernel::Linear => a.iter().zip(b).map(|(x, y)| x * y).sum(),
        Kernel::Poly {
            gamma,
            coef0,
            degree,
        } => {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            (gamma * dot + coef0).powf(degree)
        }
        Kernel::Rbf { gamma } => {
            let sq: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            (-gamma * sq).exp()
        }
    }
}

/// Computes the cross-kernel matrix `K[i][t] = k(a_i, b_t)` as `n_a x n_b` rows.
pub fn kernel_matrix(a: &[Vec<f64>], b: &[Vec<f64>], kernel: Kernel) -> Result<Vec<Vec<f64>>, String> {
    if a.is_empty() || b.is_empty() {
        return Err("kernel_matrix requires non-empty point sets".to_string());
    }
    let d = a[0].len();
    if b[0].len() != d {
        return Err(format!("A has {} features but B has {}", d, b[0].len()));
    }
    if let Kernel::Rbf { gamma } = kernel {
        if gamma <= 0.0 {
            return Err(format!("rbf gamma must be positive, got {}", gamma));
        }
    }
    let mut out = vec![vec![0.0f64; b.len()]; a.len()];
    for (i, ai) in a.iter().enumerate() {
        for (t, bt) in b.iter().enumerate() {
            out[i][t] = kernel_eval(ai, bt, kernel);
        }
    }
    Ok(out)
}

/// Symmetric eigendecomposition via the cyclic Jacobi method.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvector `k` is column `k` of the
/// returned row-major `n x n` matrix, sorted by descending eigenvalue.
fn jacobi_eigen(a_in: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let idx = |i: usize, j: usize| i * n + j;

    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[idx(p, q)] * a[idx(p, q)];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[idx(p, q)];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[idx(p, p)];
                let aqq = a[idx(q, q)];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[idx(k, p)];
                    let akq = a[idx(k, q)];
                    a[idx(k, p)] = c * akp - s * akq;
                    a[idx(k, q)] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[idx(p, k)];
                    let aqk = a[idx(q, k)];
                    a[idx(p, k)] = c * apk - s * aqk;
                    a[idx(q, k)] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[idx(k, p)];
                    let vkq = v[idx(k, q)];
                    v[idx(k, p)] = c * vkp - s * vkq;
                    v[idx(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }

    let eigvals: Vec<f64> = (0..n).map(|i| a[idx(i, i)]).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| eigvals[y].partial_cmp(&eigvals[x]).unwrap());

    let sorted_vals: Vec<f64> = order.iter().map(|&k| eigvals[k]).collect();
    let mut sorted_vecs = vec![0.0f64; n * n];
    for (k, &col) in order.iter().enumerate() {
        for i in 0..n {
            sorted_vecs[idx(i, k)] = v[idx(i, col)];
        }
    }
    (sorted_vals, sorted_vecs)
}

/// Flips each coefficient column so its largest-magnitude entry is positive.
fn fix_signs(alphas: &mut [Vec<f64>], k: usize) {
    let n = alphas.len();
    for comp in 0..k {
        let mut best_idx = 0usize;
        let mut best = 0.0f64;
        for i in 0..n {
            if alphas[i][comp].abs() > best {
                best = alphas[i][comp].abs();
                best_idx = i;
            }
        }
        if alphas[best_idx][comp] < 0.0 {
            for row in alphas.iter_mut() {
                row[comp] = -row[comp];
            }
        }
    }
}

const EIGENVALUE_TOL: f64 = 1e-12;

/// Fits a Kernel PCA model to `x`.
///
/// `n_components = None` retains every strictly-positive component, capped at
/// `n - 1` (centering removes one degree of freedom). An explicit value must lie in
/// `[1, n - 1]` and each retained eigenvalue must be positive.
pub fn fit_kernel_pca(
    x: &Matrix,
    n_components: Option<usize>,
    kernel: Kernel,
) -> Result<KernelPcaResult, String> {
    let n = x.rows;
    let d = x.cols;
    if n < 2 {
        return Err(format!("need at least 2 samples for Kernel PCA, got {}", n));
    }
    if d < 1 {
        return Err("X must have at least one feature".to_string());
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err("X contains non-finite values (nan or inf)".to_string());
    }
    let max_components = n - 1;

    // Materialize training rows for kernel evaluations.
    let x_rows: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..d).map(|j| x.get(i, j)).collect())
        .collect();

    // Gram matrix.
    let k_mat = kernel_matrix(&x_rows, &x_rows, kernel)?;

    // Row means and grand mean.
    let mut row_means = vec![0.0f64; n];
    let mut total = 0.0f64;
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += k_mat[i][j];
        }
        row_means[i] = s / n as f64;
        total += s;
    }
    let total_mean = total / (n as f64 * n as f64);

    // Centered kernel: K_tilde = H K H.
    let mut kt = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            kt[i * n + j] = k_mat[i][j] - row_means[i] - row_means[j] + total_mean;
        }
    }
    // Symmetrize before the symmetric eigensolver.
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (kt[i * n + j] + kt[j * n + i]);
            kt[i * n + j] = avg;
            kt[j * n + i] = avg;
        }
    }

    let (eigvals, eigvecs) = jacobi_eigen(&kt, n);

    let total_var: f64 = eigvals.iter().map(|&e| e.max(0.0)).sum();
    if total_var <= 0.0 {
        return Err("kernel matrix has no positive variance; Kernel PCA is undefined".to_string());
    }

    let n_positive = eigvals[..max_components]
        .iter()
        .filter(|&&e| e > EIGENVALUE_TOL)
        .count();

    let k = match n_components {
        None => {
            if n_positive < 1 {
                return Err(
                    "kernel matrix has no positive variance; Kernel PCA is undefined".to_string(),
                );
            }
            n_positive
        }
        Some(v) => {
            if v < 1 || v > max_components {
                return Err(format!(
                    "n_components must be in [1, {}] for {} samples, got {}",
                    max_components, n, v
                ));
            }
            v
        }
    };

    let mu: Vec<f64> = eigvals[..k].to_vec();
    if let Some(bad) = mu.iter().position(|&e| e <= EIGENVALUE_TOL) {
        return Err(format!(
            "component {} has non-positive eigenvalue {:.3e}; request fewer components or a different kernel",
            bad, mu[bad]
        ));
    }

    // alpha^k = (unit eigenvector column k) / sqrt(mu_k); store as rows of length k.
    let mut alphas = vec![vec![0.0f64; k]; n];
    for comp in 0..k {
        let scale = mu[comp].sqrt();
        for i in 0..n {
            alphas[i][comp] = eigvecs[i * n + comp] / scale;
        }
    }
    fix_signs(&mut alphas, k);

    let explained_variance_ratio: Vec<f64> = mu.iter().map(|&e| e / total_var).collect();

    Ok(KernelPcaResult {
        x_fit: x_rows,
        kernel,
        alphas,
        eigenvalues: mu,
        explained_variance_ratio,
        row_means,
        total_mean,
    })
}

/// Projects new points `z` onto the fitted components, returning `n_z x n_components`.
///
/// Each test point is centered with the *training* row and grand means.
pub fn transform(model: &KernelPcaResult, z: &Matrix) -> Result<Vec<Vec<f64>>, String> {
    let n = model.n_train();
    let d = model.x_fit[0].len();
    if z.cols != d {
        return Err(format!("Z has {} features but model was fit on {}", z.cols, d));
    }
    if z.data.iter().any(|v| !v.is_finite()) {
        return Err("Z contains non-finite values (nan or inf)".to_string());
    }
    let k = model.n_components();
    let nz = z.rows;

    let z_rows: Vec<Vec<f64>> = (0..nz)
        .map(|t| (0..d).map(|j| z.get(t, j)).collect())
        .collect();

    // K_z[i][t] = k(x_i, z_t).
    let k_z = kernel_matrix(&model.x_fit, &z_rows, model.kernel)?;

    // Per-test-column means (mean over training rows).
    let mut col_means = vec![0.0f64; nz];
    for t in 0..nz {
        let mut s = 0.0;
        for i in 0..n {
            s += k_z[i][t];
        }
        col_means[t] = s / n as f64;
    }

    let mut out = vec![vec![0.0f64; k]; nz];
    for t in 0..nz {
        for comp in 0..k {
            let mut acc = 0.0;
            for i in 0..n {
                let centered = k_z[i][t] - model.row_means[i] - col_means[t] + model.total_mean;
                acc += centered * model.alphas[i][comp];
            }
            out[t][comp] = acc;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn fixture() -> Matrix {
        Matrix::from_rows(&[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![1.0, 2.0],
        ])
        .unwrap()
    }

    fn z_fixture() -> Matrix {
        Matrix::from_rows(&[vec![0.5, 0.5]]).unwrap()
    }

    const RBF: Kernel = Kernel::Rbf { gamma: 0.5 };
    const LINEAR: Kernel = Kernel::Linear;
    const TOL: f64 = 1e-9;

    #[test]
    fn rbf_eigenvalues_match_fixture() {
        let m = fit_kernel_pca(&fixture(), Some(2), RBF).unwrap();
        let expected: [f64; 2] = [1.2737124650823262, 0.8646647167633879];
        assert!((m.eigenvalues[0] - expected[0]).abs() < TOL);
        assert!((m.eigenvalues[1] - expected[1]).abs() < TOL);
    }

    #[test]
    fn rbf_explained_variance_ratio_matches_fixture() {
        let m = fit_kernel_pca(&fixture(), Some(2), RBF).unwrap();
        let expected: [f64; 2] = [0.42052544762205024, 0.28547535415413783];
        assert!((m.explained_variance_ratio[0] - expected[0]).abs() < TOL);
        assert!((m.explained_variance_ratio[1] - expected[1]).abs() < TOL);
    }

    #[test]
    fn rbf_first_alpha_matches_fixture() {
        let m = fit_kernel_pca(&fixture(), Some(2), RBF).unwrap();
        let expected: [f64; 6] = [
            0.5236557939308785,
            0.24722334224292025,
            0.24722334224291986,
            -0.16977678408044233,
            -0.4241628471681383,
            -0.42416284716813824,
        ];
        for i in 0..6 {
            assert!((m.alphas[i][0] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn rbf_train_projection_first_component_matches_fixture() {
        let m = fit_kernel_pca(&fixture(), Some(2), RBF).unwrap();
        let proj = transform(&m, &fixture()).unwrap();
        let expected: [f64; 6] = [
            0.666986912142342,
            0.31489145267412133,
            0.3148914526741211,
            -0.21624680616484987,
            -0.5402615056628672,
            -0.5402615056628672,
        ];
        for i in 0..6 {
            assert!((proj[i][0] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn rbf_train_projection_second_component_up_to_sign() {
        let m = fit_kernel_pca(&fixture(), Some(2), RBF).unwrap();
        let proj = transform(&m, &fixture()).unwrap();
        let expected_abs: [f64; 6] = [
            0.0,
            0.46493674751609687,
            0.4649367475160969,
            0.0,
            0.4649367475160967,
            0.46493674751609687,
        ];
        for i in 0..6 {
            assert!((proj[i][1].abs() - expected_abs[i]).abs() < TOL);
        }
    }

    #[test]
    fn rbf_out_of_sample_projection_matches_fixture() {
        let m = fit_kernel_pca(&fixture(), Some(2), RBF).unwrap();
        let pz = transform(&m, &z_fixture()).unwrap();
        assert!((pz[0][0] - 0.3931535807027422).abs() < TOL);
    }

    #[test]
    fn train_projection_shortcut_equals_mu_times_alpha() {
        let m = fit_kernel_pca(&fixture(), Some(2), RBF).unwrap();
        let proj = transform(&m, &fixture()).unwrap();
        for i in 0..6 {
            for c in 0..2 {
                let expected = m.eigenvalues[c] * m.alphas[i][c];
                assert!((proj[i][c] - expected).abs() < TOL);
            }
        }
    }

    #[test]
    fn linear_kernel_eigenvalues_match_fixture() {
        let m = fit_kernel_pca(&fixture(), Some(2), LINEAR).unwrap();
        let expected: [f64; 2] = [3.666666666666668, 1.9999999999999991];
        assert!((m.eigenvalues[0] - expected[0]).abs() < TOL);
        assert!((m.eigenvalues[1] - expected[1]).abs() < TOL);
    }

    #[test]
    fn linear_kernel_projection_matches_fixture() {
        let m = fit_kernel_pca(&fixture(), Some(2), LINEAR).unwrap();
        let proj = transform(&m, &fixture()).unwrap();
        let expected: [f64; 6] = [
            1.1785113019775793,
            0.47140452079103184,
            0.4714045207910317,
            -0.23570226039551584,
            -0.9428090415820635,
            -0.9428090415820635,
        ];
        for i in 0..6 {
            assert!((proj[i][0] - expected[i]).abs() < TOL);
        }
        let evr: [f64; 2] = [0.6470588235294118, 0.352941176470588];
        assert!((m.explained_variance_ratio[0] - evr[0]).abs() < TOL);
        assert!((m.explained_variance_ratio[1] - evr[1]).abs() < TOL);
    }

    #[test]
    fn alpha_normalization_holds() {
        let m = fit_kernel_pca(&fixture(), Some(2), RBF).unwrap();
        for c in 0..2 {
            let mut norm_sq = 0.0;
            for i in 0..6 {
                norm_sq += m.alphas[i][c] * m.alphas[i][c];
            }
            assert!((norm_sq - 1.0 / m.eigenvalues[c]).abs() < TOL);
        }
    }

    #[test]
    fn default_n_components_respects_kernel_rank() {
        let m = fit_kernel_pca(&fixture(), None, LINEAR).unwrap();
        assert_eq!(m.n_components(), 2);
    }

    #[test]
    fn default_n_components_for_full_rank_kernel() {
        let m = fit_kernel_pca(&fixture(), None, RBF).unwrap();
        assert_eq!(m.n_components(), 5);
    }

    #[test]
    fn too_few_samples_errors() {
        let x = Matrix::from_rows(&[vec![1.0, 2.0]]).unwrap();
        assert!(fit_kernel_pca(&x, None, LINEAR).is_err());
    }

    #[test]
    fn bad_n_components_errors() {
        assert!(fit_kernel_pca(&fixture(), Some(6), LINEAR).is_err());
    }

    #[test]
    fn rbf_negative_gamma_errors() {
        let bad = Kernel::Rbf { gamma: -1.0 };
        assert!(fit_kernel_pca(&fixture(), Some(2), bad).is_err());
    }

    #[test]
    fn transform_feature_mismatch_errors() {
        let m = fit_kernel_pca(&fixture(), Some(2), RBF).unwrap();
        let bad = Matrix::from_rows(&[vec![1.0, 2.0, 3.0]]).unwrap();
        assert!(transform(&m, &bad).is_err());
    }
}
