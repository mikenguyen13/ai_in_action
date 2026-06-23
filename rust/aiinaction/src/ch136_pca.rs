//! Principal Component Analysis (PCA) from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch136_pca` and the Julia module
//! `AIInAction.Ch136Pca`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! This is a std-only implementation. The centered (and optionally scaled) data
//! matrix is decomposed by forming the small `d x d` covariance matrix and running
//! a cyclic Jacobi eigensolver on it, which is robust for the modest feature
//! counts used in the book and avoids any external linear-algebra dependency.
//!
//! Sign convention: each loading vector is flipped so that its largest-magnitude
//! entry is positive, removing the eigenvector sign ambiguity.

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

/// The fitted state of a PCA model.
#[derive(Clone, Debug)]
pub struct PcaResult {
    /// Per-feature training means, length `d`.
    pub mean: Vec<f64>,
    /// Per-feature scales applied after centering, length `d` (all 1.0 if unscaled).
    pub scale: Vec<f64>,
    /// Principal directions as rows: `n_components` rows of length `d`.
    pub components: Vec<Vec<f64>>,
    /// Variance `lambda_j = s_j^2 / (n - 1)` per retained component.
    pub explained_variance: Vec<f64>,
    /// Fraction of total variance per component.
    pub explained_variance_ratio: Vec<f64>,
    /// Whether `transform` rescales scores to unit variance.
    pub whiten: bool,
}

impl PcaResult {
    pub fn n_components(&self) -> usize {
        self.components.len()
    }
    pub fn n_features(&self) -> usize {
        self.mean.len()
    }
}

/// Symmetric eigendecomposition via the cyclic Jacobi method.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvector `k` is column `k` of
/// the returned matrix (stored row-major, `n x n`), sorted by descending eigenvalue.
fn jacobi_eigen(a_in: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    // V starts as identity.
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let idx = |i: usize, j: usize| i * n + j;

    for _sweep in 0..100 {
        // Off-diagonal Frobenius norm.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[idx(p, q)] * a[idx(p, q)];
            }
        }
        if off.sqrt() < 1e-300 || off < 1e-30 {
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
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Apply rotation to A.
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
                // Accumulate rotation into V.
                for k in 0..n {
                    let vkp = v[idx(k, p)];
                    let vkq = v[idx(k, q)];
                    v[idx(k, p)] = c * vkp - s * vkq;
                    v[idx(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut eigvals: Vec<f64> = (0..n).map(|i| a[idx(i, i)]).collect();
    // Sort indices by descending eigenvalue.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| eigvals[y].partial_cmp(&eigvals[x]).unwrap());

    let sorted_vals: Vec<f64> = order.iter().map(|&k| eigvals[k]).collect();
    // Eigenvectors row-major as columns: out[i*n + k] = v[i, order[k]].
    let mut sorted_vecs = vec![0.0f64; n * n];
    for (k, &col) in order.iter().enumerate() {
        for i in 0..n {
            sorted_vecs[idx(i, k)] = v[idx(i, col)];
        }
    }
    eigvals.copy_from_slice(&sorted_vals);
    (eigvals, sorted_vecs)
}

fn fix_signs(components: &mut [Vec<f64>]) {
    for row in components.iter_mut() {
        let mut k = 0usize;
        let mut best = 0.0f64;
        for (j, &val) in row.iter().enumerate() {
            if val.abs() > best {
                best = val.abs();
                k = j;
            }
        }
        if row[k] < 0.0 {
            for val in row.iter_mut() {
                *val = -*val;
            }
        }
    }
}

/// Fits a PCA model to `x` via the eigendecomposition of the covariance matrix.
///
/// `n_components = None` retains `min(n, d)`. `scale = true` divides each centered
/// feature by its sample standard deviation (correlation PCA).
pub fn fit_pca(
    x: &Matrix,
    n_components: Option<usize>,
    scale: bool,
    whiten: bool,
) -> Result<PcaResult, String> {
    let n = x.rows;
    let d = x.cols;
    if n < 2 {
        return Err(format!(
            "need at least 2 samples to estimate variance, got {}",
            n
        ));
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err("X contains non-finite values (nan or inf)".to_string());
    }
    let max_components = n.min(d);
    let k = n_components.unwrap_or(max_components);
    if k < 1 || k > max_components {
        return Err(format!(
            "n_components must be in [1, {}] for a {}x{} matrix, got {}",
            max_components, n, d, k
        ));
    }

    // Column means.
    let mut mean = vec![0.0f64; d];
    for i in 0..n {
        for j in 0..d {
            mean[j] += x.get(i, j);
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }

    // Scales.
    let mut scale_vec = vec![1.0f64; d];
    if scale {
        for j in 0..d {
            let mut ss = 0.0;
            for i in 0..n {
                let c = x.get(i, j) - mean[j];
                ss += c * c;
            }
            let std = (ss / (n as f64 - 1.0)).sqrt();
            if std == 0.0 {
                return Err(format!("cannot scale: feature {} has zero variance", j));
            }
            scale_vec[j] = std;
        }
    }

    // Centered (and scaled) matrix.
    let mut xc = vec![0.0f64; n * d];
    for i in 0..n {
        for j in 0..d {
            xc[i * d + j] = (x.get(i, j) - mean[j]) / scale_vec[j];
        }
    }

    // Covariance matrix C = Xc^T Xc / (n - 1), shape d x d.
    let mut cov = vec![0.0f64; d * d];
    for a in 0..d {
        for b in a..d {
            let mut acc = 0.0;
            for i in 0..n {
                acc += xc[i * d + a] * xc[i * d + b];
            }
            let val = acc / (n as f64 - 1.0);
            cov[a * d + b] = val;
            cov[b * d + a] = val;
        }
    }

    let (eigvals, eigvecs) = jacobi_eigen(&cov, d);

    let total_var: f64 = eigvals.iter().map(|&e| e.max(0.0)).sum();
    if total_var == 0.0 {
        return Err("X has zero total variance after centering; PCA is undefined".to_string());
    }

    let mut components: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut explained_variance: Vec<f64> = Vec::with_capacity(k);
    for comp in 0..k {
        let mut row = vec![0.0f64; d];
        for i in 0..d {
            row[i] = eigvecs[i * d + comp];
        }
        components.push(row);
        explained_variance.push(eigvals[comp].max(0.0));
    }
    fix_signs(&mut components);

    let explained_variance_ratio: Vec<f64> =
        explained_variance.iter().map(|&e| e / total_var).collect();

    Ok(PcaResult {
        mean,
        scale: scale_vec,
        components,
        explained_variance,
        explained_variance_ratio,
        whiten,
    })
}

/// Projects `x` onto the fitted principal components, returning `n x n_components` scores.
pub fn transform(model: &PcaResult, x: &Matrix) -> Result<Vec<Vec<f64>>, String> {
    let d = model.n_features();
    if x.cols != d {
        return Err(format!(
            "X has {} features but model was fit on {}",
            x.cols, d
        ));
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err("X contains non-finite values (nan or inf)".to_string());
    }
    let k = model.n_components();
    let mut out = vec![vec![0.0f64; k]; x.rows];
    for i in 0..x.rows {
        for (c, comp) in model.components.iter().enumerate() {
            let mut acc = 0.0;
            for j in 0..d {
                let xc = (x.get(i, j) - model.mean[j]) / model.scale[j];
                acc += xc * comp[j];
            }
            if model.whiten {
                let std = model.explained_variance[c].sqrt();
                if std == 0.0 {
                    return Err("cannot whiten: a retained component has zero variance".to_string());
                }
                acc /= std;
            }
            out[i][c] = acc;
        }
    }
    Ok(out)
}

/// Maps scores back to the original feature space (best rank-m reconstruction).
pub fn inverse_transform(model: &PcaResult, scores: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let k = model.n_components();
    let d = model.n_features();
    let mut out = vec![vec![0.0f64; d]; scores.len()];
    for (i, srow) in scores.iter().enumerate() {
        if srow.len() != k {
            return Err(format!(
                "scores has {} columns but model has {} components",
                srow.len(),
                k
            ));
        }
        for j in 0..d {
            let mut acc = 0.0;
            for c in 0..k {
                let mut t = srow[c];
                if model.whiten {
                    t *= model.explained_variance[c].sqrt();
                }
                acc += t * model.components[c][j];
            }
            out[i][j] = acc * model.scale[j] + model.mean[j];
        }
    }
    Ok(out)
}

/// Mean squared reconstruction error of `x` under the truncated model.
pub fn reconstruction_error(model: &PcaResult, x: &Matrix) -> Result<f64, String> {
    let scores = transform(model, x)?;
    let recon = inverse_transform(model, &scores)?;
    let mut acc = 0.0;
    for i in 0..x.rows {
        for j in 0..x.cols {
            let diff = x.get(i, j) - recon[i][j];
            acc += diff * diff;
        }
    }
    Ok(acc / x.rows as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn fixture() -> Matrix {
        Matrix::from_rows(&[
            vec![2.5, 2.4],
            vec![0.5, 0.7],
            vec![2.2, 2.9],
            vec![1.9, 2.2],
            vec![3.1, 3.0],
            vec![2.3, 2.7],
            vec![2.0, 1.6],
            vec![1.0, 1.1],
            vec![1.5, 1.6],
            vec![1.1, 0.9],
        ])
        .unwrap()
    }

    const TOL: f64 = 1e-9;

    #[test]
    fn mean_matches_fixture() {
        let r = fit_pca(&fixture(), Some(2), false, false).unwrap();
        assert!((r.mean[0] - 1.81).abs() < TOL);
        assert!((r.mean[1] - 1.91).abs() < TOL);
    }

    #[test]
    fn components_match_fixture() {
        let r = fit_pca(&fixture(), Some(2), false, false).unwrap();
        let expected = [
            [0.6778733985280119, 0.735178655544408],
            [0.735178655544408, -0.6778733985280119],
        ];
        for i in 0..2 {
            for j in 0..2 {
                assert!((r.components[i][j] - expected[i][j]).abs() < TOL);
            }
        }
    }

    #[test]
    fn explained_variance_matches_fixture() {
        let r = fit_pca(&fixture(), Some(2), false, false).unwrap();
        assert!((r.explained_variance[0] - 1.2840277121727839).abs() < TOL);
        assert!((r.explained_variance[1] - 0.0490833989383273).abs() < TOL);
    }

    #[test]
    fn explained_variance_ratio_matches_fixture() {
        let r = fit_pca(&fixture(), Some(2), false, false).unwrap();
        assert!((r.explained_variance_ratio[0] - 0.963181314348646).abs() < TOL);
        assert!((r.explained_variance_ratio[1] - 0.0368186856513541).abs() < TOL);
    }

    #[test]
    fn transform_first_row_matches_fixture() {
        let r = fit_pca(&fixture(), Some(2), false, false).unwrap();
        let scores = transform(&r, &fixture()).unwrap();
        assert!((scores[0][0] - 0.8279701862010882).abs() < TOL);
        assert!((scores[0][1] - 0.1751153070469155).abs() < TOL);
    }

    #[test]
    fn reconstruction_error_one_component() {
        let r = fit_pca(&fixture(), Some(1), false, false).unwrap();
        let err = reconstruction_error(&r, &fixture()).unwrap();
        assert!((err - 0.04417505904449458).abs() < TOL);
    }

    #[test]
    fn full_rank_reconstruction_is_exact() {
        let r = fit_pca(&fixture(), Some(2), false, false).unwrap();
        let err = reconstruction_error(&r, &fixture()).unwrap();
        assert!(err.abs() < 1e-12);
    }

    #[test]
    fn round_trip_inverse_transform() {
        let r = fit_pca(&fixture(), Some(2), false, false).unwrap();
        let scores = transform(&r, &fixture()).unwrap();
        let rec = inverse_transform(&r, &scores).unwrap();
        let x = fixture();
        for i in 0..x.rows {
            for j in 0..x.cols {
                assert!((rec[i][j] - x.get(i, j)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn scaled_pca_components_match() {
        let r = fit_pca(&fixture(), Some(2), true, false).unwrap();
        // First component is well-determined; the second is a tied-magnitude
        // eigenvector whose overall sign is numerically arbitrary, so compare its
        // absolute values.
        assert!((r.components[0][0] - 0.7071067811865476).abs() < TOL);
        assert!((r.components[0][1] - 0.7071067811865475).abs() < TOL);
        assert!((r.components[1][0].abs() - 0.7071067811865475).abs() < TOL);
        assert!((r.components[1][1].abs() - 0.7071067811865476).abs() < TOL);
        assert!((r.explained_variance_ratio[0] - 0.9629646363461227).abs() < TOL);
        assert!((r.explained_variance_ratio[1] - 0.0370353636538773).abs() < TOL);
    }

    #[test]
    fn too_few_samples_errors() {
        let x = Matrix::from_rows(&[vec![1.0, 2.0]]).unwrap();
        assert!(fit_pca(&x, None, false, false).is_err());
    }

    #[test]
    fn bad_n_components_errors() {
        assert!(fit_pca(&fixture(), Some(5), false, false).is_err());
    }

    #[test]
    fn scale_zero_variance_errors() {
        let x = Matrix::from_rows(&[vec![1.0, 5.0], vec![2.0, 5.0], vec![3.0, 5.0]]).unwrap();
        assert!(fit_pca(&x, Some(2), true, false).is_err());
    }
}
