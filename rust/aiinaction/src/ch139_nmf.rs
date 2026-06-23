//! Non-Negative Matrix Factorization (NMF) from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch139_nmf` and the Julia module
//! `AIInAction.Ch139Nmf`. The shared fixtures in the tests below match the
//! Python/Julia suites, which keeps the three implementations at parity.
//!
//! Given a non-negative matrix `V` of shape `(n, m)`, NMF seeks non-negative
//! factors `W` (`n x r`) and `H` (`r x m`) with `V ~= W H`, refined by the
//! Lee-Seung multiplicative updates for the squared Frobenius objective:
//!
//! ```text
//! H <- H * (W^T V) / (W^T W H + eps)
//! W <- W * (V H^T) / (W H H^T + eps)
//! ```
//!
//! Determinism: the factors are seeded by a self-contained 32-bit linear
//! congruential generator (Numerical Recipes constants), filled row-major. The
//! identical LCG, fill order, and fixed iteration count are reproduced in the
//! Python and Julia ports, so all three agree bit-for-bit to 1e-9.

/// Numerical floor added to denominators. Shared across languages.
const EPS: f64 = 1e-10;

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
            return Err("V must be non-empty".to_string());
        }
        let cols = rows[0].len();
        if cols == 0 {
            return Err("V must be non-empty".to_string());
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

    /// Builds a zero-filled `rows x cols` matrix.
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    fn validate_input(&self) -> Result<(), String> {
        if self.data.iter().any(|v| !v.is_finite()) {
            return Err("V contains non-finite values (nan or inf)".to_string());
        }
        if self.data.iter().any(|&v| v < 0.0) {
            return Err("V must be non-negative (all entries >= 0)".to_string());
        }
        Ok(())
    }
}

/// The fitted state of an NMF model.
#[derive(Clone, Debug)]
pub struct NmfResult {
    /// Basis matrix `n x r`; column `k` is latent component `k`.
    pub w: Matrix,
    /// Coefficient matrix `r x m`; column `j` holds activations for data column `j`.
    pub h: Matrix,
    /// Number of multiplicative-update iterations run.
    pub n_iter: usize,
    /// Final Frobenius reconstruction error `||V - W H||_F`.
    pub error: f64,
}

impl NmfResult {
    pub fn n_components(&self) -> usize {
        self.w.cols
    }
    pub fn n_features(&self) -> usize {
        self.w.rows
    }
}

/// Deterministic uniform fill in `(0, 1]` via a 32-bit LCG, row-major.
///
/// Entry `(i, j)` is the `(i * cols + j)`-th draw. Matches the Python and Julia
/// initializers exactly.
fn seeded_uniform(rows: usize, cols: usize, seed: u64) -> Matrix {
    let mut m = Matrix::zeros(rows, cols);
    let mut state: u64 = seed & 0xFFFF_FFFF;
    for i in 0..rows {
        for j in 0..cols {
            state = (1664525u64.wrapping_mul(state).wrapping_add(1013904223)) & 0xFFFF_FFFF;
            m.set(i, j, (state as f64 + 1.0) / 4294967297.0);
        }
    }
    m
}

/// Matrix product `a (p x q) * b (q x s) -> (p x s)`.
fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let p = a.rows;
    let q = a.cols;
    let s = b.cols;
    let mut out = Matrix::zeros(p, s);
    for i in 0..p {
        for k in 0..q {
            let aik = a.get(i, k);
            if aik == 0.0 {
                continue;
            }
            for j in 0..s {
                out.data[i * s + j] += aik * b.get(k, j);
            }
        }
    }
    out
}

/// Transpose of `a`.
fn transpose(a: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(a.cols, a.rows);
    for i in 0..a.rows {
        for j in 0..a.cols {
            out.set(j, i, a.get(i, j));
        }
    }
    out
}

/// Elementwise multiplicative update: `target[i] *= numer[i] / (denom[i] + EPS)`.
fn mult_update(target: &mut Matrix, numer: &Matrix, denom: &Matrix) {
    for idx in 0..target.data.len() {
        target.data[idx] *= numer.data[idx] / (denom.data[idx] + EPS);
    }
}

/// Frobenius reconstruction error `||V - W H||_F`.
fn frob_error(v: &Matrix, w: &Matrix, h: &Matrix) -> f64 {
    let wh = matmul(w, h);
    let mut acc = 0.0;
    for idx in 0..v.data.len() {
        let d = v.data[idx] - wh.data[idx];
        acc += d * d;
    }
    acc.sqrt()
}

/// Factors a non-negative matrix `v` as `W H` via multiplicative updates.
///
/// `n_components` is the rank `r` in `[1, min(n, m)]`; `max_iter >= 1` sweeps;
/// `seed` seeds the deterministic LCG initializer.
pub fn fit_nmf(
    v: &Matrix,
    n_components: usize,
    max_iter: usize,
    seed: u64,
) -> Result<NmfResult, String> {
    v.validate_input()?;
    let n = v.rows;
    let m = v.cols;
    let max_components = n.min(m);
    let r = n_components;
    if r < 1 || r > max_components {
        return Err(format!(
            "n_components must be in [1, {}] for a {}x{} matrix, got {}",
            max_components, n, m, r
        ));
    }
    if max_iter < 1 {
        return Err(format!("max_iter must be a positive integer, got {}", max_iter));
    }

    let mut w = seeded_uniform(n, r, seed);
    let mut h = seeded_uniform(r, m, seed + 1);

    for _ in 0..max_iter {
        // Update H: H *= (W^T V) / (W^T W H).
        let wt = transpose(&w);
        let wtv = matmul(&wt, v);
        let wtw = matmul(&wt, &w);
        let wtwh = matmul(&wtw, &h);
        mult_update(&mut h, &wtv, &wtwh);
        // Update W: W *= (V H^T) / (W H H^T).
        let ht = transpose(&h);
        let vht = matmul(v, &ht);
        let hht = matmul(&h, &ht);
        let whht = matmul(&w, &hht);
        mult_update(&mut w, &vht, &whht);
    }

    let error = frob_error(v, &w, &h);
    Ok(NmfResult {
        w,
        h,
        n_iter: max_iter,
        error,
    })
}

/// Encodes new data under the fixed model basis `W`, returning `H` (`r x m`).
pub fn transform(model: &NmfResult, v: &Matrix, max_iter: usize) -> Result<Matrix, String> {
    v.validate_input()?;
    if v.rows != model.n_features() {
        return Err(format!(
            "V has {} features but model basis has {}",
            v.rows,
            model.n_features()
        ));
    }
    if max_iter < 1 {
        return Err(format!("max_iter must be a positive integer, got {}", max_iter));
    }
    let r = model.n_components();
    let m = v.cols;
    let w = &model.w;
    let mut h = seeded_uniform(r, m, 1);
    let wt = transpose(w);
    let wtv = matmul(&wt, v);
    let wtw = matmul(&wt, w);
    for _ in 0..max_iter {
        let wtwh = matmul(&wtw, &h);
        mult_update(&mut h, &wtv, &wtwh);
    }
    Ok(h)
}

/// Returns the low-rank reconstruction `W H` of the fitted model.
pub fn reconstruct(model: &NmfResult) -> Matrix {
    matmul(&model.w, &model.h)
}

/// Frobenius reconstruction error `||V - W H||_F` for given factors.
pub fn reconstruction_error(v: &Matrix, w: &Matrix, h: &Matrix) -> Result<f64, String> {
    v.validate_input()?;
    if w.rows != v.rows {
        return Err(format!("W has {} rows but V has {}", w.rows, v.rows));
    }
    if h.cols != v.cols {
        return Err(format!("H has {} columns but V has {}", h.cols, v.cols));
    }
    if w.cols != h.rows {
        return Err(format!(
            "inner dimensions disagree: W is {}x{}, H is {}x{}",
            w.rows, w.cols, h.rows, h.cols
        ));
    }
    Ok(frob_error(v, w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn v_block() -> Matrix {
        Matrix::from_rows(&[
            vec![1.0, 1.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 2.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ])
        .unwrap()
    }

    fn v_dense() -> Matrix {
        Matrix::from_rows(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])
        .unwrap()
    }

    const TOL: f64 = 1e-9;

    #[test]
    fn seeded_uniform_matches_fixture() {
        let m = seeded_uniform(2, 2, 0);
        let expected: [f64; 4] = [
            0.2360679730223333,
            0.2785669087249397,
            0.8195337600029228,
            0.6678668978466031,
        ];
        for i in 0..4 {
            assert!((m.data[i] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn block_factors_match_fixture() {
        let r = fit_nmf(&v_block(), 2, 300, 0).unwrap();
        let expected_w: [[f64; 2]; 4] = [
            [0.8504210044717101, 0.0],
            [0.8504210044717101, 0.0],
            [0.0, 1.3943383444007382],
            [0.0, 1.3943383444007382],
        ];
        let expected_h: [[f64; 4]; 2] = [
            [1.175888171504758, 1.175888171504758, 0.0, 0.0],
            [0.0, 0.0, 1.4343720862275404, 1.4343720862275404],
        ];
        for i in 0..4 {
            for j in 0..2 {
                assert!((r.w.get(i, j) - expected_w[i][j]).abs() < TOL);
            }
        }
        for i in 0..2 {
            for j in 0..4 {
                assert!((r.h.get(i, j) - expected_h[i][j]).abs() < TOL);
            }
        }
        assert_eq!(r.n_components(), 2);
        assert_eq!(r.n_features(), 4);
    }

    #[test]
    fn block_reconstruction_is_near_exact() {
        let r = fit_nmf(&v_block(), 2, 300, 0).unwrap();
        assert!(r.error.abs() < 1e-6);
        let recon = reconstruct(&r);
        let v = v_block();
        for idx in 0..v.data.len() {
            assert!((recon.data[idx] - v.data[idx]).abs() < 1e-6);
        }
    }

    #[test]
    fn dense_factors_match_fixture() {
        let r = fit_nmf(&v_dense(), 2, 100, 7).unwrap();
        let expected_w: [[f64; 2]; 3] = [
            [0.7214042811455682, 0.2319152540718632],
            [0.3793167032779777, 0.9034766028860397],
            [0.0933041476566348, 1.5614929744526524],
        ];
        let expected_h: [[f64; 3]; 2] = [
            [1.8692478223292252e-3, 1.1082573543977547, 2.3545687125460262],
            [4.4657474657315506, 5.0631720186370552, 5.6307847244507254],
        ];
        for i in 0..3 {
            for j in 0..2 {
                assert!((r.w.get(i, j) - expected_w[i][j]).abs() < TOL);
            }
        }
        for i in 0..2 {
            for j in 0..3 {
                assert!((r.h.get(i, j) - expected_h[i][j]).abs() < TOL);
            }
        }
        assert!((r.error - 0.06848010125776947).abs() < TOL);
    }

    #[test]
    fn factors_are_non_negative() {
        let r = fit_nmf(&v_dense(), 2, 100, 7).unwrap();
        assert!(r.w.data.iter().all(|&v| v >= 0.0));
        assert!(r.h.data.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn transform_matches_fixture() {
        let model = fit_nmf(&v_block(), 2, 300, 0).unwrap();
        let hn = transform(&model, &v_block(), 300).unwrap();
        let expected: [[f64; 4]; 2] = [
            [1.1758881714856226, 1.1758881714856226, 0.0, 0.0],
            [0.0, 0.0, 1.4343720862268226, 1.4343720862268226],
        ];
        for i in 0..2 {
            for j in 0..4 {
                assert!((hn.get(i, j) - expected[i][j]).abs() < TOL);
            }
        }
        let err = reconstruction_error(&v_block(), &model.w, &hn).unwrap();
        assert!(err.abs() < 1e-6);
    }

    #[test]
    fn reconstruction_error_helper() {
        let v = Matrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        let w = Matrix::from_rows(&[vec![1.0], vec![0.0]]).unwrap();
        let h = Matrix::from_rows(&[vec![1.0, 0.0]]).unwrap();
        let err = reconstruction_error(&v, &w, &h).unwrap();
        assert!((err - 1.0).abs() < TOL);
    }

    #[test]
    fn negative_input_errors() {
        let v = Matrix::from_rows(&[vec![1.0, -1.0], vec![2.0, 3.0]]).unwrap();
        assert!(fit_nmf(&v, 1, 50, 0).is_err());
    }

    #[test]
    fn bad_n_components_errors() {
        assert!(fit_nmf(&v_block(), 0, 50, 0).is_err());
        assert!(fit_nmf(&v_dense(), 5, 50, 0).is_err());
    }

    #[test]
    fn bad_max_iter_errors() {
        assert!(fit_nmf(&v_block(), 2, 0, 0).is_err());
    }

    #[test]
    fn transform_feature_mismatch_errors() {
        let model = fit_nmf(&v_block(), 2, 50, 0).unwrap();
        let bad = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        assert!(transform(&model, &bad, 50).is_err());
    }
}
