//! Spectral clustering from scratch (chapter 131).
//!
//! A std-only port of the Python module `aiinaction.ch131_spectral_clustering`
//! and the Julia module `AIInAction.Ch131SpectralClustering`. Matrices are
//! `Vec<Vec<f64>>` (row-major). Every primitive, including the symmetric
//! eigensolver (cyclic Jacobi) and deterministic k-means, is implemented from
//! scratch so all three language ports agree to within `1e-9` on the shared
//! fixtures.

/// Squared Euclidean distance between two equal-length rows.
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn validate_matrix(x: &[Vec<f64>]) -> Result<usize, String> {
    if x.is_empty() {
        return Err("X must contain at least one sample".to_string());
    }
    let d = x[0].len();
    for row in x {
        if row.len() != d {
            return Err("X rows must all have the same length".to_string());
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err("X must contain only finite values".to_string());
        }
    }
    Ok(d)
}

/// Gaussian (RBF) similarity matrix with a zero diagonal.
///
/// `w_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))`, `w_ii = 0`.
pub fn rbf_affinity(x: &[Vec<f64>], sigma: f64) -> Result<Vec<Vec<f64>>, String> {
    validate_matrix(x)?;
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(format!("sigma must be a positive finite number; got {sigma}"));
    }
    let n = x.len();
    let denom = 2.0 * sigma * sigma;
    let mut w = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let val = (-sq_dist(&x[i], &x[j]) / denom).exp();
            w[i][j] = val;
            w[j][i] = val;
        }
    }
    Ok(w)
}

/// Symmetric normalized Laplacian `L_sym = I - D^{-1/2} W D^{-1/2}`.
pub fn normalized_laplacian(w: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = w.len();
    if n == 0 {
        return Err("W must be non-empty".to_string());
    }
    for row in w {
        if row.len() != n {
            return Err("W must be a square matrix".to_string());
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err("W must contain only finite values".to_string());
        }
        if row.iter().any(|&v| v < 0.0) {
            return Err("W must have non-negative entries".to_string());
        }
    }
    for i in 0..n {
        for j in 0..n {
            if (w[i][j] - w[j][i]).abs() > 1e-12 {
                return Err("W must be symmetric".to_string());
            }
        }
    }
    let deg: Vec<f64> = w.iter().map(|row| row.iter().sum()).collect();
    if deg.iter().any(|&d| d <= 0.0) {
        return Err("every vertex must have positive degree (no isolated vertices)".to_string());
    }
    let dinv: Vec<f64> = deg.iter().map(|&d| 1.0 / d.sqrt()).collect();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let id = if i == j { 1.0 } else { 0.0 };
            l[i][j] = id - dinv[i] * w[i][j] * dinv[j];
        }
    }
    // Symmetrize.
    for i in 0..n {
        for j in (i + 1)..n {
            let m = 0.5 * (l[i][j] + l[j][i]);
            l[i][j] = m;
            l[j][i] = m;
        }
    }
    Ok(l)
}

/// Eigen-decomposition of a real symmetric matrix via cyclic Jacobi rotations.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvalues ascending and
/// eigenvectors as columns, each sign-fixed so its first nonzero entry is
/// positive.
pub fn jacobi_eigh(a: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    let n = a.len();
    if n == 0 {
        return Err("A must be non-empty".to_string());
    }
    for row in a {
        if row.len() != n {
            return Err("A must be a square matrix".to_string());
        }
    }
    for i in 0..n {
        for j in 0..n {
            if (a[i][j] - a[j][i]).abs() > 1e-12 {
                return Err("A must be symmetric".to_string());
            }
        }
    }
    let max_sweeps = 100usize;
    let tol = 1e-12;
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    for _ in 0..max_sweeps {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[p][q] * m[p][q];
            }
        }
        if off.sqrt() <= tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p][q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let app = m[p][p];
                let aqq = m[q][q];
                let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
                let c = phi.cos();
                let s = phi.sin();
                for i in 0..n {
                    let mip = m[i][p];
                    let miq = m[i][q];
                    m[i][p] = c * mip - s * miq;
                    m[i][q] = s * mip + c * miq;
                }
                for i in 0..n {
                    let mpi = m[p][i];
                    let mqi = m[q][i];
                    m[p][i] = c * mpi - s * mqi;
                    m[q][i] = s * mpi + c * mqi;
                }
                for i in 0..n {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
        }
    }

    let eig: Vec<f64> = (0..n).map(|i| m[i][i]).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| eig[i].partial_cmp(&eig[j]).unwrap());
    let eigvals: Vec<f64> = order.iter().map(|&i| eig[i]).collect();
    let mut vecs = vec![vec![0.0; n]; n];
    for (new_j, &old_j) in order.iter().enumerate() {
        for i in 0..n {
            vecs[i][new_j] = v[i][old_j];
        }
    }
    // Sign-fix each column: first nonzero entry positive.
    for j in 0..n {
        for i in 0..n {
            if vecs[i][j].abs() > 1e-12 {
                if vecs[i][j] < 0.0 {
                    for r in 0..n {
                        vecs[r][j] = -vecs[r][j];
                    }
                }
                break;
            }
        }
    }
    Ok((eigvals, vecs))
}

/// Row-normalized spectral embedding: the `k` smallest eigenvectors of
/// `L_sym`, each row scaled to unit length.
pub fn spectral_embedding(w: &[Vec<f64>], k: usize) -> Result<Vec<Vec<f64>>, String> {
    let l = normalized_laplacian(w)?;
    let n = l.len();
    if k < 1 || k > n {
        return Err(format!("k must be an integer in [1, n]={n}; got {k}"));
    }
    let (_, vecs) = jacobi_eigh(&l)?;
    let mut u = vec![vec![0.0; k]; n];
    for i in 0..n {
        for j in 0..k {
            u[i][j] = vecs[i][j];
        }
    }
    for row in u.iter_mut() {
        let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    }
    Ok(u)
}

/// Lloyd's k-means with deterministic furthest-point seeding.
///
/// Returns `(labels, centers)`.
pub fn kmeans(x: &[Vec<f64>], k: usize) -> Result<(Vec<usize>, Vec<Vec<f64>>), String> {
    let d = validate_matrix(x)?;
    let n = x.len();
    if k < 1 || k > n {
        return Err(format!("k must be an integer in [1, n]={n}; got {k}"));
    }
    let max_iter = 100usize;
    let tol = 1e-10;

    // Seed 0: lexicographically smallest point.
    let mut first = 0usize;
    for i in 1..n {
        if lex_less(&x[i], &x[first]) {
            first = i;
        }
    }
    let mut center_idx = vec![first];
    while center_idx.len() < k {
        let mut best_i = usize::MAX;
        let mut best_d = -1.0;
        for i in 0..n {
            if center_idx.contains(&i) {
                continue;
            }
            let dist = center_idx
                .iter()
                .map(|&c| sq_dist(&x[i], &x[c]))
                .fold(f64::INFINITY, f64::min);
            if dist > best_d {
                best_d = dist;
                best_i = i;
            }
        }
        center_idx.push(best_i);
    }
    let mut centers: Vec<Vec<f64>> = center_idx.iter().map(|&i| x[i].clone()).collect();

    let mut labels = vec![0usize; n];
    for _ in 0..max_iter {
        for i in 0..n {
            let mut best_c = 0;
            let mut best_d = f64::INFINITY;
            for c in 0..k {
                let dist = sq_dist(&x[i], &centers[c]);
                if dist < best_d {
                    best_d = dist;
                    best_c = c;
                }
            }
            labels[i] = best_c;
        }
        let mut new_centers = centers.clone();
        for c in 0..k {
            let members: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
            if !members.is_empty() {
                let mut mean = vec![0.0; d];
                for &i in &members {
                    for t in 0..d {
                        mean[t] += x[i][t];
                    }
                }
                for t in 0..d {
                    mean[t] /= members.len() as f64;
                }
                new_centers[c] = mean;
            }
        }
        let mut shift = 0.0;
        for c in 0..k {
            shift += sq_dist(&new_centers[c], &centers[c]);
        }
        centers = new_centers;
        if shift.sqrt() <= tol {
            break;
        }
    }
    Ok((labels, centers))
}

fn lex_less(a: &[f64], b: &[f64]) -> bool {
    for (x, y) in a.iter().zip(b) {
        if x < y {
            return true;
        }
        if x > y {
            return false;
        }
    }
    false
}

/// End-to-end spectral clustering (Ng-Jordan-Weiss). Returns cluster labels.
pub fn spectral_clustering(x: &[Vec<f64>], k: usize, sigma: f64) -> Result<Vec<usize>, String> {
    let w = rbf_affinity(x, sigma)?;
    let embedding = spectral_embedding(&w, k)?;
    let (labels, _) = kmeans(&embedding, k)?;
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn blobs() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![0.2, 0.1],
            vec![0.1, -0.2],
            vec![5.0, 5.0],
            vec![5.2, 4.9],
            vec![4.9, 5.1],
        ]
    }
    const SIGMA: f64 = 1.0;
    const TOL: f64 = 1e-9;

    fn aff_w01() -> f64 {
        (-((0.2_f64).powi(2) + (0.1_f64).powi(2)) / (2.0 * SIGMA * SIGMA)).exp()
    }

    #[test]
    fn rbf_affinity_symmetric_zero_diag() {
        let w = rbf_affinity(&blobs(), SIGMA).unwrap();
        for i in 0..6 {
            assert!(w[i][i].abs() < TOL);
            for j in 0..6 {
                assert!((w[i][j] - w[j][i]).abs() < TOL);
            }
        }
        assert!((w[0][1] - aff_w01()).abs() < TOL);
        assert!(w[0][3] < 1e-9);
    }

    #[test]
    fn rbf_affinity_rejects_bad_sigma() {
        assert!(rbf_affinity(&blobs(), 0.0).is_err());
        assert!(rbf_affinity(&blobs(), -1.0).is_err());
    }

    #[test]
    fn normalized_laplacian_unit_diag_symmetric() {
        let w = rbf_affinity(&blobs(), SIGMA).unwrap();
        let l = normalized_laplacian(&w).unwrap();
        for i in 0..6 {
            assert!((l[i][i] - 1.0).abs() < TOL);
            for j in 0..6 {
                assert!((l[i][j] - l[j][i]).abs() < TOL);
            }
        }
    }

    #[test]
    fn normalized_laplacian_smallest_eigenvalue_zero() {
        let w = rbf_affinity(&blobs(), SIGMA).unwrap();
        let l = normalized_laplacian(&w).unwrap();
        let (vals, _) = jacobi_eigh(&l).unwrap();
        assert!(vals[0].abs() < 1e-8);
        assert!(vals.iter().all(|&v| v >= -1e-8));
    }

    #[test]
    fn normalized_laplacian_rejects_isolated() {
        let w = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        assert!(normalized_laplacian(&w).is_err());
    }

    #[test]
    fn jacobi_eigh_values_and_vectors() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (vals, vecs) = jacobi_eigh(&a).unwrap();
        assert!((vals[0] - 1.0).abs() < TOL);
        assert!((vals[1] - 3.0).abs() < TOL);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert!((vecs[0][0] - inv_sqrt2).abs() < TOL);
        assert!((vecs[1][0] + inv_sqrt2).abs() < TOL);
        // Orthonormality of columns.
        let dot = vecs[0][0] * vecs[0][1] + vecs[1][0] * vecs[1][1];
        assert!(dot.abs() < TOL);
    }

    #[test]
    fn jacobi_eigh_diagonal() {
        let a = vec![
            vec![4.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ];
        let (vals, _) = jacobi_eigh(&a).unwrap();
        assert!((vals[0] - 1.0).abs() < TOL);
        assert!((vals[1] - 2.0).abs() < TOL);
        assert!((vals[2] - 4.0).abs() < TOL);
    }

    #[test]
    fn jacobi_eigh_rejects_asymmetric() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(jacobi_eigh(&a).is_err());
    }

    #[test]
    fn spectral_embedding_unit_rows() {
        let w = rbf_affinity(&blobs(), SIGMA).unwrap();
        let u = spectral_embedding(&w, 2).unwrap();
        assert_eq!(u.len(), 6);
        assert_eq!(u[0].len(), 2);
        for row in &u {
            let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn spectral_embedding_rejects_bad_k() {
        let w = rbf_affinity(&blobs(), SIGMA).unwrap();
        assert!(spectral_embedding(&w, 0).is_err());
        assert!(spectral_embedding(&w, 7).is_err());
    }

    #[test]
    fn kmeans_labels_and_centers() {
        let pts = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 9.9],
        ];
        let (labels, centers) = kmeans(&pts, 2).unwrap();
        assert_eq!(labels, vec![0, 0, 1, 1]);
        assert!((centers[0][0] - 0.05).abs() < TOL);
        assert!((centers[0][1] - 0.0).abs() < TOL);
        assert!((centers[1][0] - 10.05).abs() < TOL);
        assert!((centers[1][1] - 9.95).abs() < TOL);
    }

    #[test]
    fn kmeans_single_cluster() {
        let pts = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 9.9],
        ];
        let (labels, _) = kmeans(&pts, 1).unwrap();
        assert_eq!(labels, vec![0, 0, 0, 0]);
    }

    #[test]
    fn spectral_clustering_separates_blobs() {
        let labels = spectral_clustering(&blobs(), 2, SIGMA).unwrap();
        assert_eq!(labels, vec![0, 0, 0, 1, 1, 1]);
    }
}
