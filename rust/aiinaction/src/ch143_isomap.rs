//! Isomap (isometric feature mapping) from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch143_isomap` and the Julia module
//! `AIInAction.Ch143Isomap`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! This is a std-only implementation. The three stages are:
//!
//! 1. Build a symmetric k-nearest-neighbor graph weighted by Euclidean distance.
//! 2. Approximate geodesics with all-pairs shortest paths (Floyd-Warshall).
//! 3. Embed via classical MDS: double-center the squared geodesic distances to
//!    form the Gram matrix `B`, diagonalize it with a cyclic Jacobi eigensolver,
//!    and set `Y = U_d Lambda_d^{1/2}`.
//!
//! Sign convention: each embedding coordinate (column of `Y`) is flipped so its
//! largest-magnitude entry is positive, removing the eigenvector sign ambiguity.

const INF: f64 = f64::INFINITY;

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

/// The fitted state of an Isomap embedding.
#[derive(Clone, Debug)]
pub struct IsomapResult {
    /// Low-dimensional coordinates: `n` rows of length `n_components`.
    pub embedding: Vec<Vec<f64>>,
    /// Top `n_components` eigenvalues of `B`, descending, clamped non-negative.
    pub eigenvalues: Vec<f64>,
    /// The `n x n` matrix of graph shortest-path (geodesic) distances, row-major.
    pub geodesic_distances: Vec<f64>,
    /// The `k` used to build the neighborhood graph.
    pub n_neighbors: usize,
}

impl IsomapResult {
    pub fn n_samples(&self) -> usize {
        self.embedding.len()
    }
    pub fn n_components(&self) -> usize {
        if self.embedding.is_empty() {
            0
        } else {
            self.embedding[0].len()
        }
    }
}

fn validate(x: &Matrix) -> Result<(), String> {
    if x.rows < 2 {
        return Err(format!("need at least 2 samples, got {}", x.rows));
    }
    if x.cols < 1 {
        return Err("X must have at least one feature".to_string());
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err("X contains non-finite values (nan or inf)".to_string());
    }
    Ok(())
}

/// Euclidean distance matrix of the rows of `x`, row-major `n x n`.
pub fn pairwise_distances(x: &Matrix) -> Result<Vec<f64>, String> {
    validate(x)?;
    let n = x.rows;
    let d = x.cols;
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut ss = 0.0;
            for c in 0..d {
                let diff = x.get(i, c) - x.get(j, c);
                ss += diff * diff;
            }
            let dist = ss.max(0.0).sqrt();
            out[i * n + j] = dist;
            out[j * n + i] = dist;
        }
    }
    Ok(out)
}

/// Symmetric weighted k-nearest-neighbor graph as a row-major `n x n` adjacency
/// matrix. Non-edges are `inf`, the diagonal is `0`.
pub fn knn_graph(x: &Matrix, n_neighbors: usize) -> Result<Vec<f64>, String> {
    validate(x)?;
    let n = x.rows;
    if n_neighbors < 1 || n_neighbors > n - 1 {
        return Err(format!(
            "n_neighbors must be in [1, {}] for {} samples, got {}",
            n - 1,
            n,
            n_neighbors
        ));
    }
    let dist = pairwise_distances(x)?;
    let mut adj = vec![INF; n * n];
    for i in 0..n {
        adj[i * n + i] = 0.0;
        // Stable sort of the other indices by distance.
        let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        order.sort_by(|&a, &b| {
            dist[i * n + a]
                .partial_cmp(&dist[i * n + b])
                .unwrap()
                .then(a.cmp(&b))
        });
        for &j in order.iter().take(n_neighbors) {
            adj[i * n + j] = dist[i * n + j];
        }
    }
    // Symmetrize: keep an edge if it exists in either direction (min weight).
    let mut sym = vec![INF; n * n];
    for i in 0..n {
        for j in 0..n {
            sym[i * n + j] = adj[i * n + j].min(adj[j * n + i]);
        }
        sym[i * n + i] = 0.0;
    }
    Ok(sym)
}

/// All-pairs shortest-path distances via Floyd-Warshall on a row-major `n x n`
/// weighted adjacency matrix. Disconnected pairs remain `inf`.
pub fn graph_shortest_paths(adj: &[f64], n: usize) -> Vec<f64> {
    let mut d = adj.to_vec();
    for k in 0..n {
        for i in 0..n {
            let dik = d[i * n + k];
            if dik == INF {
                continue;
            }
            for j in 0..n {
                let through = dik + d[k * n + j];
                if through < d[i * n + j] {
                    d[i * n + j] = through;
                }
            }
        }
    }
    d
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

/// Flips each embedding column so its largest-magnitude entry is positive.
fn fix_signs(embedding: &mut [Vec<f64>], k: usize) {
    let n = embedding.len();
    for col in 0..k {
        let mut best_row = 0usize;
        let mut best = 0.0f64;
        for (row, erow) in embedding.iter().enumerate() {
            if erow[col].abs() > best {
                best = erow[col].abs();
                best_row = row;
            }
        }
        if embedding[best_row][col] < 0.0 {
            for erow in embedding.iter_mut().take(n) {
                erow[col] = -erow[col];
            }
        }
    }
}

/// Classical (Torgerson) MDS of a row-major `n x n` distance matrix.
///
/// Returns `(embedding, eigenvalues)`: the embedding is `n` rows of length
/// `n_components`; eigenvalues are the top `n_components` (descending, clamped
/// non-negative) eigenvalues of the centered Gram matrix.
pub fn classical_mds(
    distances: &[f64],
    n: usize,
    n_components: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    if distances.iter().any(|v| !v.is_finite()) {
        return Err(
            "distance matrix contains non-finite values; the neighborhood graph is \
             likely disconnected (increase n_neighbors)"
                .to_string(),
        );
    }
    if n_components < 1 || n_components > n {
        return Err(format!(
            "n_components must be in [1, {}], got {}",
            n, n_components
        ));
    }

    let nf = n as f64;
    // Squared distances.
    let mut d2 = vec![0.0f64; n * n];
    for i in 0..(n * n) {
        d2[i] = distances[i] * distances[i];
    }
    // Row, column, and grand means.
    let mut row_mean = vec![0.0f64; n];
    let mut col_mean = vec![0.0f64; n];
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            let v = d2[i * n + j];
            row_mean[i] += v;
            col_mean[j] += v;
            total += v;
        }
    }
    for i in 0..n {
        row_mean[i] /= nf;
        col_mean[i] /= nf;
    }
    let grand = total / (nf * nf);

    // B = -1/2 (D2 - rowmean - colmean + grand), symmetrized.
    let mut b = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = -0.5 * (d2[i * n + j] - row_mean[i] - col_mean[j] + grand);
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (b[i * n + j] + b[j * n + i]);
            b[i * n + j] = avg;
            b[j * n + i] = avg;
        }
    }

    let (eigvals, eigvecs) = jacobi_eigen(&b, n);

    let mut embedding = vec![vec![0.0f64; n_components]; n];
    let mut out_vals = vec![0.0f64; n_components];
    for c in 0..n_components {
        let lam = eigvals[c].max(0.0);
        out_vals[c] = lam;
        let scale = lam.sqrt();
        for i in 0..n {
            embedding[i][c] = eigvecs[i * n + c] * scale;
        }
    }
    fix_signs(&mut embedding, n_components);
    Ok((embedding, out_vals))
}

/// Fits an Isomap embedding of `x` into `n_components` dimensions using a
/// `n_neighbors`-nearest-neighbor graph.
pub fn fit_isomap(
    x: &Matrix,
    n_components: usize,
    n_neighbors: usize,
) -> Result<IsomapResult, String> {
    validate(x)?;
    let n = x.rows;
    if n_components < 1 || n_components > n {
        return Err(format!(
            "n_components must be in [1, {}], got {}",
            n, n_components
        ));
    }
    let adj = knn_graph(x, n_neighbors)?;
    let geo = graph_shortest_paths(&adj, n);
    if geo.iter().any(|v| !v.is_finite()) {
        return Err(
            "neighborhood graph is disconnected; some geodesic distances are \
             infinite. Increase n_neighbors."
                .to_string(),
        );
    }
    let (embedding, eigenvalues) = classical_mds(&geo, n, n_components)?;
    Ok(IsomapResult {
        embedding,
        eigenvalues,
        geodesic_distances: geo,
        n_neighbors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn fixture() -> Matrix {
        Matrix::from_rows(&[
            vec![0.0, 0.0],
            vec![1.0, 0.6],
            vec![2.2, 0.0],
            vec![3.2, 0.6],
            vec![4.6, 0.0],
            vec![6.2, 0.6],
        ])
        .unwrap()
    }

    const TOL: f64 = 1e-9;

    #[test]
    fn pairwise_distance_fixture() {
        let d = pairwise_distances(&fixture()).unwrap();
        assert!((d[1] - 1.16619037896906).abs() < TOL);
        assert_eq!(d[0], 0.0);
    }

    #[test]
    fn geodesic_distances_fixture() {
        let r = fit_isomap(&fixture(), 1, 2).unwrap();
        let geo = &r.geodesic_distances;
        assert!((geo[5] - 6.36619037896906).abs() < TOL);
        assert!((geo[6 + 4] - 4.030985786641715).abs() < TOL);
    }

    #[test]
    fn embedding1_matches_fixture() {
        let r = fit_isomap(&fixture(), 1, 2).unwrap();
        let expected: [f64; 6] = [
            -2.9371839938074595,
            -2.0650406540284134,
            -0.7481485843361929,
            0.42401358188469096,
            1.9113800872248041,
            3.4149795630625714,
        ];
        for i in 0..6 {
            assert!((r.embedding[i][0] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn eigenvalue1_matches_fixture() {
        let r = fit_isomap(&fixture(), 1, 2).unwrap();
        assert!((r.eigenvalues[0] - 28.946415792110297).abs() < TOL);
    }

    #[test]
    fn eigenvalues_two_components_match_fixture() {
        let r = fit_isomap(&fixture(), 2, 2).unwrap();
        assert!((r.eigenvalues[0] - 28.946415792110297).abs() < TOL);
        assert!((r.eigenvalues[1] - 0.34784148875645043).abs() < TOL);
    }

    #[test]
    fn embedding_preserves_order() {
        let r = fit_isomap(&fixture(), 1, 2).unwrap();
        for i in 0..5 {
            assert!(r.embedding[i][0] < r.embedding[i + 1][0]);
        }
    }

    #[test]
    fn embedding_is_centered() {
        let r = fit_isomap(&fixture(), 1, 2).unwrap();
        let s: f64 = r.embedding.iter().map(|row| row[0]).sum();
        assert!(s.abs() < 1e-9);
    }

    #[test]
    fn sign_convention_is_deterministic() {
        let r = fit_isomap(&fixture(), 2, 2).unwrap();
        for c in 0..2 {
            let mut best_row = 0usize;
            let mut best = 0.0f64;
            for (row, erow) in r.embedding.iter().enumerate() {
                if erow[c].abs() > best {
                    best = erow[c].abs();
                    best_row = row;
                }
            }
            assert!(r.embedding[best_row][c] > 0.0);
        }
    }

    #[test]
    fn shortest_paths_simple_chain() {
        let inf = f64::INFINITY;
        let adj = vec![
            0.0, 1.0, inf, 1.0, 0.0, 1.0, inf, 1.0, 0.0,
        ];
        let d = graph_shortest_paths(&adj, 3);
        assert!((d[2] - 2.0).abs() < TOL);
        assert!((d[1] - 1.0).abs() < TOL);
    }

    #[test]
    fn classical_mds_recovers_euclidean_geometry() {
        let pts = Matrix::from_rows(&[vec![0.0], vec![1.0], vec![2.0], vec![4.0]]).unwrap();
        let d = pairwise_distances(&pts).unwrap();
        let (y, _vals) = classical_mds(&d, 4, 1).unwrap();
        // Reconstructed pairwise distances must match the originals.
        for i in 0..4 {
            for j in 0..4 {
                let rec = (y[i][0] - y[j][0]).abs();
                assert!((rec - d[i * 4 + j]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn too_few_samples_errors() {
        let x = Matrix::from_rows(&[vec![1.0, 2.0]]).unwrap();
        assert!(fit_isomap(&x, 1, 1).is_err());
    }

    #[test]
    fn bad_n_neighbors_errors() {
        assert!(fit_isomap(&fixture(), 2, 10).is_err());
    }

    #[test]
    fn bad_n_components_errors() {
        assert!(fit_isomap(&fixture(), 7, 2).is_err());
    }

    #[test]
    fn disconnected_graph_errors() {
        let far = Matrix::from_rows(&[
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![100.0, 0.0],
            vec![100.1, 0.0],
        ])
        .unwrap();
        assert!(fit_isomap(&far, 1, 1).is_err());
    }
}
