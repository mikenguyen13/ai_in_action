//! Local Outlier Factor (LOF) from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch146_lof` and the Julia module
//! `AIInAction.Ch146Lof`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! This is a std-only implementation of the Local Outlier Factor of Breunig,
//! Kriegel, Ng, and Sander (2000). Each point is scored by how much sparser its
//! local neighborhood is than the neighborhoods of its own `k` nearest neighbors.
//! Distance ties are broken by ascending point index, so neighborhoods are
//! deterministic and identical across the three languages.

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
                return Err("all rows must have the same number of features".to_string());
            }
            for &v in r {
                if !v.is_finite() {
                    return Err("X contains non-finite values (nan or inf)".to_string());
                }
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
    fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

/// Euclidean (L2) distance between two equal-length vectors.
pub fn euclidean(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!("length mismatch: {} != {}", a.len(), b.len()));
    }
    Ok(a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt())
}

fn check_k(k: usize, n: usize) -> Result<(), String> {
    if k < 1 || k > n - 1 {
        return Err(format!("k must be in [1, {}] for {} points, got {}", n - 1, n, k));
    }
    Ok(())
}

fn pairwise(x: &Matrix) -> Vec<Vec<f64>> {
    let n = x.rows;
    let mut dist = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean(x.row(i), x.row(j)).unwrap();
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

/// Indices of the `k` nearest neighbors of point `i`, ties broken by index.
fn neighbors(dist_row: &[f64], i: usize, k: usize, n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
    order.sort_by(|&a, &b| {
        dist_row[a]
            .partial_cmp(&dist_row[b])
            .unwrap()
            .then(a.cmp(&b))
    });
    order.truncate(k);
    order
}

/// Returns, for each point, the indices of its `k` nearest neighbors.
pub fn knn_distances(x: &Matrix, k: usize) -> Result<Vec<Vec<usize>>, String> {
    let n = x.rows;
    check_k(k, n)?;
    let dist = pairwise(x);
    Ok((0..n).map(|i| neighbors(&dist[i], i, k, n)).collect())
}

/// Returns the `k`-distance (distance to the `k`-th nearest neighbor) of each point.
pub fn k_distance(x: &Matrix, k: usize) -> Result<Vec<f64>, String> {
    let n = x.rows;
    check_k(k, n)?;
    let dist = pairwise(x);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let nbrs = neighbors(&dist[i], i, k, n);
        out.push(dist[i][*nbrs.last().unwrap()]);
    }
    Ok(out)
}

fn lrd_from(dist: &[Vec<f64>], neighbors: &[Vec<usize>], kdist: &[f64]) -> Vec<f64> {
    let n = dist.len();
    let mut lrd_vals = vec![0.0f64; n];
    for i in 0..n {
        let nbrs = &neighbors[i];
        let mut total = 0.0;
        for &y in nbrs {
            total += kdist[y].max(dist[i][y]);
        }
        let mean_reach = total / nbrs.len() as f64;
        lrd_vals[i] = if mean_reach == 0.0 {
            f64::INFINITY
        } else {
            1.0 / mean_reach
        };
    }
    lrd_vals
}

/// Local reachability density of each point.
pub fn lrd(x: &Matrix, k: usize) -> Result<Vec<f64>, String> {
    let n = x.rows;
    check_k(k, n)?;
    let dist = pairwise(x);
    let neighbors: Vec<Vec<usize>> = (0..n).map(|i| neighbors(&dist[i], i, k, n)).collect();
    let kdist: Vec<f64> = (0..n).map(|i| dist[i][*neighbors[i].last().unwrap()]).collect();
    Ok(lrd_from(&dist, &neighbors, &kdist))
}

/// Local Outlier Factor score of every point. Values near 1 are inliers;
/// values much greater than 1 are local outliers.
pub fn lof_scores(x: &Matrix, k: usize) -> Result<Vec<f64>, String> {
    let n = x.rows;
    check_k(k, n)?;
    let dist = pairwise(x);
    let neighbors: Vec<Vec<usize>> = (0..n).map(|i| neighbors(&dist[i], i, k, n)).collect();
    let kdist: Vec<f64> = (0..n).map(|i| dist[i][*neighbors[i].last().unwrap()]).collect();
    let lrd_vals = lrd_from(&dist, &neighbors, &kdist);

    let mut scores = vec![0.0f64; n];
    for i in 0..n {
        let nbrs = &neighbors[i];
        let li = lrd_vals[i];
        if li.is_infinite() {
            // x coincides with its neighbors: ratio collapses to 0.
            scores[i] = 0.0;
        } else {
            let mut total = 0.0;
            for &y in nbrs {
                let ly = lrd_vals[y];
                total += if ly.is_infinite() {
                    f64::INFINITY
                } else {
                    ly / li
                };
            }
            scores[i] = total / nbrs.len() as f64;
        }
    }
    Ok(scores)
}

/// Indices of the `m` highest-LOF points, most anomalous first.
/// Ties in score are broken by ascending point index.
pub fn top_anomalies(x: &Matrix, k: usize, m: usize) -> Result<Vec<usize>, String> {
    if m < 1 {
        return Err(format!("m must be a positive integer, got {}", m));
    }
    let scores = lof_scores(x, k)?;
    let n = scores.len();
    if m > n {
        return Err(format!("m must be in [1, {}] for {} points, got {}", n, n, m));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap()
            .then(a.cmp(&b))
    });
    order.truncate(m);
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn fixture() -> Matrix {
        Matrix::from_rows(&[
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![8.0, 8.0],
        ])
        .unwrap()
    }

    const K: usize = 2;
    const TOL: f64 = 1e-9;

    #[test]
    fn euclidean_basic() {
        let a: [f64; 2] = [0.0, 0.0];
        let b: [f64; 2] = [3.0, 4.0];
        assert!((euclidean(&a, &b).unwrap() - 5.0).abs() < TOL);
    }

    #[test]
    fn knn_indices_match_fixture() {
        let nbrs = knn_distances(&fixture(), K).unwrap();
        let expected: [[usize; 2]; 5] = [[1, 2], [0, 3], [0, 3], [1, 2], [3, 1]];
        for i in 0..5 {
            assert_eq!(nbrs[i], expected[i].to_vec());
        }
    }

    #[test]
    fn k_distance_matches_fixture() {
        let kd = k_distance(&fixture(), K).unwrap();
        let expected: [f64; 5] = [1.0, 1.0, 1.0, 1.0, 10.63014581273465];
        for i in 0..5 {
            assert!((kd[i] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn lrd_matches_fixture() {
        let v = lrd(&fixture(), K).unwrap();
        let expected: [f64; 5] = [1.0, 1.0, 1.0, 1.0, 0.09742011681639788];
        for i in 0..5 {
            assert!((v[i] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn lof_scores_match_fixture() {
        let s = lof_scores(&fixture(), K).unwrap();
        let expected: [f64; 5] = [1.0, 1.0, 1.0, 1.0, 10.264820374673157];
        for i in 0..5 {
            assert!((s[i] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn inliers_have_lof_near_one() {
        let s = lof_scores(&fixture(), K).unwrap();
        for i in 0..4 {
            assert!((s[i] - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn outlier_has_largest_lof() {
        let s = lof_scores(&fixture(), K).unwrap();
        let mut best = 0usize;
        for i in 1..s.len() {
            if s[i] > s[best] {
                best = i;
            }
        }
        assert_eq!(best, 4);
    }

    #[test]
    fn top_anomalies_orders_by_score() {
        assert_eq!(top_anomalies(&fixture(), K, 1).unwrap(), vec![4]);
        assert_eq!(top_anomalies(&fixture(), K, 2).unwrap()[0], 4);
    }

    #[test]
    fn duplicate_point_finite() {
        let x = Matrix::from_rows(&[
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ])
        .unwrap();
        let s = lof_scores(&x, 2).unwrap();
        assert_eq!(s.len(), 4);
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bad_k_errors() {
        assert!(lof_scores(&fixture(), 5).is_err());
        assert!(lof_scores(&fixture(), 0).is_err());
    }

    #[test]
    fn non_finite_errors() {
        assert!(Matrix::from_rows(&[vec![0.0, 0.0], vec![f64::NAN, 1.0]]).is_err());
    }

    #[test]
    fn ragged_rows_error() {
        assert!(Matrix::from_rows(&[vec![0.0, 0.0], vec![1.0]]).is_err());
    }

    #[test]
    fn euclidean_length_mismatch_errors() {
        let a: [f64; 2] = [1.0, 2.0];
        let b: [f64; 1] = [1.0];
        assert!(euclidean(&a, &b).is_err());
    }

    #[test]
    fn bad_m_errors() {
        assert!(top_anomalies(&fixture(), K, 0).is_err());
        assert!(top_anomalies(&fixture(), K, 99).is_err());
    }
}
