//! Agglomerative hierarchical clustering from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch125_agglomerative_clustering` and the
//! Julia module `AIInAction.Ch125AgglomerativeClustering`. The shared fixtures in
//! the tests below match the Python/Julia suites, which keeps the three at parity.
//!
//! `linkage_matrix` returns SciPy-style rows `[node_a, node_b, height, size]`:
//! original points have ids `0..n-1` and the merge formed at step `t` has id
//! `n + t`. Supported linkages: single, complete, average, Ward. Memory is
//! `O(n^2)` and time `O(n^3)`, which keeps the code transparent for the small
//! didactic datasets in this chapter.

/// Supported linkage names, in the same order as the Python `LINKAGES` tuple.
pub const LINKAGES: [&str; 4] = ["single", "complete", "average", "ward"];

/// One linkage-matrix row: `[node_a, node_b, height, size]`.
pub type LinkRow = [f64; 4];

fn lance_williams(
    d_ak: f64,
    d_bk: f64,
    d_ab: f64,
    n_a: usize,
    n_b: usize,
    n_k: usize,
    linkage: &str,
) -> f64 {
    match linkage {
        "single" => 0.5 * d_ak + 0.5 * d_bk - 0.5 * (d_ak - d_bk).abs(),
        "complete" => 0.5 * d_ak + 0.5 * d_bk + 0.5 * (d_ak - d_bk).abs(),
        "average" => {
            let total = (n_a + n_b) as f64;
            (n_a as f64 / total) * d_ak + (n_b as f64 / total) * d_bk
        }
        "ward" => {
            let total = (n_a + n_b + n_k) as f64;
            (n_a + n_k) as f64 / total * d_ak + (n_b + n_k) as f64 / total * d_bk
                - n_k as f64 / total * d_ab
        }
        _ => unreachable!("unknown linkage validated by caller"),
    }
}

fn validate_points(points: &[Vec<f64>]) -> Result<usize, String> {
    if points.is_empty() {
        return Err("inputs must be non-empty".to_string());
    }
    let width = points[0].len();
    if width == 0 {
        return Err("points must have at least one feature".to_string());
    }
    for row in points {
        if row.len() != width {
            return Err("all points must have the same number of features".to_string());
        }
    }
    Ok(width)
}

fn pairwise_sq_euclidean(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s: f64 = points[i]
                .iter()
                .zip(&points[j])
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            dist[i][j] = s;
            dist[j][i] = s;
        }
    }
    dist
}

/// Agglomerative clustering of `points`; returns a SciPy-style linkage matrix.
///
/// For Ward the reported height is the Euclidean (non-squared) merge distance.
pub fn linkage_matrix(points: &[Vec<f64>], linkage: &str) -> Result<Vec<LinkRow>, String> {
    if !LINKAGES.contains(&linkage) {
        return Err(format!(
            "unknown linkage {:?}; expected one of {:?}",
            linkage, LINKAGES
        ));
    }
    validate_points(points)?;
    let n = points.len();
    if n < 2 {
        return Err("need at least 2 points to cluster".to_string());
    }

    // Ward works in squared-Euclidean space; the others use raw distances.
    let mut dist = pairwise_sq_euclidean(points);
    if linkage != "ward" {
        for row in dist.iter_mut() {
            for v in row.iter_mut() {
                *v = v.sqrt();
            }
        }
    }

    let mut active: Vec<usize> = (0..n).collect();
    let mut sizes: Vec<usize> = vec![1; n]; // indexed by cluster id
    let mut next_id = n;
    let mut result: Vec<LinkRow> = Vec::with_capacity(n - 1);

    for _ in 0..(n - 1) {
        // Closest active pair.
        let mut best = f64::INFINITY;
        let (mut bi, mut bj) = (0usize, 1usize);
        for a_idx in 0..active.len() {
            for b_idx in (a_idx + 1)..active.len() {
                let d = dist[active[a_idx]][active[b_idx]];
                if d < best {
                    best = d;
                    bi = a_idx;
                    bj = b_idx;
                }
            }
        }
        let ca = active[bi];
        let cb = active[bj];
        let (n_a, n_b) = (sizes[ca], sizes[cb]);

        let height = if linkage == "ward" { best.sqrt() } else { best };
        let (node_a, node_b) = if ca < cb { (ca, cb) } else { (cb, ca) };
        result.push([node_a as f64, node_b as f64, height, (n_a + n_b) as f64]);

        // Lance-Williams update, computed before growing the matrix.
        let new_id = next_id;
        next_id += 1;
        let mut updates: Vec<(usize, f64)> = Vec::new();
        for &ck in &active {
            if ck == ca || ck == cb {
                continue;
            }
            let d_new =
                lance_williams(dist[ca][ck], dist[cb][ck], best, n_a, n_b, sizes[ck], linkage);
            updates.push((ck, d_new));
        }
        // Grow the matrix by one row/column for new_id.
        for row in dist.iter_mut() {
            row.push(0.0);
        }
        dist.push(vec![0.0; new_id + 1]);
        for (ck, d_new) in updates {
            dist[new_id][ck] = d_new;
            dist[ck][new_id] = d_new;
        }

        active.retain(|&c| c != ca && c != cb);
        active.push(new_id);
        sizes.push(n_a + n_b);
    }

    Ok(result)
}

/// Cut the tree to obtain exactly `n_clusters` flat clusters.
///
/// Labels are assigned in increasing order of the smallest original-point id in
/// each cluster, matching the Python/Julia implementations.
pub fn fcluster(linkage_mat: &[LinkRow], n_clusters: usize) -> Result<Vec<usize>, String> {
    let n = linkage_mat.len() + 1;
    if n_clusters < 1 || n_clusters > n {
        return Err(format!("n_clusters must be in 1..{}, got {}", n, n_clusters));
    }

    let mut members: Vec<Option<Vec<usize>>> = vec![None; 2 * n - 1];
    for (i, slot) in members.iter_mut().enumerate().take(n) {
        *slot = Some(vec![i]);
    }
    let merges_to_apply = n - n_clusters;
    for t in 0..merges_to_apply {
        let a = linkage_mat[t][0] as usize;
        let b = linkage_mat[t][1] as usize;
        let ma = members[a].take().unwrap();
        let mb = members[b].take().unwrap();
        let mut merged = ma;
        merged.extend(mb);
        members[n + t] = Some(merged);
    }

    let mut clusters: Vec<Vec<usize>> = members.into_iter().flatten().collect();
    clusters.sort_by_key(|g| *g.iter().min().unwrap());
    let mut labels = vec![0usize; n];
    for (cid, group) in clusters.iter().enumerate() {
        for &p in group {
            labels[p] = cid;
        }
    }
    Ok(labels)
}

/// Cophenetic distance matrix induced by a linkage matrix.
pub fn cophenetic_distances(linkage_mat: &[LinkRow]) -> Vec<Vec<f64>> {
    let n = linkage_mat.len() + 1;
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); 2 * n - 1];
    for i in 0..n {
        members[i] = vec![i];
    }
    let mut coph = vec![vec![0.0; n]; n];
    for (t, row) in linkage_mat.iter().enumerate() {
        let a = row[0] as usize;
        let b = row[1] as usize;
        let height = row[2];
        let ma = members[a].clone();
        let mb = members[b].clone();
        for &x in &ma {
            for &y in &mb {
                coph[x][y] = height;
                coph[y][x] = height;
            }
        }
        let mut merged = ma;
        merged.extend(mb);
        members[n + t] = merged;
    }
    coph
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn pts() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0],
            vec![10.0, 11.0],
        ]
    }

    const TOL: f64 = 1e-9;

    // (third-merge height, root height) per linkage.
    fn expected(linkage: &str) -> (f64, f64) {
        match linkage {
            "single" => (1.0, 13.45362404707371),
            "complete" => (1.414213562373095, 14.866068747318508),
            "average" => (1.2071067811865475, 14.045043082079953),
            "ward" => (1.2909944487358056, 21.73323108360405),
            _ => unreachable!(),
        }
    }

    #[test]
    fn first_merge_is_nearest_pair() {
        for lk in LINKAGES {
            let m = linkage_matrix(&pts(), lk).unwrap();
            assert_eq!(m[0][0] as usize, 0);
            assert_eq!(m[0][1] as usize, 1);
            assert!((m[0][2] - 1.0).abs() < TOL);
            assert!((m[0][3] - 2.0).abs() < TOL);
        }
    }

    #[test]
    fn final_two_heights_match_fixture() {
        for lk in LINKAGES {
            let m = linkage_matrix(&pts(), lk).unwrap();
            let (h_third, h_root) = expected(lk);
            assert!((m[2][2] - h_third).abs() < TOL);
            assert!((m[3][2] - h_root).abs() < TOL);
            assert!((m[3][3] - 5.0).abs() < TOL);
        }
    }

    #[test]
    fn linkage_shape_and_monotone() {
        for lk in LINKAGES {
            let m = linkage_matrix(&pts(), lk).unwrap();
            assert_eq!(m.len(), 4);
            for row in &m {
                assert!(row[0] < row[1]);
            }
            for w in m.windows(2) {
                assert!(w[1][2] >= w[0][2] - TOL);
            }
        }
    }

    #[test]
    fn fcluster_two_groups() {
        for lk in LINKAGES {
            let m = linkage_matrix(&pts(), lk).unwrap();
            assert_eq!(fcluster(&m, 2).unwrap(), vec![0, 0, 0, 1, 1]);
            assert_eq!(fcluster(&m, 5).unwrap(), vec![0, 1, 2, 3, 4]);
            assert_eq!(fcluster(&m, 1).unwrap(), vec![0, 0, 0, 0, 0]);
        }
    }

    #[test]
    fn cophenetic_single() {
        let m = linkage_matrix(&pts(), "single").unwrap();
        let coph = cophenetic_distances(&m);
        let root = expected("single").1;
        assert!((coph[0][1] - 1.0).abs() < TOL);
        assert!((coph[3][4] - 1.0).abs() < TOL);
        assert!((coph[0][3] - root).abs() < TOL);
        assert_eq!(coph[0][0], 0.0);
    }

    #[test]
    fn collinear_example() {
        let m = linkage_matrix(&vec![vec![0.0], vec![0.0], vec![5.0]], "single").unwrap();
        assert_eq!(m[0][0] as usize, 0);
        assert_eq!(m[0][1] as usize, 1);
        assert!((m[0][2] - 0.0).abs() < TOL);
        assert!((m[1][2] - 5.0).abs() < TOL);
        assert!((m[1][3] - 3.0).abs() < TOL);
    }

    #[test]
    fn errors() {
        assert!(linkage_matrix(&pts(), "centroid").is_err());
        assert!(linkage_matrix(&vec![vec![1.0, 2.0]], "single").is_err());
        assert!(linkage_matrix(&Vec::<Vec<f64>>::new(), "single").is_err());
        assert!(linkage_matrix(&vec![vec![0.0, 0.0], vec![1.0]], "single").is_err());
        let m = linkage_matrix(&pts(), "ward").unwrap();
        assert!(fcluster(&m, 0).is_err());
        assert!(fcluster(&m, 6).is_err());
    }
}
