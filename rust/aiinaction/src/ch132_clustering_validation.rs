//! Clustering validation metrics (Chapter 132).
//!
//! From-scratch, std-only implementations of four clustering validation indices,
//! mirroring the Python module `aiinaction.ch132_clustering_validation` and the
//! Julia module `AIInAction.Ch132ClusteringValidation`. The shared fixtures in the
//! tests below match the Python/Julia suites to keep the three at parity.
//!
//! Points are passed as `&[Vec<f64>]` (each inner vector a sample of equal
//! dimension); labels are an `&[i64]` of the same length. Label values are
//! arbitrary; only the induced partition matters.

/// Number of distinct labels and a stable, sorted list of them.
fn unique_sorted(labels: &[i64]) -> Vec<i64> {
    let mut v: Vec<i64> = labels.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

fn check_matrix(x: &[Vec<f64>]) -> Result<usize, String> {
    if x.is_empty() {
        return Err("X must contain at least one sample".to_string());
    }
    let dim = x[0].len();
    for row in x {
        if row.len() != dim {
            return Err("all samples in X must have the same dimension".to_string());
        }
    }
    Ok(dim)
}

fn check_labels(x: &[Vec<f64>], labels: &[i64]) -> Result<Vec<i64>, String> {
    check_matrix(x)?;
    if labels.len() != x.len() {
        return Err(format!(
            "length mismatch: len(labels)={} != n_samples={}",
            labels.len(),
            x.len()
        ));
    }
    let uniq = unique_sorted(labels);
    let k = uniq.len();
    if k < 2 {
        return Err(format!("need at least 2 clusters, got {}", k));
    }
    if k > x.len() {
        return Err(format!(
            "number of clusters ({}) cannot exceed number of samples ({})",
            k,
            x.len()
        ));
    }
    Ok(uniq)
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q) * (p - q))
        .sum::<f64>()
        .sqrt()
}

fn centroid(points: &[&Vec<f64>], dim: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; dim];
    for p in points {
        for (ci, pi) in c.iter_mut().zip(p.iter()) {
            *ci += pi;
        }
    }
    let n = points.len() as f64;
    for ci in c.iter_mut() {
        *ci /= n;
    }
    c
}

/// Mean silhouette coefficient over all samples (Euclidean distance).
///
/// `s(i) = (b(i) - a(i)) / max(a(i), b(i))`, with `a` the mean intra-cluster
/// distance and `b` the minimum mean distance to another cluster. Points alone in
/// their cluster contribute `0`.
pub fn silhouette_score(x: &[Vec<f64>], labels: &[i64]) -> Result<f64, String> {
    let uniq = check_labels(x, labels)?;
    let n = x.len();

    // Pairwise distance matrix.
    let mut dist = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean(&x[i], &x[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // Members per cluster.
    let members: Vec<Vec<usize>> = uniq
        .iter()
        .map(|&c| {
            (0..n)
                .filter(|&i| labels[i] == c)
                .collect::<Vec<usize>>()
        })
        .collect();
    // label value -> cluster index
    let cluster_of = |lbl: i64| uniq.iter().position(|&c| c == lbl).unwrap();

    let mut total = 0.0_f64;
    for i in 0..n {
        let own = cluster_of(labels[i]);
        let own_members = &members[own];
        if own_members.len() <= 1 {
            // s = 0
            continue;
        }
        let a_i: f64 = own_members.iter().map(|&j| dist[i][j]).sum::<f64>()
            / (own_members.len() as f64 - 1.0);
        let mut b_i = f64::INFINITY;
        for (ci, mem) in members.iter().enumerate() {
            if ci == own {
                continue;
            }
            let mean_to_c: f64 = mem.iter().map(|&j| dist[i][j]).sum::<f64>() / mem.len() as f64;
            if mean_to_c < b_i {
                b_i = mean_to_c;
            }
        }
        let denom = a_i.max(b_i);
        if denom != 0.0 {
            total += (b_i - a_i) / denom;
        }
    }
    Ok(total / n as f64)
}

/// Davies-Bouldin index (lower is better; 0 is ideal).
pub fn davies_bouldin_index(x: &[Vec<f64>], labels: &[i64]) -> Result<f64, String> {
    let uniq = check_labels(x, labels)?;
    let dim = x[0].len();
    let k = uniq.len();

    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut scatter = vec![0.0_f64; k];
    for (j, &c) in uniq.iter().enumerate() {
        let pts: Vec<&Vec<f64>> = x
            .iter()
            .zip(labels)
            .filter(|(_, &l)| l == c)
            .map(|(p, _)| p)
            .collect();
        let mu = centroid(&pts, dim);
        scatter[j] = pts.iter().map(|p| euclidean(p, &mu)).sum::<f64>() / pts.len() as f64;
        centroids.push(mu);
    }

    let mut total = 0.0_f64;
    for j in 0..k {
        let mut worst = 0.0_f64;
        for m in 0..k {
            if m == j {
                continue;
            }
            let sep = euclidean(&centroids[j], &centroids[m]);
            if sep == 0.0 {
                return Err(format!(
                    "clusters {} and {} have identical centroids; Davies-Bouldin is \
                     undefined (zero separation)",
                    uniq[j], uniq[m]
                ));
            }
            let ratio = (scatter[j] + scatter[m]) / sep;
            if ratio > worst {
                worst = ratio;
            }
        }
        total += worst;
    }
    Ok(total / k as f64)
}

/// Calinski-Harabasz index, a.k.a. variance ratio criterion (higher is better).
pub fn calinski_harabasz_index(x: &[Vec<f64>], labels: &[i64]) -> Result<f64, String> {
    let uniq = check_labels(x, labels)?;
    let dim = x[0].len();
    let n = x.len();
    let k = uniq.len();

    let all: Vec<&Vec<f64>> = x.iter().collect();
    let grand = centroid(&all, dim);

    let mut between = 0.0_f64;
    let mut within = 0.0_f64;
    for &c in &uniq {
        let pts: Vec<&Vec<f64>> = x
            .iter()
            .zip(labels)
            .filter(|(_, &l)| l == c)
            .map(|(p, _)| p)
            .collect();
        let mu = centroid(&pts, dim);
        let sq_to_grand: f64 = mu
            .iter()
            .zip(&grand)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        between += pts.len() as f64 * sq_to_grand;
        for p in &pts {
            within += p
                .iter()
                .zip(&mu)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>();
        }
    }
    if within == 0.0 {
        return Err("within-cluster scatter is zero; Calinski-Harabasz is undefined".to_string());
    }
    Ok((between / within) * (n as f64 - k as f64) / (k as f64 - 1.0))
}

/// Adjusted Rand Index between two labelings (external, chance-corrected).
///
/// Returns 1.0 for perfect agreement, ~0.0 for chance, possibly negative for
/// worse-than-chance. Degenerate partitions (both trivial) yield 1.0.
pub fn adjusted_rand_index(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64, String> {
    if labels_true.len() != labels_pred.len() {
        return Err(format!(
            "length mismatch: len(labels_true)={} != len(labels_pred)={}",
            labels_true.len(),
            labels_pred.len()
        ));
    }
    let n = labels_true.len();
    if n == 0 {
        return Err("inputs must be non-empty".to_string());
    }

    let rows = unique_sorted(labels_true);
    let cols = unique_sorted(labels_pred);
    let row_of = |l: i64| rows.iter().position(|&v| v == l).unwrap();
    let col_of = |l: i64| cols.iter().position(|&v| v == l).unwrap();

    let mut table = vec![vec![0_i64; cols.len()]; rows.len()];
    for i in 0..n {
        table[row_of(labels_true[i])][col_of(labels_pred[i])] += 1;
    }

    let comb2 = |x: i64| -> f64 {
        let x = x as f64;
        x * (x - 1.0) / 2.0
    };

    let mut sum_ij = 0.0_f64;
    for row in &table {
        for &cnt in row {
            sum_ij += comb2(cnt);
        }
    }
    let sum_a: f64 = table
        .iter()
        .map(|row| comb2(row.iter().sum::<i64>()))
        .sum();
    let sum_b: f64 = (0..cols.len())
        .map(|j| comb2(table.iter().map(|row| row[j]).sum::<i64>()))
        .sum();

    let total_pairs = n as f64 * (n as f64 - 1.0) / 2.0;
    let expected = if total_pairs > 0.0 {
        sum_a * sum_b / total_pairs
    } else {
        0.0
    };
    let max_index = 0.5 * (sum_a + sum_b);
    let denom = max_index - expected;
    if denom == 0.0 {
        return Ok(1.0);
    }
    Ok((sum_ij - expected) / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn fixture_x() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 1.0],
            vec![1.5, 2.0],
            vec![1.0, 0.5],
            vec![8.0, 8.0],
            vec![8.5, 7.5],
            vec![7.5, 8.5],
        ]
    }
    const LABELS: [i64; 6] = [0, 0, 0, 1, 1, 1];
    const LABELS_TRUE: [i64; 6] = [0, 0, 0, 1, 1, 1];
    const LABELS_PRED: [i64; 6] = [0, 0, 1, 1, 2, 2];
    const TOL: f64 = 1e-9;

    #[test]
    fn silhouette_matches_fixture() {
        let s = silhouette_score(&fixture_x(), &LABELS).unwrap();
        assert!((s - 0.8954900167230767).abs() < TOL);
    }

    #[test]
    fn davies_bouldin_matches_fixture() {
        let db = davies_bouldin_index(&fixture_x(), &LABELS).unwrap();
        assert!((db - 0.11157205284841143).abs() < TOL);
    }

    #[test]
    fn calinski_harabasz_matches_fixture() {
        let ch = calinski_harabasz_index(&fixture_x(), &LABELS).unwrap();
        assert!((ch - 240.14285714285714).abs() < 1e-7);
    }

    #[test]
    fn ari_matches_fixture() {
        let ari = adjusted_rand_index(&LABELS_TRUE, &LABELS_PRED).unwrap();
        assert!((ari - 0.24242424242424246).abs() < TOL);
    }

    #[test]
    fn silhouette_one_d_fixture() {
        let x = vec![vec![0.0], vec![0.1], vec![10.0], vec![10.1]];
        let s = silhouette_score(&x, &[0, 0, 1, 1]).unwrap();
        assert!((s - 0.9899997499937498).abs() < TOL);
    }

    #[test]
    fn ari_perfect_and_relabel() {
        assert!((adjusted_rand_index(&LABELS_TRUE, &LABELS_TRUE).unwrap() - 1.0).abs() < TOL);
        assert!((adjusted_rand_index(&[0, 0, 1, 1], &[1, 1, 0, 0]).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn ari_worse_than_chance() {
        let val =
            adjusted_rand_index(&[0, 0, 0, 0, 1, 1, 1, 1], &[0, 1, 0, 1, 0, 1, 0, 1]).unwrap();
        assert!(val <= 1e-9);
    }

    #[test]
    fn ari_degenerate() {
        assert!((adjusted_rand_index(&[0, 0, 0], &[5, 5, 5]).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn single_cluster_errors() {
        assert!(silhouette_score(&vec![vec![0.0], vec![1.0], vec![2.0]], &[0, 0, 0]).is_err());
        assert!(davies_bouldin_index(&vec![vec![0.0], vec![1.0], vec![2.0]], &[0, 0, 0]).is_err());
        assert!(calinski_harabasz_index(&vec![vec![0.0], vec![1.0], vec![2.0]], &[0, 0, 0]).is_err());
    }

    #[test]
    fn length_mismatch_errors() {
        assert!(silhouette_score(&vec![vec![0.0], vec![1.0], vec![2.0]], &[0, 1]).is_err());
        assert!(adjusted_rand_index(&[0, 0, 1], &[0, 1]).is_err());
    }

    #[test]
    fn davies_bouldin_identical_centroids_errors() {
        let x = vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]];
        assert!(davies_bouldin_index(&x, &[0, 1, 0, 1]).is_err());
    }

    #[test]
    fn calinski_harabasz_zero_within_errors() {
        assert!(calinski_harabasz_index(&vec![vec![0.0], vec![5.0]], &[0, 1]).is_err());
    }
}
