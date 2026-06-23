//! K-Means variants and extensions (Chapter 124).
//!
//! Mirrors the Python module `aiinaction.ch124_kmeans_variants` and the Julia
//! module `AIInAction.Ch124KMeansVariants`. The inline tests use the identical
//! shared fixtures (1e-9 tolerance) that the Python and Julia suites assert
//! against, keeping the three implementations at parity.
//!
//! std-only: data is passed as `&[Vec<f64>]` row-major matrices and `&[f64]`
//! vectors; every routine is deterministic given its inputs.

/// Squared Euclidean distance between two equal-length points.
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn check_matrix(x: &[Vec<f64>], name: &str) -> Result<usize, String> {
    if x.is_empty() {
        return Err(format!("{name} must contain at least one sample"));
    }
    let d = x[0].len();
    if d == 0 {
        return Err(format!("{name} must have at least one feature"));
    }
    for row in x {
        if row.len() != d {
            return Err(format!("{name} rows must all have the same length"));
        }
    }
    Ok(d)
}

/// One iteration of Lloyd's algorithm: assign to nearest centroid (ties to the
/// lowest index), then recompute each centroid as the mean of its members. A
/// centroid with no members is left unchanged.
pub fn lloyd_step(
    x: &[Vec<f64>],
    centroids: &[Vec<f64>],
) -> Result<(Vec<usize>, Vec<Vec<f64>>), String> {
    let d = check_matrix(x, "X")?;
    let k = check_matrix(centroids, "centroids")?;
    if k != d {
        return Err(format!("centroid dimension {k} does not match data dimension {d}"));
    }
    let labels: Vec<usize> = x
        .iter()
        .map(|xi| {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let dd = sq_dist(xi, c);
                if dd < best_d {
                    best_d = dd;
                    best = j;
                }
            }
            best
        })
        .collect();
    let mut new_c = centroids.to_vec();
    for j in 0..centroids.len() {
        let members: Vec<&Vec<f64>> = x
            .iter()
            .zip(&labels)
            .filter(|(_, &l)| l == j)
            .map(|(xi, _)| xi)
            .collect();
        if !members.is_empty() {
            let mut mean = vec![0.0; d];
            for m in &members {
                for t in 0..d {
                    mean[t] += m[t];
                }
            }
            for t in 0..d {
                mean[t] /= members.len() as f64;
            }
            new_c[j] = mean;
        }
    }
    Ok((labels, new_c))
}

/// Within-cluster sum of squared distances to the nearest centroid (the K-Means
/// objective).
pub fn inertia(x: &[Vec<f64>], centroids: &[Vec<f64>]) -> Result<f64, String> {
    let d = check_matrix(x, "X")?;
    let k = check_matrix(centroids, "centroids")?;
    if k != d {
        return Err(format!("centroid dimension {k} does not match data dimension {d}"));
    }
    let mut total = 0.0;
    for xi in x {
        let mut best = f64::INFINITY;
        for c in centroids {
            let dd = sq_dist(xi, c);
            if dd < best {
                best = dd;
            }
        }
        total += best;
    }
    Ok(total)
}

/// One mini-batch K-Means update (Sculley, 2010). Points are processed in order;
/// each is assigned to its nearest centroid, its count is incremented, and the
/// centroid moves toward the point with learning rate `1 / count`.
pub fn mini_batch_update(
    centroids: &[Vec<f64>],
    counts: &[f64],
    batch: &[Vec<f64>],
) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    let k = check_matrix(centroids, "centroids")?;
    if counts.len() != centroids.len() {
        return Err(format!(
            "counts length {} does not match number of centroids {}",
            counts.len(),
            centroids.len()
        ));
    }
    if counts.iter().any(|&c| c < 0.0) {
        return Err("counts must be non-negative".to_string());
    }
    let bd = check_matrix(batch, "batch")?;
    if bd != k {
        return Err(format!("batch dimension {bd} does not match centroid dimension {k}"));
    }
    let mut c = centroids.to_vec();
    let mut cnt = counts.to_vec();
    for x in batch {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (j, cj) in c.iter().enumerate() {
            let dd = sq_dist(x, cj);
            if dd < best_d {
                best_d = dd;
                best = j;
            }
        }
        cnt[best] += 1.0;
        let eta = 1.0 / cnt[best];
        for t in 0..k {
            c[best][t] = (1.0 - eta) * c[best][t] + eta * x[t];
        }
    }
    Ok((c, cnt))
}

/// Coordinatewise (lower) median: the L1-optimal k-medians representative.
pub fn kmedians_centroid(members: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    let d = check_matrix(members, "members")?;
    let n = members.len();
    let idx = (n - 1) / 2; // lower median for even n
    let mut out = vec![0.0; d];
    for t in 0..d {
        let mut col: Vec<f64> = members.iter().map(|m| m[t]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out[t] = col[idx];
    }
    Ok(out)
}

/// Assign every point to its nearest medoid given an `n x n` dissimilarity matrix.
/// Returns `(labels, total_cost)` where labels index into `medoid_indices`.
pub fn pam_assign_cost(
    distances: &[Vec<f64>],
    medoid_indices: &[usize],
) -> Result<(Vec<usize>, f64), String> {
    let n = distances.len();
    if n == 0 || distances.iter().any(|r| r.len() != n) {
        return Err("distances must be a square n x n dissimilarity matrix".to_string());
    }
    if medoid_indices.is_empty() {
        return Err("medoid_indices must be non-empty".to_string());
    }
    for &m in medoid_indices {
        if m >= n {
            return Err(format!("medoid index {m} out of range [0, {n})"));
        }
    }
    let mut labels = Vec::with_capacity(n);
    let mut total = 0.0;
    for i in 0..n {
        let mut best_pos = 0usize;
        let mut best_cost = f64::INFINITY;
        for (pos, &m) in medoid_indices.iter().enumerate() {
            let c = distances[i][m];
            if c < best_cost {
                best_cost = c;
                best_pos = pos;
            }
        }
        labels.push(best_pos);
        total += best_cost;
    }
    Ok((labels, total))
}

/// Gaussian (RBF) kernel matrix `K[i][j] = exp(-gamma ||x_i - x_j||^2)`.
pub fn rbf_kernel_matrix(x: &[Vec<f64>], gamma: f64) -> Result<Vec<Vec<f64>>, String> {
    if gamma <= 0.0 {
        return Err(format!("gamma must be positive, got {gamma}"));
    }
    check_matrix(x, "X")?;
    let n = x.len();
    let mut k = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            k[i][j] = (-gamma * sq_dist(&x[i], &x[j])).exp();
        }
    }
    Ok(k)
}

/// Feature-space squared distances from each point to every cluster mean via the
/// kernel trick. An empty cluster yields `+inf` for every point.
pub fn kernel_assignment_distances(
    kernel: &[Vec<f64>],
    labels: &[usize],
    n_clusters: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let n = kernel.len();
    if n == 0 || kernel.iter().any(|r| r.len() != n) {
        return Err("kernel must be a square n x n matrix".to_string());
    }
    if labels.len() != n {
        return Err(format!("labels length {} does not match kernel size {n}", labels.len()));
    }
    if n_clusters == 0 {
        return Err("n_clusters must be positive".to_string());
    }
    for &c in labels {
        if c >= n_clusters {
            return Err(format!("label {c} out of range [0, {n_clusters})"));
        }
    }
    let members: Vec<Vec<usize>> = (0..n_clusters)
        .map(|j| (0..n).filter(|&i| labels[i] == j).collect())
        .collect();
    let third: Vec<f64> = members
        .iter()
        .map(|s| {
            if s.is_empty() {
                f64::INFINITY
            } else {
                let mut sum = 0.0;
                for &l in s {
                    for &m in s {
                        sum += kernel[l][m];
                    }
                }
                sum / (s.len() as f64).powi(2)
            }
        })
        .collect();
    let mut out = vec![vec![0.0; n_clusters]; n];
    for i in 0..n {
        for j in 0..n_clusters {
            let s = &members[j];
            if s.is_empty() {
                out[i][j] = f64::INFINITY;
                continue;
            }
            let cross: f64 = s.iter().map(|&l| kernel[i][l]).sum();
            out[i][j] = kernel[i][i] - 2.0 * cross / s.len() as f64 + third[j];
        }
    }
    Ok(out)
}

/// Membership-weighted fuzzy c-means centroids
/// `c_j = sum_i u_ij^m x_i / sum_i u_ij^m`.
pub fn fuzzy_centroids(
    x: &[Vec<f64>],
    memberships: &[Vec<f64>],
    m: f64,
) -> Result<Vec<Vec<f64>>, String> {
    if m <= 1.0 {
        return Err(format!("fuzziness exponent m must be greater than 1, got {m}"));
    }
    let d = check_matrix(x, "X")?;
    if memberships.len() != x.len() {
        return Err("memberships must have shape (n_samples, n_clusters)".to_string());
    }
    let k = memberships[0].len();
    let mut num = vec![vec![0.0; d]; k];
    let mut den = vec![0.0; k];
    for (xi, ui) in x.iter().zip(memberships) {
        for j in 0..k {
            let w = ui[j].powf(m);
            den[j] += w;
            for t in 0..d {
                num[j][t] += w * xi[t];
            }
        }
    }
    for j in 0..k {
        if den[j] == 0.0 {
            return Err("a cluster has zero total membership; cannot form its centroid".to_string());
        }
        for t in 0..d {
            num[j][t] /= den[j];
        }
    }
    Ok(num)
}

/// Update fuzzy memberships from distances to all centroids (Bezdek FCM).
/// A point coinciding with one or more centroids splits its membership uniformly
/// across those zero-distance centroids.
pub fn fuzzy_memberships(
    x: &[Vec<f64>],
    centroids: &[Vec<f64>],
    m: f64,
) -> Result<Vec<Vec<f64>>, String> {
    if m <= 1.0 {
        return Err(format!("fuzziness exponent m must be greater than 1, got {m}"));
    }
    let d = check_matrix(x, "X")?;
    let kd = check_matrix(centroids, "centroids")?;
    if kd != d {
        return Err(format!("centroid dimension {kd} does not match data dimension {d}"));
    }
    let k = centroids.len();
    let p = 2.0 / (m - 1.0);
    let mut out = vec![vec![0.0; k]; x.len()];
    for (i, xi) in x.iter().enumerate() {
        let dist: Vec<f64> = centroids.iter().map(|c| sq_dist(xi, c).sqrt()).collect();
        let zeros: Vec<usize> = (0..k).filter(|&j| dist[j] == 0.0).collect();
        if !zeros.is_empty() {
            for &j in &zeros {
                out[i][j] = 1.0 / zeros.len() as f64;
            }
            continue;
        }
        for j in 0..k {
            let denom: f64 = (0..k).map(|l| (dist[j] / dist[l]).powf(p)).sum();
            out[i][j] = 1.0 / denom;
        }
    }
    Ok(out)
}

/// Run 2-means to convergence to bisect a cluster. Returns
/// `(labels in {0,1}, two_centroids, sse)`.
pub fn bisecting_split(
    x: &[Vec<f64>],
    init_two_centroids: &[Vec<f64>],
) -> Result<(Vec<usize>, Vec<Vec<f64>>, f64), String> {
    if init_two_centroids.len() != 2 {
        return Err(format!(
            "bisecting requires exactly 2 initial centroids, got {}",
            init_two_centroids.len()
        ));
    }
    let mut centroids = init_two_centroids.to_vec();
    let mut labels: Vec<usize> = Vec::new();
    let mut prev: Option<Vec<usize>> = None;
    for _ in 0..1000 {
        let (l, c) = lloyd_step(x, &centroids)?;
        labels = l;
        centroids = c;
        if let Some(ref p) = prev {
            if *p == labels {
                break;
            }
        }
        prev = Some(labels.clone());
    }
    let sse = inertia(x, &centroids)?;
    Ok((labels, centroids, sse))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < TOL, "expected {b}, got {a}");
    }

    fn approx2d(a: &[Vec<f64>], b: &[Vec<f64>]) {
        assert_eq!(a.len(), b.len());
        for (ra, rb) in a.iter().zip(b) {
            assert_eq!(ra.len(), rb.len());
            for (x, y) in ra.iter().zip(rb) {
                approx(*x, *y);
            }
        }
    }

    fn x_blobs() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.5],
            vec![0.5, 1.0],
            vec![8.0, 8.0],
            vec![9.0, 8.5],
            vec![8.5, 9.0],
        ]
    }

    #[test]
    fn lloyd_step_fixture() {
        let (labels, c) = lloyd_step(&x_blobs(), &[vec![0.0, 0.0], vec![9.0, 9.0]]).unwrap();
        assert_eq!(labels, vec![0, 0, 0, 1, 1, 1]);
        approx2d(&c, &[vec![0.5, 0.5], vec![8.5, 8.5]]);
        approx(inertia(&x_blobs(), &c).unwrap(), 2.0);
    }

    #[test]
    fn mini_batch_update_fixture() {
        let batch = vec![
            vec![1.0, 1.0],
            vec![9.0, 9.0],
            vec![2.0, 0.0],
            vec![8.0, 10.0],
        ];
        let (c, cnt) =
            mini_batch_update(&[vec![0.0, 0.0], vec![10.0, 10.0]], &[0.0, 0.0], &batch).unwrap();
        approx2d(&c, &[vec![1.5, 0.5], vec![8.5, 9.5]]);
        assert_eq!(cnt, vec![2.0, 2.0]);
    }

    #[test]
    fn kmedians_fixture() {
        let c = kmedians_centroid(&[vec![1.0, 5.0], vec![2.0, 100.0], vec![3.0, 6.0]]).unwrap();
        approx2d(&[c], &[vec![2.0, 6.0]]);
        let c2 = kmedians_centroid(&[vec![1.0], vec![2.0], vec![3.0], vec![100.0]]).unwrap();
        approx(c2[0], 2.0);
    }

    #[test]
    fn pam_fixture() {
        let d = vec![
            vec![0.0, 2.0, 5.0, 9.0],
            vec![2.0, 0.0, 4.0, 7.0],
            vec![5.0, 4.0, 0.0, 3.0],
            vec![9.0, 7.0, 3.0, 0.0],
        ];
        let (labels, cost) = pam_assign_cost(&d, &[0, 3]).unwrap();
        assert_eq!(labels, vec![0, 0, 1, 1]);
        approx(cost, 5.0);
    }

    #[test]
    fn kernel_fixture() {
        let xk = vec![vec![0.0], vec![0.2], vec![5.0], vec![5.2]];
        let k = rbf_kernel_matrix(&xk, 0.5).unwrap();
        approx(k[0][1], 0.9801986733067553);
        approx(k[0][0], 1.0);
        let kd = kernel_assignment_distances(&k, &[0, 0, 1, 1], 2).unwrap();
        approx(kd[0][0], 0.009900663346622318);
        approx(kd[0][1], 1.990094266187928);
    }

    #[test]
    fn fuzzy_fixture() {
        let xf = vec![vec![0.0], vec![1.0], vec![5.0], vec![6.0]];
        let u = fuzzy_memberships(&xf, &[vec![0.5], vec![5.5]], 2.0).unwrap();
        approx(u[0][0], 0.9918032786885246);
        approx(u[0][1], 0.00819672131147541);
        for row in &u {
            approx(row.iter().sum::<f64>(), 1.0);
        }
        let c = fuzzy_centroids(&xf, &u, 2.0).unwrap();
        approx(c[0][0], 0.4985105160463291);
        approx(c[1][0], 5.501489483953671);
    }

    #[test]
    fn bisecting_fixture() {
        let xb = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.3],
            vec![8.0, 8.0],
            vec![8.4, 7.6],
        ];
        let (labels, c, sse) = bisecting_split(&xb, &[vec![0.0, 0.0], vec![8.0, 8.0]]).unwrap();
        assert_eq!(labels, vec![0, 0, 1, 1]);
        approx2d(&c, &[vec![0.25, 0.15], vec![8.2, 7.8]]);
        approx(sse, 0.33);
    }

    #[test]
    fn fuzzy_on_centroid() {
        let u = fuzzy_memberships(&[vec![0.5], vec![5.5]], &[vec![0.5], vec![5.5]], 2.0).unwrap();
        approx2d(&u, &[vec![1.0, 0.0], vec![0.0, 1.0]]);
    }

    #[test]
    fn kernel_empty_cluster_inf() {
        let xk = vec![vec![0.0], vec![0.2], vec![5.0], vec![5.2]];
        let k = rbf_kernel_matrix(&xk, 0.5).unwrap();
        let kd = kernel_assignment_distances(&k, &[0, 0, 0, 0], 2).unwrap();
        for row in &kd {
            assert!(row[1].is_infinite());
        }
    }

    #[test]
    fn validation_errors() {
        assert!(rbf_kernel_matrix(&[vec![0.0]], 0.0).is_err());
        assert!(fuzzy_memberships(&[vec![0.0]], &[vec![0.0]], 1.0).is_err());
        assert!(lloyd_step(&[vec![0.0, 0.0]], &[vec![0.0]]).is_err());
        assert!(pam_assign_cost(&[vec![0.0, 1.0], vec![1.0, 0.0]], &[0, 99]).is_err());
        assert!(bisecting_split(&[vec![0.0]], &[vec![0.0]]).is_err());
    }
}
