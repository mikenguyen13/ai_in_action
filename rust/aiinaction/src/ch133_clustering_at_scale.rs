//! Clustering at scale: mini-batch k-means, BIRCH features, canopy, k-means||.
//!
//! Mirrors the Python module `aiinaction.ch133_clustering_at_scale` and the Julia
//! module `AIInAction.Ch133ClusteringAtScale`. The inline tests use fixtures
//! identical to the Python/Julia suites (1e-9 tolerance), which keeps the three
//! libraries at parity. A tiny self-contained LCG makes sampling reproducible
//! byte-for-byte across the three languages.

/// 32-bit linear congruential generator (Numerical Recipes constants).
pub struct Lcg {
    state: u32,
}

impl Lcg {
    const A: u32 = 1664525;
    const C: u32 = 1013904223;

    pub fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    /// Advance the state and return the new 32-bit value.
    pub fn next_u32(&mut self) -> u32 {
        // Wrapping arithmetic reproduces Python's `% 2**32`.
        self.state = Self::A.wrapping_mul(self.state).wrapping_add(Self::C);
        self.state
    }

    /// Return an integer in `[0, bound)`.
    pub fn next_below(&mut self, bound: u32) -> Result<u32, String> {
        if bound == 0 {
            return Err("bound must be positive".to_string());
        }
        Ok(self.next_u32() % bound)
    }

    /// Uniform draw in `[0, 1)` from the 32-bit stream.
    pub fn next_unit(&mut self) -> f64 {
        self.next_u32() as f64 / 4294967296.0_f64
    }
}

fn validate_matrix(points: &[Vec<f64>], name: &str) -> Result<usize, String> {
    if points.is_empty() {
        return Err(format!("{} must be non-empty", name));
    }
    let dim = points[0].len();
    if dim == 0 {
        return Err(format!("{} rows must have at least one dimension", name));
    }
    for (i, row) in points.iter().enumerate() {
        if row.len() != dim {
            return Err(format!(
                "{} are ragged: row 0 has {} dims but row {} has {}",
                name,
                dim,
                i,
                row.len()
            ));
        }
    }
    Ok(dim)
}

/// Squared Euclidean distance between two equal-length vectors.
pub fn squared_distance(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!("dimension mismatch: {} != {}", a.len(), b.len()));
    }
    if a.is_empty() {
        return Err("vectors must be non-empty".to_string());
    }
    Ok(a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum())
}

/// Return `(index, squared_distance)` of the nearest centroid. Ties -> lowest index.
pub fn nearest_centroid(point: &[f64], centroids: &[Vec<f64>]) -> Result<(usize, f64), String> {
    validate_matrix(centroids, "centroids")?;
    let mut best_j = 0usize;
    let mut best_d = squared_distance(point, &centroids[0])?;
    for j in 1..centroids.len() {
        let d = squared_distance(point, &centroids[j])?;
        if d < best_d {
            best_d = d;
            best_j = j;
        }
    }
    Ok((best_j, best_d))
}

/// Total within-cluster sum of squared distances (the k-means objective).
pub fn inertia(points: &[Vec<f64>], centroids: &[Vec<f64>]) -> Result<f64, String> {
    validate_matrix(points, "points")?;
    validate_matrix(centroids, "centroids")?;
    let mut total = 0.0;
    for p in points {
        total += nearest_centroid(p, centroids)?.1;
    }
    Ok(total)
}

/// BIRCH clustering feature `CF = (N, LS, SS)`.
#[derive(Clone, Debug)]
pub struct ClusteringFeature {
    pub n: usize,
    pub ls: Vec<f64>,
    pub ss: f64,
}

impl ClusteringFeature {
    pub fn new(n: usize, ls: Vec<f64>, ss: f64) -> Result<Self, String> {
        if ls.is_empty() {
            return Err("linear sum LS must be non-empty".to_string());
        }
        Ok(ClusteringFeature { n, ls, ss })
    }

    pub fn from_points(points: &[Vec<f64>]) -> Result<Self, String> {
        let dim = validate_matrix(points, "points")?;
        let mut ls = vec![0.0; dim];
        let mut ss = 0.0;
        for p in points {
            for d in 0..dim {
                ls[d] += p[d];
            }
            ss += p.iter().map(|v| v * v).sum::<f64>();
        }
        Ok(ClusteringFeature {
            n: points.len(),
            ls,
            ss,
        })
    }

    pub fn dim(&self) -> usize {
        self.ls.len()
    }

    pub fn merge(&self, other: &ClusteringFeature) -> Result<ClusteringFeature, String> {
        if self.dim() != other.dim() {
            return Err(format!(
                "dimension mismatch: {} != {}",
                self.dim(),
                other.dim()
            ));
        }
        let ls = self
            .ls
            .iter()
            .zip(&other.ls)
            .map(|(a, b)| a + b)
            .collect();
        ClusteringFeature::new(self.n + other.n, ls, self.ss + other.ss)
    }

    pub fn centroid(&self) -> Result<Vec<f64>, String> {
        if self.n == 0 {
            return Err("centroid is undefined for an empty CF (N=0)".to_string());
        }
        Ok(self.ls.iter().map(|v| v / self.n as f64).collect())
    }

    pub fn radius(&self) -> Result<f64, String> {
        if self.n == 0 {
            return Err("radius is undefined for an empty CF (N=0)".to_string());
        }
        let n = self.n as f64;
        let mean_ss = self.ss / n;
        let centroid_norm_sq = self.ls.iter().map(|v| v * v).sum::<f64>() / (n * n);
        let var = mean_ss - centroid_norm_sq;
        Ok(if var > 0.0 { var.sqrt() } else { 0.0 })
    }
}

/// Run mini-batch k-means and return the final centroids.
pub fn mini_batch_kmeans(
    points: &[Vec<f64>],
    centroids: &[Vec<f64>],
    batch_size: usize,
    n_iter: usize,
    seed: u32,
) -> Result<Vec<Vec<f64>>, String> {
    let dim = validate_matrix(points, "points")?;
    let cdim = validate_matrix(centroids, "centroids")?;
    if cdim != dim {
        return Err(format!(
            "centroid dim {} does not match point dim {}",
            cdim, dim
        ));
    }
    if batch_size == 0 {
        return Err("batch_size must be positive".to_string());
    }
    let mut cs: Vec<Vec<f64>> = centroids.to_vec();
    let mut counts = vec![0u64; cs.len()];
    let mut rng = Lcg::new(seed);
    let n = points.len() as u32;
    for _ in 0..n_iter {
        let batch: Vec<&Vec<f64>> = (0..batch_size)
            .map(|_| &points[rng.next_below(n).unwrap() as usize])
            .collect();
        let assignments: Vec<usize> = batch
            .iter()
            .map(|x| nearest_centroid(x, &cs).unwrap().0)
            .collect();
        for (x, &j) in batch.iter().zip(&assignments) {
            counts[j] += 1;
            let eta = 1.0 / counts[j] as f64;
            for d in 0..dim {
                cs[j][d] = (1.0 - eta) * cs[j][d] + eta * x[d];
            }
        }
    }
    Ok(cs)
}

/// Partition points into overlapping canopies using two squared-distance thresholds.
pub fn canopy_clustering(
    points: &[Vec<f64>],
    t1: f64,
    t2: f64,
    seed: u32,
) -> Result<Vec<Vec<usize>>, String> {
    validate_matrix(points, "points")?;
    if t1 < 0.0 || t2 < 0.0 {
        return Err(format!("thresholds must be non-negative, got t1={}, t2={}", t1, t2));
    }
    if t2 > t1 {
        return Err(format!("require t2 <= t1, got t1={}, t2={}", t1, t2));
    }
    let n = points.len();
    let mut pool: Vec<usize> = (0..n).collect();
    let mut rng = Lcg::new(seed);
    let mut canopies: Vec<Vec<usize>> = Vec::new();
    while !pool.is_empty() {
        let pick = rng.next_below(pool.len() as u32).unwrap() as usize;
        let center_idx = pool[pick];
        let center = &points[center_idx];
        let mut members: Vec<usize> = Vec::new();
        let mut survivors: Vec<usize> = Vec::new();
        for &idx in &pool {
            let d = squared_distance(center, &points[idx])?;
            if d <= t1 {
                members.push(idx);
            }
            if d > t2 {
                survivors.push(idx);
            }
        }
        members.sort_unstable();
        canopies.push(members);
        pool = survivors;
    }
    Ok(canopies)
}

/// Seed `k` centers with the scalable k-means|| oversampling scheme.
pub fn kmeans_parallel_init(
    points: &[Vec<f64>],
    k: usize,
    oversampling: f64,
    n_rounds: usize,
    seed: u32,
) -> Result<Vec<Vec<f64>>, String> {
    validate_matrix(points, "points")?;
    let n = points.len();
    if k == 0 {
        return Err("k must be positive".to_string());
    }
    if k > n {
        return Err(format!("k={} exceeds number of points n={}", k, n));
    }
    if oversampling <= 0.0 {
        return Err("oversampling must be positive".to_string());
    }
    let mut rng = Lcg::new(seed);
    let mut chosen: Vec<usize> = vec![rng.next_below(n as u32).unwrap() as usize];

    let min_sq = |idx: usize, chosen: &[usize]| -> f64 {
        chosen
            .iter()
            .map(|&c| squared_distance(&points[idx], &points[c]).unwrap())
            .fold(f64::INFINITY, f64::min)
    };

    for _ in 0..n_rounds {
        let dists: Vec<f64> = (0..n).map(|i| min_sq(i, &chosen)).collect();
        let phi: f64 = dists.iter().sum();
        if phi <= 0.0 {
            break;
        }
        for i in 0..n {
            if chosen.contains(&i) {
                continue;
            }
            let mut prob = oversampling * dists[i] / phi;
            if prob > 1.0 {
                prob = 1.0;
            }
            if rng.next_unit() < prob {
                chosen.push(i);
            }
        }
    }

    // De-duplicate preserving order.
    let mut candidates: Vec<usize> = Vec::new();
    for &c in &chosen {
        if !candidates.contains(&c) {
            candidates.push(c);
        }
    }

    let mut seeds: Vec<usize> = vec![candidates[0]];
    while seeds.len() < k {
        let mut best_idx: i64 = -1;
        let mut best_gain = -1.0;
        for &c in &candidates {
            if seeds.contains(&c) {
                continue;
            }
            let d = seeds
                .iter()
                .map(|&s| squared_distance(&points[c], &points[s]).unwrap())
                .fold(f64::INFINITY, f64::min);
            if d > best_gain {
                best_gain = d;
                best_idx = c as i64;
            }
        }
        if best_idx < 0 {
            for i in 0..n {
                if !seeds.contains(&i) {
                    best_idx = i as i64;
                    break;
                }
            }
        }
        seeds.push(best_idx as usize);
    }

    Ok(seeds[..k].iter().map(|&s| points[s].clone()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn points() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0], vec![0.2, 0.1], vec![0.1, 0.2],
            vec![5.0, 5.0], vec![5.2, 4.9], vec![4.9, 5.1],
            vec![0.0, 5.0], vec![0.1, 4.8], vec![-0.1, 5.2],
        ]
    }

    fn init() -> Vec<Vec<f64>> {
        vec![vec![1.0, 1.0], vec![4.0, 4.0], vec![1.0, 4.0]]
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < TOL, "{} != {}", a, b);
    }

    #[test]
    fn lcg_deterministic() {
        assert_eq!(Lcg::new(7).next_below(10).unwrap(), 8);
    }

    #[test]
    fn squared_distance_fixture() {
        approx(squared_distance(&[0.0, 0.0], &[3.0, 4.0]).unwrap(), 25.0);
    }

    #[test]
    fn nearest_centroid_fixture() {
        let (j, d) = nearest_centroid(&[0.1, 0.1], &init()).unwrap();
        assert_eq!(j, 0);
        approx(d, 1.62);
    }

    #[test]
    fn cf_fixture() {
        let cf = ClusteringFeature::from_points(&points()).unwrap();
        assert_eq!(cf.n, 9);
        approx(cf.ls[0], 15.4);
        approx(cf.ls[1], 30.299999999999997);
        approx(cf.ss, 226.27000000000004);
        let c = cf.centroid().unwrap();
        approx(c[0], 1.7111111111111112);
        approx(c[1], 3.3666666666666663);
        approx(cf.radius().unwrap(), 3.29829735349904);
    }

    #[test]
    fn cf_additivity() {
        let pts = points();
        let whole = ClusteringFeature::from_points(&pts).unwrap();
        let left = ClusteringFeature::from_points(&pts[..3]).unwrap();
        let right = ClusteringFeature::from_points(&pts[3..]).unwrap();
        let merged = left.merge(&right).unwrap();
        assert_eq!(merged.n, whole.n);
        approx(merged.ls[0], whole.ls[0]);
        approx(merged.ls[1], whole.ls[1]);
        approx(merged.ss, whole.ss);
    }

    #[test]
    fn cf_single_point_radius_zero() {
        let cf = ClusteringFeature::from_points(&[vec![2.0, 3.0]]).unwrap();
        approx(cf.radius().unwrap(), 0.0);
    }

    #[test]
    fn mini_batch_kmeans_fixture() {
        let out = mini_batch_kmeans(&points(), &init(), 4, 20, 42).unwrap();
        let expected = [
            [0.09999999999999999, 0.10769230769230771],
            [5.042307692307693, 5.000000000000001],
            [-0.0035714285714285696, 5.007142857142858],
        ];
        for (row, exp) in out.iter().zip(expected.iter()) {
            approx(row[0], exp[0]);
            approx(row[1], exp[1]);
        }
        approx(inertia(&points(), &out).unwrap(), 0.20727712534718032);
    }

    #[test]
    fn mini_batch_kmeans_zero_iter() {
        let out = mini_batch_kmeans(&points(), &init(), 4, 0, 1).unwrap();
        assert_eq!(out, init());
    }

    #[test]
    fn mini_batch_kmeans_bad_batch() {
        assert!(mini_batch_kmeans(&points(), &init(), 0, 1, 1).is_err());
    }

    #[test]
    fn canopy_fixture() {
        let out = canopy_clustering(&points(), 2.0, 1.0, 7).unwrap();
        assert_eq!(
            out,
            vec![vec![3, 4, 5], vec![6, 7, 8], vec![0, 1, 2]]
        );
    }

    #[test]
    fn canopy_threshold_order_errors() {
        assert!(canopy_clustering(&points(), 1.0, 2.0, 7).is_err());
    }

    #[test]
    fn kmeans_parallel_fixture() {
        let seeds = kmeans_parallel_init(&points(), 3, 2.0, 3, 123).unwrap();
        let expected = [[0.2, 0.1], [5.2, 4.9], [-0.1, 5.2]];
        assert_eq!(seeds.len(), 3);
        for (row, exp) in seeds.iter().zip(expected.iter()) {
            approx(row[0], exp[0]);
            approx(row[1], exp[1]);
        }
    }

    #[test]
    fn kmeans_parallel_k_too_large() {
        assert!(kmeans_parallel_init(&points(), 99, 2.0, 1, 1).is_err());
    }
}
