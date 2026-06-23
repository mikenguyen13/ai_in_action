//! SMOTE: Synthetic Minority Over-sampling Technique (Chapter 118).
//!
//! Mirrors the Python module `aiinaction.ch118_smote` and the Julia module
//! `AIInAction.Ch118Smote`. SMOTE rebalances an imbalanced training set by
//! synthesizing new minority-class examples through linear interpolation between a
//! minority point and one of its `k` nearest minority neighbors:
//!
//! ```text
//! x_new = x_i + lambda * (x_nn - x_i),   lambda ~ Uniform(0, 1).
//! ```
//!
//! All randomness flows through a fixed linear-congruential generator (the
//! Numerical Recipes constants) so the three languages emit identical synthetic
//! points to floating-point tolerance for a given seed.

const LCG_A: u64 = 1664525;
const LCG_C: u64 = 1013904223;
const LCG_M: u64 = 1 << 32;

/// Deterministic linear-congruential generator shared across all three languages.
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Construct from a non-negative seed.
    pub fn new(seed: u64) -> Self {
        Lcg {
            state: seed % LCG_M,
        }
    }

    /// Advance the generator and return the raw 32-bit state.
    pub fn next_uint(&mut self) -> u64 {
        self.state = (LCG_A.wrapping_mul(self.state).wrapping_add(LCG_C)) % LCG_M;
        self.state
    }

    /// Next pseudo-random float in `[0, 1)`.
    pub fn next_float(&mut self) -> f64 {
        self.next_uint() as f64 / LCG_M as f64
    }

    /// Next pseudo-random integer in `[0, n)`.
    pub fn next_index(&mut self, n: usize) -> Result<usize, String> {
        if n == 0 {
            return Err("n must be positive".to_string());
        }
        Ok((self.next_uint() % n as u64) as usize)
    }
}

/// Euclidean distance between two equal-length feature vectors.
pub fn euclidean(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!("dimension mismatch: {} != {}", a.len(), b.len()));
    }
    if a.is_empty() {
        return Err("vectors must be non-empty".to_string());
    }
    let total: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    Ok(total.sqrt())
}

/// Indices of the `k` nearest neighbors of `points[idx]`, excluding itself.
/// Ties are broken by the lower point index, so the result is deterministic.
pub fn k_nearest(points: &[Vec<f64>], idx: usize, k: usize) -> Result<Vec<usize>, String> {
    let n = points.len();
    if n == 0 {
        return Err("points must be non-empty".to_string());
    }
    if idx >= n {
        return Err(format!("idx {} out of range for {} points", idx, n));
    }
    if k == 0 {
        return Err("k must be positive".to_string());
    }
    if k > n - 1 {
        return Err(format!("k={} exceeds available neighbors {}", k, n - 1));
    }
    let mut dists: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
    for j in 0..n {
        if j != idx {
            dists.push((euclidean(&points[idx], &points[j])?, j));
        }
    }
    dists.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    Ok(dists.iter().take(k).map(|&(_, j)| j).collect())
}

/// One synthetic point on the segment from `x_i` toward `x_nn`:
/// `x_i + lam * (x_nn - x_i)` componentwise.
pub fn smote_sample(x_i: &[f64], x_nn: &[f64], lam: f64) -> Result<Vec<f64>, String> {
    if x_i.len() != x_nn.len() {
        return Err(format!("dimension mismatch: {} != {}", x_i.len(), x_nn.len()));
    }
    if !(0.0..=1.0).contains(&lam) {
        return Err(format!("lam must be in [0, 1], got {}", lam));
    }
    Ok(x_i
        .iter()
        .zip(x_nn)
        .map(|(xi, xn)| xi + lam * (xn - xi))
        .collect())
}

/// Generate `n_synthetic` synthetic minority examples via SMOTE.
///
/// For each synthetic point: pick a base minority index round-robin, draw one of
/// its `k` nearest neighbors via the LCG, then draw `lambda` via the LCG. The draw
/// order (neighbor first, then lambda) is fixed and shared across languages.
pub fn smote(
    minority: &[Vec<f64>],
    n_synthetic: usize,
    k: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, String> {
    let n = minority.len();
    if n == 0 {
        return Err("minority set must be non-empty".to_string());
    }
    let dim = minority[0].len();
    if dim == 0 {
        return Err("feature vectors must be non-empty".to_string());
    }
    for row in minority {
        if row.len() != dim {
            return Err("all minority rows must have the same dimension".to_string());
        }
    }
    if n_synthetic == 0 {
        return Ok(Vec::new());
    }
    if k == 0 {
        return Err("k must be positive".to_string());
    }
    if k > n - 1 {
        return Err(format!(
            "k={} exceeds available neighbors {}; need at least k+1 minority points",
            k,
            n - 1
        ));
    }

    let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n);
    for i in 0..n {
        neighbors.push(k_nearest(minority, i, k)?);
    }

    let mut rng = Lcg::new(seed);
    let mut synthetic: Vec<Vec<f64>> = Vec::with_capacity(n_synthetic);
    for s in 0..n_synthetic {
        let base = s % n;
        let nn_choice = rng.next_index(k)?;
        let nn = neighbors[base][nn_choice];
        let lam = rng.next_float();
        synthetic.push(smote_sample(&minority[base], &minority[nn], lam)?);
    }
    Ok(synthetic)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;
    const SEED: u64 = 42;
    const K: usize = 2;

    fn minority() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![3.0, 1.0],
        ]
    }

    // Synthetic points from smote(minority, 4, k=2, seed=42).
    const EXPECTED_SMOTE: [[f64; 2]; 4] = [
        [0.17625009082257748, 0.0],
        [1.222554265987128, 0.777445734012872],
        [2.0256639048457146, 0.02566390484571457],
        [2.763079992495477, 1.0],
    ];

    // LCG(42).next_float() sequence.
    const EXPECTED_LCG: [f64; 4] = [
        0.252345174784,
        0.088125045411,
        0.577281198232,
        0.222554265987,
    ];

    #[test]
    fn euclidean_3_4_5() {
        assert!((euclidean(&[0.0, 0.0], &[3.0, 4.0]).unwrap() - 5.0).abs() < TOL);
    }

    #[test]
    fn lcg_sequence_matches_fixture() {
        let mut rng = Lcg::new(SEED);
        for &expected in EXPECTED_LCG.iter() {
            assert!((rng.next_float() - expected).abs() < TOL);
        }
    }

    #[test]
    fn lcg_floats_in_unit_interval() {
        let mut rng = Lcg::new(123);
        for _ in 0..100 {
            let v = rng.next_float();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn k_nearest_matches_fixture() {
        let m = minority();
        assert_eq!(k_nearest(&m, 0, K).unwrap(), vec![1, 2]);
        assert_eq!(k_nearest(&m, 1, K).unwrap(), vec![0, 2]);
        assert_eq!(k_nearest(&m, 2, K).unwrap(), vec![1, 3]);
        assert_eq!(k_nearest(&m, 3, K).unwrap(), vec![2, 1]);
    }

    #[test]
    fn smote_sample_midpoint() {
        assert_eq!(smote_sample(&[0.0, 0.0], &[2.0, 4.0], 0.5).unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn smote_matches_fixture() {
        let m = minority();
        let pts = smote(&m, 4, K, SEED).unwrap();
        assert_eq!(pts.len(), 4);
        for (got, expected) in pts.iter().zip(EXPECTED_SMOTE.iter()) {
            assert!((got[0] - expected[0]).abs() < TOL);
            assert!((got[1] - expected[1]).abs() < TOL);
        }
    }

    #[test]
    fn smote_zero_synthetic_empty() {
        let m = minority();
        assert!(smote(&m, 0, K, SEED).unwrap().is_empty());
    }

    #[test]
    fn euclidean_dimension_mismatch_errors() {
        assert!(euclidean(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn k_nearest_k_too_large_errors() {
        let m = minority();
        assert!(k_nearest(&m, 0, 4).is_err());
    }

    #[test]
    fn smote_k_too_large_errors() {
        let m = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        assert!(smote(&m, 3, 5, SEED).is_err());
    }

    #[test]
    fn smote_empty_minority_errors() {
        let m: Vec<Vec<f64>> = Vec::new();
        assert!(smote(&m, 3, 2, SEED).is_err());
    }

    #[test]
    fn smote_sample_lambda_out_of_range_errors() {
        assert!(smote_sample(&[0.0], &[1.0], 1.5).is_err());
    }
}
