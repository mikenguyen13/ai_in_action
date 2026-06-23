//! External clustering-comparison metrics (Chapter 164, Rust).
//!
//! Mirrors the Python module `aiinaction.ch164_clustering_metrics` and the Julia
//! module `AIInAction.Ch164ClusteringMetrics`. The shared fixtures in the tests
//! below match the Python/Julia suites (1e-9 tolerance), which is what keeps the
//! three implementations at parity.
//!
//! Implemented here: the contingency table, Shannon entropy, mutual information,
//! normalized mutual information (four averaging methods), homogeneity,
//! completeness, the V-measure, and the Fowlkes-Mallows index. The silhouette
//! coefficient and the adjusted Rand index already live in
//! `aiinaction::ch132_clustering_validation` and are not duplicated here.
//!
//! Labels are integer slices of equal length; only the induced partition matters.
//! All logarithms are natural (nats), and since every metric is a log ratio the
//! base cancels.

use std::collections::BTreeMap;

/// How the two marginal entropies are combined when normalizing mutual information.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AverageMethod {
    Arithmetic,
    Geometric,
    Min,
    Max,
}

fn validate(labels_true: &[i64], labels_pred: &[i64]) -> Result<(), String> {
    if labels_true.len() != labels_pred.len() {
        return Err(format!(
            "length mismatch: {} != {}",
            labels_true.len(),
            labels_pred.len()
        ));
    }
    if labels_true.is_empty() {
        return Err("inputs must be non-empty".to_string());
    }
    Ok(())
}

/// Maps each distinct label to a contiguous 0-based index, in ascending order.
fn index_map(labels: &[i64]) -> BTreeMap<i64, usize> {
    let mut keys: Vec<i64> = labels.to_vec();
    keys.sort_unstable();
    keys.dedup();
    keys.into_iter().enumerate().map(|(i, v)| (v, i)).collect()
}

/// Contingency table `n_ij = |U_i intersect V_j|` as a row-major `rows x cols` grid.
///
/// Rows index the sorted distinct values of `labels_true`, columns those of
/// `labels_pred`. Returns `(table, rows, cols)`.
pub fn contingency_matrix(
    labels_true: &[i64],
    labels_pred: &[i64],
) -> Result<(Vec<Vec<u64>>, usize, usize), String> {
    validate(labels_true, labels_pred)?;
    let row_idx = index_map(labels_true);
    let col_idx = index_map(labels_pred);
    let rows = row_idx.len();
    let cols = col_idx.len();
    let mut table = vec![vec![0u64; cols]; rows];
    for (&a, &b) in labels_true.iter().zip(labels_pred.iter()) {
        table[row_idx[&a]][col_idx[&b]] += 1;
    }
    Ok((table, rows, cols))
}

/// Shannon entropy (in nats) of the partition induced by `labels`.
pub fn entropy(labels: &[i64]) -> Result<f64, String> {
    if labels.is_empty() {
        return Err("inputs must be non-empty".to_string());
    }
    let n = labels.len() as f64;
    let mut counts: BTreeMap<i64, u64> = BTreeMap::new();
    for &v in labels {
        *counts.entry(v).or_insert(0) += 1;
    }
    let mut h = 0.0;
    for &c in counts.values() {
        let p = c as f64 / n;
        h -= p * p.ln();
    }
    Ok(h)
}

/// Mutual information `I(U; V)` (in nats) between two labelings.
pub fn mutual_information(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64, String> {
    let (table, rows, cols) = contingency_matrix(labels_true, labels_pred)?;
    let n = labels_true.len() as f64;

    let mut a = vec![0.0f64; rows]; // true class sizes
    let mut b = vec![0.0f64; cols]; // predicted cluster sizes
    for i in 0..rows {
        for j in 0..cols {
            a[i] += table[i][j] as f64;
            b[j] += table[i][j] as f64;
        }
    }

    let mut mi = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let nij = table[i][j] as f64;
            if nij == 0.0 {
                continue;
            }
            mi += (nij / n) * ((n * nij) / (a[i] * b[j])).ln();
        }
    }
    // MI is provably nonnegative; floor tiny negative round-off.
    Ok(mi.max(0.0))
}

/// Normalized mutual information `I(U; V) / mean(H(U), H(V))` in `[0, 1]`.
///
/// When both labelings are trivial (a single cluster each) both entropies are 0
/// and the result is defined to be 1.0.
pub fn normalized_mutual_information(
    labels_true: &[i64],
    labels_pred: &[i64],
    method: AverageMethod,
) -> Result<f64, String> {
    let h_true = entropy(labels_true)?;
    let h_pred = entropy(labels_pred)?;
    let mi = mutual_information(labels_true, labels_pred)?;

    let denom = match method {
        AverageMethod::Arithmetic => (h_true + h_pred) / 2.0,
        AverageMethod::Geometric => (h_true * h_pred).sqrt(),
        AverageMethod::Min => h_true.min(h_pred),
        AverageMethod::Max => h_true.max(h_pred),
    };
    if denom == 0.0 {
        return Ok(1.0);
    }
    Ok(mi / denom)
}

/// Returns `(homogeneity, completeness)` for the two labelings.
fn homogeneity_completeness(
    labels_true: &[i64],
    labels_pred: &[i64],
) -> Result<(f64, f64), String> {
    let h_true = entropy(labels_true)?;
    let h_pred = entropy(labels_pred)?;
    let mi = mutual_information(labels_true, labels_pred)?;
    let homog = if h_true == 0.0 { 1.0 } else { mi / h_true };
    let compl = if h_pred == 0.0 { 1.0 } else { mi / h_pred };
    Ok((homog, compl))
}

/// Homogeneity: `1 - H(true | pred) / H(true) = I(U; V) / H(true)` in `[0, 1]`.
pub fn homogeneity(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64, String> {
    Ok(homogeneity_completeness(labels_true, labels_pred)?.0)
}

/// Completeness: `1 - H(pred | true) / H(pred) = I(U; V) / H(pred)` in `[0, 1]`.
pub fn completeness(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64, String> {
    Ok(homogeneity_completeness(labels_true, labels_pred)?.1)
}

/// V-measure: weighted harmonic mean `(1 + beta) h c / (beta h + c)`.
pub fn v_measure(labels_true: &[i64], labels_pred: &[i64], beta: f64) -> Result<f64, String> {
    if beta < 0.0 {
        return Err(format!("beta must be nonnegative, got {}", beta));
    }
    let (h, c) = homogeneity_completeness(labels_true, labels_pred)?;
    let denom = beta * h + c;
    if denom == 0.0 {
        return Ok(0.0);
    }
    Ok((1.0 + beta) * h * c / denom)
}

/// Fowlkes-Mallows index: `a / sqrt((a + b)(a + c))`, the geometric mean of
/// pairwise precision and recall. Returns 0.0 when no pair is co-clustered.
pub fn fowlkes_mallows_index(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64, String> {
    let (table, rows, cols) = contingency_matrix(labels_true, labels_pred)?;

    let comb2 = |x: f64| x * (x - 1.0) / 2.0;

    let mut a = 0.0; // pairs together in both
    for i in 0..rows {
        for j in 0..cols {
            a += comb2(table[i][j] as f64);
        }
    }
    let mut a_plus_b = 0.0; // pairs together in prediction (cluster sizes)
    for j in 0..cols {
        let mut col_sum = 0.0;
        for i in 0..rows {
            col_sum += table[i][j] as f64;
        }
        a_plus_b += comb2(col_sum);
    }
    let mut a_plus_c = 0.0; // pairs together in reference (class sizes)
    for i in 0..rows {
        let mut row_sum = 0.0;
        for j in 0..cols {
            row_sum += table[i][j] as f64;
        }
        a_plus_c += comb2(row_sum);
    }

    let denom = a_plus_b * a_plus_c;
    if denom == 0.0 {
        return Ok(0.0);
    }
    Ok(a / denom.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const LT: [i64; 10] = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2];
    const LP: [i64; 10] = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3];
    const TOL: f64 = 1e-9;

    #[test]
    fn contingency_matches_fixture() {
        let (table, rows, cols) = contingency_matrix(&LT, &LP).unwrap();
        assert_eq!(rows, 3);
        assert_eq!(cols, 4);
        let expected: [[u64; 4]; 3] = [[2, 2, 0, 0], [0, 1, 2, 0], [0, 0, 0, 3]];
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(table[i][j], expected[i][j]);
            }
        }
    }

    #[test]
    fn entropy_matches_fixture() {
        assert!((entropy(&LT).unwrap() - 1.0888999753452238).abs() < TOL);
        assert!((entropy(&LP).unwrap() - 1.366158847569202).abs() < TOL);
    }

    #[test]
    fn mutual_information_matches_fixture() {
        assert!((mutual_information(&LT, &LP).unwrap() - 0.8979457248567799).abs() < TOL);
    }

    #[test]
    fn mutual_information_symmetric() {
        let forward = mutual_information(&LT, &LP).unwrap();
        let backward = mutual_information(&LP, &LT).unwrap();
        assert!((forward - backward).abs() < TOL);
    }

    #[test]
    fn nmi_methods_match_fixture() {
        assert!(
            (normalized_mutual_information(&LT, &LP, AverageMethod::Arithmetic).unwrap()
                - 0.7315064848758445)
                .abs()
                < TOL
        );
        assert!(
            (normalized_mutual_information(&LT, &LP, AverageMethod::Geometric).unwrap()
                - 0.7362164101692431)
                .abs()
                < TOL
        );
        assert!(
            (normalized_mutual_information(&LT, &LP, AverageMethod::Min).unwrap()
                - 0.8246356370538958)
                .abs()
                < TOL
        );
        assert!(
            (normalized_mutual_information(&LT, &LP, AverageMethod::Max).unwrap()
                - 0.6572776851348504)
                .abs()
                < TOL
        );
    }

    #[test]
    fn homogeneity_completeness_match_fixture() {
        assert!((homogeneity(&LT, &LP).unwrap() - 0.8246356370538958).abs() < TOL);
        assert!((completeness(&LT, &LP).unwrap() - 0.6572776851348504).abs() < TOL);
    }

    #[test]
    fn v_measure_matches_fixture() {
        assert!((v_measure(&LT, &LP, 1.0).unwrap() - 0.7315064848758445).abs() < TOL);
        assert!((v_measure(&LT, &LP, 2.0).unwrap() - 0.7049682606092936).abs() < TOL);
    }

    #[test]
    fn fowlkes_mallows_matches_fixture() {
        assert!((fowlkes_mallows_index(&LT, &LP).unwrap() - 0.6123724356957946).abs() < TOL);
    }

    #[test]
    fn perfect_agreement_up_to_relabeling() {
        let a: [i64; 6] = [0, 0, 1, 1, 2, 2];
        let b: [i64; 6] = [2, 2, 0, 0, 1, 1];
        assert!((mutual_information(&a, &b).unwrap() - entropy(&a).unwrap()).abs() < TOL);
        assert!(
            (normalized_mutual_information(&a, &b, AverageMethod::Arithmetic).unwrap() - 1.0).abs()
                < TOL
        );
        assert!((homogeneity(&a, &b).unwrap() - 1.0).abs() < TOL);
        assert!((completeness(&a, &b).unwrap() - 1.0).abs() < TOL);
        assert!((v_measure(&a, &b, 1.0).unwrap() - 1.0).abs() < TOL);
        assert!((fowlkes_mallows_index(&a, &b).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn independent_partitions_have_zero_mi() {
        let a: [i64; 4] = [0, 0, 1, 1];
        let b: [i64; 4] = [0, 1, 0, 1];
        assert!(mutual_information(&a, &b).unwrap().abs() < TOL);
        assert!(
            normalized_mutual_information(&a, &b, AverageMethod::Arithmetic)
                .unwrap()
                .abs()
                < TOL
        );
        assert!(v_measure(&a, &b, 1.0).unwrap().abs() < TOL);
    }

    #[test]
    fn single_cluster_each_is_degenerate_perfect() {
        let a: [i64; 3] = [0, 0, 0];
        let b: [i64; 3] = [0, 0, 0];
        assert_eq!(entropy(&a).unwrap(), 0.0);
        assert_eq!(
            normalized_mutual_information(&a, &b, AverageMethod::Arithmetic).unwrap(),
            1.0
        );
        assert_eq!(homogeneity(&a, &b).unwrap(), 1.0);
        assert_eq!(completeness(&a, &b).unwrap(), 1.0);
    }

    #[test]
    fn all_singletons_fm_is_zero() {
        let a: [i64; 4] = [0, 1, 2, 3];
        let b: [i64; 4] = [3, 2, 1, 0];
        assert_eq!(fowlkes_mallows_index(&a, &b).unwrap(), 0.0);
    }

    #[test]
    fn length_mismatch_errors() {
        assert!(mutual_information(&[0, 0, 1], &[0, 1]).is_err());
    }

    #[test]
    fn empty_errors() {
        let empty: [i64; 0] = [];
        assert!(entropy(&empty).is_err());
    }

    #[test]
    fn negative_beta_errors() {
        assert!(v_measure(&LT, &LP, -1.0).is_err());
    }
}
