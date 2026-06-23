//! Ranking metrics from scratch: MRR, MAP, and NDCG (Rust).
//!
//! Mirrors the Python module `aiinaction.ch162_ranking_metrics` and the Julia
//! module `AIInAction.Ch162RankingMetrics`. The shared fixtures in the tests
//! below match the Python/Julia suites, which is what keeps the three
//! implementations at parity. std-only.
//!
//! Conventions: a *relevance list* gives the per-position relevance of items in
//! ranked order (position 1 is the top). An item is a "hit" when its relevance is
//! strictly positive. The optional cutoff `k` (`Option<usize>`) restricts to the
//! top `k` positions; `None` means the whole list. Reciprocal rank, average
//! precision, and NDCG are all `0` for a query with no relevant item.

/// Validates a relevance slice: all entries finite and non-negative.
fn validate_rel(relevances: &[f64]) -> Result<(), String> {
    for &r in relevances {
        if !r.is_finite() {
            return Err("relevances must be finite".to_string());
        }
        if r < 0.0 {
            return Err("relevances must be non-negative".to_string());
        }
    }
    Ok(())
}

fn cutoff(len: usize, k: Option<usize>) -> Result<usize, String> {
    match k {
        None => Ok(len),
        Some(0) => Err("k must be a positive integer or None, got 0".to_string()),
        Some(kk) => Ok(kk.min(len)),
    }
}

/// Reciprocal rank of the first relevant item within the top `k`, or `0.0`.
pub fn reciprocal_rank(relevances: &[f64], k: Option<usize>) -> Result<f64, String> {
    validate_rel(relevances)?;
    let cut = cutoff(relevances.len(), k)?;
    for i in 0..cut {
        if relevances[i] > 0.0 {
            return Ok(1.0 / (i as f64 + 1.0));
        }
    }
    Ok(0.0)
}

/// Mean reciprocal rank over a non-empty set of queries.
pub fn mean_reciprocal_rank(queries: &[Vec<f64>], k: Option<usize>) -> Result<f64, String> {
    if queries.is_empty() {
        return Err("queries must be non-empty".to_string());
    }
    let mut total = 0.0;
    for q in queries {
        total += reciprocal_rank(q, k)?;
    }
    Ok(total / queries.len() as f64)
}

/// Precision at cutoff `k`: fraction of the top `k` items that are relevant.
///
/// `k` may exceed the list length; the denominator is still `k`.
pub fn precision_at_k(relevances: &[f64], k: usize) -> Result<f64, String> {
    if k == 0 {
        return Err("k must be a positive integer, got 0".to_string());
    }
    validate_rel(relevances)?;
    let cut = k.min(relevances.len());
    let hits = (0..cut).filter(|&i| relevances[i] > 0.0).count();
    Ok(hits as f64 / k as f64)
}

/// Average precision for a single query under binary relevance.
///
/// `n_relevant = None` infers `R` from the positive entries; otherwise `R` must
/// be `>=` the number of hits observed within the cutoff. Returns `0.0` for
/// `R == 0`.
pub fn average_precision(
    relevances: &[f64],
    n_relevant: Option<usize>,
    k: Option<usize>,
) -> Result<f64, String> {
    validate_rel(relevances)?;
    let cut = cutoff(relevances.len(), k)?;

    let observed_hits = (0..cut).filter(|&i| relevances[i] > 0.0).count();
    let r = match n_relevant {
        None => relevances.iter().filter(|&&v| v > 0.0).count(),
        Some(nr) => {
            if nr < observed_hits {
                return Err(format!(
                    "n_relevant={} is smaller than the {} relevant items observed",
                    nr, observed_hits
                ));
            }
            nr
        }
    };
    if r == 0 {
        return Ok(0.0);
    }

    let mut hits = 0usize;
    let mut score = 0.0;
    for i in 0..cut {
        if relevances[i] > 0.0 {
            hits += 1;
            score += hits as f64 / (i as f64 + 1.0);
        }
    }
    Ok(score / r as f64)
}

/// Mean average precision over a non-empty set of queries.
///
/// `n_relevant`, when `Some`, must supply one `R` per query.
pub fn mean_average_precision(
    queries: &[Vec<f64>],
    n_relevant: Option<&[usize]>,
    k: Option<usize>,
) -> Result<f64, String> {
    if queries.is_empty() {
        return Err("queries must be non-empty".to_string());
    }
    if let Some(nr) = n_relevant {
        if nr.len() != queries.len() {
            return Err(format!(
                "n_relevant has length {} but there are {} queries",
                nr.len(),
                queries.len()
            ));
        }
    }
    let mut total = 0.0;
    for (idx, q) in queries.iter().enumerate() {
        let r = n_relevant.map(|nr| nr[idx]);
        total += average_precision(q, r, k)?;
    }
    Ok(total / queries.len() as f64)
}

/// Discounted cumulative gain at cutoff `k`.
///
/// `exponential = true` uses gain `2^rel - 1`; `false` uses raw `rel`. The
/// discount at rank `j` (1-based) is `1 / log2(j + 1)`.
pub fn dcg(relevances: &[f64], k: Option<usize>, exponential: bool) -> Result<f64, String> {
    validate_rel(relevances)?;
    let cut = cutoff(relevances.len(), k)?;
    let mut total = 0.0;
    for j in 0..cut {
        let gain = if exponential {
            2.0f64.powf(relevances[j]) - 1.0
        } else {
            relevances[j]
        };
        total += gain / (j as f64 + 2.0).log2();
    }
    Ok(total)
}

/// Normalized discounted cumulative gain at cutoff `k`.
///
/// Divides `dcg` by the ideal DCG (relevances sorted descending). Returns `0.0`
/// when the ideal DCG is `0`. Result lies in `[0, 1]`.
pub fn ndcg(relevances: &[f64], k: Option<usize>, exponential: bool) -> Result<f64, String> {
    let actual = dcg(relevances, k, exponential)?;
    let mut ideal_rel = relevances.to_vec();
    ideal_rel.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let ideal = dcg(&ideal_rel, k, exponential)?;
    if ideal == 0.0 {
        return Ok(0.0);
    }
    Ok(actual / ideal)
}

/// Mean NDCG over a non-empty set of queries.
pub fn mean_ndcg(queries: &[Vec<f64>], k: Option<usize>, exponential: bool) -> Result<f64, String> {
    if queries.is_empty() {
        return Err("queries must be non-empty".to_string());
    }
    let mut total = 0.0;
    for q in queries {
        total += ndcg(q, k, exponential)?;
    }
    Ok(total / queries.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn mrr_queries() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ]
    }

    fn ap_q1() -> [f64; 6] {
        [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    }
    fn ap_q2() -> [f64; 4] {
        [0.0, 1.0, 1.0, 0.0]
    }
    fn ndcg_q1() -> [f64; 5] {
        [3.0, 2.0, 0.0, 1.0, 2.0]
    }
    fn ndcg_q2() -> [f64; 4] {
        [0.0, 0.0, 2.0, 1.0]
    }

    #[test]
    fn reciprocal_rank_per_query() {
        let q = mrr_queries();
        assert!((reciprocal_rank(&q[0], None).unwrap() - 1.0).abs() < TOL);
        assert!((reciprocal_rank(&q[1], None).unwrap() - 0.3333333333333333).abs() < TOL);
        assert!((reciprocal_rank(&q[2], None).unwrap() - 0.5).abs() < TOL);
    }

    #[test]
    fn mean_reciprocal_rank_matches_fixture() {
        assert!((mean_reciprocal_rank(&mrr_queries(), None).unwrap() - 0.6111111111111112).abs() < TOL);
    }

    #[test]
    fn reciprocal_rank_cutoff_excludes_late_hit() {
        let xs: [f64; 4] = [0.0, 0.0, 1.0, 0.0];
        assert_eq!(reciprocal_rank(&xs, Some(2)).unwrap(), 0.0);
        assert!((reciprocal_rank(&xs, Some(3)).unwrap() - 1.0 / 3.0).abs() < TOL);
    }

    #[test]
    fn average_precision_matches_fixture() {
        assert!((average_precision(&ap_q1(), None, None).unwrap() - 0.7222222222222222).abs() < TOL);
        assert!((average_precision(&ap_q2(), None, None).unwrap() - 0.5833333333333333).abs() < TOL);
    }

    #[test]
    fn mean_average_precision_matches_fixture() {
        let queries = vec![ap_q1().to_vec(), ap_q2().to_vec()];
        assert!((mean_average_precision(&queries, None, None).unwrap() - 0.6527777777777778).abs() < TOL);
    }

    #[test]
    fn average_precision_with_explicit_n_relevant() {
        assert!((average_precision(&ap_q1(), Some(5), None).unwrap() - 0.43333333333333335).abs() < TOL);
    }

    #[test]
    fn average_precision_cutoff() {
        assert!((average_precision(&ap_q1(), None, Some(3)).unwrap() - 5.0 / 9.0).abs() < TOL);
    }

    #[test]
    fn average_precision_no_relevant_is_zero() {
        let xs: [f64; 3] = [0.0, 0.0, 0.0];
        assert_eq!(average_precision(&xs, None, None).unwrap(), 0.0);
    }

    #[test]
    fn precision_at_k_matches_fixture() {
        assert!((precision_at_k(&ap_q1(), 3).unwrap() - 0.6666666666666666).abs() < TOL);
    }

    #[test]
    fn precision_at_k_beyond_length() {
        let xs: [f64; 3] = [1.0, 0.0, 1.0];
        assert!((precision_at_k(&xs, 10).unwrap() - 0.2).abs() < TOL);
    }

    #[test]
    fn dcg_matches_fixture() {
        assert!((dcg(&ndcg_q1(), None, true).unwrap() - 10.484024240491392).abs() < TOL);
    }

    #[test]
    fn idcg_matches_fixture() {
        let ideal: [f64; 5] = [3.0, 2.0, 2.0, 1.0, 0.0];
        assert!((dcg(&ideal, None, true).unwrap() - 10.823465818787767).abs() < TOL);
    }

    #[test]
    fn ndcg_matches_fixture() {
        assert!((ndcg(&ndcg_q1(), None, true).unwrap() - 0.9686383655679718).abs() < TOL);
        assert!((ndcg(&ndcg_q2(), None, true).unwrap() - 0.531730627995306).abs() < TOL);
    }

    #[test]
    fn mean_ndcg_matches_fixture() {
        let queries = vec![ndcg_q1().to_vec(), ndcg_q2().to_vec()];
        assert!((mean_ndcg(&queries, None, true).unwrap() - 0.7501844967816389).abs() < TOL);
    }

    #[test]
    fn ndcg_ideal_ordering_is_one() {
        let xs: [f64; 5] = [3.0, 2.0, 2.0, 1.0, 0.0];
        assert!((ndcg(&xs, None, true).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn dcg_linear_gain() {
        let xs: [f64; 3] = [3.0, 0.0, 0.0];
        assert!((dcg(&xs, None, false).unwrap() - 3.0).abs() < TOL);
    }

    #[test]
    fn empty_queries_error() {
        let empty: Vec<Vec<f64>> = vec![];
        assert!(mean_reciprocal_rank(&empty, None).is_err());
        assert!(mean_average_precision(&empty, None, None).is_err());
        assert!(mean_ndcg(&empty, None, true).is_err());
    }

    #[test]
    fn negative_relevance_errors() {
        let xs: [f64; 3] = [1.0, -1.0, 0.0];
        assert!(dcg(&xs, None, true).is_err());
    }

    #[test]
    fn bad_cutoff_errors() {
        let xs: [f64; 2] = [1.0, 0.0];
        assert!(reciprocal_rank(&xs, Some(0)).is_err());
        assert!(precision_at_k(&xs, 0).is_err());
    }

    #[test]
    fn n_relevant_too_small_errors() {
        let xs: [f64; 3] = [1.0, 1.0, 1.0];
        assert!(average_precision(&xs, Some(2), None).is_err());
    }
}
