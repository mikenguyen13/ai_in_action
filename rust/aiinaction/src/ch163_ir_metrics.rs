//! Information retrieval (IR) ranking metrics from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch163_ir_metrics` and the Julia module
//! `AIInAction.Ch163IrMetrics`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! Two input conventions are used. Binary ranked metrics (`precision_at_k`,
//! `recall_at_k`, `average_precision`, `reciprocal_rank` and their means) take a
//! slice of `0/1` relevance labels in rank order. Graded metrics (`dcg_at_k`,
//! `ndcg_at_k`) take a slice of non-negative integer relevance grades. Recall and
//! MAP additionally take `num_relevant = |R_q|`, the total relevant count for the
//! query. std-only.

/// Validates that a slice contains only `0/1` labels.
fn check_binary(relevances: &[i64]) -> Result<(), String> {
    if relevances.is_empty() {
        return Err("relevances must be non-empty".to_string());
    }
    for (i, &r) in relevances.iter().enumerate() {
        if r != 0 && r != 1 {
            return Err(format!("relevances must be binary 0/1 labels, got {} at index {}", r, i));
        }
    }
    Ok(())
}

/// Validates that a slice contains only non-negative grades.
fn check_grades(grades: &[i64]) -> Result<(), String> {
    if grades.is_empty() {
        return Err("grades must be non-empty".to_string());
    }
    for (i, &g) in grades.iter().enumerate() {
        if g < 0 {
            return Err(format!("grades must be non-negative, got {} at index {}", g, i));
        }
    }
    Ok(())
}

/// Clamps `k` to `n` after checking it is a positive integer.
fn check_k(k: usize, n: usize) -> Result<usize, String> {
    if k < 1 {
        return Err(format!("k must be a positive integer, got {}", k));
    }
    Ok(k.min(n))
}

/// Precision@k: fraction of the top `k` ranked items that are relevant.
///
/// `k` is clamped to the list length.
pub fn precision_at_k(relevances: &[i64], k: usize) -> Result<f64, String> {
    check_binary(relevances)?;
    let kk = check_k(k, relevances.len())?;
    let hits: i64 = relevances[..kk].iter().sum();
    Ok(hits as f64 / kk as f64)
}

/// Recall@k: fraction of all `num_relevant` relevant documents in the top `k`.
pub fn recall_at_k(relevances: &[i64], k: usize, num_relevant: i64) -> Result<f64, String> {
    check_binary(relevances)?;
    if num_relevant <= 0 {
        return Err(format!("num_relevant must be a positive integer, got {}", num_relevant));
    }
    let found_total: i64 = relevances.iter().sum();
    if num_relevant < found_total {
        return Err(format!(
            "num_relevant={} is smaller than the {} relevant labels present",
            num_relevant, found_total
        ));
    }
    let kk = check_k(k, relevances.len())?;
    let hits: i64 = relevances[..kk].iter().sum();
    Ok(hits as f64 / num_relevant as f64)
}

/// Average Precision (AP) for a single query.
///
/// Pass `num_relevant = None` to divide by the number of relevant labels present.
pub fn average_precision(relevances: &[i64], num_relevant: Option<i64>) -> Result<f64, String> {
    check_binary(relevances)?;
    let found_total: i64 = relevances.iter().sum();
    let nrel = num_relevant.unwrap_or(found_total);
    if nrel <= 0 {
        return Err(format!("num_relevant must be a positive integer, got {}", nrel));
    }
    if nrel < found_total {
        return Err(format!(
            "num_relevant={} is smaller than the {} relevant labels present",
            nrel, found_total
        ));
    }
    let mut hits: i64 = 0;
    let mut precision_sum = 0.0f64;
    for (idx, &r) in relevances.iter().enumerate() {
        if r == 1 {
            hits += 1;
            precision_sum += hits as f64 / (idx as f64 + 1.0);
        }
    }
    Ok(precision_sum / nrel as f64)
}

/// Mean Average Precision (MAP): AP averaged over a set of queries.
///
/// `num_relevant = None` infers each query's relevant count from its own labels.
pub fn mean_average_precision(
    rankings: &[Vec<i64>],
    num_relevant: Option<&[i64]>,
) -> Result<f64, String> {
    if rankings.is_empty() {
        return Err("rankings must contain at least one query".to_string());
    }
    let mut total = 0.0f64;
    match num_relevant {
        None => {
            for r in rankings {
                total += average_precision(r, None)?;
            }
        }
        Some(nrel) => {
            if nrel.len() != rankings.len() {
                return Err(format!(
                    "length mismatch: {} rankings != {} num_relevant entries",
                    rankings.len(),
                    nrel.len()
                ));
            }
            for (r, &n) in rankings.iter().zip(nrel) {
                total += average_precision(r, Some(n))?;
            }
        }
    }
    Ok(total / rankings.len() as f64)
}

/// Discounted Cumulative Gain at cutoff `k` (exponential-gain form).
///
/// Gain `2^rel - 1`, discount `log2(i + 1)` for 1-based rank `i`. `k` is clamped.
pub fn dcg_at_k(grades: &[i64], k: usize) -> Result<f64, String> {
    check_grades(grades)?;
    let kk = check_k(k, grades.len())?;
    let mut total = 0.0f64;
    for i in 1..=kk {
        let gain = (2.0f64).powi(grades[i - 1] as i32) - 1.0;
        total += gain / ((i as f64 + 1.0).log2());
    }
    Ok(total)
}

/// Normalized DCG at cutoff `k`, in `[0, 1]`.
///
/// `ideal_grades = None` uses `grades` sorted descending as the ideal ordering.
pub fn ndcg_at_k(grades: &[i64], k: usize, ideal_grades: Option<&[i64]>) -> Result<f64, String> {
    check_grades(grades)?;
    let mut ideal: Vec<i64> = match ideal_grades {
        None => grades.to_vec(),
        Some(g) => {
            check_grades(g)?;
            g.to_vec()
        }
    };
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let dcg = dcg_at_k(grades, k)?;
    let idcg = dcg_at_k(&ideal, k)?;
    if idcg == 0.0 {
        return Err("IDCG@k is zero (no relevant documents); NDCG is undefined".to_string());
    }
    Ok(dcg / idcg)
}

/// Reciprocal Rank: `1 / rank` of the first relevant document, or `0.0` if none.
pub fn reciprocal_rank(relevances: &[i64]) -> Result<f64, String> {
    check_binary(relevances)?;
    for (idx, &r) in relevances.iter().enumerate() {
        if r == 1 {
            return Ok(1.0 / (idx as f64 + 1.0));
        }
    }
    Ok(0.0)
}

/// Mean Reciprocal Rank (MRR): reciprocal rank averaged over queries.
pub fn mean_reciprocal_rank(rankings: &[Vec<i64>]) -> Result<f64, String> {
    if rankings.is_empty() {
        return Err("rankings must contain at least one query".to_string());
    }
    let mut total = 0.0f64;
    for r in rankings {
        total += reciprocal_rank(r)?;
    }
    Ok(total / rankings.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const RANKING: [i64; 6] = [1, 0, 1, 1, 0, 1];
    const NUM_RELEVANT: i64 = 5;
    const GRADES: [i64; 6] = [3, 2, 3, 0, 1, 2];
    const IDEAL_GRADES: [i64; 6] = [3, 3, 2, 2, 1, 0];
    const TOL: f64 = 1e-9;

    fn query_set() -> Vec<Vec<i64>> {
        vec![
            vec![1, 0, 1, 1, 0, 1],
            vec![0, 1, 0, 0, 1, 0],
            vec![0, 0, 0, 0, 0, 1],
        ]
    }

    #[test]
    fn precision_at_k_matches_fixture() {
        assert!((precision_at_k(&RANKING, 1).unwrap() - 1.0).abs() < TOL);
        assert!((precision_at_k(&RANKING, 3).unwrap() - 0.6666666666666666).abs() < TOL);
        assert!((precision_at_k(&RANKING, 6).unwrap() - 0.6666666666666666).abs() < TOL);
    }

    #[test]
    fn precision_k_is_clamped() {
        assert!((precision_at_k(&RANKING, 100).unwrap() - 0.6666666666666666).abs() < TOL);
    }

    #[test]
    fn recall_at_k_matches_fixture() {
        assert!((recall_at_k(&RANKING, 3, NUM_RELEVANT).unwrap() - 0.4).abs() < TOL);
        assert!((recall_at_k(&RANKING, 6, NUM_RELEVANT).unwrap() - 0.8).abs() < TOL);
    }

    #[test]
    fn average_precision_matches_fixture() {
        assert!((average_precision(&RANKING, Some(NUM_RELEVANT)).unwrap() - 0.6166666666666666).abs() < TOL);
    }

    #[test]
    fn average_precision_default_num_relevant() {
        assert!((average_precision(&RANKING, None).unwrap() - 0.7708333333333333).abs() < TOL);
    }

    #[test]
    fn average_precision_perfect_is_one() {
        assert!((average_precision(&[1, 1, 1], None).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn dcg_at_k_matches_fixture() {
        assert!((dcg_at_k(&GRADES, 3).unwrap() - 12.392789260714373).abs() < TOL);
    }

    #[test]
    fn ndcg_at_k_self_ideal_matches_fixture() {
        assert!((ndcg_at_k(&GRADES, 3, None).unwrap() - 0.9594535145926796).abs() < TOL);
    }

    #[test]
    fn ndcg_ideal_ranking_is_one() {
        assert!((ndcg_at_k(&IDEAL_GRADES, 6, None).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn ndcg_with_external_pool_matches_fixture() {
        assert!((ndcg_at_k(&GRADES, 6, Some(&IDEAL_GRADES)).unwrap() - 0.9488107485678985).abs() < TOL);
    }

    #[test]
    fn reciprocal_rank_matches_fixture() {
        assert!((reciprocal_rank(&RANKING).unwrap() - 1.0).abs() < TOL);
        assert!((reciprocal_rank(&[0, 0, 1, 0]).unwrap() - 1.0 / 3.0).abs() < TOL);
    }

    #[test]
    fn reciprocal_rank_no_hit_is_zero() {
        assert_eq!(reciprocal_rank(&[0, 0, 0]).unwrap(), 0.0);
    }

    #[test]
    fn mean_average_precision_matches_fixture() {
        assert!((mean_average_precision(&query_set(), None).unwrap() - 0.46249999999999997).abs() < TOL);
    }

    #[test]
    fn mean_reciprocal_rank_matches_fixture() {
        assert!((mean_reciprocal_rank(&query_set()).unwrap() - 0.5555555555555556).abs() < TOL);
    }

    #[test]
    fn precision_rejects_non_binary() {
        assert!(precision_at_k(&[1, 2, 0], 3).is_err());
    }

    #[test]
    fn precision_rejects_nonpositive_k() {
        assert!(precision_at_k(&[1, 0, 1], 0).is_err());
    }

    #[test]
    fn recall_rejects_num_relevant_too_small() {
        assert!(recall_at_k(&[1, 1, 1], 3, 2).is_err());
    }

    #[test]
    fn recall_rejects_nonpositive_num_relevant() {
        assert!(recall_at_k(&[1, 0, 1], 3, 0).is_err());
    }

    #[test]
    fn grades_reject_negative() {
        assert!(dcg_at_k(&[1, -1, 2], 3).is_err());
    }

    #[test]
    fn ndcg_zero_idcg_errors() {
        assert!(ndcg_at_k(&[0, 0, 0], 3, None).is_err());
    }

    #[test]
    fn empty_inputs_error() {
        assert!(precision_at_k(&[], 1).is_err());
        assert!(dcg_at_k(&[], 1).is_err());
    }

    #[test]
    fn empty_query_set_errors() {
        let empty: Vec<Vec<i64>> = vec![];
        assert!(mean_average_precision(&empty, None).is_err());
        assert!(mean_reciprocal_rank(&empty).is_err());
    }

    #[test]
    fn map_length_mismatch_errors() {
        let rankings = vec![vec![1, 0], vec![0, 1]];
        let nrel = [1i64];
        assert!(mean_average_precision(&rankings, Some(&nrel)).is_err());
    }
}
