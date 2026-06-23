//! Precision-Recall curves and Average Precision from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch155_pr_curves` and the Julia module
//! `AIInAction.Ch155PrCurves`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! Given binary labels `y_i in {0, 1}` and real-valued `scores` (larger means more
//! positive), thresholding at `tau` predicts positive iff `score >= tau`. Sweeping
//! `tau` over the distinct score values from high to low traces the PR curve:
//!
//! ```text
//! precision = TP / (TP + FP),   recall = TP / P
//! ```
//!
//! `average_precision` is the rank-based AP estimator (the scikit-learn `AP`), and
//! `auprc_trapezoid` integrates the curve produced here by the trapezoidal rule.
//! This is a std-only implementation.

/// A precision-recall curve, one point per distinct threshold (decreasing order).
#[derive(Clone, Debug)]
pub struct PrCurve {
    /// Precision at each operating point, in order of decreasing threshold.
    pub precision: Vec<f64>,
    /// Recall at each operating point (non-decreasing), decreasing-threshold order.
    pub recall: Vec<f64>,
    /// Distinct score thresholds in decreasing order.
    pub thresholds: Vec<f64>,
}

impl PrCurve {
    /// Number of operating points (distinct thresholds).
    pub fn len(&self) -> usize {
        self.thresholds.len()
    }
    /// Whether the curve has no points (cannot happen for valid input).
    pub fn is_empty(&self) -> bool {
        self.thresholds.is_empty()
    }
}

fn validate(y_true: &[i32], scores: &[f64]) -> Result<(), String> {
    if y_true.len() != scores.len() {
        return Err(format!(
            "length mismatch: len(y_true)={} != len(scores)={}",
            y_true.len(),
            scores.len()
        ));
    }
    if y_true.is_empty() {
        return Err("inputs must be non-empty".to_string());
    }
    for &v in y_true {
        if v != 0 && v != 1 {
            return Err(format!("y_true must contain only 0/1 labels, found {}", v));
        }
    }
    if scores.iter().any(|v| !v.is_finite()) {
        return Err("scores contains non-finite values (nan or inf)".to_string());
    }
    if y_true.iter().sum::<i32>() == 0 {
        return Err("y_true must contain at least one positive (label 1)".to_string());
    }
    Ok(())
}

/// Distinct score values in decreasing order.
fn distinct_descending(scores: &[f64]) -> Vec<f64> {
    let mut sorted: Vec<f64> = scores.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut out: Vec<f64> = Vec::new();
    for &v in &sorted {
        if out.last().map_or(true, |&last| last != v) {
            out.push(v);
        }
    }
    out
}

/// Computes the precision-recall curve over distinct score thresholds.
///
/// `y_true` are binary labels (0/1) with at least one positive; `scores` are the
/// matching real-valued classifier scores. Points are ordered by decreasing
/// threshold; recall is non-decreasing.
pub fn pr_curve(y_true: &[i32], scores: &[f64]) -> Result<PrCurve, String> {
    validate(y_true, scores)?;
    let p_total = y_true.iter().sum::<i32>() as f64;
    let thresholds = distinct_descending(scores);

    let mut precision = Vec::with_capacity(thresholds.len());
    let mut recall = Vec::with_capacity(thresholds.len());
    for &tau in &thresholds {
        let mut tp = 0i32;
        let mut fp = 0i32;
        for (&label, &score) in y_true.iter().zip(scores.iter()) {
            if score >= tau {
                if label == 1 {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }
        let predicted_pos = tp + fp;
        let prec = if predicted_pos == 0 {
            1.0
        } else {
            tp as f64 / predicted_pos as f64
        };
        precision.push(prec);
        recall.push(tp as f64 / p_total);
    }

    Ok(PrCurve {
        precision,
        recall,
        thresholds,
    })
}

/// Rank-based average precision (the scikit-learn `AP` estimator).
///
/// Sorts instances by descending score (ties broken by original index) and
/// averages `Precision@k` at each rank where a positive appears.
pub fn average_precision(y_true: &[i32], scores: &[f64]) -> Result<f64, String> {
    validate(y_true, scores)?;
    let p_total = y_true.iter().sum::<i32>() as f64;

    // Indices sorted by descending score; ties keep ascending index order.
    let mut order: Vec<usize> = (0..y_true.len()).collect();
    order.sort_by(|&i, &j| match scores[j].partial_cmp(&scores[i]).unwrap() {
        std::cmp::Ordering::Equal => i.cmp(&j),
        other => other,
    });

    let mut ap = 0.0;
    let mut tp = 0i32;
    let mut seen = 0i32;
    for &i in &order {
        seen += 1;
        if y_true[i] == 1 {
            tp += 1;
            ap += tp as f64 / seen as f64;
        }
    }
    Ok(ap / p_total)
}

/// Area under the PR curve by the trapezoidal rule over [`pr_curve`] points.
///
/// Generally differs from (and is mildly optimistic relative to)
/// [`average_precision`] because of the curve's sawtooth structure.
pub fn auprc_trapezoid(y_true: &[i32], scores: &[f64]) -> Result<f64, String> {
    let curve = pr_curve(y_true, scores)?;
    let rec = &curve.recall;
    let prec = &curve.precision;
    let mut area = 0.0;
    for k in 1..rec.len() {
        let width = (rec[k] - rec[k - 1]).abs();
        area += width * (prec[k] + prec[k - 1]) / 2.0;
    }
    Ok(area)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn fixture() -> ([i32; 8], [f64; 8]) {
        let y: [i32; 8] = [1, 0, 1, 1, 0, 1, 0, 0];
        let s: [f64; 8] = [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.51];
        (y, s)
    }

    const TOL: f64 = 1e-9;

    #[test]
    fn pr_curve_matches_fixture() {
        let (y, s) = fixture();
        let c = pr_curve(&y, &s).unwrap();
        let exp_thr: [f64; 8] = [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.51];
        let exp_prec: [f64; 8] = [
            1.0,
            0.5,
            0.6666666666666666,
            0.75,
            0.6,
            0.6666666666666666,
            0.5714285714285714,
            0.5,
        ];
        let exp_rec: [f64; 8] = [0.25, 0.25, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0];
        assert_eq!(c.len(), 8);
        for k in 0..8 {
            assert!((c.thresholds[k] - exp_thr[k]).abs() < TOL);
            assert!((c.precision[k] - exp_prec[k]).abs() < TOL);
            assert!((c.recall[k] - exp_rec[k]).abs() < TOL);
        }
    }

    #[test]
    fn average_precision_matches_fixture() {
        let (y, s) = fixture();
        let ap = average_precision(&y, &s).unwrap();
        assert!((ap - 0.7708333333333333).abs() < TOL);
    }

    #[test]
    fn auprc_trapezoid_matches_fixture() {
        let (y, s) = fixture();
        let area = auprc_trapezoid(&y, &s).unwrap();
        assert!((area - 0.48125).abs() < TOL);
    }

    #[test]
    fn recall_is_non_decreasing() {
        let (y, s) = fixture();
        let c = pr_curve(&y, &s).unwrap();
        for k in 1..c.recall.len() {
            assert!(c.recall[k] >= c.recall[k - 1]);
        }
    }

    #[test]
    fn perfect_ranking_gives_ap_one() {
        let y: [i32; 4] = [1, 1, 0, 0];
        let s: [f64; 4] = [0.9, 0.8, 0.2, 0.1];
        assert!((average_precision(&y, &s).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn worst_ranking_gives_low_ap() {
        let y: [i32; 4] = [1, 1, 0, 0];
        let s: [f64; 4] = [0.2, 0.1, 0.9, 0.8];
        let expected = (1.0 / 3.0 + 2.0 / 4.0) / 2.0;
        assert!((average_precision(&y, &s).unwrap() - expected).abs() < TOL);
    }

    #[test]
    fn ap_invariant_to_monotone_transform() {
        let (y, s) = fixture();
        let ap1 = average_precision(&y, &s).unwrap();
        let shifted: Vec<f64> = s.iter().map(|x| 10.0 * x + 3.0).collect();
        let ap2 = average_precision(&y, &shifted).unwrap();
        assert!((ap1 - ap2).abs() < TOL);
    }

    #[test]
    fn tie_handling_is_deterministic() {
        let y: [i32; 2] = [1, 0];
        let s: [f64; 2] = [0.5, 0.5];
        assert!((average_precision(&y, &s).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn single_threshold_when_all_scores_equal() {
        let y: [i32; 3] = [1, 0, 1];
        let s: [f64; 3] = [0.5, 0.5, 0.5];
        let c = pr_curve(&y, &s).unwrap();
        assert_eq!(c.len(), 1);
        assert!((c.recall[0] - 1.0).abs() < TOL);
        assert!((c.precision[0] - 2.0 / 3.0).abs() < TOL);
    }

    #[test]
    fn length_mismatch_errors() {
        assert!(average_precision(&[1, 0], &[0.5]).is_err());
    }

    #[test]
    fn empty_errors() {
        assert!(average_precision(&[], &[]).is_err());
    }

    #[test]
    fn non_binary_label_errors() {
        assert!(pr_curve(&[1, 2, 0], &[0.1, 0.2, 0.3]).is_err());
    }

    #[test]
    fn no_positive_errors() {
        assert!(average_precision(&[0, 0, 0], &[0.1, 0.2, 0.3]).is_err());
    }

    #[test]
    fn non_finite_score_errors() {
        assert!(pr_curve(&[1, 0], &[f64::NAN, 0.3]).is_err());
    }
}
