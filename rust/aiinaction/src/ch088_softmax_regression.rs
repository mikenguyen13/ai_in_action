//! Multiclass logistic regression (softmax regression) from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch088_softmax_regression` and the
//! Julia module `AIInAction.Ch088SoftmaxRegression`. The shared fixtures in the
//! tests below match the Python/Julia suites (1e-9 tolerance), which is what
//! keeps the three implementations at parity. std-only, row-major matrices
//! represented as `Vec<Vec<f64>>`.

/// Row-wise numerically stable softmax.
///
/// Each inner vector of `z` (the logits for one example) is mapped to a
/// probability distribution. The per-row maximum is subtracted before
/// exponentiating to avoid overflow without changing the result.
pub fn softmax(z: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    if z.is_empty() {
        return Err("z must be non-empty".to_string());
    }
    let ncols = z[0].len();
    if ncols == 0 {
        return Err("z must be non-empty".to_string());
    }
    let mut out = Vec::with_capacity(z.len());
    for row in z {
        if row.len() != ncols {
            return Err("z rows must all have the same length".to_string());
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err("z must contain only finite values".to_string());
        }
        let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = exps.iter().sum();
        out.push(exps.iter().map(|e| e / s).collect());
    }
    Ok(out)
}

/// Mean multiclass cross-entropy (negative log likelihood).
pub fn cross_entropy(probs: &[Vec<f64>], y: &[usize]) -> Result<f64, String> {
    if probs.is_empty() || probs[0].is_empty() {
        return Err("probs must be non-empty".to_string());
    }
    let k = probs[0].len();
    if probs.len() != y.len() {
        return Err(format!(
            "length mismatch: probs has {} rows but y has {}",
            probs.len(),
            y.len()
        ));
    }
    let eps = 1e-15_f64;
    let mut total = 0.0;
    for (row, &label) in probs.iter().zip(y) {
        if label >= k {
            return Err(format!("labels must lie in [0, {}); got {}", k, label));
        }
        let p = row[label].max(eps).min(1.0);
        total += -p.ln();
    }
    Ok(total / probs.len() as f64)
}

/// Multinomial logistic regression trained by full-batch gradient descent.
pub struct SoftmaxRegression {
    learning_rate: f64,
    n_iter: usize,
    l2: f64,
    /// Weights of shape (n_features, n_classes), row-major.
    pub w: Vec<Vec<f64>>,
    /// Bias of length n_classes.
    pub b: Vec<f64>,
    pub n_classes: usize,
    fitted: bool,
}

impl SoftmaxRegression {
    /// Construct with hyperparameters. Errors on invalid values.
    pub fn new(learning_rate: f64, n_iter: usize, l2: f64) -> Result<Self, String> {
        if !(learning_rate > 0.0) {
            return Err(format!("learning_rate must be positive, got {}", learning_rate));
        }
        if n_iter == 0 {
            return Err("n_iter must be positive, got 0".to_string());
        }
        if l2 < 0.0 {
            return Err(format!("l2 must be non-negative, got {}", l2));
        }
        Ok(SoftmaxRegression {
            learning_rate,
            n_iter,
            l2,
            w: Vec::new(),
            b: Vec::new(),
            n_classes: 0,
            fitted: false,
        })
    }

    /// Fit on features `x` (n x d) and integer labels `y` (n).
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[usize]) -> Result<(), String> {
        if x.is_empty() || x[0].is_empty() {
            return Err("X must be non-empty".to_string());
        }
        let n = x.len();
        let d = x[0].len();
        if x.iter().any(|r| r.len() != d) {
            return Err("X rows must all have the same length".to_string());
        }
        if n != y.len() {
            return Err(format!("length mismatch: X has {} rows but y has {}", n, y.len()));
        }
        let k = *y.iter().max().unwrap() + 1;
        if k < 2 {
            return Err(format!("need at least 2 classes, got {}", k));
        }

        // one-hot targets
        let mut onehot = vec![vec![0.0_f64; k]; n];
        for (i, &label) in y.iter().enumerate() {
            onehot[i][label] = 1.0;
        }

        let mut w = vec![vec![0.0_f64; k]; d];
        let mut b = vec![0.0_f64; k];

        for _ in 0..self.n_iter {
            // logits = X * W + b, then softmax
            let mut logits = vec![vec![0.0_f64; k]; n];
            for i in 0..n {
                for c in 0..k {
                    let mut acc = b[c];
                    for j in 0..d {
                        acc += x[i][j] * w[j][c];
                    }
                    logits[i][c] = acc;
                }
            }
            let probs = softmax(&logits)?;

            // diff = probs - onehot
            // grad_W[j][c] = sum_i x[i][j] * diff[i][c] / n + 2*l2*w[j][c]
            // grad_b[c]    = mean_i diff[i][c]
            let mut grad_w = vec![vec![0.0_f64; k]; d];
            let mut grad_b = vec![0.0_f64; k];
            for i in 0..n {
                for c in 0..k {
                    let diff = probs[i][c] - onehot[i][c];
                    grad_b[c] += diff;
                    for j in 0..d {
                        grad_w[j][c] += x[i][j] * diff;
                    }
                }
            }
            let nf = n as f64;
            for c in 0..k {
                grad_b[c] /= nf;
                b[c] -= self.learning_rate * grad_b[c];
                for j in 0..d {
                    let g = grad_w[j][c] / nf + 2.0 * self.l2 * w[j][c];
                    w[j][c] -= self.learning_rate * g;
                }
            }
        }

        self.w = w;
        self.b = b;
        self.n_classes = k;
        self.fitted = true;
        Ok(())
    }

    /// Predicted class-probability matrix (n x n_classes).
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if !self.fitted {
            return Err("model is not fitted; call fit() first".to_string());
        }
        let d = self.w.len();
        if x.is_empty() || x[0].is_empty() {
            return Err("X must be non-empty".to_string());
        }
        if x.iter().any(|r| r.len() != d) {
            return Err(format!("X has wrong number of features; model was fit on {}", d));
        }
        let n = x.len();
        let k = self.n_classes;
        let mut logits = vec![vec![0.0_f64; k]; n];
        for i in 0..n {
            for c in 0..k {
                let mut acc = self.b[c];
                for j in 0..d {
                    acc += x[i][j] * self.w[j][c];
                }
                logits[i][c] = acc;
            }
        }
        softmax(&logits)
    }

    /// Predicted integer class labels (argmax of the probabilities).
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<usize>, String> {
        let probs = self.predict_proba(x)?;
        Ok(probs
            .iter()
            .map(|row| {
                let mut best = 0usize;
                for c in 1..row.len() {
                    if row[c] > row[best] {
                        best = c;
                    }
                }
                best
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // --- Shared fixtures: identical to the Python and Julia test suites. ---

    fn softmax_z() -> Vec<Vec<f64>> {
        vec![vec![1.0, 2.0, 3.0], vec![1.0, 1.0, 1.0]]
    }
    const SOFTMAX_ROW0: [f64; 3] = [
        0.09003057317038046,
        0.24472847105479764,
        0.6652409557748218,
    ];

    fn ce_probs() -> Vec<Vec<f64>> {
        vec![vec![0.7, 0.2, 0.1], vec![0.1, 0.8, 0.1]]
    }
    const CE_Y: [usize; 2] = [0, 1];
    const CE_EXPECTED: f64 = 0.2899092476264711;

    fn train_x() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![2.0, 0.0],
        ]
    }
    const TRAIN_Y: [usize; 6] = [0, 1, 2, 1, 2, 1];
    const TRAIN_LR: f64 = 0.5;
    const TRAIN_ITERS: usize = 200;
    const TRAIN_PRED: [usize; 6] = [0, 1, 2, 1, 2, 1];
    const TRAIN_PROBA_ROW0: [f64; 3] = [
        0.8222160377229213,
        0.16537455654418495,
        0.012409405732893796,
    ];

    #[test]
    fn softmax_matches_fixture() {
        let p = softmax(&softmax_z()).unwrap();
        for c in 0..3 {
            assert!((p[0][c] - SOFTMAX_ROW0[c]).abs() < TOL);
        }
        for c in 0..3 {
            assert!((p[1][c] - 1.0 / 3.0).abs() < TOL);
        }
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let p = softmax(&softmax_z()).unwrap();
        for row in &p {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn cross_entropy_matches_fixture() {
        assert!((cross_entropy(&ce_probs(), &CE_Y).unwrap() - CE_EXPECTED).abs() < TOL);
    }

    #[test]
    fn fit_predicts_training_labels() {
        let mut m = SoftmaxRegression::new(TRAIN_LR, TRAIN_ITERS, 0.0).unwrap();
        m.fit(&train_x(), &TRAIN_Y).unwrap();
        assert_eq!(m.predict(&train_x()).unwrap(), TRAIN_PRED.to_vec());
    }

    #[test]
    fn fit_proba_matches_fixture() {
        let mut m = SoftmaxRegression::new(TRAIN_LR, TRAIN_ITERS, 0.0).unwrap();
        m.fit(&train_x(), &TRAIN_Y).unwrap();
        let proba = m.predict_proba(&vec![vec![0.0, 0.0]]).unwrap();
        for c in 0..3 {
            assert!((proba[0][c] - TRAIN_PROBA_ROW0[c]).abs() < TOL);
        }
        assert!((proba[0].iter().sum::<f64>() - 1.0).abs() < TOL);
    }

    #[test]
    fn n_classes_inferred() {
        let mut m = SoftmaxRegression::new(0.5, 10, 0.0).unwrap();
        m.fit(&train_x(), &TRAIN_Y).unwrap();
        assert_eq!(m.n_classes, 3);
        assert_eq!(m.w.len(), 2);
        assert_eq!(m.w[0].len(), 3);
    }

    #[test]
    fn softmax_rejects_non_finite() {
        assert!(softmax(&vec![vec![1.0, f64::INFINITY]]).is_err());
    }

    #[test]
    fn cross_entropy_length_mismatch() {
        assert!(cross_entropy(&vec![vec![0.5, 0.5]], &[0, 1]).is_err());
    }

    #[test]
    fn cross_entropy_label_out_of_range() {
        assert!(cross_entropy(&vec![vec![0.5, 0.5]], &[2]).is_err());
    }

    #[test]
    fn new_rejects_bad_hyperparameters() {
        assert!(SoftmaxRegression::new(0.0, 100, 0.0).is_err());
        assert!(SoftmaxRegression::new(0.5, 0, 0.0).is_err());
        assert!(SoftmaxRegression::new(0.5, 100, -1.0).is_err());
    }

    #[test]
    fn fit_length_mismatch() {
        let mut m = SoftmaxRegression::new(0.5, 10, 0.0).unwrap();
        assert!(m.fit(&vec![vec![0.0, 0.0], vec![1.0, 1.0]], &[0]).is_err());
    }

    #[test]
    fn predict_before_fit_errors() {
        let m = SoftmaxRegression::new(0.5, 10, 0.0).unwrap();
        assert!(m.predict(&vec![vec![0.0, 0.0]]).is_err());
    }

    #[test]
    fn predict_feature_mismatch() {
        let mut m = SoftmaxRegression::new(0.5, 10, 0.0).unwrap();
        m.fit(&train_x(), &TRAIN_Y).unwrap();
        assert!(m.predict(&vec![vec![0.0, 0.0, 0.0]]).is_err());
    }
}
