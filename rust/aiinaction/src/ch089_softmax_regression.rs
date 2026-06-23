//! Softmax regression from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch089_softmax_regression` and the Julia
//! module `AIInAction.Ch089SoftmaxRegression`. The shared fixtures in the tests
//! below match the Python/Julia suites, which keeps the three at parity.
//!
//! Implemented with the standard library only (`Vec<f64>` row-major matrices).

/// Numerically stable softmax with optional temperature `T > 0`.
///
/// Subtracts the max logit before exponentiating so nothing overflows.
pub fn softmax(z: &[f64], temperature: f64) -> Result<Vec<f64>, String> {
    if z.is_empty() {
        return Err("z must be non-empty".to_string());
    }
    if !(temperature > 0.0) {
        return Err(format!("temperature must be positive, got {temperature}"));
    }
    let scaled: Vec<f64> = z.iter().map(|v| v / temperature).collect();
    let m = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|v| (v - m).exp()).collect();
    let total: f64 = exps.iter().sum();
    Ok(exps.iter().map(|e| e / total).collect())
}

/// Stable `log(sum(exp(z)))` via the max-subtraction trick.
pub fn log_sum_exp(z: &[f64]) -> Result<f64, String> {
    if z.is_empty() {
        return Err("z must be non-empty".to_string());
    }
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let s: f64 = z.iter().map(|v| (v - m).exp()).sum();
    Ok(m + s.ln())
}

/// Fused, stable cross-entropy loss from logits: `log_sum_exp(z) - z[label]`.
pub fn cross_entropy_from_logits(z: &[f64], label: usize) -> Result<f64, String> {
    if z.is_empty() {
        return Err("z must be non-empty".to_string());
    }
    if label >= z.len() {
        return Err(format!("label {} out of range for {} classes", label, z.len()));
    }
    Ok(log_sum_exp(z)? - z[label])
}

/// Multinomial logistic (softmax) regression trained by batch gradient descent.
///
/// `w` is a row-major `(k, d)` weight matrix and `b` is a length-`k` bias
/// vector. Both start at zero so fitting is deterministic.
pub struct SoftmaxRegression {
    learning_rate: f64,
    n_iter: usize,
    l2: f64,
    k: usize,
    d: usize,
    pub w: Vec<f64>, // row-major (k, d)
    pub b: Vec<f64>, // (k,)
    fitted: bool,
}

impl SoftmaxRegression {
    /// Construct an unfitted model. `learning_rate > 0`, `n_iter >= 1`, `l2 >= 0`.
    pub fn new(learning_rate: f64, n_iter: usize, l2: f64) -> Result<Self, String> {
        if !(learning_rate > 0.0) {
            return Err(format!("learning_rate must be positive, got {learning_rate}"));
        }
        if n_iter < 1 {
            return Err("n_iter must be >= 1".to_string());
        }
        if l2 < 0.0 {
            return Err(format!("l2 must be non-negative, got {l2}"));
        }
        Ok(Self {
            learning_rate,
            n_iter,
            l2,
            k: 0,
            d: 0,
            w: Vec::new(),
            b: Vec::new(),
            fitted: false,
        })
    }

    fn softmax_row(z: &[f64]) -> Vec<f64> {
        let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    }

    /// Logits for one feature row: `z = W x + b`, length `k`.
    fn logits(&self, x: &[f64]) -> Vec<f64> {
        let mut z = vec![0.0; self.k];
        for c in 0..self.k {
            let mut acc = self.b[c];
            for j in 0..self.d {
                acc += self.w[c * self.d + j] * x[j];
            }
            z[c] = acc;
        }
        z
    }

    /// Fit on `x` (n rows of length `d`) and integer labels `y` (length `n`).
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[usize]) -> Result<(), String> {
        let n = x.len();
        if n == 0 {
            return Err("x must contain at least one row".to_string());
        }
        if y.len() != n {
            return Err(format!("x and y disagree on N: {} != {}", n, y.len()));
        }
        let d = x[0].len();
        for row in x {
            if row.len() != d {
                return Err("all rows of x must have the same length".to_string());
            }
        }
        let k = y.iter().cloned().max().unwrap() + 1;
        self.k = k;
        self.d = d;
        self.w = vec![0.0; k * d];
        self.b = vec![0.0; k];

        let nf = n as f64;
        for _ in 0..self.n_iter {
            let mut grad_w = vec![0.0; k * d];
            let mut grad_b = vec![0.0; k];
            for (xi, &yi) in x.iter().zip(y.iter()) {
                let p = Self::softmax_row(&self.logits(xi));
                for c in 0..k {
                    let g = p[c] - if c == yi { 1.0 } else { 0.0 };
                    grad_b[c] += g;
                    for j in 0..d {
                        grad_w[c * d + j] += g * xi[j];
                    }
                }
            }
            for c in 0..k {
                self.b[c] -= self.learning_rate * (grad_b[c] / nf);
                for j in 0..d {
                    let gw = grad_w[c * d + j] / nf + self.l2 * self.w[c * d + j];
                    self.w[c * d + j] -= self.learning_rate * gw;
                }
            }
        }
        self.fitted = true;
        Ok(())
    }

    /// Class probabilities for one feature row.
    pub fn predict_proba_row(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        if !self.fitted {
            return Err("model is not fitted; call fit() first".to_string());
        }
        if x.len() != self.d {
            return Err(format!("x has {} features but model expects {}", x.len(), self.d));
        }
        Ok(Self::softmax_row(&self.logits(x)))
    }

    /// Predicted class index (argmax) for one feature row.
    pub fn predict_row(&self, x: &[f64]) -> Result<usize, String> {
        let p = self.predict_proba_row(x)?;
        let mut best = 0usize;
        for c in 1..p.len() {
            if p[c] > p[best] {
                best = c;
            }
        }
        Ok(best)
    }

    /// Mean cross-entropy loss (excluding L2) over the dataset.
    pub fn loss(&self, x: &[Vec<f64>], y: &[usize]) -> Result<f64, String> {
        if !self.fitted {
            return Err("model is not fitted; call fit() first".to_string());
        }
        if y.len() != x.len() {
            return Err(format!("x and y disagree on N: {} != {}", x.len(), y.len()));
        }
        let mut total = 0.0;
        for (xi, &yi) in x.iter().zip(y.iter()) {
            total += cross_entropy_from_logits(&self.logits(xi), yi)?;
        }
        Ok(total / x.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // Shared scalar fixtures: identical to the Python and Julia suites.
    const SOFTMAX_Z: [f64; 3] = [1.0, 2.0, 3.0];
    const SOFTMAX_EXPECTED: [f64; 3] =
        [0.09003057317038046, 0.24472847105479764, 0.6652409557748218];
    const SOFTMAX_T2_EXPECTED: [f64; 3] =
        [0.1863237232258476, 0.3071958857184984, 0.506480391055654];
    const LSE_EXPECTED: f64 = 0.6931471805599453;
    const CE_Z: [f64; 3] = [2.0, 1.0, 0.0];
    const CE_EXPECTED: f64 = 0.4076059644443806;

    // Shared classifier fixture.
    const FIT_W: [f64; 6] = [
        -0.20927700627356688,
        -0.20927700627356688,
        0.41855401254713376,
        -0.20927700627356688,
        -0.20927700627356688,
        0.41855401254713376,
    ];
    const FIT_B: [f64; 3] = [0.0258904318973107, -0.012945215948655314, -0.012945215948655314];
    const FIT_LOSS: f64 = 0.8486364761645943;
    const FIT_PROBA: [f64; 9] = [
        0.3420186020672228,
        0.3289906989663886,
        0.3289906989663886,
        0.265668759951062,
        0.4787821251716869,
        0.25554911487725124,
        0.265668759951062,
        0.25554911487725124,
        0.4787821251716869,
    ];

    fn fit_data() -> (Vec<Vec<f64>>, Vec<usize>) {
        (
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![0, 1, 2],
        )
    }

    #[test]
    fn softmax_matches_fixture() {
        let out = softmax(&SOFTMAX_Z, 1.0).unwrap();
        for (a, b) in out.iter().zip(SOFTMAX_EXPECTED.iter()) {
            assert!((a - b).abs() < TOL);
        }
        assert!((out.iter().sum::<f64>() - 1.0).abs() < TOL);
    }

    #[test]
    fn softmax_temperature() {
        let out = softmax(&SOFTMAX_Z, 2.0).unwrap();
        for (a, b) in out.iter().zip(SOFTMAX_T2_EXPECTED.iter()) {
            assert!((a - b).abs() < TOL);
        }
    }

    #[test]
    fn softmax_no_overflow() {
        let out = softmax(&[1000.0, 1000.0, 1000.0], 1.0).unwrap();
        for p in out {
            assert!((p - 1.0 / 3.0).abs() < TOL);
        }
    }

    #[test]
    fn log_sum_exp_matches_fixture() {
        assert!((log_sum_exp(&[0.0, 0.0]).unwrap() - LSE_EXPECTED).abs() < TOL);
    }

    #[test]
    fn cross_entropy_matches_fixture() {
        assert!((cross_entropy_from_logits(&CE_Z, 0).unwrap() - CE_EXPECTED).abs() < TOL);
    }

    #[test]
    fn classifier_parameters_match_fixture() {
        let (x, y) = fit_data();
        let mut m = SoftmaxRegression::new(1.0, 2, 0.0).unwrap();
        m.fit(&x, &y).unwrap();
        for (a, b) in m.w.iter().zip(FIT_W.iter()) {
            assert!((a - b).abs() < TOL);
        }
        for (a, b) in m.b.iter().zip(FIT_B.iter()) {
            assert!((a - b).abs() < TOL);
        }
    }

    #[test]
    fn classifier_loss_matches_fixture() {
        let (x, y) = fit_data();
        let mut m = SoftmaxRegression::new(1.0, 2, 0.0).unwrap();
        m.fit(&x, &y).unwrap();
        assert!((m.loss(&x, &y).unwrap() - FIT_LOSS).abs() < TOL);
    }

    #[test]
    fn classifier_proba_matches_fixture() {
        let (x, y) = fit_data();
        let mut m = SoftmaxRegression::new(1.0, 2, 0.0).unwrap();
        m.fit(&x, &y).unwrap();
        let mut flat = Vec::new();
        for row in &x {
            flat.extend(m.predict_proba_row(row).unwrap());
        }
        for (a, b) in flat.iter().zip(FIT_PROBA.iter()) {
            assert!((a - b).abs() < TOL);
        }
    }

    #[test]
    fn classifier_separable_converges() {
        let x = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![2.0, 0.0],
        ];
        let y = vec![0usize, 1, 2, 2, 2, 1];
        let mut m = SoftmaxRegression::new(0.5, 300, 0.0).unwrap();
        m.fit(&x, &y).unwrap();
        for (row, &yi) in x.iter().zip(y.iter()) {
            assert_eq!(m.predict_row(row).unwrap(), yi);
        }
    }

    #[test]
    fn errors() {
        assert!(softmax(&[], 1.0).is_err());
        assert!(softmax(&[1.0, 2.0], 0.0).is_err());
        assert!(cross_entropy_from_logits(&[1.0, 2.0], 2).is_err());
        assert!(SoftmaxRegression::new(0.0, 1, 0.0).is_err());
        assert!(SoftmaxRegression::new(1.0, 0, 0.0).is_err());
        let mut m = SoftmaxRegression::new(1.0, 1, 0.0).unwrap();
        assert!(m.predict_row(&[1.0]).is_err()); // not fitted
        assert!(m.fit(&[vec![0.0]], &[0, 1]).is_err()); // N mismatch
    }
}
