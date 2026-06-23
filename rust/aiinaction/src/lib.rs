//! AI in Action: reusable, tested reference implementations (Rust).
//!
//! Mirrors the Python package `aiinaction` and the Julia package `AIInAction`.
//! The shared fixtures in the tests below match the Python/Julia suites, which is
//! what keeps the three libraries at parity.

pub mod metrics {
    //! Regression and classification metrics.

    /// Validates that the two slices are the same non-zero length.
    fn validate(y_true: &[f64], y_pred: &[f64]) -> Result<(), String> {
        if y_true.len() != y_pred.len() {
            return Err(format!(
                "length mismatch: {} != {}",
                y_true.len(),
                y_pred.len()
            ));
        }
        if y_true.is_empty() {
            return Err("inputs must be non-empty".to_string());
        }
        Ok(())
    }

    /// Root mean squared error.
    pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> Result<f64, String> {
        validate(y_true, y_pred)?;
        let n = y_true.len() as f64;
        let mse: f64 = y_true
            .iter()
            .zip(y_pred)
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>()
            / n;
        Ok(mse.sqrt())
    }

    /// Mean absolute error.
    pub fn mae(y_true: &[f64], y_pred: &[f64]) -> Result<f64, String> {
        validate(y_true, y_pred)?;
        let n = y_true.len() as f64;
        Ok(y_true
            .iter()
            .zip(y_pred)
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / n)
    }

    /// Coefficient of determination R^2. Errors if the target variance is zero.
    pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64, String> {
        validate(y_true, y_pred)?;
        let n = y_true.len() as f64;
        let mean = y_true.iter().sum::<f64>() / n;
        let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
        if ss_tot == 0.0 {
            return Err("R^2 is undefined when all y_true values are equal (zero variance)".to_string());
        }
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred)
            .map(|(t, p)| (t - p).powi(2))
            .sum();
        Ok(1.0 - ss_res / ss_tot)
    }

    /// Classification accuracy.
    pub fn accuracy(y_true: &[i64], y_pred: &[i64]) -> Result<f64, String> {
        if y_true.len() != y_pred.len() {
            return Err(format!("length mismatch: {} != {}", y_true.len(), y_pred.len()));
        }
        if y_true.is_empty() {
            return Err("inputs must be non-empty".to_string());
        }
        let correct = y_true.iter().zip(y_pred).filter(|(t, p)| t == p).count();
        Ok(correct as f64 / y_true.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::metrics::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const Y_TRUE: [f64; 4] = [3.0, -0.5, 2.0, 7.0];
    const Y_PRED: [f64; 4] = [2.5, 0.0, 2.0, 8.0];
    const TOL: f64 = 1e-12;

    #[test]
    fn rmse_matches_fixture() {
        assert!((rmse(&Y_TRUE, &Y_PRED).unwrap() - 0.6123724356957945).abs() < TOL);
    }

    #[test]
    fn mae_matches_fixture() {
        assert!((mae(&Y_TRUE, &Y_PRED).unwrap() - 0.5).abs() < TOL);
    }

    #[test]
    fn r2_matches_fixture() {
        assert!((r2_score(&Y_TRUE, &Y_PRED).unwrap() - 0.9486081370449679).abs() < TOL);
    }

    #[test]
    fn perfect_prediction() {
        assert_eq!(rmse(&Y_TRUE, &Y_TRUE).unwrap(), 0.0);
        assert!((r2_score(&Y_TRUE, &Y_TRUE).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn accuracy_half() {
        assert!((accuracy(&[1, 0, 1, 1], &[1, 1, 1, 0]).unwrap() - 0.5).abs() < TOL);
    }

    #[test]
    fn length_mismatch_errors() {
        assert!(rmse(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn zero_variance_errors() {
        assert!(r2_score(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]).is_err());
    }
}
