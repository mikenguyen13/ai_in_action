//! Regression metrics: MSE, RMSE, MAE, and the Huber loss (Rust).
//!
//! Mirrors the Python module `aiinaction.ch157_regression_metrics` and the Julia
//! module `AIInAction.Ch157RegressionMetrics`. The shared fixtures in the tests
//! below match the Python/Julia suites, which keeps the three implementations at
//! parity.
//!
//! RMSE and MAE already live in the crate's `metrics` module; this module *reuses*
//! `crate::metrics::{rmse, mae}` rather than re-deriving them, and adds the
//! squared-error (`mse`) and robust Huber-loss (`huber_loss`, `huber_loss_mean`)
//! pieces the chapter introduces. `rmse`/`mae` are re-exported for convenience.
//!
//! std-only; no external dependencies.

pub use crate::metrics::{mae, rmse};

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

/// Mean squared error: the average of the squared residuals.
///
/// Lives in the squared units of the target and weights large residuals
/// quadratically, so a single outlier can dominate the score.
pub fn mse(y_true: &[f64], y_pred: &[f64]) -> Result<f64, String> {
    validate(y_true, y_pred)?;
    let n = y_true.len() as f64;
    Ok(y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>()
        / n)
}

/// Per-observation Huber loss with threshold `delta`.
///
/// Quadratic for residuals with `|r| <= delta` and linear beyond, the two branches
/// meeting with a continuous value and derivative at `|r| = delta`. Returns one
/// loss value per observation (elementwise, not averaged). Errors if `delta <= 0`.
pub fn huber_loss(y_true: &[f64], y_pred: &[f64], delta: f64) -> Result<Vec<f64>, String> {
    validate(y_true, y_pred)?;
    if !delta.is_finite() || delta <= 0.0 {
        return Err(format!("delta must be a positive finite number, got {}", delta));
    }
    let out = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| {
            let a = (t - p).abs();
            if a <= delta {
                0.5 * a * a
            } else {
                delta * (a - 0.5 * delta)
            }
        })
        .collect();
    Ok(out)
}

/// Mean Huber loss over all observations: the scalar objective minimized by Huber
/// (robust) regression.
pub fn huber_loss_mean(y_true: &[f64], y_pred: &[f64], delta: f64) -> Result<f64, String> {
    let losses = huber_loss(y_true, y_pred, delta)?;
    let n = losses.len() as f64;
    Ok(losses.iter().sum::<f64>() / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const Y_TRUE: [f64; 4] = [3.0, -0.5, 2.0, 7.0];
    const Y_PRED: [f64; 4] = [2.5, 0.0, 2.0, 8.0];
    const DELTA: f64 = 0.75;
    const TOL: f64 = 1e-9;

    #[test]
    fn mse_matches_fixture() {
        assert!((mse(&Y_TRUE, &Y_PRED).unwrap() - 0.375).abs() < TOL);
    }

    #[test]
    fn rmse_matches_fixture() {
        assert!((rmse(&Y_TRUE, &Y_PRED).unwrap() - 0.6123724356957945).abs() < TOL);
    }

    #[test]
    fn mae_matches_fixture() {
        assert!((mae(&Y_TRUE, &Y_PRED).unwrap() - 0.5).abs() < TOL);
    }

    #[test]
    fn rmse_is_sqrt_mse() {
        let m = mse(&Y_TRUE, &Y_PRED).unwrap();
        assert!((rmse(&Y_TRUE, &Y_PRED).unwrap() - m.sqrt()).abs() < TOL);
    }

    #[test]
    fn huber_each_matches_fixture() {
        let expected: [f64; 4] = [0.125, 0.125, 0.0, 0.46875];
        let got = huber_loss(&Y_TRUE, &Y_PRED, DELTA).unwrap();
        for i in 0..4 {
            assert!((got[i] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn huber_mean_matches_fixture() {
        assert!((huber_loss_mean(&Y_TRUE, &Y_PRED, DELTA).unwrap() - 0.1796875).abs() < TOL);
    }

    #[test]
    fn huber_continuity_at_threshold() {
        let d: f64 = 1.5;
        let quad = huber_loss(&[0.0], &[d], d).unwrap()[0];
        let lin = huber_loss(&[0.0], &[d + 1e-9], d).unwrap()[0];
        assert!((quad - 0.5 * d * d).abs() < TOL);
        assert!((lin - 0.5 * d * d).abs() < 1e-6);
    }

    #[test]
    fn huber_large_delta_approaches_half_mse() {
        let big = huber_loss_mean(&Y_TRUE, &Y_PRED, 1000.0).unwrap();
        let half_mse = 0.5 * mse(&Y_TRUE, &Y_PRED).unwrap();
        assert!((big - half_mse).abs() < TOL);
    }

    #[test]
    fn perfect_prediction_is_zero() {
        assert_eq!(mse(&Y_TRUE, &Y_TRUE).unwrap(), 0.0);
        assert_eq!(rmse(&Y_TRUE, &Y_TRUE).unwrap(), 0.0);
        assert_eq!(huber_loss_mean(&Y_TRUE, &Y_TRUE, 1.0).unwrap(), 0.0);
    }

    #[test]
    fn length_mismatch_errors() {
        assert!(mse(&[1.0, 2.0], &[1.0]).is_err());
        assert!(huber_loss(&[1.0, 2.0], &[1.0], 1.0).is_err());
    }

    #[test]
    fn empty_errors() {
        assert!(mse(&[], &[]).is_err());
        assert!(huber_loss_mean(&[], &[], 1.0).is_err());
    }

    #[test]
    fn bad_delta_errors() {
        assert!(huber_loss(&Y_TRUE, &Y_PRED, 0.0).is_err());
        assert!(huber_loss(&Y_TRUE, &Y_PRED, -1.0).is_err());
        assert!(huber_loss(&Y_TRUE, &Y_PRED, f64::INFINITY).is_err());
        assert!(huber_loss(&Y_TRUE, &Y_PRED, f64::NAN).is_err());
    }
}
