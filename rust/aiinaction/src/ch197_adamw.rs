//! AdamW: Adam with decoupled weight decay, from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch197_adamw` and the Julia module
//! `AIInAction.Ch197Adamw`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! The update rule for a parameter vector `theta` at step `t` with gradient `g`:
//!
//! 1. `m = beta1 * m + (1 - beta1) * g`
//! 2. `v = beta2 * v + (1 - beta2) * g * g`
//! 3. `mhat = m / (1 - beta1^t)`, `vhat = v / (1 - beta2^t)`
//! 4. `theta = theta - lr * (mhat / (sqrt(vhat) + eps) + wd * theta)`
//!
//! The weight-decay term `wd * theta` in step 4 is applied directly to the
//! parameters, outside the adaptive preconditioner `sqrt(vhat) + eps`. This is
//! the defining feature of AdamW versus L2-regularized Adam.

/// Hyperparameters for the AdamW optimizer.
#[derive(Clone, Debug)]
pub struct AdamWConfig {
    /// Learning rate (step size) `alpha`. Must be positive.
    pub lr: f64,
    /// First-moment decay rate, in `[0, 1)`.
    pub beta1: f64,
    /// Second-moment decay rate, in `[0, 1)`.
    pub beta2: f64,
    /// Numerical floor added to `sqrt(vhat)`. Must be positive.
    pub eps: f64,
    /// Decoupled weight-decay coefficient `lambda`. Must be non-negative.
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        AdamWConfig {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

impl AdamWConfig {
    /// Validates the hyperparameters, returning an error string on the first problem.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.lr > 0.0) {
            return Err(format!("lr must be positive, got {}", self.lr));
        }
        if !(self.beta1 >= 0.0 && self.beta1 < 1.0) {
            return Err(format!("beta1 must be in [0, 1), got {}", self.beta1));
        }
        if !(self.beta2 >= 0.0 && self.beta2 < 1.0) {
            return Err(format!("beta2 must be in [0, 1), got {}", self.beta2));
        }
        if !(self.eps > 0.0) {
            return Err(format!("eps must be positive, got {}", self.eps));
        }
        if !(self.weight_decay >= 0.0) {
            return Err(format!(
                "weight_decay must be non-negative, got {}",
                self.weight_decay
            ));
        }
        Ok(())
    }
}

/// Mutable optimizer state for one parameter vector.
#[derive(Clone, Debug)]
pub struct AdamWState {
    /// First-moment estimate.
    pub m: Vec<f64>,
    /// Second-moment estimate.
    pub v: Vec<f64>,
    /// Number of update steps taken so far (starts at 0).
    pub t: u64,
}

/// Creates a zero-initialized optimizer state for `n_params` parameters.
pub fn init_state(n_params: usize) -> Result<AdamWState, String> {
    if n_params < 1 {
        return Err(format!("n_params must be >= 1, got {}", n_params));
    }
    Ok(AdamWState {
        m: vec![0.0; n_params],
        v: vec![0.0; n_params],
        t: 0,
    })
}

fn check_vector(x: &[f64], name: &str, expected_len: Option<usize>) -> Result<(), String> {
    if x.is_empty() {
        return Err(format!("{} must be non-empty", name));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(format!("{} contains non-finite values (nan or inf)", name));
    }
    if let Some(n) = expected_len {
        if x.len() != n {
            return Err(format!("{} has length {} but expected {}", name, x.len(), n));
        }
    }
    Ok(())
}

/// Applies one AdamW update in place on `state` and returns the new parameters.
///
/// The state's `m`, `v`, and `t` fields are advanced one step. `params` itself
/// is not modified; a fresh `Vec` of updated parameters is returned.
pub fn adamw_step(
    params: &[f64],
    grad: &[f64],
    state: &mut AdamWState,
    config: &AdamWConfig,
) -> Result<Vec<f64>, String> {
    config.validate()?;
    check_vector(params, "params", None)?;
    let d = params.len();
    check_vector(grad, "grad", Some(d))?;
    if state.m.len() != d || state.v.len() != d {
        return Err(format!(
            "state has length m={}, v={} but params has length {}",
            state.m.len(),
            state.v.len(),
            d
        ));
    }

    let b1 = config.beta1;
    let b2 = config.beta2;
    state.t += 1;
    let t = state.t as i32;

    for i in 0..d {
        state.m[i] = b1 * state.m[i] + (1.0 - b1) * grad[i];
        state.v[i] = b2 * state.v[i] + (1.0 - b2) * (grad[i] * grad[i]);
    }

    let bc1 = 1.0 - b1.powi(t);
    let bc2 = 1.0 - b2.powi(t);

    let mut out = vec![0.0f64; d];
    for i in 0..d {
        let mhat = state.m[i] / bc1;
        let vhat = state.v[i] / bc2;
        let adaptive = mhat / (vhat.sqrt() + config.eps);
        out[i] = params[i] - config.lr * (adaptive + config.weight_decay * params[i]);
    }
    Ok(out)
}

/// Runs AdamW for `n_steps` iterations starting from `x0`.
///
/// `grad_fn` maps a parameter vector to its gradient vector of the same length.
pub fn minimize<F>(
    grad_fn: F,
    x0: &[f64],
    config: &AdamWConfig,
    n_steps: u64,
) -> Result<Vec<f64>, String>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if n_steps < 1 {
        return Err(format!("n_steps must be >= 1, got {}", n_steps));
    }
    check_vector(x0, "x0", None)?;
    let mut x = x0.to_vec();
    let mut state = init_state(x.len())?;
    for _ in 0..n_steps {
        let g = grad_fn(&x);
        check_vector(&g, "grad_fn(x)", Some(x.len()))?;
        x = adamw_step(&x, &g, &mut state, config)?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn step_config() -> AdamWConfig {
        AdamWConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }

    // Diagonal quadratic used by the minimization fixtures: grad(x) = A * (x - target).
    fn quad_grad(x: &[f64]) -> Vec<f64> {
        let a: [f64; 2] = [3.0, 1.0];
        let target: [f64; 2] = [2.0, -1.0];
        vec![a[0] * (x[0] - target[0]), a[1] * (x[1] - target[1])]
    }

    #[test]
    fn single_step_theta_matches_fixture() {
        let params: [f64; 3] = [1.0, -2.0, 0.5];
        let grad: [f64; 3] = [0.5, -1.0, 2.0];
        let mut st = init_state(3).unwrap();
        let theta = adamw_step(&params, &grad, &mut st, &step_config()).unwrap();
        let expected: [f64; 3] = [0.899000002, -1.898000001, 0.3995000005];
        for i in 0..3 {
            assert!((theta[i] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn single_step_moments_match_fixture() {
        let params: [f64; 3] = [1.0, -2.0, 0.5];
        let grad: [f64; 3] = [0.5, -1.0, 2.0];
        let mut st = init_state(3).unwrap();
        adamw_step(&params, &grad, &mut st, &step_config()).unwrap();
        let expected_m: [f64; 3] = [
            0.04999999999999999,
            -0.09999999999999998,
            0.19999999999999996,
        ];
        let expected_v: [f64; 3] = [
            0.0002500000000000002,
            0.0010000000000000009,
            0.0040000000000000036,
        ];
        for i in 0..3 {
            assert!((st.m[i] - expected_m[i]).abs() < TOL);
            assert!((st.v[i] - expected_v[i]).abs() < TOL);
        }
        assert_eq!(st.t, 1);
    }

    #[test]
    fn minimize_quadratic_matches_fixture() {
        let cfg = AdamWConfig {
            lr: 0.1,
            ..AdamWConfig::default()
        };
        let x0: [f64; 2] = [0.0, 0.0];
        let x = minimize(quad_grad, &x0, &cfg, 200).unwrap();
        let expected: [f64; 2] = [1.999943097328645, -1.000007218001001];
        assert!((x[0] - expected[0]).abs() < TOL);
        assert!((x[1] - expected[1]).abs() < TOL);
    }

    #[test]
    fn minimize_with_weight_decay_matches_fixture() {
        let cfg = AdamWConfig {
            lr: 0.1,
            weight_decay: 0.01,
            ..AdamWConfig::default()
        };
        let x0: [f64; 2] = [5.0, 5.0];
        let x = minimize(quad_grad, &x0, &cfg, 50).unwrap();
        let expected: [f64; 2] = [1.7895443334390244, 0.604294026757625];
        assert!((x[0] - expected[0]).abs() < TOL);
        assert!((x[1] - expected[1]).abs() < TOL);
    }

    #[test]
    fn minimize_converges_to_target() {
        let cfg = AdamWConfig {
            lr: 0.1,
            ..AdamWConfig::default()
        };
        let x0: [f64; 2] = [0.0, 0.0];
        let x = minimize(quad_grad, &x0, &cfg, 500).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-4);
        assert!((x[1] - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn weight_decay_shrinks_when_gradient_zero() {
        let cfg = AdamWConfig {
            lr: 0.1,
            weight_decay: 0.2,
            ..AdamWConfig::default()
        };
        let mut st = init_state(2).unwrap();
        let params: [f64; 2] = [4.0, -6.0];
        let grad: [f64; 2] = [0.0, 0.0];
        let theta = adamw_step(&params, &grad, &mut st, &cfg).unwrap();
        assert!((theta[0] - 4.0 * (1.0 - 0.02)).abs() < TOL);
        assert!((theta[1] - (-6.0) * (1.0 - 0.02)).abs() < TOL);
    }

    #[test]
    fn step_increments_t() {
        let cfg = AdamWConfig {
            lr: 0.1,
            ..AdamWConfig::default()
        };
        let mut st = init_state(2).unwrap();
        let mut params: Vec<f64> = vec![1.0, 1.0];
        let grad: [f64; 2] = [0.1, 0.1];
        for expected_t in 1..=3u64 {
            params = adamw_step(&params, &grad, &mut st, &cfg).unwrap();
            assert_eq!(st.t, expected_t);
        }
    }

    #[test]
    fn bad_lr_errors() {
        let cfg = AdamWConfig {
            lr: 0.0,
            ..AdamWConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn negative_weight_decay_errors() {
        let cfg = AdamWConfig {
            weight_decay: -0.1,
            ..AdamWConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn init_state_bad_n_errors() {
        assert!(init_state(0).is_err());
    }

    #[test]
    fn grad_length_mismatch_errors() {
        let mut st = init_state(3).unwrap();
        let params: [f64; 3] = [1.0, 2.0, 3.0];
        let grad: [f64; 2] = [0.1, 0.2];
        assert!(adamw_step(&params, &grad, &mut st, &AdamWConfig::default()).is_err());
    }

    #[test]
    fn non_finite_grad_errors() {
        let mut st = init_state(2).unwrap();
        let params: [f64; 2] = [1.0, 2.0];
        let grad: [f64; 2] = [f64::NAN, 0.1];
        assert!(adamw_step(&params, &grad, &mut st, &AdamWConfig::default()).is_err());
    }

    #[test]
    fn minimize_bad_n_steps_errors() {
        let x0: [f64; 2] = [0.0, 0.0];
        assert!(minimize(quad_grad, &x0, &AdamWConfig::default(), 0).is_err());
    }
}
