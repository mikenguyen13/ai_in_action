//! RMSProp adaptive-learning-rate optimizer from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch195_rmsprop` and the Julia module
//! `AIInAction.Ch195Rmsprop`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! Plain (uncentered, no-momentum) RMSProp maintains, per coordinate `i`, an
//! exponential moving average of the squared gradient
//!
//! ```text
//! v_t,i = beta * v_{t-1,i} + (1 - beta) * g_t,i^2,   v_0 = 0,
//! ```
//!
//! and updates each coordinate with its own effective step size
//!
//! ```text
//! theta_{t+1,i} = theta_t,i - eta * g_t,i / (sqrt(v_t,i) + eps).
//! ```
//!
//! The epsilon is placed *outside* the square root, matching the chapter. This is
//! a std-only implementation.

/// State of an RMSProp optimizer.
#[derive(Clone, Debug)]
pub struct RmsPropState {
    /// Current parameter vector, length `d`.
    pub params: Vec<f64>,
    /// Exponential moving average of squared gradients, length `d`.
    pub v: Vec<f64>,
    /// Global learning rate `eta` (positive).
    pub lr: f64,
    /// Squared-gradient EMA decay in `[0, 1)`.
    pub beta: f64,
    /// Numerical-stability constant added to `sqrt(v)` (positive).
    pub eps: f64,
    /// Number of update steps applied so far.
    pub step_count: usize,
}

impl RmsPropState {
    /// Dimensionality of the parameter vector.
    pub fn n_params(&self) -> usize {
        self.params.len()
    }
}

fn validate_vector(x: &[f64], name: &str) -> Result<(), String> {
    if x.is_empty() {
        return Err(format!("{} must be non-empty", name));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(format!("{} contains non-finite values (nan or inf)", name));
    }
    Ok(())
}

fn validate_hparams(lr: f64, beta: f64, eps: f64) -> Result<(), String> {
    if !(lr > 0.0) {
        return Err(format!("lr must be positive, got {}", lr));
    }
    if !(beta >= 0.0 && beta < 1.0) {
        return Err(format!("beta must be in [0, 1), got {}", beta));
    }
    if !(eps > 0.0) {
        return Err(format!("eps must be positive, got {}", eps));
    }
    Ok(())
}

/// Initializes an RMSProp state from a starting parameter vector.
///
/// `v` is set to zeros and `step_count` to 0.
pub fn init_state(
    params: &[f64],
    lr: f64,
    beta: f64,
    eps: f64,
) -> Result<RmsPropState, String> {
    validate_vector(params, "params")?;
    validate_hparams(lr, beta, eps)?;
    Ok(RmsPropState {
        params: params.to_vec(),
        v: vec![0.0; params.len()],
        lr,
        beta,
        eps,
        step_count: 0,
    })
}

/// Applies one RMSProp update and returns the new state.
///
/// Performs, coordinate-wise:
/// `v <- beta*v + (1-beta)*g^2` then `params <- params - lr*g/(sqrt(v)+eps)`.
pub fn rmsprop_step(state: &RmsPropState, grad: &[f64]) -> Result<RmsPropState, String> {
    validate_vector(grad, "grad")?;
    if grad.len() != state.n_params() {
        return Err(format!(
            "grad has {} entries but state has {} parameters",
            grad.len(),
            state.n_params()
        ));
    }
    let d = state.n_params();
    let mut v_new = vec![0.0f64; d];
    let mut params_new = vec![0.0f64; d];
    for i in 0..d {
        let g = grad[i];
        let vi = state.beta * state.v[i] + (1.0 - state.beta) * g * g;
        v_new[i] = vi;
        params_new[i] = state.params[i] - state.lr * g / (vi.sqrt() + state.eps);
    }
    Ok(RmsPropState {
        params: params_new,
        v: v_new,
        lr: state.lr,
        beta: state.beta,
        eps: state.eps,
        step_count: state.step_count + 1,
    })
}

/// Runs RMSProp for `n_steps` iterations on the objective with gradient `grad_fn`.
///
/// `grad_fn` maps the current parameter vector to the gradient at that point and
/// must return a finite vector of the same length as `params0`.
pub fn minimize<F>(
    grad_fn: F,
    params0: &[f64],
    n_steps: usize,
    lr: f64,
    beta: f64,
    eps: f64,
) -> Result<RmsPropState, String>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut state = init_state(params0, lr, beta, eps)?;
    for _ in 0..n_steps {
        let g = grad_fn(&state.params);
        state = rmsprop_step(&state, &g)?;
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const TOL: f64 = 1e-9;

    fn params0() -> [f64; 3] {
        [1.0, -2.0, 0.5]
    }
    fn grad() -> [f64; 3] {
        [0.1, -0.3, 2.0]
    }

    #[test]
    fn init_state_zeros_v_and_count() {
        let s = init_state(&params0(), 0.01, 0.9, 1e-8).unwrap();
        assert_eq!(s.n_params(), 3);
        assert_eq!(s.v, vec![0.0, 0.0, 0.0]);
        assert_eq!(s.step_count, 0);
    }

    #[test]
    fn single_step_v_matches_fixture() {
        let s = init_state(&params0(), 0.01, 0.9, 1e-8).unwrap();
        let s1 = rmsprop_step(&s, &grad()).unwrap();
        let expected_v: [f64; 3] =
            [0.0009999999999999998, 0.008999999999999998, 0.3999999999999999];
        for i in 0..3 {
            assert!((s1.v[i] - expected_v[i]).abs() < TOL);
        }
        assert_eq!(s1.step_count, 1);
    }

    #[test]
    fn single_step_params_match_fixture() {
        let s = init_state(&params0(), 0.01, 0.9, 1e-8).unwrap();
        let s1 = rmsprop_step(&s, &grad()).unwrap();
        let expected_p: [f64; 3] =
            [0.968377233398313, -1.9683772267316493, 0.4683772238983162];
        for i in 0..3 {
            assert!((s1.params[i] - expected_p[i]).abs() < TOL);
        }
    }

    #[test]
    fn two_steps_match_fixture() {
        let s = init_state(&params0(), 0.01, 0.9, 1e-8).unwrap();
        let s2 = rmsprop_step(&rmsprop_step(&s, &grad()).unwrap(), &grad()).unwrap();
        let expected_v: [f64; 3] =
            [0.0018999999999999998, 0.017099999999999997, 0.7599999999999998];
        let expected_p: [f64; 3] =
            [0.9454356652744136, -1.9454356550989789, 0.44543565077441793];
        for i in 0..3 {
            assert!((s2.v[i] - expected_v[i]).abs() < TOL);
            assert!((s2.params[i] - expected_p[i]).abs() < TOL);
        }
        assert_eq!(s2.step_count, 2);
    }

    #[test]
    fn minimize_quadratic_matches_fixture() {
        let c: [f64; 2] = [1.0, 4.0];
        let s = minimize(
            |x: &[f64]| vec![c[0] * x[0], c[1] * x[1]],
            &[2.0, 2.0],
            10,
            0.1,
            0.9,
            1e-8,
        )
        .unwrap();
        let expected_p: [f64; 2] = [0.6027968620415073, 0.6027968532310017];
        let expected_v: [f64; 2] = [0.8446178332851308, 13.513885189461563];
        for i in 0..2 {
            assert!((s.params[i] - expected_p[i]).abs() < TOL);
            assert!((s.v[i] - expected_v[i]).abs() < TOL);
        }
        assert_eq!(s.step_count, 10);
    }

    #[test]
    fn beta_zero_is_signed_gradient_normalization() {
        let s = init_state(&[5.0], 0.1, 0.0, 1e-12).unwrap();
        let s1 = rmsprop_step(&s, &[2.0]).unwrap();
        let expected = 5.0 - 0.1 * 2.0 / (2.0 + 1e-12);
        assert!((s1.params[0] - expected).abs() < TOL);
    }

    #[test]
    fn zero_steps_returns_initial_params() {
        let c: [f64; 2] = [1.0, 4.0];
        let s = minimize(
            |x: &[f64]| vec![c[0] * x[0], c[1] * x[1]],
            &[2.0, 2.0],
            0,
            0.1,
            0.9,
            1e-8,
        )
        .unwrap();
        assert_eq!(s.params, vec![2.0, 2.0]);
        assert_eq!(s.step_count, 0);
    }

    #[test]
    fn grad_length_mismatch_errors() {
        let s = init_state(&params0(), 0.01, 0.9, 1e-8).unwrap();
        assert!(rmsprop_step(&s, &[0.1, 0.2]).is_err());
    }

    #[test]
    fn non_finite_grad_errors() {
        let s = init_state(&params0(), 0.01, 0.9, 1e-8).unwrap();
        assert!(rmsprop_step(&s, &[0.1, f64::INFINITY, 0.2]).is_err());
    }

    #[test]
    fn bad_hparams_error() {
        assert!(init_state(&params0(), 0.0, 0.9, 1e-8).is_err());
        assert!(init_state(&params0(), 0.01, 1.0, 1e-8).is_err());
        assert!(init_state(&params0(), 0.01, 0.9, -1e-8).is_err());
    }

    #[test]
    fn empty_params_errors() {
        assert!(init_state(&[], 0.01, 0.9, 1e-8).is_err());
    }
}
