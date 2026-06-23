//! AdaGrad adaptive learning rates from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch194_adagrad` and the Julia module
//! `AIInAction.Ch194Adagrad`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! AdaGrad gives every coordinate its own learning rate by accumulating the sum of
//! squared gradients `G[i] = sum_tau g[tau, i]^2` and stepping
//! `theta[i] -= eta / (sqrt(G[i]) + eps) * g[i]`. This is a std-only
//! implementation.

/// Mutable optimizer state.
#[derive(Clone, Debug)]
pub struct AdaGradState {
    /// Current parameter vector, length `d`.
    pub theta: Vec<f64>,
    /// Per-coordinate sum of squared gradients `G`, length `d` (nondecreasing).
    pub accumulator: Vec<f64>,
    /// Global base learning rate `eta` (positive).
    pub learning_rate: f64,
    /// Numerical-stability constant added to `sqrt(G)` (positive).
    pub epsilon: f64,
}

/// The outcome of a [`minimize`] run.
#[derive(Clone, Debug)]
pub struct AdaGradResult {
    /// Final parameter vector.
    pub theta: Vec<f64>,
    /// Final gradient-square accumulator.
    pub accumulator: Vec<f64>,
    /// Number of update steps actually performed.
    pub n_steps: usize,
    /// Euclidean norm of the gradient at the final `theta`.
    pub grad_norm: f64,
    /// `true` if stopped because `grad_norm <= tol`.
    pub converged: bool,
}

fn check_vector(v: &[f64], name: &str) -> Result<(), String> {
    if v.is_empty() {
        return Err(format!("{} must be non-empty", name));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(format!("{} contains non-finite values (nan or inf)", name));
    }
    Ok(())
}

fn check_hyperparams(learning_rate: f64, epsilon: f64) -> Result<(), String> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(format!(
            "learning_rate must be a positive finite number, got {}",
            learning_rate
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(format!(
            "epsilon must be a positive finite number, got {}",
            epsilon
        ));
    }
    Ok(())
}

/// Create a fresh state with a zero accumulator.
pub fn init_state(
    theta0: &[f64],
    learning_rate: f64,
    epsilon: f64,
) -> Result<AdaGradState, String> {
    check_vector(theta0, "theta0")?;
    check_hyperparams(learning_rate, epsilon)?;
    Ok(AdaGradState {
        theta: theta0.to_vec(),
        accumulator: vec![0.0; theta0.len()],
        learning_rate,
        epsilon,
    })
}

/// Per-coordinate effective learning rate `eta / (sqrt(G) + eps)`.
pub fn effective_learning_rate(state: &AdaGradState) -> Vec<f64> {
    state
        .accumulator
        .iter()
        .map(|&g| state.learning_rate / (g.sqrt() + state.epsilon))
        .collect()
}

/// Apply one in-place AdaGrad update for the given gradient.
pub fn adagrad_step(state: &mut AdaGradState, grad: &[f64]) -> Result<(), String> {
    check_vector(grad, "grad")?;
    if grad.len() != state.theta.len() {
        return Err(format!(
            "grad has length {} but theta has length {}",
            grad.len(),
            state.theta.len()
        ));
    }
    for i in 0..state.theta.len() {
        let g = grad[i];
        state.accumulator[i] += g * g;
        let step = state.learning_rate / (state.accumulator[i].sqrt() + state.epsilon);
        state.theta[i] -= step * g;
    }
    Ok(())
}

/// Minimize an objective by AdaGrad using only its gradient closure.
///
/// `grad_fn` maps a parameter vector to its gradient (same length). Iterates until
/// the gradient norm drops to `tol` or `max_iter` steps have been taken.
pub fn minimize<F>(
    grad_fn: F,
    theta0: &[f64],
    learning_rate: f64,
    epsilon: f64,
    max_iter: usize,
    tol: f64,
) -> Result<AdaGradResult, String>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if max_iter < 1 {
        return Err(format!("max_iter must be a positive integer, got {}", max_iter));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(format!("tol must be a nonnegative finite number, got {}", tol));
    }
    let mut state = init_state(theta0, learning_rate, epsilon)?;
    let d = state.theta.len();

    let mut grad_norm = f64::INFINITY;
    let mut steps = 0usize;
    let mut converged = false;
    for _ in 0..max_iter {
        let g = grad_fn(&state.theta);
        if g.len() != d {
            return Err(format!(
                "grad_fn returned length {} but expected {}",
                g.len(),
                d
            ));
        }
        if g.iter().any(|x| !x.is_finite()) {
            return Err("grad_fn returned non-finite values (nan or inf)".to_string());
        }
        grad_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        if grad_norm <= tol {
            converged = true;
            break;
        }
        adagrad_step(&mut state, &g)?;
        steps += 1;
    }

    Ok(AdaGradResult {
        theta: state.theta,
        accumulator: state.accumulator,
        n_steps: steps,
        grad_norm,
        converged,
    })
}

/// Gradient of the separable quadratic `f = 0.5 * sum a_i (theta_i - b_i)^2`.
pub fn quadratic_grad(theta: &[f64], a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
    if theta.len() != a.len() || theta.len() != b.len() {
        return Err(format!(
            "theta, a, b must share length; got {}, {}, {}",
            theta.len(),
            a.len(),
            b.len()
        ));
    }
    if a.iter().any(|&x| x <= 0.0) {
        return Err("curvatures a must be strictly positive".to_string());
    }
    Ok((0..theta.len()).map(|i| a[i] * (theta[i] - b[i])).collect())
}

/// Value of `f = 0.5 * sum a_i (theta_i - b_i)^2` (the test objective).
pub fn quadratic_value(theta: &[f64], a: &[f64], b: &[f64]) -> Result<f64, String> {
    if theta.len() != a.len() || theta.len() != b.len() {
        return Err(format!(
            "theta, a, b must share length; got {}, {}, {}",
            theta.len(),
            a.len(),
            b.len()
        ));
    }
    if a.iter().any(|&x| x <= 0.0) {
        return Err("curvatures a must be strictly positive".to_string());
    }
    Ok(0.5
        * (0..theta.len())
            .map(|i| a[i] * (theta[i] - b[i]).powi(2))
            .sum::<f64>())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const A: [f64; 3] = [1.0, 4.0, 0.25];
    const B: [f64; 3] = [2.0, -1.0, 5.0];
    const THETA0: [f64; 3] = [0.0, 0.0, 0.0];
    const ETA: f64 = 0.5;
    const EPS: f64 = 1e-8;
    const TOL: f64 = 1e-9;

    fn grad(theta: &[f64]) -> Vec<f64> {
        quadratic_grad(theta, &A, &B).unwrap()
    }

    #[test]
    fn initial_accumulator_is_zero() {
        let s = init_state(&THETA0, ETA, EPS).unwrap();
        assert_eq!(s.accumulator, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn grad0_matches_fixture() {
        let s = init_state(&THETA0, ETA, EPS).unwrap();
        let g = grad(&s.theta);
        let expected: [f64; 3] = [-2.0, 4.0, -1.25];
        for i in 0..3 {
            assert!((g[i] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn single_step_matches_fixture() {
        let mut s = init_state(&THETA0, ETA, EPS).unwrap();
        let g = grad(&s.theta);
        adagrad_step(&mut s, &g).unwrap();
        let theta_exp: [f64; 3] = [0.4999999975, -0.49999999875, 0.49999999600000006];
        let acc_exp: [f64; 3] = [4.0, 16.0, 1.5625];
        let elr_exp: [f64; 3] = [0.24999999875, 0.1249999996875, 0.39999999680000003];
        let elr = effective_learning_rate(&s);
        for i in 0..3 {
            assert!((s.theta[i] - theta_exp[i]).abs() < TOL);
            assert!((s.accumulator[i] - acc_exp[i]).abs() < TOL);
            assert!((elr[i] - elr_exp[i]).abs() < TOL);
        }
    }

    #[test]
    fn three_steps_match_fixture() {
        let mut s = init_state(&THETA0, ETA, EPS).unwrap();
        for _ in 0..3 {
            let g = grad(&s.theta);
            adagrad_step(&mut s, &g).unwrap();
        }
        let theta_exp: [f64; 3] = [1.0163655300219518, -0.843601262889007, 1.0977190862943662];
        let acc_exp: [f64; 3] = [7.690000015612001, 21.22229126752294, 3.9125960778289572];
        for i in 0..3 {
            assert!((s.theta[i] - theta_exp[i]).abs() < TOL);
            assert!((s.accumulator[i] - acc_exp[i]).abs() < TOL);
        }
    }

    #[test]
    fn minimize_converges_to_minimizer() {
        let res = minimize(grad, &THETA0, ETA, EPS, 10000, 1e-9).unwrap();
        assert!(res.converged);
        let min_exp: [f64; 3] = [2.0, -1.0, 5.0];
        for i in 0..3 {
            assert!((res.theta[i] - min_exp[i]).abs() < 1e-6);
        }
        assert_eq!(res.n_steps, 641);
        assert!(res.grad_norm <= 1e-9);
    }

    #[test]
    fn constant_gradient_decays_like_sqrt_t() {
        let mut s = init_state(&[0.0], 1.0, 1e-8).unwrap();
        let g: [f64; 1] = [1.0];
        for t in 1..=100usize {
            adagrad_step(&mut s, &g).unwrap();
            let expected = 1.0 / ((t as f64).sqrt() + 1e-8);
            let elr = effective_learning_rate(&s);
            assert!((elr[0] - expected).abs() < TOL);
        }
    }

    #[test]
    fn accumulator_is_monotone_nondecreasing() {
        let mut s = init_state(&THETA0, ETA, EPS).unwrap();
        let mut prev = s.accumulator.clone();
        for _ in 0..20 {
            let g = grad(&s.theta);
            adagrad_step(&mut s, &g).unwrap();
            for i in 0..3 {
                assert!(s.accumulator[i] >= prev[i] - 1e-15);
            }
            prev = s.accumulator.clone();
        }
    }

    #[test]
    fn quadratic_value_matches() {
        let v = quadratic_value(&[1.5, 0.0, 4.0], &A, &B).unwrap();
        let expected = 0.5 * (1.0 * 0.25 + 4.0 * 1.0 + 0.25 * 1.0);
        assert!((v - expected).abs() < TOL);
    }

    #[test]
    fn minimize_runs_full_iters_when_tol_zero() {
        let res = minimize(grad, &THETA0, ETA, EPS, 5, 0.0).unwrap();
        assert_eq!(res.n_steps, 5);
        assert!(!res.converged);
    }

    #[test]
    fn rejects_bad_hyperparams() {
        assert!(init_state(&THETA0, 0.0, EPS).is_err());
        assert!(init_state(&THETA0, ETA, -1e-8).is_err());
        assert!(init_state(&[], ETA, EPS).is_err());
        assert!(init_state(&[0.0, f64::NAN], ETA, EPS).is_err());
    }

    #[test]
    fn step_rejects_length_mismatch() {
        let mut s = init_state(&THETA0, ETA, EPS).unwrap();
        assert!(adagrad_step(&mut s, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn step_rejects_non_finite() {
        let mut s = init_state(&THETA0, ETA, EPS).unwrap();
        assert!(adagrad_step(&mut s, &[1.0, f64::INFINITY, 0.0]).is_err());
    }

    #[test]
    fn minimize_rejects_bad_args() {
        assert!(minimize(grad, &THETA0, ETA, EPS, 0, 1e-9).is_err());
        assert!(minimize(grad, &THETA0, ETA, EPS, 10, -1.0).is_err());
    }

    #[test]
    fn quadratic_rejects_non_positive_curvature() {
        assert!(quadratic_grad(&[0.0, 0.0], &[1.0, 0.0], &[0.0, 0.0]).is_err());
    }
}
