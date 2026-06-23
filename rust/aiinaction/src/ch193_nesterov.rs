//! Nesterov Accelerated Gradient (NAG) from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch193_nesterov` and the Julia module
//! `AIInAction.Ch193Nesterov`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! This is a std-only implementation. The gradient is supplied as a closure
//! mapping a point (`&[f64]`) to its gradient (`Vec<f64>` of the same length).
//! Two schedules are provided:
//!
//! * `nesterov_convex` -- the two-sequence (FISTA-style) form for smooth convex
//!   objectives, with the momentum schedule `t_{k+1} = (1 + sqrt(1 + 4 t_k^2))/2`.
//! * `nesterov_momentum` -- the constant-momentum velocity form for strongly
//!   convex objectives, with lookahead `y_k = x_k + beta * v_k`.

/// The outcome of a Nesterov optimization run.
#[derive(Clone, Debug)]
pub struct OptimizeResult {
    /// Best iterate found, length `n`.
    pub x: Vec<f64>,
    /// Number of iterations actually performed (`<= max_iter`).
    pub n_iter: usize,
    /// Euclidean norm of the gradient at `x`.
    pub grad_norm: f64,
    /// `true` if `grad_norm <= tol` was reached within the iteration budget.
    pub converged: bool,
    /// Gradient norm recorded after each iteration, length `n_iter`.
    pub history: Vec<f64>,
}

fn validate_inputs(
    x0: &[f64],
    step_size: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(), String> {
    if x0.is_empty() {
        return Err("x0 must have at least one entry".to_string());
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err("x0 contains non-finite values (nan or inf)".to_string());
    }
    if !(step_size > 0.0) || !step_size.is_finite() {
        return Err(format!(
            "step_size must be a positive finite number, got {}",
            step_size
        ));
    }
    if max_iter < 1 {
        return Err(format!("max_iter must be a positive integer, got {}", max_iter));
    }
    if !(tol >= 0.0) || !tol.is_finite() {
        return Err(format!("tol must be a non-negative finite number, got {}", tol));
    }
    Ok(())
}

fn eval_grad<F>(grad: &F, y: &[f64]) -> Result<Vec<f64>, String>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let g = grad(y);
    if g.len() != y.len() {
        return Err(format!(
            "gradient has length {} but the point has length {}; \
             the gradient oracle must return a vector matching x0",
            g.len(),
            y.len()
        ));
    }
    if g.iter().any(|v| !v.is_finite()) {
        return Err("gradient returned non-finite values (nan or inf)".to_string());
    }
    Ok(g)
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Minimizes a smooth convex function with the two-sequence Nesterov schedule.
///
/// `step_size` is the constant `eta` (`1/L` is canonical). Returns an error on
/// invalid inputs or a non-conforming gradient oracle.
pub fn nesterov_convex<F>(
    grad: &F,
    x0: &[f64],
    step_size: f64,
    max_iter: usize,
    tol: f64,
) -> Result<OptimizeResult, String>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_inputs(x0, step_size, max_iter, tol)?;
    let eta = step_size;
    let mut x = x0.to_vec();
    let mut y = x0.to_vec();
    let mut t = 1.0_f64;
    let mut history: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;

    for k in 1..=max_iter {
        n_iter = k;
        let g = eval_grad(grad, &y)?;
        let x_next: Vec<f64> = y.iter().zip(&g).map(|(yi, gi)| yi - eta * gi).collect();
        let t_next = (1.0 + (1.0 + 4.0 * t * t).sqrt()) / 2.0;
        let gamma = (t - 1.0) / t_next;
        // y_{k+1} = x_next + gamma * (x_next - x)
        y = x_next
            .iter()
            .zip(&x)
            .map(|(xn, xp)| xn + gamma * (xn - xp))
            .collect();
        x = x_next;
        t = t_next;

        let gx = eval_grad(grad, &x)?;
        let gn = norm(&gx);
        history.push(gn);
        if gn <= tol {
            converged = true;
            break;
        }
    }

    let grad_norm = match history.last() {
        Some(&g) => g,
        None => norm(&eval_grad(grad, &x)?),
    };
    Ok(OptimizeResult {
        x,
        n_iter,
        grad_norm,
        converged,
        history,
    })
}

/// Minimizes with the constant-momentum (velocity) Nesterov form.
///
/// The lookahead point is `y_k = x_k + beta * v_k` and the gradient is taken
/// there before the velocity is updated. `momentum` must lie in `[0, 1)`.
pub fn nesterov_momentum<F>(
    grad: &F,
    x0: &[f64],
    step_size: f64,
    momentum: f64,
    max_iter: usize,
    tol: f64,
) -> Result<OptimizeResult, String>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_inputs(x0, step_size, max_iter, tol)?;
    if !(0.0..1.0).contains(&momentum) {
        return Err(format!("momentum must lie in [0, 1), got {}", momentum));
    }
    let eta = step_size;
    let beta = momentum;
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut v = vec![0.0_f64; n];
    let mut history: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;

    for k in 1..=max_iter {
        n_iter = k;
        // y_k = x_k + beta * v_k
        let y: Vec<f64> = x.iter().zip(&v).map(|(xi, vi)| xi + beta * vi).collect();
        let g = eval_grad(grad, &y)?;
        // v_{k+1} = beta * v_k - eta * grad
        for i in 0..n {
            v[i] = beta * v[i] - eta * g[i];
        }
        // x_{k+1} = x_k + v_{k+1}
        for i in 0..n {
            x[i] += v[i];
        }

        let gx = eval_grad(grad, &x)?;
        let gn = norm(&gx);
        history.push(gn);
        if gn <= tol {
            converged = true;
            break;
        }
    }

    let grad_norm = match history.last() {
        Some(&g) => g,
        None => norm(&eval_grad(grad, &x)?),
    };
    Ok(OptimizeResult {
        x,
        n_iter,
        grad_norm,
        converged,
        history,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    // Quadratic f(x) = 0.5 x^T A x - b^T x, grad f(x) = A x - b, with
    //   A = [[5, 1], [1, 3]],  b = [1, 2],  x* = [1/14, 9/14].
    fn grad_fixture(x: &[f64]) -> Vec<f64> {
        let a: [[f64; 2]; 2] = [[5.0, 1.0], [1.0, 3.0]];
        let b: [f64; 2] = [1.0, 2.0];
        vec![
            a[0][0] * x[0] + a[0][1] * x[1] - b[0],
            a[1][0] * x[0] + a[1][1] * x[1] - b[1],
        ]
    }

    const ETA: f64 = 1.0 / 6.0;
    const BETA: f64 = 0.5;
    const X_STAR: [f64; 2] = [0.07142857142857142, 0.6428571428571429];
    const TOL: f64 = 1e-9;

    #[test]
    fn convex_five_iterations_matches_fixture() {
        let res = nesterov_convex(&grad_fixture, &[0.0, 0.0], ETA, 5, 0.0).unwrap();
        let expected_x: [f64; 2] = [0.06917044734055175, 0.6483203299202533];
        let expected_gn: f64 = 0.015285826582542654;
        let expected_hist: [f64; 5] = [
            0.8498365855987974,
            0.4746668747398631,
            0.2123582666221123,
            0.05611771750996081,
            0.015285826582542654,
        ];
        assert_eq!(res.n_iter, 5);
        assert!(!res.converged);
        assert!((res.x[0] - expected_x[0]).abs() < TOL);
        assert!((res.x[1] - expected_x[1]).abs() < TOL);
        assert!((res.grad_norm - expected_gn).abs() < TOL);
        for i in 0..5 {
            assert!((res.history[i] - expected_hist[i]).abs() < TOL);
        }
    }

    #[test]
    fn momentum_five_iterations_matches_fixture() {
        let res = nesterov_momentum(&grad_fixture, &[0.0, 0.0], ETA, BETA, 5, 0.0).unwrap();
        let expected_x: [f64; 2] = [0.061631944444444475, 0.666087962962963];
        let expected_gn: f64 = 0.06519733559752104;
        let expected_hist: [f64; 5] = [
            0.8498365855987974,
            0.3004626062886657,
            0.021960261528947072,
            0.07158169488919522,
            0.06519733559752104,
        ];
        assert_eq!(res.n_iter, 5);
        assert!(!res.converged);
        assert!((res.x[0] - expected_x[0]).abs() < TOL);
        assert!((res.x[1] - expected_x[1]).abs() < TOL);
        assert!((res.grad_norm - expected_gn).abs() < TOL);
        for i in 0..5 {
            assert!((res.history[i] - expected_hist[i]).abs() < TOL);
        }
    }

    #[test]
    fn convex_converges_to_minimizer() {
        let res = nesterov_convex(&grad_fixture, &[0.0, 0.0], ETA, 10000, 1e-10).unwrap();
        assert!(res.converged);
        assert!((res.x[0] - X_STAR[0]).abs() < 1e-7);
        assert!((res.x[1] - X_STAR[1]).abs() < 1e-7);
        assert!(res.grad_norm <= 1e-10);
    }

    #[test]
    fn momentum_converges_to_minimizer() {
        let res = nesterov_momentum(&grad_fixture, &[0.0, 0.0], ETA, BETA, 10000, 1e-10).unwrap();
        assert!(res.converged);
        assert!((res.x[0] - X_STAR[0]).abs() < 1e-7);
        assert!((res.x[1] - X_STAR[1]).abs() < 1e-7);
        assert!(res.grad_norm <= 1e-10);
    }

    #[test]
    fn already_at_optimum_converges_immediately() {
        let res = nesterov_convex(&grad_fixture, &X_STAR, ETA, 100, 1e-9).unwrap();
        assert!(res.converged);
        assert_eq!(res.n_iter, 1);
        assert!(res.grad_norm <= 1e-9);
    }

    #[test]
    fn zero_momentum_is_plain_gradient_descent() {
        let res = nesterov_momentum(&grad_fixture, &[0.0, 0.0], ETA, 0.0, 1, 0.0).unwrap();
        let g = grad_fixture(&[0.0, 0.0]);
        assert!((res.x[0] - (-ETA * g[0])).abs() < TOL);
        assert!((res.x[1] - (-ETA * g[1])).abs() < TOL);
    }

    #[test]
    fn history_length_equals_n_iter() {
        let res = nesterov_convex(&grad_fixture, &[0.0, 0.0], ETA, 7, 0.0).unwrap();
        assert_eq!(res.history.len(), res.n_iter);
        assert_eq!(res.n_iter, 7);
    }

    #[test]
    fn bad_step_size_errors() {
        assert!(nesterov_convex(&grad_fixture, &[0.0, 0.0], 0.0, 10, 0.0).is_err());
        assert!(nesterov_momentum(&grad_fixture, &[0.0, 0.0], -1.0, 0.5, 10, 0.0).is_err());
    }

    #[test]
    fn bad_momentum_errors() {
        assert!(nesterov_momentum(&grad_fixture, &[0.0, 0.0], ETA, 1.0, 10, 0.0).is_err());
        assert!(nesterov_momentum(&grad_fixture, &[0.0, 0.0], ETA, -0.1, 10, 0.0).is_err());
    }

    #[test]
    fn bad_max_iter_errors() {
        assert!(nesterov_convex(&grad_fixture, &[0.0, 0.0], ETA, 0, 0.0).is_err());
    }

    #[test]
    fn empty_x0_errors() {
        assert!(nesterov_convex(&grad_fixture, &[], ETA, 10, 0.0).is_err());
    }

    #[test]
    fn grad_shape_mismatch_errors() {
        let bad = |_x: &[f64]| vec![0.0_f64];
        assert!(nesterov_convex(&bad, &[0.0, 0.0], ETA, 10, 0.0).is_err());
    }
}
