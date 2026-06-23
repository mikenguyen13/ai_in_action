//! Momentum (heavy-ball) gradient descent from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch192_momentum` and the Julia module
//! `AIInAction.Ch192Momentum`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! The optimizer maintains a velocity vector that is an exponentially weighted
//! accumulation of past gradients and steps the parameters along it:
//!
//! ```text
//! v <- beta * v + grad(theta)
//! theta <- theta - alpha * v
//! ```
//!
//! `beta = 0` recovers plain gradient descent. This is a std-only implementation;
//! the gradient is supplied as a closure so the driver works for any objective.

/// The outcome of a momentum optimization run.
#[derive(Clone, Debug)]
pub struct MomentumResult {
    /// Final parameter vector, length `d`.
    pub theta: Vec<f64>,
    /// Final velocity vector, length `d`.
    pub velocity: Vec<f64>,
    /// Number of update steps actually performed (`<= max_iter`).
    pub n_iter: usize,
    /// True if stopped because `||grad|| <= tol` rather than hitting `max_iter`.
    pub converged: bool,
    /// Euclidean gradient norm at `theta` on the final iteration.
    pub grad_norm: f64,
    /// Gradient norm at the start of each performed step, length `n_iter`.
    pub history: Vec<f64>,
}

impl MomentumResult {
    pub fn n_features(&self) -> usize {
        self.theta.len()
    }
}

fn check_vector(x: &[f64], name: &str) -> Result<(), String> {
    if x.is_empty() {
        return Err(format!("{} must have at least one entry", name));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(format!("{} contains non-finite values (nan or inf)", name));
    }
    Ok(())
}

fn check_hyperparams(alpha: f64, beta: f64) -> Result<(), String> {
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(format!(
            "alpha (learning rate) must be a positive finite number, got {}",
            alpha
        ));
    }
    if !beta.is_finite() || beta < 0.0 || beta >= 1.0 {
        return Err(format!("beta (momentum) must be in [0, 1), got {}", beta));
    }
    Ok(())
}

/// Applies a single heavy-ball update, returning `(new_theta, new_velocity)`.
///
/// `v' = beta*v + g`, `theta' = theta - alpha*v'`. The three vectors must share a
/// length.
pub fn momentum_step(
    theta: &[f64],
    velocity: &[f64],
    grad: &[f64],
    alpha: f64,
    beta: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    check_hyperparams(alpha, beta)?;
    check_vector(theta, "theta")?;
    check_vector(velocity, "velocity")?;
    check_vector(grad, "grad")?;
    if !(theta.len() == velocity.len() && velocity.len() == grad.len()) {
        return Err(format!(
            "length mismatch: theta={}, velocity={}, grad={}",
            theta.len(),
            velocity.len(),
            grad.len()
        ));
    }
    let new_v: Vec<f64> = velocity
        .iter()
        .zip(grad)
        .map(|(v, g)| beta * v + g)
        .collect();
    let new_theta: Vec<f64> = theta
        .iter()
        .zip(&new_v)
        .map(|(t, v)| t - alpha * v)
        .collect();
    Ok((new_theta, new_v))
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Minimizes an objective with heavy-ball momentum from initial point `theta0`.
///
/// `grad_fn` maps a parameter vector to its gradient. `beta = 0` is plain gradient
/// descent. The loop stops as soon as `||grad(theta)|| <= tol`.
pub fn minimize<F>(
    grad_fn: F,
    theta0: &[f64],
    alpha: f64,
    beta: f64,
    max_iter: usize,
    tol: f64,
) -> Result<MomentumResult, String>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    check_hyperparams(alpha, beta)?;
    check_vector(theta0, "theta0")?;
    if max_iter < 1 {
        return Err(format!("max_iter must be a positive integer, got {}", max_iter));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(format!("tol must be a non-negative finite number, got {}", tol));
    }

    let d = theta0.len();
    let mut theta = theta0.to_vec();
    let mut velocity = vec![0.0f64; d];
    let mut history: Vec<f64> = Vec::new();
    let mut grad_norm = f64::INFINITY;
    let mut converged = false;
    let mut n_iter = 0usize;

    for _ in 0..max_iter {
        let g = grad_fn(&theta);
        if g.len() != d {
            return Err(format!(
                "grad_fn returned length {} but theta has length {}",
                g.len(),
                d
            ));
        }
        if g.iter().any(|v| !v.is_finite()) {
            return Err("grad_fn returned non-finite values (nan or inf)".to_string());
        }
        grad_norm = norm2(&g);
        history.push(grad_norm);
        if grad_norm <= tol {
            converged = true;
            break;
        }
        for j in 0..d {
            velocity[j] = beta * velocity[j] + g[j];
            theta[j] -= alpha * velocity[j];
        }
        n_iter += 1;
    }

    Ok(MomentumResult {
        theta,
        velocity,
        n_iter,
        converged,
        grad_norm,
        history,
    })
}

/// Builds the gradient closure of `f(theta) = 1/2 (theta - b)^T H (theta - b)`.
///
/// Returns a closure computing `grad f(theta) = H (theta - b)`. `h` must be a
/// square `d x d` matrix (rows) and `b` a length-`d` vector.
pub fn quadratic_gradient(
    h: &[Vec<f64>],
    b: &[f64],
) -> Result<impl Fn(&[f64]) -> Vec<f64>, String> {
    let d = h.len();
    if d == 0 {
        return Err("H must have at least one row".to_string());
    }
    for row in h {
        if row.len() != d {
            return Err(format!("H must be a square matrix; got a row of length {}", row.len()));
        }
    }
    if h.iter().any(|row| row.iter().any(|v| !v.is_finite())) {
        return Err("H contains non-finite values (nan or inf)".to_string());
    }
    check_vector(b, "b")?;
    if b.len() != d {
        return Err(format!("H is {}x{} but b has length {}", d, d, b.len()));
    }
    let hm: Vec<Vec<f64>> = h.to_vec();
    let bv: Vec<f64> = b.to_vec();
    Ok(move |theta: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0f64; d];
        for i in 0..d {
            let mut acc = 0.0;
            for j in 0..d {
                acc += hm[i][j] * (theta[j] - bv[j]);
            }
            out[i] = acc;
        }
        out
    })
}

/// Polyak's optimal momentum for a quadratic with curvature in `[lambda_min, lambda_max]`.
///
/// `beta* = ((sqrt(hi) - sqrt(lo)) / (sqrt(hi) + sqrt(lo)))^2`.
pub fn optimal_beta(lambda_min: f64, lambda_max: f64) -> Result<f64, String> {
    if !lambda_min.is_finite() || !lambda_max.is_finite() || lambda_min <= 0.0 || lambda_max <= 0.0 {
        return Err("lambda_min and lambda_max must be positive finite numbers".to_string());
    }
    if lambda_max < lambda_min {
        return Err(format!(
            "lambda_max ({}) must be >= lambda_min ({})",
            lambda_max, lambda_min
        ));
    }
    let r = (lambda_max.sqrt() - lambda_min.sqrt()) / (lambda_max.sqrt() + lambda_min.sqrt());
    Ok(r * r)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // Shared fixtures: identical to the Python and Julia test suites.
    #[test]
    fn single_step_matches_fixture() {
        let theta: [f64; 2] = [1.0, 2.0];
        let velocity: [f64; 2] = [0.5, -0.5];
        let grad: [f64; 2] = [0.2, 0.4];
        let (th, v) = momentum_step(&theta, &velocity, &grad, 0.1, 0.9).unwrap();
        let exp_th: [f64; 2] = [0.935, 2.005];
        let exp_v: [f64; 2] = [0.65, -0.04999999999999999];
        for i in 0..2 {
            assert!((th[i] - exp_th[i]).abs() < TOL);
            assert!((v[i] - exp_v[i]).abs() < TOL);
        }
    }

    #[test]
    fn beta_zero_step_is_plain_gd() {
        let (th, v) = momentum_step(&[1.0], &[7.0], &[2.0], 0.1, 0.0).unwrap();
        assert!((v[0] - 2.0).abs() < TOL);
        assert!((th[0] - 0.8).abs() < TOL);
    }

    fn quad_grad() -> impl Fn(&[f64]) -> Vec<f64> {
        let h: [Vec<f64>; 2] = [vec![3.0, 0.2], vec![0.2, 1.0]];
        let b: [f64; 2] = [1.0, -2.0];
        quadratic_gradient(&h, &b).unwrap()
    }

    #[test]
    fn fixed_five_iterations_match_fixture() {
        let g = quad_grad();
        let theta0: [f64; 2] = [0.0, 0.0];
        let r = minimize(g, &theta0, 0.2, 0.9, 5, 0.0).unwrap();
        assert_eq!(r.n_iter, 5);
        assert!(!r.converged);
        let exp_theta: [f64; 2] = [1.2893499712, -3.1595351615999996];
        let exp_vel: [f64; 2] = [1.884893344, 3.068733408];
        let exp_hist: [f64; 5] = [
            3.1622776601683795,
            1.90275589606234,
            1.339506583783745,
            2.072914156257514,
            1.9343282392460306,
        ];
        for i in 0..2 {
            assert!((r.theta[i] - exp_theta[i]).abs() < TOL);
            assert!((r.velocity[i] - exp_vel[i]).abs() < TOL);
        }
        for i in 0..5 {
            assert!((r.history[i] - exp_hist[i]).abs() < TOL);
        }
    }

    #[test]
    fn converges_to_quadratic_minimum() {
        let g = quad_grad();
        let theta0: [f64; 2] = [0.0, 0.0];
        let r = minimize(g, &theta0, 0.2, 0.9, 5000, 1e-10).unwrap();
        assert!(r.converged);
        assert!((r.theta[0] - 1.0).abs() < 1e-7);
        assert!((r.theta[1] - (-2.0)).abs() < 1e-7);
        assert!(r.grad_norm <= 1e-10);
    }

    #[test]
    fn beta_zero_recovers_gd_minimum() {
        let h: [Vec<f64>; 1] = [vec![2.0]];
        let b: [f64; 1] = [5.0];
        let g = quadratic_gradient(&h, &b).unwrap();
        let theta0: [f64; 1] = [0.0];
        let r = minimize(g, &theta0, 0.3, 0.0, 500, 1e-12).unwrap();
        assert!(r.converged);
        assert!((r.theta[0] - 5.0).abs() < TOL);
    }

    #[test]
    fn momentum_beats_plain_gd() {
        let h: [Vec<f64>; 2] = [vec![50.0, 0.0], vec![0.0, 1.0]];
        let b: [f64; 2] = [0.0, 0.0];
        let theta0: [f64; 2] = [1.0, 1.0];
        let plain = minimize(quadratic_gradient(&h, &b).unwrap(), &theta0, 0.02, 0.0, 10000, 1e-6).unwrap();
        let fast = minimize(quadratic_gradient(&h, &b).unwrap(), &theta0, 0.02, 0.9, 10000, 1e-6).unwrap();
        assert!(plain.converged && fast.converged);
        assert!(fast.n_iter < plain.n_iter);
    }

    #[test]
    fn optimal_beta_matches_fixture() {
        let b = optimal_beta(1.0, 100.0).unwrap();
        assert!((b - 0.6694214876033059).abs() < TOL);
    }

    #[test]
    fn optimal_beta_isotropic_is_zero() {
        let b = optimal_beta(2.0, 2.0).unwrap();
        assert!(b.abs() < 1e-15);
    }

    #[test]
    fn bad_alpha_errors() {
        let g = quad_grad();
        let theta0: [f64; 2] = [1.0, 1.0];
        assert!(minimize(g, &theta0, 0.0, 0.5, 10, 1e-8).is_err());
    }

    #[test]
    fn bad_beta_errors() {
        assert!(momentum_step(&[1.0], &[0.0], &[1.0], 0.1, 1.0).is_err());
        assert!(momentum_step(&[1.0], &[0.0], &[1.0], 0.1, -0.1).is_err());
    }

    #[test]
    fn step_length_mismatch_errors() {
        assert!(momentum_step(&[1.0, 2.0], &[0.0], &[1.0, 1.0], 0.1, 0.5).is_err());
    }

    #[test]
    fn quadratic_gradient_shape_mismatch_errors() {
        let h: [Vec<f64>; 2] = [vec![1.0, 0.0], vec![0.0, 1.0]];
        let b: [f64; 1] = [1.0];
        assert!(quadratic_gradient(&h, &b).is_err());
    }

    #[test]
    fn optimal_beta_bad_order_errors() {
        assert!(optimal_beta(10.0, 1.0).is_err());
    }
}
