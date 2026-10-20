//! Why depth is not free: signal propagation and depth diagnostics from
//! scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch2071_depth` and the Julia module
//! `AIInAction.Ch2071Depth`. The shared fixtures in the tests below match the
//! Python and Julia suites, which is what keeps the three implementations at
//! parity.
//!
//! Contents: the mean field length map, its fixed point `q*`, the correlation
//! map, the slope `chi_1` and the correlation depth scale (Poole et al. 2016,
//! Schoenholz et al. 2017); the closed form ReLU correlation map (the
//! normalized arc cosine kernel of Cho and Saul 2009), its iteration in plain
//! and residual networks, the `9 pi^2 / (2 l^2)` collapse law and the
//! continuous depth limit; residual stream variance growth; the learning time
//! of a scalar deep linear network (Saxe et al. 2014); residual path
//! statistics (Veit et al. 2016); and Telgarsky's sawtooth. Activations are
//! named by the strings `"linear"`, `"relu"` and `"tanh"`. Everything is
//! deterministic: quadrature is composite Simpson with fixed nodes and the ODE
//! uses fixed step RK4.

use std::f64::consts::PI;

/// Supported activation names.
pub const ACTIVATIONS: [&str; 3] = ["linear", "relu", "tanh"];
/// Default half width of the quadrature interval `[-lim, lim]`.
pub const DEFAULT_LIM: f64 = 10.0;
/// Default number of Simpson panels for one dimensional expectations.
pub const DEFAULT_N1: usize = 4000;
/// Default number of Simpson panels per axis for two dimensional expectations.
pub const DEFAULT_N2: usize = 400;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    Linear,
    Relu,
    Tanh,
}

fn parse_activation(name: &str) -> Result<Act, String> {
    match name {
        "linear" => Ok(Act::Linear),
        "relu" => Ok(Act::Relu),
        "tanh" => Ok(Act::Tanh),
        _ => Err(format!(
            "unknown activation {:?}; expected one of {:?}",
            name, ACTIVATIONS
        )),
    }
}

fn act_eval(a: Act, x: f64) -> f64 {
    match a {
        Act::Linear => x,
        Act::Relu => {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }
        Act::Tanh => x.tanh(),
    }
}

fn act_deriv(a: Act, x: f64) -> f64 {
    match a {
        Act::Linear => 1.0,
        Act::Relu => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Act::Tanh => {
            let t = x.tanh();
            1.0 - t * t
        }
    }
}

/// Evaluate `phi(x)` for `name` in [`ACTIVATIONS`].
pub fn activation(name: &str, x: f64) -> Result<f64, String> {
    Ok(act_eval(parse_activation(name)?, x))
}

/// Evaluate `phi'(x)`; the ReLU derivative at 0 is taken as 0.
pub fn activation_derivative(name: &str, x: f64) -> Result<f64, String> {
    Ok(act_deriv(parse_activation(name)?, x))
}

fn simpson_weights(n: usize) -> Result<Vec<f64>, String> {
    if n < 2 || n % 2 != 0 {
        return Err("number of Simpson panels must be an even integer >= 2".to_string());
    }
    let mut w: Vec<f64> = (0..=n).map(|i| if i % 2 == 0 { 2.0 } else { 4.0 }).collect();
    w[0] = 1.0;
    w[n] = 1.0;
    Ok(w)
}

/// `E[f(z)]` for `z ~ N(0, 1)` by composite Simpson quadrature of
/// `f(z) * exp(-z^2 / 2) / sqrt(2 pi)` on `[-lim, lim]` with `n` panels.
pub fn gaussian_expectation<F: Fn(f64) -> f64>(f: F, n: usize, lim: f64) -> Result<f64, String> {
    let w = simpson_weights(n)?;
    let h = 2.0 * lim / n as f64;
    let c = 1.0 / (2.0 * PI).sqrt();
    let mut total = 0.0;
    for (i, wi) in w.iter().enumerate() {
        let z = -lim + i as f64 * h;
        total += wi * f(z) * (-0.5 * z * z).exp();
    }
    Ok(total * h / 3.0 * c)
}

/// `E[f(z1, z2)]` for independent standard normals by tensor Simpson
/// quadrature with `n` panels per axis.
pub fn gaussian_expectation_2d<F: Fn(f64, f64) -> f64>(
    f: F,
    n: usize,
    lim: f64,
) -> Result<f64, String> {
    let w = simpson_weights(n)?;
    let h = 2.0 * lim / n as f64;
    let c = 1.0 / (2.0 * PI);
    let nodes: Vec<f64> = (0..=n).map(|i| -lim + i as f64 * h).collect();
    let dens: Vec<f64> = nodes.iter().map(|z| (-0.5 * z * z).exp()).collect();
    let mut total = 0.0;
    for i in 0..=n {
        let z1 = nodes[i];
        let wi = w[i] * dens[i];
        let mut row = 0.0;
        for j in 0..=n {
            row += w[j] * dens[j] * f(z1, nodes[j]);
        }
        total += wi * row;
    }
    Ok(total * (h / 3.0).powi(2) * c)
}

fn check_variances(sigma_w2: f64, sigma_b2: f64) -> Result<(), String> {
    if sigma_w2 <= 0.0 || sigma_b2 < 0.0 || sigma_w2.is_nan() || sigma_b2.is_nan() {
        return Err("need sigma_w2 > 0 and sigma_b2 >= 0".to_string());
    }
    Ok(())
}

/// One step of the mean field length map
/// `q_{l+1} = sigma_w^2 E[phi(sqrt(q_l) z)^2] + sigma_b^2`.
pub fn length_map(q: f64, sigma_w2: f64, sigma_b2: f64, phi: &str) -> Result<f64, String> {
    check_variances(sigma_w2, sigma_b2)?;
    if q < 0.0 {
        return Err("q must be nonnegative".to_string());
    }
    let a = parse_activation(phi)?;
    let s = q.sqrt();
    let e = gaussian_expectation(
        |z| {
            let v = act_eval(a, s * z);
            v * v
        },
        DEFAULT_N1,
        DEFAULT_LIM,
    )?;
    Ok(sigma_w2 * e + sigma_b2)
}

/// Iterate the length map from `q0` to its fixed point `q*`, stopping when
/// `|q_next - q| <= tol * max(1, q)`. Errors if the map diverges (for example
/// ReLU with `sigma_w^2 > 2`) or has not converged within `max_iter` steps.
pub fn length_fixed_point(
    sigma_w2: f64,
    sigma_b2: f64,
    phi: &str,
    q0: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    let mut q = q0;
    for _ in 0..max_iter {
        let q_next = length_map(q, sigma_w2, sigma_b2, phi)?;
        if !q_next.is_finite() || q_next > 1e12 {
            return Err("length map diverged; no finite fixed point".to_string());
        }
        if (q_next - q).abs() <= tol * q.max(1.0) {
            return Ok(q_next);
        }
        q = q_next;
    }
    Err("length map did not converge".to_string())
}

/// One step of the mean field correlation map at fixed length `q`:
/// `c_{l+1} = (sigma_w^2 E[phi(u1) phi(u2)] + sigma_b^2) / q` with
/// `u1 = sqrt(q) z1` and `u2 = sqrt(q) (c z1 + sqrt(1 - c^2) z2)`, using `n`
/// Simpson panels per axis.
pub fn correlation_map(
    c: f64,
    q: f64,
    sigma_w2: f64,
    sigma_b2: f64,
    phi: &str,
    n: usize,
) -> Result<f64, String> {
    check_variances(sigma_w2, sigma_b2)?;
    if !(-1.0..=1.0).contains(&c) {
        return Err("correlation must lie in [-1, 1]".to_string());
    }
    if !(q > 0.0) {
        return Err("q must be positive".to_string());
    }
    let a = parse_activation(phi)?;
    let s = q.sqrt();
    let r = (1.0 - c * c).max(0.0).sqrt();
    let e = gaussian_expectation_2d(
        |z1, z2| act_eval(a, s * z1) * act_eval(a, s * (c * z1 + r * z2)),
        n,
        DEFAULT_LIM,
    )?;
    Ok((sigma_w2 * e + sigma_b2) / q)
}

/// Slope of the correlation map at `c = 1`,
/// `chi_1 = sigma_w^2 E[phi'(sqrt(q) z)^2]`. `chi_1 < 1` is the ordered phase
/// and `chi_1 > 1` the chaotic phase.
pub fn chi1(q: f64, sigma_w2: f64, phi: &str) -> Result<f64, String> {
    if !(sigma_w2 > 0.0) || !(q >= 0.0) {
        return Err("need sigma_w2 > 0 and q >= 0".to_string());
    }
    let a = parse_activation(phi)?;
    let s = q.sqrt();
    let e = gaussian_expectation(
        |z| {
            let v = act_deriv(a, s * z);
            v * v
        },
        DEFAULT_N1,
        DEFAULT_LIM,
    )?;
    Ok(sigma_w2 * e)
}

/// `xi_c = -1 / log(chi_c)`; infinite at criticality `chi_c = 1`.
pub fn correlation_depth_scale(chi_c: f64) -> Result<f64, String> {
    if !(chi_c > 0.0) {
        return Err("chi_c must be positive".to_string());
    }
    if chi_c == 1.0 {
        return Ok(f64::INFINITY);
    }
    Ok(-1.0 / chi_c.ln())
}

/// Closed form ReLU correlation map at criticality,
/// `f(c) = (sqrt(1 - c^2) + (pi - acos(c)) c) / pi`.
pub fn relu_correlation_map(c: f64) -> Result<f64, String> {
    if !(-1.0..=1.0).contains(&c) {
        return Err("correlation must lie in [-1, 1]".to_string());
    }
    Ok(((1.0 - c * c).max(0.0).sqrt() + (PI - c.acos()) * c) / PI)
}

/// Correlation after each of `depth` layers, returned as
/// `[c_0, c_1, ..., c_depth]`. `alpha = None` is a plain critical ReLU
/// network, `c_{l+1} = f(c_l)`; `Some(alpha)` is a residual network with
/// branch scale `alpha`, `c_{l+1} = (c_l + alpha^2 f(c_l)) / (1 + alpha^2)`.
pub fn relu_correlation_trajectory(
    c0: f64,
    depth: usize,
    alpha: Option<f64>,
) -> Result<Vec<f64>, String> {
    if let Some(a) = alpha {
        if !(a >= 0.0) {
            return Err("alpha must be nonnegative".to_string());
        }
    }
    let mut out = Vec::with_capacity(depth + 1);
    out.push(c0);
    let mut c = c0;
    for _ in 0..depth {
        let fc = relu_correlation_map(c)?;
        c = match alpha {
            None => fc,
            Some(a) => {
                let a2 = a * a;
                (c + a2 * fc) / (1.0 + a2)
            }
        };
        c = c.max(-1.0).min(1.0);
        out.push(c);
    }
    Ok(out)
}

/// Asymptotic `1 - c_l` for large depth: `9 pi^2 / (2 l^2)` for a plain
/// network and `9 pi^2 (1 + alpha^2)^2 / (2 alpha^4 l^2)` for a residual one.
pub fn relu_collapse_asymptotic(depth: f64, alpha: Option<f64>) -> Result<f64, String> {
    if !(depth > 0.0) {
        return Err("depth must be positive".to_string());
    }
    let base = 9.0 * PI.powi(2) / (2.0 * depth.powi(2));
    match alpha {
        None => Ok(base),
        Some(a) => {
            if !(a > 0.0) {
                return Err("alpha must be positive".to_string());
            }
            let a2 = a * a;
            Ok(base * (1.0 + a2).powi(2) / (a2 * a2))
        }
    }
}

/// Integrate `dc/dt = f(c) - c` from `c(0) = c0` to time `t` with `steps`
/// fixed RK4 steps. This is the continuous depth limit of a residual ReLU
/// network with branch scale `alpha = 1 / sqrt(L)`.
pub fn relu_kernel_ode(c0: f64, t: f64, steps: usize) -> Result<f64, String> {
    if steps == 0 || !(t >= 0.0) {
        return Err("need steps > 0 and t >= 0".to_string());
    }
    let g = |c: f64| -> f64 {
        let cc = c.max(-1.0).min(1.0);
        ((1.0 - cc * cc).max(0.0).sqrt() + (PI - cc.acos()) * cc) / PI - cc
    };
    let h = t / steps as f64;
    let mut c = c0;
    for _ in 0..steps {
        let k1 = g(c);
        let k2 = g(c + 0.5 * h * k1);
        let k3 = g(c + 0.5 * h * k2);
        let k4 = g(c + h * k3);
        c += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
    }
    Ok(c)
}

/// Residual stream variance `q_0 (1 + alpha^2)^depth` for ReLU branches.
pub fn residual_variance(q0: f64, depth: usize, alpha: f64) -> Result<f64, String> {
    if !(q0 >= 0.0) || !(alpha >= 0.0) {
        return Err("need q0 >= 0, depth >= 0, alpha >= 0".to_string());
    }
    Ok(q0 * (1.0 + alpha * alpha).powf(depth as f64))
}

/// Gradient flow time for the scalar depth `N` linear net `u = a_1 ... a_N`,
/// started at `a_i = a0`, to reach `u = frac * s`:
/// `T = int_{u0}^{frac s} du / (N u^{2 - 2/N} (s - u))` with `u0 = a0^N`,
/// evaluated by Simpson's rule with `n` panels in `v = log u`.
pub fn deep_linear_learning_time(
    depth: usize,
    a0: f64,
    s: f64,
    frac: f64,
    n: usize,
) -> Result<f64, String> {
    if depth < 1 {
        return Err("depth must be at least 1".to_string());
    }
    if !(a0 > 0.0) || !(s > 0.0) || !(frac > 0.0 && frac < 1.0) {
        return Err("need a0 > 0, s > 0 and 0 < frac < 1".to_string());
    }
    let nf = depth as f64;
    let u0 = a0.powf(nf);
    let u1 = frac * s;
    if u0 >= u1 {
        return Ok(0.0);
    }
    let (v0, v1) = (u0.ln(), u1.ln());
    let w = simpson_weights(n)?;
    let h = (v1 - v0) / n as f64;
    let p = 2.0 - 2.0 / nf;
    let mut total = 0.0;
    for (i, wi) in w.iter().enumerate() {
        let v = v0 + i as f64 * h;
        let u = v.exp();
        total += wi * u / (nf * u.powf(p) * (s - u));
    }
    Ok(total * h / 3.0)
}

/// Leading order learning time `a0^{-(N-2)} / (s (N - 2))` for `N > 2`.
pub fn deep_linear_time_asymptotic(depth: usize, a0: f64, s: f64) -> Result<f64, String> {
    if depth <= 2 {
        return Err("asymptotic formula needs depth > 2".to_string());
    }
    if !(a0 > 0.0) || !(s > 0.0) {
        return Err("need a0 > 0 and s > 0".to_string());
    }
    let m = (depth - 2) as f64;
    Ok(a0.powf(-m) / (s * m))
}

/// Row `n` of Pascal's triangle as floats (exact while entries stay below
/// `2^53`, that is for `n <= 56`).
fn binomial_row(n: usize) -> Vec<f64> {
    let mut row = vec![1.0_f64];
    for _ in 0..n {
        let mut next = vec![1.0_f64; row.len() + 1];
        for k in 1..row.len() {
            next[k] = row[k - 1] + row[k];
        }
        row = next;
    }
    row
}

/// Fraction of the `2^n` paths through `n` residual blocks that cross exactly
/// `k` branches, `C(n, k) / 2^n` for `k = 0, ..., n`.
pub fn path_length_distribution(n_blocks: usize) -> Vec<f64> {
    let total = 2.0_f64.powf(n_blocks as f64);
    binomial_row(n_blocks).into_iter().map(|c| c / total).collect()
}

/// Share of the gradient carried by paths of each length when each residual
/// branch scales a backpropagated signal by `rho`: Binomial(n, rho / (1 + rho)).
pub fn effective_path_distribution(n_blocks: usize, rho: f64) -> Result<Vec<f64>, String> {
    if !(rho >= 0.0) {
        return Err("need n_blocks >= 0 and rho >= 0".to_string());
    }
    let p = rho / (1.0 + rho);
    Ok(binomial_row(n_blocks)
        .into_iter()
        .enumerate()
        .map(|(k, c)| c * p.powf(k as f64) * (1.0 - p).powf((n_blocks - k) as f64))
        .collect())
}

/// Telgarsky's construction: `depth` compositions of the tent map
/// `m(x) = 2 relu(x) - 4 relu(x - 1/2)`, giving `2^depth` linear pieces on
/// `[0, 1]`.
pub fn sawtooth(x: f64, depth: usize) -> f64 {
    let mut y = x;
    for _ in 0..depth {
        y = 2.0 * y.max(0.0) - 4.0 * (y - 0.5).max(0.0);
    }
    y
}

/// Count maximal linear pieces of a function sampled on a uniform grid: each
/// maximal run of grid points whose second difference exceeds `tol` in
/// magnitude marks one breakpoint.
pub fn count_linear_pieces(ys: &[f64], tol: f64) -> Result<usize, String> {
    if ys.len() < 3 {
        return Err("need at least three samples".to_string());
    }
    let mut pieces = 1;
    let mut in_kink = false;
    for i in 1..ys.len() - 1 {
        let d2 = ys[i + 1] - 2.0 * ys[i] + ys[i - 1];
        let kink = d2.abs() > tol;
        if kink && !in_kink {
            pieces += 1;
        }
        in_kink = kink;
    }
    Ok(pieces)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const TOL: f64 = 1e-9;

    const GE_Z2: f64 = 0.9999999999999986;
    const GE_Z4: f64 = 2.9999999999999942;
    const GE_COS: f64 = 0.6065306597126348;
    const GE_COS_N40_LIM6: f64 = 0.6065306577125951;
    const GE2_SQ_N40_LIM6: f64 = 1.999999841083864;
    const GE2_COSPROD_N20_LIM5: f64 = 0.08206418533372323;

    const LM_RELU: f64 = 0.9999999999999994;
    const LM_TANH: f64 = 0.6414417355967605;
    const LM_LINEAR: f64 = 0.9399999999999971;
    const LM_TANH_Q2: f64 = 1.2999393641598669;
    const QSTAR_TANH: f64 = 0.41803720053418575;
    const QSTAR_RELU: f64 = 0.40000000000256286;
    const QSTAR_TANH_CHAOS: f64 = 2.195493928033813;
    const CM_RELU: f64 = 0.6089975654758696;
    const CM_TANH: f64 = 0.54974715475882;
    const CM_TANH_N40: f64 = 0.37652618700891255;
    const CM_LINEAR_N40: f64 = -0.27000003475872764;
    const CHI1_TANH: f64 = 0.9386362681983815;
    const CHI1_RELU: f64 = 0.9986701923986636;
    const CHI1_LINEAR: f64 = 0.8999999999999972;
    const XI_09: f64 = 9.491221581029905;

    const RCM_INPUTS: [f64; 6] = [-1.0, -0.5, 0.0, 0.5, 0.9, 1.0];
    const RCM: [f64; 6] = [
        0.0,
        0.1089977810442294,
        0.3183098861837907,
        0.6089977810442293,
        0.9095383988446721,
        1.0,
    ];
    const TRAJ_PLAIN: [f64; 6] = [
        0.2,
        0.4246975638632654,
        0.5598220427417753,
        0.6495461163619815,
        0.7129656727390657,
        0.7598110813130384,
    ];
    const TRAJ_RES: [f64; 6] = [
        0.2,
        0.24493951277265308,
        0.2860269749375514,
        0.3237085998221533,
        0.35836528648096716,
        0.39032419037801597,
    ];
    const TRAJ_NEG: [f64; 4] = [-0.9, 0.009538398844672072, 0.3230935658001504, 0.49662004436592994];
    const COLLAPSE_PLAIN: f64 = 0.44413219804902115;
    const COLLAPSE_RES: f64 = 11.103304951225528;
    const ODE: f64 = 0.3839566908351801;
    const ODE_COARSE: f64 = 0.2991227756181538;
    const RV1: f64 = 1024.0;
    const RV2: f64 = 1.3524069147107642;

    const DLT_DEPTHS: [usize; 5] = [1, 2, 3, 5, 10];
    const DLT: [f64; 5] = [
        0.0,
        0.5493061443340553,
        1.0179617754504038,
        2.4934025172272927,
        32.14439124998376,
    ];
    const DLT_OPTS: f64 = 2.9016436644053982;
    const DLA_DEPTHS: [usize; 3] = [3, 5, 10];
    const DLA: [f64; 3] = [2.0, 2.6666666666666665, 32.0];
    const DLA_S: f64 = 2.777777777777778;

    const PLD4: [f64; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    const EPD: [f64; 5] = [
        0.19753086419753096,
        0.3950617283950618,
        0.29629629629629634,
        0.09876543209876541,
        0.012345679012345677,
    ];
    const EPD_ZERO: [f64; 4] = [1.0, 0.0, 0.0, 0.0];
    const SAW_XS: [f64; 6] = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0];
    const SAW: [f64; 6] = [0.0, 0.8, 0.3999999999999999, 0.0, 0.40000000000000036, 0.0];

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn all_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (k, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(close(*x, *y, tol), "index {}: {} vs {}", k, x, y);
        }
    }

    #[test]
    fn activations_match_definitions() {
        assert_eq!(ACTIVATIONS, ["linear", "relu", "tanh"]);
        assert_eq!(activation("linear", -1.5).unwrap(), -1.5);
        assert_eq!(activation("relu", -1.5).unwrap(), 0.0);
        assert_eq!(activation("relu", 2.0).unwrap(), 2.0);
        assert_eq!(activation("relu", 0.0).unwrap(), 0.0);
        assert!(close(activation("tanh", 0.5).unwrap(), 0.5_f64.tanh(), 1e-15));
        assert_eq!(activation_derivative("linear", 3.0).unwrap(), 1.0);
        assert_eq!(activation_derivative("relu", 0.0).unwrap(), 0.0);
        assert_eq!(activation_derivative("relu", 0.1).unwrap(), 1.0);
        let t = 0.5_f64.tanh();
        assert!(close(activation_derivative("tanh", 0.5).unwrap(), 1.0 - t * t, 1e-15));
        assert!(activation("sigmoid", 0.0).is_err());
        assert!(activation_derivative("gelu", 0.0).is_err());
        assert!(length_map(1.0, 1.0, 0.0, "softplus").is_err());
    }

    #[test]
    fn quadrature_matches_fixture() {
        let ge = |f: fn(f64) -> f64| gaussian_expectation(f, DEFAULT_N1, DEFAULT_LIM).unwrap();
        assert!(close(ge(|z| z * z), GE_Z2, TOL));
        assert!(close(ge(|z| z.powi(4)), GE_Z4, TOL));
        assert!(close(ge(f64::cos), GE_COS, TOL));
        assert!(close(gaussian_expectation(f64::cos, 40, 6.0).unwrap(), GE_COS_N40_LIM6, TOL));
        assert!(close(GE_COS, (-0.5_f64).exp(), 1e-12));
        let sq = gaussian_expectation_2d(|a, b| (a + b) * (a + b), 40, 6.0).unwrap();
        assert!(close(sq, GE2_SQ_N40_LIM6, TOL));
        let cp = gaussian_expectation_2d(|a, b| a.cos() * (2.0 * b).cos(), 20, 5.0).unwrap();
        assert!(close(cp, GE2_COSPROD_N20_LIM5, TOL));
        assert!(gaussian_expectation(f64::cos, 3, DEFAULT_LIM).is_err());
        assert!(gaussian_expectation(f64::cos, 0, DEFAULT_LIM).is_err());
        assert!(gaussian_expectation_2d(|a, b| a * b, 5, DEFAULT_LIM).is_err());
    }

    #[test]
    fn length_map_matches_fixture() {
        assert!(close(length_map(1.0, 2.0, 0.0, "relu").unwrap(), LM_RELU, TOL));
        assert!(close(length_map(1.0, 1.5, 0.05, "tanh").unwrap(), LM_TANH, TOL));
        assert!(close(length_map(0.7, 1.2, 0.1, "linear").unwrap(), LM_LINEAR, TOL));
        assert!(close(length_map(2.0, 2.5, 0.0, "tanh").unwrap(), LM_TANH_Q2, TOL));
        // sigma_w^2 = 2 preserves the ReLU length.
        for q in [0.25, 1.0, 3.0] {
            assert!(close(length_map(q, 2.0, 0.0, "relu").unwrap(), q, 1e-12 * q));
        }
        assert!(length_map(1.0, 0.0, 0.0, "relu").is_err());
        assert!(length_map(1.0, 1.0, -0.1, "relu").is_err());
        assert!(length_map(-1.0, 1.0, 0.0, "relu").is_err());
    }

    #[test]
    fn length_fixed_point_matches_fixture() {
        let fp = |w: f64, b: f64, phi: &str, q0: f64, it: usize| length_fixed_point(w, b, phi, q0, 1e-12, it);
        assert!(close(fp(1.5, 0.05, "tanh", 1.0, 10_000).unwrap(), QSTAR_TANH, TOL));
        assert!(close(fp(1.5, 0.1, "relu", 1.0, 10_000).unwrap(), QSTAR_RELU, TOL));
        assert!(close(fp(4.0, 0.05, "tanh", 0.5, 10_000).unwrap(), QSTAR_TANH_CHAOS, TOL));
        assert!(close(QSTAR_RELU, 0.1 / (1.0 - 0.75), 1e-9));
        assert!(fp(3.0, 0.0, "relu", 1.0, 10_000).unwrap_err().contains("diverged"));
        assert!(fp(1.5, 0.05, "tanh", 1.0, 2).unwrap_err().contains("did not converge"));
    }

    #[test]
    fn correlation_map_matches_fixture() {
        let cm = correlation_map(0.5, 1.0, 2.0, 0.0, "relu", DEFAULT_N2).unwrap();
        assert!(close(cm, CM_RELU, TOL));
        assert!(close(cm, relu_correlation_map(0.5).unwrap(), 1e-6));
        let ct = correlation_map(0.5, QSTAR_TANH, 1.5, 0.05, "tanh", DEFAULT_N2).unwrap();
        assert!(close(ct, CM_TANH, TOL));
        let c40 = correlation_map(0.3, QSTAR_TANH, 1.5, 0.05, "tanh", 40).unwrap();
        assert!(close(c40, CM_TANH_N40, TOL));
        let cl = correlation_map(-0.4, 0.8, 1.3, 0.2, "linear", 40).unwrap();
        assert!(close(cl, CM_LINEAR_N40, TOL));
        assert!(correlation_map(1.5, 1.0, 2.0, 0.0, "relu", 4).is_err());
        assert!(correlation_map(0.5, 1.0, -2.0, 0.0, "relu", 4).is_err());
        assert!(correlation_map(0.5, 0.0, 2.0, 0.0, "relu", 4).is_err());
    }

    #[test]
    fn chi1_and_depth_scale_match_fixture() {
        assert!(close(chi1(QSTAR_TANH, 1.5, "tanh").unwrap(), CHI1_TANH, TOL));
        assert!(close(chi1(1.0, 2.0, "relu").unwrap(), CHI1_RELU, TOL));
        assert!(close(chi1(1.0, 0.9, "linear").unwrap(), CHI1_LINEAR, TOL));
        assert!(chi1(1.0, 0.0, "tanh").is_err());
        assert!(chi1(-1.0, 1.0, "tanh").is_err());
        assert!(close(correlation_depth_scale(0.9).unwrap(), XI_09, TOL));
        assert_eq!(correlation_depth_scale(1.0).unwrap(), f64::INFINITY);
        assert!(correlation_depth_scale(0.0).is_err());
    }

    #[test]
    fn relu_kernel_and_trajectories_match_fixture() {
        let got: Vec<f64> = RCM_INPUTS.iter().map(|&c| relu_correlation_map(c).unwrap()).collect();
        all_close(&got, &RCM, TOL);
        assert!(relu_correlation_map(1.01).is_err());
        assert!(relu_correlation_map(-1.01).is_err());
        all_close(&relu_correlation_trajectory(0.2, 5, None).unwrap(), &TRAJ_PLAIN, TOL);
        all_close(&relu_correlation_trajectory(0.2, 5, Some(0.5)).unwrap(), &TRAJ_RES, TOL);
        all_close(&relu_correlation_trajectory(-0.9, 3, None).unwrap(), &TRAJ_NEG, TOL);
        assert_eq!(relu_correlation_trajectory(0.3, 0, None).unwrap(), vec![0.3]);
        assert!(relu_correlation_trajectory(0.2, 3, Some(-0.1)).is_err());
        assert!(close(relu_collapse_asymptotic(10.0, None).unwrap(), COLLAPSE_PLAIN, TOL));
        assert!(close(relu_collapse_asymptotic(10.0, Some(0.5)).unwrap(), COLLAPSE_RES, TOL));
        assert!(relu_collapse_asymptotic(0.0, None).is_err());
        assert!(relu_collapse_asymptotic(10.0, Some(0.0)).is_err());
    }

    #[test]
    fn kernel_ode_matches_fixture_and_residual_limit() {
        let target = relu_kernel_ode(0.2, 1.0, 10_000).unwrap();
        assert!(close(target, ODE, TOL));
        assert!(close(relu_kernel_ode(-0.3, 2.0, 50).unwrap(), ODE_COARSE, TOL));
        assert!(relu_kernel_ode(0.2, 1.0, 0).is_err());
        assert!(relu_kernel_ode(0.2, -1.0, 10_000).is_err());
        let errs: Vec<f64> = [10usize, 100, 1000]
            .iter()
            .map(|&l| {
                let traj = relu_correlation_trajectory(0.2, l, Some(1.0 / (l as f64).sqrt())).unwrap();
                (traj[l] - target).abs()
            })
            .collect();
        assert!(errs[0] > errs[1] && errs[1] > errs[2]);
        assert!(errs[2] < 2e-4);
    }

    #[test]
    fn residual_variance_matches_fixture() {
        assert!(close(residual_variance(1.0, 10, 1.0).unwrap(), RV1, TOL));
        assert!(close(residual_variance(0.5, 100, 0.1).unwrap(), RV2, TOL));
        assert!(residual_variance(-1.0, 10, 1.0).is_err());
        assert!(residual_variance(1.0, 10, -1.0).is_err());
    }

    #[test]
    fn deep_linear_learning_time_matches_fixture() {
        for (&n, &e) in DLT_DEPTHS.iter().zip(DLT.iter()) {
            assert!(close(deep_linear_learning_time(n, 0.5, 1.0, 0.5, 4000).unwrap(), e, TOL));
        }
        assert!(close(deep_linear_learning_time(4, 0.3, 2.0, 0.9, 200).unwrap(), DLT_OPTS, TOL));
        assert_eq!(deep_linear_learning_time(1, 0.9, 1.0, 0.5, 4000).unwrap(), 0.0);
        for (&n, &e) in DLA_DEPTHS.iter().zip(DLA.iter()) {
            assert!(close(deep_linear_time_asymptotic(n, 0.5, 1.0).unwrap(), e, TOL));
        }
        assert!(close(deep_linear_time_asymptotic(4, 0.3, 2.0).unwrap(), DLA_S, TOL));
        let gaps: Vec<f64> = [5usize, 10, 20]
            .iter()
            .map(|&n| {
                let t = deep_linear_learning_time(n, 0.5, 1.0, 0.5, 4000).unwrap();
                (t / deep_linear_time_asymptotic(n, 0.5, 1.0).unwrap() - 1.0).abs()
            })
            .collect();
        assert!(gaps[0] > gaps[1] && gaps[1] > gaps[2]);
        assert!(gaps[2] < 1e-4);
        assert!(deep_linear_learning_time(0, 0.5, 1.0, 0.5, 4000).is_err());
        assert!(deep_linear_learning_time(3, 0.0, 1.0, 0.5, 4000).is_err());
        assert!(deep_linear_learning_time(3, 0.5, 0.0, 0.5, 4000).is_err());
        assert!(deep_linear_learning_time(3, 0.5, 1.0, 1.0, 4000).is_err());
        assert!(deep_linear_learning_time(3, 0.5, 1.0, 0.5, 7).is_err());
        assert!(deep_linear_time_asymptotic(2, 0.5, 1.0).is_err());
        assert!(deep_linear_time_asymptotic(3, -0.5, 1.0).is_err());
    }

    #[test]
    fn path_distributions_match_fixture() {
        all_close(&path_length_distribution(4), &PLD4, TOL);
        assert_eq!(path_length_distribution(0), vec![1.0]);
        all_close(&effective_path_distribution(4, 0.5).unwrap(), &EPD, TOL);
        all_close(&effective_path_distribution(3, 0.0).unwrap(), &EPD_ZERO, TOL);
        for n in [1usize, 7, 54] {
            let s: f64 = path_length_distribution(n).iter().sum();
            assert!(close(s, 1.0, 1e-12));
            let s: f64 = effective_path_distribution(n, 0.2).unwrap().iter().sum();
            assert!(close(s, 1.0, 1e-12));
        }
        assert!(effective_path_distribution(4, -0.5).is_err());
    }

    #[test]
    fn sawtooth_matches_fixture_and_piece_count() {
        let got: Vec<f64> = SAW_XS.iter().map(|&x| sawtooth(x, 3)).collect();
        all_close(&got, &SAW, TOL);
        assert_eq!(sawtooth(0.37, 0), 0.37);
        for depth in 1..=6usize {
            let grid = 4 * (1usize << depth);
            let ys: Vec<f64> = (0..=grid).map(|i| sawtooth(i as f64 / grid as f64, depth)).collect();
            assert_eq!(count_linear_pieces(&ys, 1e-9).unwrap(), 1usize << depth);
        }
        let xs: Vec<f64> = (0..=64).map(|i| i as f64 / 64.0).collect();
        let line: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
        assert_eq!(count_linear_pieces(&line, 1e-9).unwrap(), 1);
        let vee: Vec<f64> = xs.iter().map(|x| (x - 0.33).abs()).collect();
        assert_eq!(count_linear_pieces(&vee, 1e-9).unwrap(), 2);
        assert!(count_linear_pieces(&[1.0, 2.0], 1e-9).is_err());
    }
}
