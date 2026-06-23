//! Learning rate schedules from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch198_lr_schedules` and the Julia module
//! `AIInAction.Ch198LrSchedules`. The shared fixtures in the tests below match the
//! Python/Julia suites, which keeps the three implementations at parity to 1e-9.
//!
//! All schedules are pure functions of the integer step `t` and a few
//! hyperparameters. Steps are zero-indexed: `t = 0` is the first update and
//! `t = total_steps` is the final point of the horizon. This is a std-only
//! implementation.

use std::f64::consts::PI;

fn check_horizon(total_steps: i64, name: &str) -> Result<(), String> {
    if total_steps < 1 {
        return Err(format!("{} must be >= 1, got {}", name, total_steps));
    }
    Ok(())
}

fn check_step(t: i64) -> Result<(), String> {
    if t < 0 {
        return Err(format!("t must be >= 0, got {}", t));
    }
    Ok(())
}

fn check_bounds(eta_max: f64, eta_min: f64) -> Result<(), String> {
    if eta_max < eta_min {
        return Err(format!(
            "eta_max ({}) must be >= eta_min ({})",
            eta_max, eta_min
        ));
    }
    if eta_min < 0.0 {
        return Err(format!("eta_min must be >= 0, got {}", eta_min));
    }
    Ok(())
}

/// Cosine annealing learning rate at step `t` over a horizon of `total_steps`.
///
/// Half a cosine wave from `eta_max` at `t = 0` down to `eta_min` at
/// `t = total_steps`. Steps beyond the horizon clamp to `eta_min`.
pub fn cosine_annealing(
    t: i64,
    total_steps: i64,
    eta_max: f64,
    eta_min: f64,
) -> Result<f64, String> {
    check_step(t)?;
    check_horizon(total_steps, "total_steps")?;
    check_bounds(eta_max, eta_min)?;
    if t >= total_steps {
        return Ok(eta_min);
    }
    let cos = (PI * t as f64 / total_steps as f64).cos();
    Ok(eta_min + 0.5 * (eta_max - eta_min) * (1.0 + cos))
}

/// Linear warmup from `eta_start` to `eta_target` over `warmup_steps`.
///
/// At `t = 0` returns `eta_start`; at `t >= warmup_steps` returns `eta_target`.
pub fn linear_warmup(
    t: i64,
    warmup_steps: i64,
    eta_target: f64,
    eta_start: f64,
) -> Result<f64, String> {
    check_step(t)?;
    check_horizon(warmup_steps, "warmup_steps")?;
    if eta_target < 0.0 || eta_start < 0.0 {
        return Err("learning rates must be non-negative".to_string());
    }
    if t >= warmup_steps {
        return Ok(eta_target);
    }
    let frac = t as f64 / warmup_steps as f64;
    Ok(eta_start + (eta_target - eta_start) * frac)
}

/// Linear warmup followed by cosine annealing: the standard modern recipe.
///
/// For `t < warmup_steps` the rate ramps linearly from `eta_start` to `eta_max`.
/// For `warmup_steps <= t <= total_steps` it follows a cosine descent from
/// `eta_max` to `eta_min`. Requires `0 <= warmup_steps < total_steps`.
pub fn warmup_cosine(
    t: i64,
    warmup_steps: i64,
    total_steps: i64,
    eta_max: f64,
    eta_min: f64,
    eta_start: f64,
) -> Result<f64, String> {
    check_step(t)?;
    check_horizon(total_steps, "total_steps")?;
    check_bounds(eta_max, eta_min)?;
    if eta_start < 0.0 {
        return Err(format!("eta_start must be >= 0, got {}", eta_start));
    }
    if warmup_steps < 0 || warmup_steps >= total_steps {
        return Err(format!(
            "warmup_steps must satisfy 0 <= warmup_steps < total_steps={}, got {}",
            total_steps, warmup_steps
        ));
    }
    if t < warmup_steps {
        let frac = t as f64 / warmup_steps as f64;
        return Ok(eta_start + (eta_max - eta_start) * frac);
    }
    if t >= total_steps {
        return Ok(eta_min);
    }
    let progress = (t - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
    let cos = (PI * progress).cos();
    Ok(eta_min + 0.5 * (eta_max - eta_min) * (1.0 + cos))
}

/// One-cycle learning rate (Smith and Topin) with cosine ramps.
///
/// Rises from `eta_min` to `eta_max` over the first `pct_start` fraction of the
/// horizon, then descends back to `eta_min`, both legs half-cosines. Pass
/// `eta_min = None` to default to `eta_max / 25`. `pct_start` must be in `(0, 1)`.
pub fn one_cycle(
    t: i64,
    total_steps: i64,
    eta_max: f64,
    eta_min: Option<f64>,
    pct_start: f64,
) -> Result<f64, String> {
    check_step(t)?;
    check_horizon(total_steps, "total_steps")?;
    let eta_min = eta_min.unwrap_or(eta_max / 25.0);
    check_bounds(eta_max, eta_min)?;
    if !(pct_start > 0.0 && pct_start < 1.0) {
        return Err(format!("pct_start must be in (0, 1), got {}", pct_start));
    }
    let t1 = pct_start * total_steps as f64;
    if t >= total_steps {
        return Ok(eta_min);
    }
    if (t as f64) <= t1 {
        let cos = (PI * t as f64 / t1).cos();
        return Ok(eta_min + 0.5 * (eta_max - eta_min) * (1.0 - cos));
    }
    let progress = (t as f64 - t1) / (total_steps as f64 - t1);
    let cos = (PI * progress).cos();
    Ok(eta_min + 0.5 * (eta_max - eta_min) * (1.0 + cos))
}

/// Antiphase momentum schedule for the one-cycle policy.
///
/// Momentum falls from `mom_max` to `mom_min` while the learning rate climbs, then
/// rises back. Both `mom_max` and `mom_min` must lie in `[0, 1)` with
/// `mom_max >= mom_min`.
pub fn one_cycle_momentum(
    t: i64,
    total_steps: i64,
    mom_max: f64,
    mom_min: f64,
    pct_start: f64,
) -> Result<f64, String> {
    check_step(t)?;
    check_horizon(total_steps, "total_steps")?;
    if !(0.0 <= mom_min && mom_min <= mom_max && mom_max < 1.0) {
        return Err(format!(
            "momenta must satisfy 0 <= mom_min <= mom_max < 1, got mom_min={}, mom_max={}",
            mom_min, mom_max
        ));
    }
    if !(pct_start > 0.0 && pct_start < 1.0) {
        return Err(format!("pct_start must be in (0, 1), got {}", pct_start));
    }
    let t1 = pct_start * total_steps as f64;
    if t >= total_steps {
        return Ok(mom_max);
    }
    if (t as f64) <= t1 {
        let cos = (PI * t as f64 / t1).cos();
        return Ok(mom_max - 0.5 * (mom_max - mom_min) * (1.0 - cos));
    }
    let progress = (t as f64 - t1) / (total_steps as f64 - t1);
    let cos = (PI * progress).cos();
    Ok(mom_max - 0.5 * (mom_max - mom_min) * (1.0 + cos))
}

/// Materialize a schedule over all steps `0, 1, ..., total_steps - 1`.
///
/// `name` is one of `"cosine"`, `"warmup_cosine"`, or `"one_cycle"`. The
/// remaining parameters are passed through positionally; unused ones are ignored
/// by the chosen schedule.
pub fn schedule_curve(
    name: &str,
    total_steps: i64,
    warmup_steps: i64,
    eta_max: f64,
    eta_min: f64,
    pct_start: f64,
) -> Result<Vec<f64>, String> {
    check_horizon(total_steps, "total_steps")?;
    let mut out = Vec::with_capacity(total_steps as usize);
    for t in 0..total_steps {
        let v = match name {
            "cosine" => cosine_annealing(t, total_steps, eta_max, eta_min)?,
            "warmup_cosine" => {
                warmup_cosine(t, warmup_steps, total_steps, eta_max, eta_min, 0.0)?
            }
            "one_cycle" => one_cycle(t, total_steps, eta_max, Some(eta_min), pct_start)?,
            _ => {
                return Err(format!(
                    "unknown schedule name {:?}; expected 'cosine', 'warmup_cosine', or 'one_cycle'",
                    name
                ))
            }
        };
        out.push(v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // --- Shared fixtures: identical to the Python and Julia test suites. ---

    // cosine_annealing(t, 8, 0.2, 0.01), t = 0..=8
    fn cosine_expected() -> [f64; 9] {
        [
            0.2,
            0.19276855558857230,
            0.17217514421272200,
            0.14135492607468360,
            0.105,
            0.06864507392531650,
            0.03782485578727799,
            0.01723144441142780,
            0.01,
        ]
    }

    // warmup_cosine(t, 3, 10, 0.5, 0.0, 0.0), t = 0..=10
    fn wc_expected() -> [f64; 11] {
        [
            0.0,
            0.16666666666666666,
            0.33333333333333330,
            0.5,
            0.47524221697560480,
            0.40587245046468340,
            0.30563023348907860,
            0.19436976651092142,
            0.09412754953531660,
            0.02475778302439520,
            0.0,
        ]
    }

    // one_cycle(t, 10, 1.0, Some(0.04), 0.3), t = 0..=10
    fn oc_expected() -> [f64; 11] {
        [
            0.04,
            0.27999999999999990,
            0.75999999999999990,
            1.0,
            0.95246505659316120,
            0.81927510489219210,
            0.62681004829903090,
            0.41318995170096910,
            0.22072489510780793,
            0.08753494340683890,
            0.04,
        ]
    }

    // one_cycle_momentum(t, 10, 0.95, 0.85, 0.3), t = 0..=10
    fn mom_expected() -> [f64; 11] {
        [
            0.95,
            0.92499999999999990,
            0.875,
            0.85,
            0.85495155660487900,
            0.86882550990706330,
            0.88887395330218420,
            0.91112604669781570,
            0.93117449009293660,
            0.94504844339512100,
            0.95,
        ]
    }

    #[test]
    fn cosine_matches_fixture() {
        let exp = cosine_expected();
        for t in 0..=8i64 {
            let v = cosine_annealing(t, 8, 0.2, 0.01).unwrap();
            assert!((v - exp[t as usize]).abs() < TOL);
        }
    }

    #[test]
    fn warmup_cosine_matches_fixture() {
        let exp = wc_expected();
        for t in 0..=10i64 {
            let v = warmup_cosine(t, 3, 10, 0.5, 0.0, 0.0).unwrap();
            assert!((v - exp[t as usize]).abs() < TOL);
        }
    }

    #[test]
    fn one_cycle_matches_fixture() {
        let exp = oc_expected();
        for t in 0..=10i64 {
            let v = one_cycle(t, 10, 1.0, Some(0.04), 0.3).unwrap();
            assert!((v - exp[t as usize]).abs() < TOL);
        }
    }

    #[test]
    fn one_cycle_momentum_matches_fixture() {
        let exp = mom_expected();
        for t in 0..=10i64 {
            let v = one_cycle_momentum(t, 10, 0.95, 0.85, 0.3).unwrap();
            assert!((v - exp[t as usize]).abs() < TOL);
        }
    }

    #[test]
    fn cosine_endpoints_and_midpoint() {
        assert!((cosine_annealing(0, 10, 0.1, 0.0).unwrap() - 0.1).abs() < TOL);
        assert!((cosine_annealing(5, 10, 0.1, 0.0).unwrap() - 0.05).abs() < TOL);
        assert!((cosine_annealing(10, 10, 0.1, 0.0).unwrap() - 0.0).abs() < TOL);
    }

    #[test]
    fn cosine_clamps_past_horizon() {
        assert!((cosine_annealing(50, 10, 0.1, 0.01).unwrap() - 0.01).abs() < TOL);
    }

    #[test]
    fn cosine_monotone_decreasing() {
        let mut prev = f64::INFINITY;
        for t in 0..=20i64 {
            let v = cosine_annealing(t, 20, 0.3, 0.0).unwrap();
            assert!(v <= prev + TOL);
            prev = v;
        }
    }

    #[test]
    fn linear_warmup_endpoints() {
        assert!((linear_warmup(0, 4, 0.1, 0.0).unwrap() - 0.0).abs() < TOL);
        assert!((linear_warmup(2, 4, 0.1, 0.0).unwrap() - 0.05).abs() < TOL);
        assert!((linear_warmup(4, 4, 0.1, 0.0).unwrap() - 0.1).abs() < TOL);
        assert!((linear_warmup(7, 4, 0.1, 0.0).unwrap() - 0.1).abs() < TOL);
    }

    #[test]
    fn warmup_cosine_continuous_at_handoff() {
        let left = linear_warmup(3, 3, 0.5, 0.0).unwrap();
        let right = warmup_cosine(3, 3, 10, 0.5, 0.0, 0.0).unwrap();
        assert!((left - right).abs() < TOL);
        assert!((right - 0.5).abs() < TOL);
    }

    #[test]
    fn one_cycle_default_eta_min() {
        assert!((one_cycle(0, 10, 1.0, None, 0.3).unwrap() - 0.04).abs() < TOL);
        assert!((one_cycle(0, 10, 0.5, None, 0.3).unwrap() - 0.02).abs() < TOL);
    }

    #[test]
    fn one_cycle_peak_at_pct_start() {
        assert!((one_cycle(3, 10, 1.0, Some(0.04), 0.3).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn momentum_antiphase() {
        assert!((one_cycle_momentum(3, 10, 0.95, 0.85, 0.3).unwrap() - 0.85).abs() < TOL);
        assert!((one_cycle_momentum(0, 10, 0.95, 0.85, 0.3).unwrap() - 0.95).abs() < TOL);
        assert!((one_cycle_momentum(10, 10, 0.95, 0.85, 0.3).unwrap() - 0.95).abs() < TOL);
    }

    #[test]
    fn schedule_curve_cosine_length_and_values() {
        let curve = schedule_curve("cosine", 8, 0, 0.2, 0.01, 0.3).unwrap();
        assert_eq!(curve.len(), 8);
        let exp = cosine_expected();
        for t in 0..8usize {
            assert!((curve[t] - exp[t]).abs() < TOL);
        }
    }

    #[test]
    fn negative_step_errors() {
        assert!(cosine_annealing(-1, 10, 0.1, 0.0).is_err());
    }

    #[test]
    fn zero_horizon_errors() {
        assert!(cosine_annealing(0, 0, 0.1, 0.0).is_err());
    }

    #[test]
    fn eta_max_below_min_errors() {
        assert!(cosine_annealing(0, 10, 0.01, 0.1).is_err());
    }

    #[test]
    fn warmup_not_less_than_total_errors() {
        assert!(warmup_cosine(0, 10, 10, 0.1, 0.0, 0.0).is_err());
    }

    #[test]
    fn bad_pct_start_errors() {
        assert!(one_cycle(0, 10, 1.0, Some(0.04), 1.0).is_err());
    }

    #[test]
    fn bad_momentum_bounds_error() {
        assert!(one_cycle_momentum(0, 10, 0.85, 0.95, 0.3).is_err());
        assert!(one_cycle_momentum(0, 10, 1.0, 0.85, 0.3).is_err());
    }

    #[test]
    fn unknown_schedule_name_errors() {
        assert!(schedule_curve("triangular", 10, 0, 0.1, 0.0, 0.3).is_err());
    }
}
