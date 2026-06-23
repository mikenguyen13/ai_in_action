//! Inverted dropout with Bernoulli masking, from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch204_dropout` and the Julia module
//! `AIInAction.Ch204Dropout`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! Randomness comes from a tiny 64-bit linear congruential generator using the
//! Numerical Recipes constants. Fixing this generator across all three languages
//! makes the dropout masks reproducible: given the same seed and length, the same
//! units are dropped everywhere. Each draw is formed from the top 53 bits of the
//! 64-bit state, so every uniform is an exact multiple of `2^-53`.
//!
//! Convention: `p` is the *retention* probability (`1 - p` is the drop
//! probability). A retained unit is scaled by `1/p`; a dropped unit becomes `0.0`.
//! This is inverted dropout, for which `E[mask] = 1` and `E[mask * h] = h`.

/// Numerical Recipes LCG multiplier.
const LCG_A: u64 = 6364136223846793005;
/// Numerical Recipes LCG increment.
const LCG_C: u64 = 1442695040888963407;
/// `2^53`, the divisor mapping a 53-bit mantissa to a uniform in `[0, 1)`.
const UNIT: f64 = 9007199254740992.0;

/// A minimal, fully reproducible 64-bit linear congruential generator.
///
/// The recurrence is `state <- (a * state + c) mod 2^64` (wrapping) with the
/// Numerical Recipes constants. Mirrors the Python `Lcg` and Julia `Lcg` bit for
/// bit.
#[derive(Clone, Debug)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Creates a generator seeded with `seed`.
    pub fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    /// Advances the generator and returns the next uniform draw in `[0, 1)`.
    pub fn next_uniform(&mut self) -> f64 {
        self.state = LCG_A.wrapping_mul(self.state).wrapping_add(LCG_C);
        // Top 53 bits -> exact multiple of 2^-53 in [0, 1).
        let top = self.state >> 11;
        (top as f64) / UNIT
    }
}

/// Returns `1 / p`, the survivor scaling that keeps the mask mean at 1.
///
/// Errors unless `0 < p <= 1`.
pub fn expected_scale(p: f64) -> Result<f64, String> {
    if !(p > 0.0 && p <= 1.0) {
        return Err(format!(
            "retention probability p must satisfy 0 < p <= 1, got {}",
            p
        ));
    }
    Ok(1.0 / p)
}

/// Builds a length-`n` inverted-dropout mask with retention probability `p`.
///
/// Unit `i` is retained when `u_i < p` (the `i`-th draw of `Lcg::new(seed)`) and
/// assigned `1/p`; otherwise it is dropped and assigned `0.0`.
pub fn bernoulli_mask(n: usize, p: f64, seed: u64) -> Result<Vec<f64>, String> {
    if n < 1 {
        return Err(format!("n must be >= 1, got {}", n));
    }
    let scale = expected_scale(p)?;
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let u = rng.next_uniform();
        out.push(if u < p { scale } else { 0.0 });
    }
    Ok(out)
}

/// Applies inverted dropout to the activation vector `h`.
///
/// Returns `(masked, mask)` where `masked[i] = mask[i] * h[i]`. Dropped units are
/// forced to exactly `0.0`; survivors are scaled by `1/p`.
pub fn inverted_dropout(h: &[f64], p: f64, seed: u64) -> Result<(Vec<f64>, Vec<f64>), String> {
    if h.is_empty() {
        return Err("h must be non-empty".to_string());
    }
    if h.iter().any(|v| !v.is_finite()) {
        return Err("h contains non-finite values (nan or inf)".to_string());
    }
    let mask = bernoulli_mask(h.len(), p, seed)?;
    let masked: Vec<f64> = h.iter().zip(&mask).map(|(hi, mi)| hi * mi).collect();
    Ok((masked, mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // Shared fixtures: identical to the Python and Julia test suites.
    const EXPECTED_UNIFORMS: [f64; 6] = [
        0.5682303266439076,
        0.2254634289477513,
        0.41283831882951183,
        0.6303980498395979,
        0.6801478072421157,
        0.02622891069993838,
    ];
    const EXPECTED_MASK_8: [f64; 8] = [0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0];
    const EXPECTED_H: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    const EXPECTED_OUT_8: [f64; 8] = [0.0, 4.0, 6.0, 0.0, 0.0, 12.0, 14.0, 16.0];

    #[test]
    fn lcg_uniform_sequence_matches_fixture() {
        let mut rng = Lcg::new(42);
        for &expected in EXPECTED_UNIFORMS.iter() {
            assert!((rng.next_uniform() - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn lcg_uniforms_in_unit_interval() {
        let mut rng = Lcg::new(123);
        for _ in 0..1000 {
            let u = rng.next_uniform();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn lcg_is_deterministic_for_same_seed() {
        let mut a = Lcg::new(7);
        let mut b = Lcg::new(7);
        for _ in 0..5 {
            assert_eq!(a.next_uniform(), b.next_uniform());
        }
    }

    #[test]
    fn expected_scale_is_reciprocal() {
        assert!((expected_scale(0.5).unwrap() - 2.0).abs() < TOL);
        assert!((expected_scale(0.8).unwrap() - 1.25).abs() < TOL);
        assert!((expected_scale(1.0).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn expected_scale_rejects_out_of_range() {
        assert!(expected_scale(0.0).is_err());
        assert!(expected_scale(1.5).is_err());
        assert!(expected_scale(-0.2).is_err());
    }

    #[test]
    fn bernoulli_mask_matches_fixture() {
        let m = bernoulli_mask(8, 0.5, 42).unwrap();
        for i in 0..8 {
            assert!((m[i] - EXPECTED_MASK_8[i]).abs() < TOL);
        }
    }

    #[test]
    fn bernoulli_mask_p_one_is_all_ones() {
        let m = bernoulli_mask(5, 1.0, 7).unwrap();
        for &v in m.iter() {
            assert!((v - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn bernoulli_mask_entries_are_zero_or_inverse_p() {
        let p = 0.3;
        let inv = 1.0 / p;
        let m = bernoulli_mask(50, p, 99).unwrap();
        for &v in m.iter() {
            assert!(v.abs() < TOL || (v - inv).abs() < TOL);
        }
    }

    #[test]
    fn bernoulli_mask_mean_approaches_one() {
        let m = bernoulli_mask(20000, 0.5, 2024).unwrap();
        let mean = m.iter().sum::<f64>() / m.len() as f64;
        assert!((mean - 1.0).abs() < 0.05);
    }

    #[test]
    fn inverted_dropout_matches_fixture() {
        let (out, mask) = inverted_dropout(&EXPECTED_H, 0.5, 42).unwrap();
        for i in 0..8 {
            assert!((out[i] - EXPECTED_OUT_8[i]).abs() < TOL);
            assert!((mask[i] - EXPECTED_MASK_8[i]).abs() < TOL);
        }
    }

    #[test]
    fn inverted_dropout_p_one_is_identity() {
        let (out, mask) = inverted_dropout(&EXPECTED_H, 1.0, 42).unwrap();
        for i in 0..8 {
            assert!((out[i] - EXPECTED_H[i]).abs() < TOL);
            assert!((mask[i] - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn inverted_dropout_preserves_expectation_over_seeds() {
        let h: [f64; 4] = [3.0, -1.0, 4.0, 2.0];
        let trials = 6000u64;
        let mut acc = [0.0f64; 4];
        for seed in 0..trials {
            let (out, _) = inverted_dropout(&h, 0.5, seed).unwrap();
            for j in 0..4 {
                acc[j] += out[j];
            }
        }
        for j in 0..4 {
            let avg = acc[j] / trials as f64;
            assert!((avg - h[j]).abs() < 0.15);
        }
    }

    #[test]
    fn dropped_units_are_exactly_zero() {
        let (out, mask) = inverted_dropout(&EXPECTED_H, 0.5, 42).unwrap();
        for i in 0..8 {
            if mask[i] == 0.0 {
                assert_eq!(out[i], 0.0);
            }
        }
    }

    #[test]
    fn bernoulli_mask_rejects_zero_n() {
        assert!(bernoulli_mask(0, 0.5, 1).is_err());
    }

    #[test]
    fn inverted_dropout_rejects_empty() {
        let empty: [f64; 0] = [];
        assert!(inverted_dropout(&empty, 0.5, 1).is_err());
    }

    #[test]
    fn inverted_dropout_rejects_non_finite() {
        let bad: [f64; 2] = [1.0, f64::NAN];
        assert!(inverted_dropout(&bad, 0.5, 1).is_err());
    }

    #[test]
    fn inverted_dropout_rejects_bad_p() {
        let h: [f64; 2] = [1.0, 2.0];
        assert!(inverted_dropout(&h, 0.0, 1).is_err());
    }
}
