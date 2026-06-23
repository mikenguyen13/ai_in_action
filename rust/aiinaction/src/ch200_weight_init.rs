//! Xavier (Glorot) and He (Kaiming) weight initialization from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch200_weight_init` and the Julia module
//! `AIInAction.Ch200WeightInit`. The shared fixtures in the tests below match the
//! Python/Julia suites, which keeps the three implementations at parity.
//!
//! A layer's weight variance must scale inversely with its fan to keep the
//! variance of activations stable forward and the variance of gradients stable
//! backward. Xavier/Glorot uses `Var(W) = gain^2 * 2 / (fan_in + fan_out)`;
//! He/Kaiming uses `Var(W) = gain^2 / fan` with `gain = sqrt(2)` for ReLU.
//!
//! Sampling uses a self-contained, deterministic SplitMix64 PRNG plus the
//! Box-Muller transform so that, for a fixed seed, the Python, Julia, and Rust
//! implementations emit the identical weight matrix (to floating-point
//! tolerance), not merely matching summary statistics. This is std-only.

/// The theoretical spread of an initialization scheme.
#[derive(Clone, Copy, Debug)]
pub struct InitScale {
    /// Standard deviation of the matching zero-mean Gaussian, `sqrt(Var(W))`.
    pub std: f64,
    /// Half-width `r` of the matching uniform support `U(-r, r)`; `r = std*sqrt(3)`.
    pub bound: f64,
}

/// Recommended gain `g` for a nonlinearity, matching PyTorch conventions.
///
/// Supported: `linear`, `sigmoid`, `tanh`, `relu`, `leaky_relu` (uses `param` as
/// the negative slope, default 0.01) and `selu`.
pub fn calculate_gain(nonlinearity: &str, param: Option<f64>) -> Result<f64, String> {
    match nonlinearity.to_lowercase().as_str() {
        "linear" | "conv1d" | "conv2d" | "conv3d" | "sigmoid" => Ok(1.0),
        "tanh" => Ok(5.0 / 3.0),
        "relu" => Ok(2.0_f64.sqrt()),
        "leaky_relu" => {
            let slope = param.unwrap_or(0.01);
            if slope <= -1.0 {
                return Err(format!("leaky_relu negative slope must be > -1, got {}", slope));
            }
            Ok((2.0 / (1.0 + slope * slope)).sqrt())
        }
        "selu" => Ok(3.0 / 4.0),
        other => Err(format!("unsupported nonlinearity: {:?}", other)),
    }
}

fn check_fan(fan_in: usize, fan_out: usize) -> Result<(), String> {
    if fan_in < 1 {
        return Err(format!("fan_in must be a positive integer, got {}", fan_in));
    }
    if fan_out < 1 {
        return Err(format!("fan_out must be a positive integer, got {}", fan_out));
    }
    Ok(())
}

/// Xavier/Glorot scale: `std = gain * sqrt(2 / (fan_in + fan_out))`.
pub fn xavier_scale(fan_in: usize, fan_out: usize, gain: f64) -> Result<InitScale, String> {
    check_fan(fan_in, fan_out)?;
    if gain <= 0.0 {
        return Err(format!("gain must be positive, got {}", gain));
    }
    let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
    Ok(InitScale {
        std,
        bound: std * 3.0_f64.sqrt(),
    })
}

/// He/Kaiming scale: `std = gain / sqrt(fan)`. `fan` is the chosen mode.
pub fn he_scale(fan: usize, gain: f64) -> Result<InitScale, String> {
    if fan < 1 {
        return Err(format!("fan must be a positive integer, got {}", fan));
    }
    if gain <= 0.0 {
        return Err(format!("gain must be positive, got {}", gain));
    }
    let std = gain / (fan as f64).sqrt();
    Ok(InitScale {
        std,
        bound: std * 3.0_f64.sqrt(),
    })
}

/// Deterministic 64-bit SplitMix64 generator producing doubles in [0, 1).
/// Reproduced identically in Python and Julia so seeded matrices match.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_double(&mut self) -> f64 {
        // Top 53 bits give a uniform double in [0, 1).
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

/// One standard normal via the basic Box-Muller transform.
fn next_normal(rng: &mut SplitMix64) -> f64 {
    let mut u1 = rng.next_double();
    let u2 = rng.next_double();
    if u1 <= 0.0 {
        u1 = 5e-324; // smallest positive subnormal double
    }
    let r = (-2.0 * u1.ln()).sqrt();
    r * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Weight matrices are returned `(fan_out, fan_in)` (rows = output units).
fn fill_normal(fan_in: usize, fan_out: usize, std: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    let rows = fan_out;
    let cols = fan_in;
    let mut out = vec![vec![0.0f64; cols]; rows];
    for row in out.iter_mut() {
        for v in row.iter_mut() {
            *v = next_normal(&mut rng) * std;
        }
    }
    out
}

fn fill_uniform(fan_in: usize, fan_out: usize, bound: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    let rows = fan_out;
    let cols = fan_in;
    let mut out = vec![vec![0.0f64; cols]; rows];
    for row in out.iter_mut() {
        for v in row.iter_mut() {
            *v = (rng.next_double() * 2.0 - 1.0) * bound;
        }
    }
    out
}

/// Sample a `(fan_out, fan_in)` weight matrix from Xavier normal.
pub fn xavier_normal(
    fan_in: usize,
    fan_out: usize,
    gain: f64,
    seed: u64,
) -> Result<Vec<Vec<f64>>, String> {
    let s = xavier_scale(fan_in, fan_out, gain)?;
    Ok(fill_normal(fan_in, fan_out, s.std, seed))
}

/// Sample a `(fan_out, fan_in)` weight matrix from Xavier uniform `U(-r, r)`.
pub fn xavier_uniform(
    fan_in: usize,
    fan_out: usize,
    gain: f64,
    seed: u64,
) -> Result<Vec<Vec<f64>>, String> {
    let s = xavier_scale(fan_in, fan_out, gain)?;
    Ok(fill_uniform(fan_in, fan_out, s.bound, seed))
}

/// He initialization mode: which fan drives the variance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FanMode {
    FanIn,
    FanOut,
}

/// Sample a `(fan_out, fan_in)` weight matrix from He normal.
pub fn he_normal(
    fan_in: usize,
    fan_out: usize,
    gain: f64,
    mode: FanMode,
    seed: u64,
) -> Result<Vec<Vec<f64>>, String> {
    check_fan(fan_in, fan_out)?;
    if gain <= 0.0 {
        return Err(format!("gain must be positive, got {}", gain));
    }
    let fan = match mode {
        FanMode::FanIn => fan_in,
        FanMode::FanOut => fan_out,
    };
    let s = he_scale(fan, gain)?;
    Ok(fill_normal(fan_in, fan_out, s.std, seed))
}

/// Sample a `(fan_out, fan_in)` weight matrix from He uniform `U(-r, r)`.
pub fn he_uniform(
    fan_in: usize,
    fan_out: usize,
    gain: f64,
    mode: FanMode,
    seed: u64,
) -> Result<Vec<Vec<f64>>, String> {
    check_fan(fan_in, fan_out)?;
    if gain <= 0.0 {
        return Err(format!("gain must be positive, got {}", gain));
    }
    let fan = match mode {
        FanMode::FanIn => fan_in,
        FanMode::FanOut => fan_out,
    };
    let s = he_scale(fan, gain)?;
    Ok(fill_uniform(fan_in, fan_out, s.bound, seed))
}

/// The default ReLU gain `sqrt(2)`, convenient for callers of the He functions.
pub fn relu_gain() -> f64 {
    2.0_f64.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // Shared fixtures: identical to the Python and Julia test suites.
    const GAIN_RELU: f64 = 1.4142135623730951;
    const GAIN_TANH: f64 = 1.6666666666666667;
    const GAIN_LEAKY_02: f64 = 1.3867504905630728;

    const XAVIER_46_STD: f64 = 0.4472135954999579;
    const XAVIER_46_BOUND: f64 = 0.7745966692414833;
    const HE_8_STD: f64 = 0.5;
    const HE_8_BOUND: f64 = 0.8660254037844386;

    // xavier_normal(fan_in=3, fan_out=2, seed=42) -> shape (2, 3)
    const XAVIER_NORMAL_3_2_SEED42: [[f64; 3]; 2] = [
        [0.262291800404047, -0.5640783697534434, 1.0938907166332181],
        [0.34508066325448, -0.683313150259559, -1.1250423158375467],
    ];

    // he_uniform(fan_in=3, fan_out=2, seed=7) -> shape (2, 3)
    const HE_UNIFORM_3_2_SEED7: [[f64; 3]; 2] = [
        [-0.3116085279902403, -1.3667290947514303, 1.1335223795602536],
        [0.23456229026376593, -0.13451463415109, -0.7087146789818501],
    ];

    // he_normal(fan_in=4, fan_out=3, seed=123, FanIn) -> shape (3, 4)
    const HE_NORMAL_4_3_SEED123: [[f64; 4]; 3] = [
        [
            0.5830829313806632,
            -0.1503944856454256,
            -0.30548437167556053,
            -0.00772328473476067,
        ],
        [
            0.43836777367788293,
            0.45770047377992695,
            0.6982243250890897,
            -0.151260320705838,
        ],
        [
            -0.16215580034083707,
            -1.1099931389726774,
            0.23040596666525975,
            -0.4700322147558535,
        ],
    ];

    #[test]
    fn gain_relu_matches() {
        assert!((calculate_gain("relu", None).unwrap() - GAIN_RELU).abs() < TOL);
    }

    #[test]
    fn gain_tanh_matches() {
        assert!((calculate_gain("tanh", None).unwrap() - GAIN_TANH).abs() < TOL);
    }

    #[test]
    fn gain_linear_and_sigmoid() {
        assert_eq!(calculate_gain("linear", None).unwrap(), 1.0);
        assert_eq!(calculate_gain("sigmoid", None).unwrap(), 1.0);
    }

    #[test]
    fn gain_leaky_param_matches() {
        assert!((calculate_gain("leaky_relu", Some(0.2)).unwrap() - GAIN_LEAKY_02).abs() < TOL);
    }

    #[test]
    fn gain_leaky_alpha_one_is_linear() {
        assert!((calculate_gain("leaky_relu", Some(1.0)).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn gain_unsupported_errors() {
        assert!(calculate_gain("gelu_bananas", None).is_err());
    }

    #[test]
    fn gain_leaky_bad_slope_errors() {
        assert!(calculate_gain("leaky_relu", Some(-1.0)).is_err());
    }

    #[test]
    fn xavier_scale_matches_fixture() {
        let s = xavier_scale(4, 6, 1.0).unwrap();
        assert!((s.std - XAVIER_46_STD).abs() < TOL);
        assert!((s.bound - XAVIER_46_BOUND).abs() < TOL);
    }

    #[test]
    fn xavier_bound_is_std_times_sqrt3() {
        let s = xavier_scale(7, 11, 1.3).unwrap();
        assert!((s.bound - s.std * 3.0_f64.sqrt()).abs() < TOL);
    }

    #[test]
    fn he_scale_matches_fixture() {
        let s = he_scale(8, relu_gain()).unwrap();
        assert!((s.std - HE_8_STD).abs() < TOL);
        assert!((s.bound - HE_8_BOUND).abs() < TOL);
    }

    #[test]
    fn he_scale_relu_gain_gives_unit_std() {
        let s = he_scale(2, relu_gain()).unwrap();
        assert!((s.std - 1.0).abs() < TOL);
    }

    #[test]
    fn xavier_normal_seed_fixture() {
        let w = xavier_normal(3, 2, 1.0, 42).unwrap();
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].len(), 3);
        for i in 0..2 {
            for j in 0..3 {
                assert!((w[i][j] - XAVIER_NORMAL_3_2_SEED42[i][j]).abs() < TOL);
            }
        }
    }

    #[test]
    fn he_uniform_seed_fixture() {
        let w = he_uniform(3, 2, relu_gain(), FanMode::FanIn, 7).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                assert!((w[i][j] - HE_UNIFORM_3_2_SEED7[i][j]).abs() < TOL);
            }
        }
    }

    #[test]
    fn he_normal_seed_fixture() {
        let w = he_normal(4, 3, relu_gain(), FanMode::FanIn, 123).unwrap();
        assert_eq!(w.len(), 3);
        assert_eq!(w[0].len(), 4);
        for i in 0..3 {
            for j in 0..4 {
                assert!((w[i][j] - HE_NORMAL_4_3_SEED123[i][j]).abs() < TOL);
            }
        }
    }

    #[test]
    fn same_seed_is_reproducible() {
        let a = he_normal(5, 5, relu_gain(), FanMode::FanIn, 99).unwrap();
        let b = he_normal(5, 5, relu_gain(), FanMode::FanIn, 99).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn he_uniform_respects_bound() {
        let bound = he_scale(16, relu_gain()).unwrap().bound;
        let w = he_uniform(16, 16, relu_gain(), FanMode::FanIn, 5).unwrap();
        for row in &w {
            for &v in row {
                assert!(v.abs() <= bound + TOL);
            }
        }
    }

    #[test]
    fn bad_fan_errors() {
        assert!(xavier_normal(0, 4, 1.0, 0).is_err());
        assert!(he_normal(4, 0, relu_gain(), FanMode::FanIn, 0).is_err());
    }

    #[test]
    fn nonpositive_gain_errors() {
        assert!(xavier_normal(4, 4, 0.0, 0).is_err());
        assert!(he_normal(4, 4, 0.0, FanMode::FanIn, 0).is_err());
    }

    #[test]
    fn he_scale_bad_fan_errors() {
        assert!(he_scale(0, relu_gain()).is_err());
    }
}
