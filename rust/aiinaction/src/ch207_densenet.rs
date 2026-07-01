//! Dense connectivity (DenseNet) mechanics, from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch207_densenet` and the Julia module
//! `AIInAction.Ch207Densenet`. The shared fixtures in the tests below match
//! the Python/Julia suites, which is what keeps the three implementations at
//! parity.
//!
//! Channel and parameter arithmetic (`dense_block_channel_sizes`,
//! `dense_block_param_count`, `transition_output_channels`,
//! `plain_block_param_count`, `densenet_dense_param_total`) is exact integer
//! arithmetic and needs no randomness. The toy `dense_block_forward` stands
//! in for a real dense block's batch-norm/ReLU/convolution composite `H_l`
//! with a single seeded linear-plus-ReLU layer, using the same 64-bit linear
//! congruential generator (LCG) used elsewhere in this book so that weights
//! are reproducible bit for bit across languages given the same seed.

/// Numerical Recipes LCG multiplier.
const LCG_A: u64 = 6364136223846793005;
/// Numerical Recipes LCG increment.
const LCG_C: u64 = 1442695040888963407;
/// `2^53`, the divisor mapping a 53-bit mantissa to a uniform in `[0, 1)`.
const UNIT: f64 = 9007199254740992.0;

/// A minimal, fully reproducible 64-bit linear congruential generator.
///
/// Mirrors the Python `Lcg` and Julia `Lcg` bit for bit.
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
        let top = self.state >> 11;
        (top as f64) / UNIT
    }
}

// --------------------------------------------------------------------------
// Channel and parameter arithmetic
// --------------------------------------------------------------------------

/// Returns the input-channel count seen by each layer of a dense block, plus
/// the block's total output width as the final entry. Layer `l` (0-indexed)
/// sees `c0 + growth_rate * l` channels.
pub fn dense_block_channel_sizes(c0: i64, growth_rate: i64, num_layers: i64) -> Result<Vec<i64>, String> {
    if c0 < 0 {
        return Err(format!("c0 must be >= 0, got {}", c0));
    }
    if growth_rate < 1 {
        return Err(format!("growth_rate must be >= 1, got {}", growth_rate));
    }
    if num_layers < 1 {
        return Err(format!("num_layers must be >= 1, got {}", num_layers));
    }
    Ok((0..=num_layers).map(|l| c0 + growth_rate * l).collect())
}

/// Counts parameters in a DenseNet-BC style dense block: each layer applies a
/// `1x1` bottleneck to `bn_size * growth_rate` channels followed by a `3x3`
/// convolution down to `growth_rate` new channels (biases and batch norm
/// omitted). Sums `params_l = c_l * (b*k) + (b*k) * k * 9` over the block.
pub fn dense_block_param_count(
    c0: i64,
    growth_rate: i64,
    num_layers: i64,
    bn_size: i64,
) -> Result<i64, String> {
    if bn_size < 1 {
        return Err(format!("bn_size must be >= 1, got {}", bn_size));
    }
    let sizes = dense_block_channel_sizes(c0, growth_rate, num_layers)?;
    let k = growth_rate;
    let bw = bn_size * k;
    let mut total = 0i64;
    for &c_l in &sizes[..sizes.len() - 1] {
        total += c_l * bw;
        total += bw * k * 9;
    }
    Ok(total)
}

/// Returns the compressed channel count `floor(theta * c_in)` after a
/// DenseNet transition layer, `0 < theta <= 1`.
pub fn transition_output_channels(c_in: i64, theta: f64) -> Result<i64, String> {
    if !(theta > 0.0 && theta <= 1.0) {
        return Err(format!("theta must satisfy 0 < theta <= 1, got {}", theta));
    }
    if c_in < 1 {
        return Err(format!("c_in must be >= 1, got {}", c_in));
    }
    Ok((theta * c_in as f64).floor() as i64)
}

/// Parameter count of a plain (additive, ResNet-style) stack matched on
/// output width: the first `3x3` layer maps `c0 -> width`, every later layer
/// maps `width -> width`. The parameter-cost baseline for Section 4.2's
/// efficiency comparison.
pub fn plain_block_param_count(c0: i64, width: i64, num_layers: i64) -> Result<i64, String> {
    if c0 < 1 {
        return Err(format!("c0 must be >= 1, got {}", c0));
    }
    if width < 1 {
        return Err(format!("width must be >= 1, got {}", width));
    }
    if num_layers < 1 {
        return Err(format!("num_layers must be >= 1, got {}", num_layers));
    }
    Ok(c0 * width * 9 + (num_layers - 1) * (width * width * 9))
}

// --------------------------------------------------------------------------
// Toy forward pass demonstrating the concatenation mechanic
// --------------------------------------------------------------------------

/// Deterministically initializes one dense layer's weight matrix
/// (`growth_rate x c_in`, row-major) and zero bias, drawing every entry from
/// `Lcg(seed)` mapped to `[-lim, lim]` with `lim = sqrt(6 / (c_in +
/// growth_rate))`.
pub fn init_layer_weights(
    c_in: usize,
    growth_rate: usize,
    seed: u64,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    if c_in < 1 {
        return Err("c_in must be >= 1".to_string());
    }
    if growth_rate < 1 {
        return Err("growth_rate must be >= 1".to_string());
    }
    let lim = (6.0 / (c_in + growth_rate) as f64).sqrt();
    let mut rng = Lcg::new(seed);
    let mut w = vec![vec![0.0; c_in]; growth_rate];
    for row in w.iter_mut() {
        for entry in row.iter_mut() {
            let u = rng.next_uniform();
            *entry = (2.0 * u - 1.0) * lim;
        }
    }
    let b = vec![0.0; growth_rate];
    Ok((w, b))
}

/// Applies one dense-block layer `H_l`: `relu(w * x + b)`.
pub fn dense_layer_forward(x: &[f64], w: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; w.len()];
    for i in 0..w.len() {
        let mut s = b[i];
        for j in 0..x.len() {
            s += w[i][j] * x[j];
        }
        out[i] = s.max(0.0);
    }
    out
}

/// Runs a toy dense block forward pass, literally implementing `x_l =
/// H_l([x_0, ..., x_{l-1}])` by concatenating each layer's output onto the
/// running feature vector. Layer `l` uses `init_layer_weights` seeded with
/// `seed + l`. Returns `(final_features, channel_sizes)`.
pub fn dense_block_forward(
    x0: &[f64],
    growth_rate: usize,
    num_layers: usize,
    seed: u64,
) -> Result<(Vec<f64>, Vec<i64>), String> {
    if x0.is_empty() {
        return Err("x0 must be non-empty".to_string());
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err("x0 contains non-finite values (nan or inf)".to_string());
    }
    let c0 = x0.len() as i64;
    let sizes = dense_block_channel_sizes(c0, growth_rate as i64, num_layers as i64)?;
    let mut features: Vec<f64> = x0.to_vec();
    for l in 0..num_layers {
        let (w, b) = init_layer_weights(features.len(), growth_rate, seed + l as u64)?;
        let new_features = dense_layer_forward(&features, &w, &b);
        features.extend(new_features);
        assert_eq!(features.len() as i64, sizes[l + 1]);
    }
    Ok((features, sizes))
}

// --------------------------------------------------------------------------
// DenseNet-BC architecture variants (Huang et al. 2017, Table 1)
// --------------------------------------------------------------------------

/// Block configs (dense layers per block) for the four ImageNet variants,
/// growth rate k=32, initial channels k0=64, compression theta=0.5.
pub fn densenet_variant_blocks(variant: &str) -> Result<[i64; 4], String> {
    match variant {
        "121" => Ok([6, 12, 24, 16]),
        "169" => Ok([6, 12, 32, 32]),
        "201" => Ok([6, 12, 48, 32]),
        "264" => Ok([6, 12, 64, 48]),
        _ => Err(format!("unknown variant {:?}", variant)),
    }
}

/// Sums dense-block and transition-layer parameters for a DenseNet-BC variant
/// (`"121"`, `"169"`, `"201"`, or `"264"`). Excludes the stem convolution,
/// final batch norm, and classifier head; see the Python docstring for
/// details.
pub fn densenet_dense_param_total(
    variant: &str,
    growth_rate: i64,
    k0: i64,
    theta: f64,
    bn_size: i64,
) -> Result<i64, String> {
    let blocks = densenet_variant_blocks(variant)?;
    let mut c = k0;
    let mut total = 0i64;
    for (i, &num_layers) in blocks.iter().enumerate() {
        total += dense_block_param_count(c, growth_rate, num_layers, bn_size)?;
        c += growth_rate * num_layers;
        if i < blocks.len() - 1 {
            let c_out = transition_output_channels(c, theta)?;
            total += c * c_out;
            c = c_out;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    #[test]
    fn channel_sizes_match_fixture() {
        assert_eq!(dense_block_channel_sizes(4, 4, 3).unwrap(), vec![4, 8, 12, 16]);
        assert_eq!(dense_block_channel_sizes(4, 3, 3).unwrap(), vec![4, 7, 10, 13]);
    }

    #[test]
    fn dense_block_param_count_matches_fixture() {
        assert_eq!(dense_block_param_count(4, 4, 3, 4).unwrap(), 2112);
        assert_eq!(dense_block_param_count(8, 12, 4, 4).unwrap(), 25728);
        assert_eq!(dense_block_param_count(64, 32, 6, 4).unwrap(), 331776);
    }

    #[test]
    fn transition_output_channels_matches_fixture() {
        assert_eq!(transition_output_channels(64, 0.5).unwrap(), 32);
        assert_eq!(transition_output_channels(128, 0.5).unwrap(), 64);
    }

    #[test]
    fn plain_block_param_count_matches_fixture() {
        assert_eq!(plain_block_param_count(16, 16, 3).unwrap(), 6912);
        assert_eq!(plain_block_param_count(64, 64, 6).unwrap(), 221184);
    }

    #[test]
    fn dense_beats_plain_at_matched_output_width() {
        // Same total output width (c0 + k*L == width), dense connectivity
        // uses far fewer parameters than a plain stack of matched width.
        let dense = dense_block_param_count(16, 12, 8, 4).unwrap();
        let plain = plain_block_param_count(16, 16 + 12 * 8, 8).unwrap();
        assert!(dense < plain);
    }

    #[test]
    fn lcg_uniform_matches_fixture() {
        let mut rng = Lcg::new(0);
        let expected = [
            0.07820865487829387,
            0.10169876029679303,
            0.60532332262523347,
            0.40121620369530075,
        ];
        for &e in expected.iter() {
            assert!((rng.next_uniform() - e).abs() < 1e-12);
        }
    }

    #[test]
    fn dense_block_forward_matches_fixture() {
        let x0 = [1.0, 0.6, -0.3, 0.9];
        let (out, sizes) = dense_block_forward(&x0, 3, 3, 2).unwrap();
        assert_eq!(sizes, vec![4, 7, 10, 13]);
        let expected = [
            1.0,
            0.6,
            -0.3,
            0.9,
            0.6279286728409486,
            0.11130697455664373,
            0.08343734603269459,
            0.0,
            0.9451360437126258,
            1.4047640097915077,
            0.0,
            0.0,
            0.8720018399583354,
        ];
        assert_eq!(out.len(), expected.len());
        for i in 0..expected.len() {
            assert!((out[i] - expected[i]).abs() < TOL, "index {}: {} vs {}", i, out[i], expected[i]);
        }
    }

    #[test]
    fn dense_block_forward_matches_channel_size_trace() {
        let x0 = [1.0, -1.0, 0.5, 0.5];
        let (out, sizes) = dense_block_forward(&x0, 2, 2, 0).unwrap();
        assert_eq!(sizes, vec![4, 6, 8]);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn densenet_variant_param_totals_match_fixture() {
        assert_eq!(densenet_dense_param_total("121", 32, 64, 0.5, 4).unwrap(), 6_860_800);
        assert_eq!(densenet_dense_param_total("169", 32, 64, 0.5, 4).unwrap(), 12_316_672);
        assert_eq!(densenet_dense_param_total("201", 32, 64, 0.5, 4).unwrap(), 17_854_464);
        assert_eq!(densenet_dense_param_total("264", 32, 64, 0.5, 4).unwrap(), 30_240_768);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(dense_block_channel_sizes(-1, 4, 3).is_err());
        assert!(dense_block_channel_sizes(4, 0, 3).is_err());
        assert!(dense_block_channel_sizes(4, 4, 0).is_err());
        assert!(transition_output_channels(64, 0.0).is_err());
        assert!(transition_output_channels(64, 1.5).is_err());
        assert!(densenet_dense_param_total("bogus", 32, 64, 0.5, 4).is_err());
        let bad = [1.0, f64::NAN];
        assert!(dense_block_forward(&bad, 2, 2, 0).is_err());
        let empty: [f64; 0] = [];
        assert!(dense_block_forward(&empty, 2, 2, 0).is_err());
    }
}
