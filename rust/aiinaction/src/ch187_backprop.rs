//! Backpropagation from scratch for a feedforward neural network (Rust).
//!
//! Mirrors the Python module `aiinaction.ch187_backprop` and the Julia module
//! `AIInAction.Ch187Backprop`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! The network is a plain multilayer perceptron: each hidden layer applies the
//! logistic sigmoid elementwise, the output layer is linear (identity), and the
//! loss is one-half squared error `C = 1/2 ||a^L - y||^2`. This std-only
//! implementation stores each weight matrix row-major as `Vec<Vec<f64>>`.
//!
//! Equations (identical to the chapter and the other languages):
//! BP1: `delta^L = a^L - y` (linear output).
//! BP2: `delta^l = ((W^{l+1})^T delta^{l+1}) .* sigma'(z^l)`.
//! BP3: `dC/db^l = delta^l`.
//! BP4: `dC/dW^l = delta^l (a^{l-1})^T`.

/// A feedforward network with sigmoid hidden layers and a linear output.
///
/// `weights[l]` has shape `(n_{l+1}, n_l)` (row-major) and `biases[l]` length `n_{l+1}`.
#[derive(Clone, Debug)]
pub struct Mlp {
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
}

impl Mlp {
    /// Validates layer shapes and finiteness, returning an `Mlp` or an error message.
    pub fn new(weights: Vec<Vec<Vec<f64>>>, biases: Vec<Vec<f64>>) -> Result<Mlp, String> {
        if weights.len() != biases.len() {
            return Err(format!(
                "weights and biases must have equal length, got {} and {}",
                weights.len(),
                biases.len()
            ));
        }
        if weights.is_empty() {
            return Err("network must have at least one layer".to_string());
        }
        let mut prev_cols: Option<usize> = None;
        for (l, (wl, bl)) in weights.iter().zip(biases.iter()).enumerate() {
            if wl.is_empty() {
                return Err(format!("weights[{}] must have at least one row", l));
            }
            let cols = wl[0].len();
            if cols == 0 {
                return Err(format!("weights[{}] rows must be non-empty", l));
            }
            for row in wl.iter() {
                if row.len() != cols {
                    return Err(format!("weights[{}] rows must all have the same length", l));
                }
                if row.iter().any(|v| !v.is_finite()) {
                    return Err(format!("layer {}: parameters contain non-finite values (nan or inf)", l));
                }
            }
            if wl.len() != bl.len() {
                return Err(format!(
                    "layer {}: weights has {} rows but bias has length {}",
                    l,
                    wl.len(),
                    bl.len()
                ));
            }
            if bl.iter().any(|v| !v.is_finite()) {
                return Err(format!("layer {}: parameters contain non-finite values (nan or inf)", l));
            }
            if let Some(pc) = prev_cols {
                if cols != pc {
                    return Err(format!(
                        "layer {}: weights expects {} inputs but previous layer emits {}",
                        l, cols, pc
                    ));
                }
            }
            prev_cols = Some(wl.len());
        }
        Ok(Mlp { weights, biases })
    }

    /// Number of weight layers `L`.
    pub fn num_layers(&self) -> usize {
        self.weights.len()
    }
    /// Dimensionality of the input feature vector.
    pub fn n_input(&self) -> usize {
        self.weights[0][0].len()
    }
    /// Dimensionality of the network output.
    pub fn n_output(&self) -> usize {
        self.weights[self.weights.len() - 1].len()
    }
}

/// Numerically stable logistic sigmoid.
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

/// Derivative `sigma'(z) = sigma(z) (1 - sigma(z))`.
pub fn sigmoid_prime(z: f64) -> f64 {
    let s = sigmoid(z);
    s * (1.0 - s)
}

fn matvec(w: &[Vec<f64>], b: &[f64], a: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; w.len()];
    for (i, row) in w.iter().enumerate() {
        let mut acc = b[i];
        for (j, &wij) in row.iter().enumerate() {
            acc += wij * a[j];
        }
        out[i] = acc;
    }
    out
}

/// Runs the forward pass, returning `(zs, activations)`.
///
/// `zs[l]` is the pre-activation `z^{l+1}`; `activations[l]` is `a^l` with
/// `activations[0] == x`. Hidden layers use the sigmoid, the output is linear.
pub fn forward(net: &Mlp, x: &[f64]) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
    if x.len() != net.n_input() {
        return Err(format!(
            "x has length {} but network expects {}",
            x.len(),
            net.n_input()
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err("x contains non-finite values (nan or inf)".to_string());
    }
    let l_total = net.num_layers();
    let mut activations: Vec<Vec<f64>> = Vec::with_capacity(l_total + 1);
    let mut zs: Vec<Vec<f64>> = Vec::with_capacity(l_total);
    let mut a = x.to_vec();
    activations.push(a.clone());
    for l in 0..l_total {
        let z = matvec(&net.weights[l], &net.biases[l], &a);
        a = if l == l_total - 1 {
            z.clone()
        } else {
            z.iter().map(|&v| sigmoid(v)).collect()
        };
        zs.push(z);
        activations.push(a.clone());
    }
    Ok((zs, activations))
}

/// One-half squared-error loss `1/2 ||a^L - y||^2` for one example.
pub fn squared_error_loss(net: &Mlp, x: &[f64], y: &[f64]) -> Result<f64, String> {
    if y.len() != net.n_output() {
        return Err(format!(
            "y has length {} but network outputs {}",
            y.len(),
            net.n_output()
        ));
    }
    let (_, activations) = forward(net, x)?;
    let out = &activations[activations.len() - 1];
    let mut acc = 0.0;
    for (o, t) in out.iter().zip(y.iter()) {
        let d = o - t;
        acc += d * d;
    }
    Ok(0.5 * acc)
}

/// Computes parameter gradients for one example via backpropagation.
///
/// Returns `(grad_w, grad_b)` with the same shapes as `net.weights`/`net.biases`.
pub fn backprop(
    net: &Mlp,
    x: &[f64],
    y: &[f64],
) -> Result<(Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>), String> {
    if y.len() != net.n_output() {
        return Err(format!(
            "y has length {} but network outputs {}",
            y.len(),
            net.n_output()
        ));
    }
    let (zs, activations) = forward(net, x)?;
    let l_total = net.num_layers();

    let mut grad_w: Vec<Vec<Vec<f64>>> = net
        .weights
        .iter()
        .map(|w| w.iter().map(|r| vec![0.0f64; r.len()]).collect())
        .collect();
    let mut grad_b: Vec<Vec<f64>> = net.biases.iter().map(|b| vec![0.0f64; b.len()]).collect();

    // BP1: linear output, delta^L = a^L - y.
    let out = &activations[l_total];
    let mut delta: Vec<f64> = out.iter().zip(y.iter()).map(|(o, t)| o - t).collect();

    // Read off gradients for the output layer (BP3, BP4).
    let last = l_total - 1;
    grad_b[last] = delta.clone();
    for (j, &dj) in delta.iter().enumerate() {
        for (k, &ak) in activations[last].iter().enumerate() {
            grad_w[last][j][k] = dj * ak;
        }
    }

    // BP2: propagate backward through the hidden (sigmoid) layers.
    for l in (0..last).rev() {
        let w_next = &net.weights[l + 1]; // shape (n_{l+2}, n_{l+1})
        let n_l = w_next[0].len();
        let mut new_delta = vec![0.0f64; n_l];
        for j in 0..n_l {
            let mut acc = 0.0;
            for (i, row) in w_next.iter().enumerate() {
                acc += row[j] * delta[i];
            }
            new_delta[j] = acc * sigmoid_prime(zs[l][j]);
        }
        delta = new_delta;
        grad_b[l] = delta.clone();
        for (j, &dj) in delta.iter().enumerate() {
            for (k, &ak) in activations[l].iter().enumerate() {
                grad_w[l][j][k] = dj * ak;
            }
        }
    }

    Ok((grad_w, grad_b))
}

/// Central-difference estimate of the loss gradient, for gradient checking.
pub fn numerical_gradient(
    net: &Mlp,
    x: &[f64],
    y: &[f64],
    eps: f64,
) -> Result<(Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>), String> {
    if eps <= 0.0 {
        return Err(format!("eps must be positive, got {}", eps));
    }
    let mut work = net.clone();
    let mut grad_w: Vec<Vec<Vec<f64>>> = net
        .weights
        .iter()
        .map(|w| w.iter().map(|r| vec![0.0f64; r.len()]).collect())
        .collect();
    let mut grad_b: Vec<Vec<f64>> = net.biases.iter().map(|b| vec![0.0f64; b.len()]).collect();

    for l in 0..work.num_layers() {
        let rows = work.weights[l].len();
        let cols = work.weights[l][0].len();
        for j in 0..rows {
            for k in 0..cols {
                let orig = work.weights[l][j][k];
                work.weights[l][j][k] = orig + eps;
                let cp = squared_error_loss(&work, x, y)?;
                work.weights[l][j][k] = orig - eps;
                let cm = squared_error_loss(&work, x, y)?;
                work.weights[l][j][k] = orig;
                grad_w[l][j][k] = (cp - cm) / (2.0 * eps);
            }
        }
        let blen = work.biases[l].len();
        for j in 0..blen {
            let orig = work.biases[l][j];
            work.biases[l][j] = orig + eps;
            let cp = squared_error_loss(&work, x, y)?;
            work.biases[l][j] = orig - eps;
            let cm = squared_error_loss(&work, x, y)?;
            work.biases[l][j] = orig;
            grad_b[l][j] = (cp - cm) / (2.0 * eps);
        }
    }
    Ok((grad_w, grad_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn fixture() -> Mlp {
        let w1: Vec<Vec<f64>> = vec![vec![0.10, 0.20, -0.30], vec![0.40, -0.50, 0.60]];
        let b1: Vec<f64> = vec![0.10, -0.20];
        let w2: Vec<Vec<f64>> = vec![vec![0.70, -0.80], vec![-0.10, 0.30]];
        let b2: Vec<f64> = vec![0.05, -0.05];
        Mlp::new(vec![w1, w2], vec![b1, b2]).unwrap()
    }

    fn fx() -> [f64; 3] {
        [1.0, -2.0, 0.5]
    }
    fn fy() -> [f64; 2] {
        [0.3, -0.7]
    }

    const TOL: f64 = 1e-9;

    #[test]
    fn forward_hidden_preactivation() {
        let (zs, _) = forward(&fixture(), &fx()).unwrap();
        let expected: [f64; 2] = [-0.3500000000000001, 1.5];
        for i in 0..2 {
            assert!((zs[0][i] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn forward_hidden_activation() {
        let (_, acts) = forward(&fixture(), &fx()).unwrap();
        let expected: [f64; 2] = [0.4133824210826699, 0.8175744761936437];
        for i in 0..2 {
            assert!((acts[1][i] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn forward_output_is_linear() {
        let (zs, acts) = forward(&fixture(), &fx()).unwrap();
        let expected: [f64; 2] = [-0.3146918861970461, 0.15393410074982605];
        let last = acts.len() - 1;
        for i in 0..2 {
            assert!((acts[last][i] - expected[i]).abs() < TOL);
            assert!((acts[last][i] - zs[zs.len() - 1][i]).abs() < TOL);
        }
    }

    #[test]
    fn loss_matches_fixture() {
        let c = squared_error_loss(&fixture(), &fx(), &fy()).unwrap();
        assert!((c - 0.5535247816899481).abs() < TOL);
    }

    #[test]
    fn grad_w_layer0_matches_fixture() {
        let (gw, _) = backprop(&fixture(), &fx(), &fy()).unwrap();
        let expected: [[f64; 3]; 2] = [
            [-0.12505050629624692, 0.25010101259249384, -0.06252525314812346],
            [0.11155166358278021, -0.22310332716556042, 0.055775831791390104],
        ];
        for i in 0..2 {
            for j in 0..3 {
                assert!((gw[0][i][j] - expected[i][j]).abs() < TOL);
            }
        }
    }

    #[test]
    fn grad_b_layer0_matches_fixture() {
        let (_, gb) = backprop(&fixture(), &fx(), &fy()).unwrap();
        let expected: [f64; 2] = [-0.12505050629624692, 0.11155166358278021];
        for i in 0..2 {
            assert!((gb[0][i] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn grad_w_layer1_matches_fixture() {
        let (gw, _) = backprop(&fixture(), &fx(), &fy()).unwrap();
        let expected: [[f64; 2]; 2] = [
            [-0.25410282013600793, -0.5025563968780328],
            [0.35300134601301564, 0.6981547251244291],
        ];
        for i in 0..2 {
            for j in 0..2 {
                assert!((gw[1][i][j] - expected[i][j]).abs() < TOL);
            }
        }
    }

    #[test]
    fn grad_b_layer1_matches_fixture() {
        let (_, gb) = backprop(&fixture(), &fx(), &fy()).unwrap();
        let expected: [f64; 2] = [-0.6146918861970461, 0.853934100749826];
        for i in 0..2 {
            assert!((gb[1][i] - expected[i]).abs() < TOL);
        }
    }

    #[test]
    fn analytic_matches_numerical_gradient() {
        let net = fixture();
        let (gw, gb) = backprop(&net, &fx(), &fy()).unwrap();
        let (ngw, ngb) = numerical_gradient(&net, &fx(), &fy(), 1e-6).unwrap();
        for l in 0..gw.len() {
            for i in 0..gw[l].len() {
                for j in 0..gw[l][i].len() {
                    assert!((gw[l][i][j] - ngw[l][i][j]).abs() < 1e-7);
                }
                assert!((gb[l][i] - ngb[l][i]).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn zero_residual_gives_zero_gradient() {
        let net = fixture();
        let (_, acts) = forward(&net, &fx()).unwrap();
        let out = acts[acts.len() - 1].clone();
        let (gw, gb) = backprop(&net, &fx(), &out).unwrap();
        for layer in gw.iter() {
            for row in layer.iter() {
                for &v in row.iter() {
                    assert!(v.abs() < 1e-12);
                }
            }
        }
        for layer in gb.iter() {
            for &v in layer.iter() {
                assert!(v.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn sigmoid_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < TOL);
        assert!((sigmoid_prime(0.0) - 0.25).abs() < TOL);
        assert!((sigmoid(1000.0) - 1.0).abs() < TOL);
        assert!(sigmoid(-1000.0).abs() < TOL);
    }

    #[test]
    fn mismatched_weights_biases_errors() {
        let w1: Vec<Vec<f64>> = vec![vec![1.0, 2.0]];
        assert!(Mlp::new(vec![w1.clone(), w1], vec![vec![0.0]]).is_err());
    }

    #[test]
    fn layer_dim_mismatch_errors() {
        let w1: Vec<Vec<f64>> = vec![vec![1.0, 2.0]];
        let w2: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0]];
        assert!(Mlp::new(vec![w1, w2], vec![vec![0.0], vec![0.0]]).is_err());
    }

    #[test]
    fn bad_input_length_errors() {
        assert!(forward(&fixture(), &[1.0, 2.0]).is_err());
    }

    #[test]
    fn non_finite_param_errors() {
        let w1: Vec<Vec<f64>> = vec![vec![f64::NAN, 0.0]];
        assert!(Mlp::new(vec![w1], vec![vec![0.0]]).is_err());
    }

    #[test]
    fn eps_must_be_positive() {
        assert!(numerical_gradient(&fixture(), &fx(), &fy(), 0.0).is_err());
    }
}
