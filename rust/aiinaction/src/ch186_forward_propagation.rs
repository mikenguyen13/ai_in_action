//! Feedforward neural network forward propagation from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch186_forward_propagation` and the Julia
//! module `AIInAction.Ch186ForwardPropagation`. The shared fixtures in the tests
//! below match the Python/Julia suites, which keeps the three implementations at
//! parity.
//!
//! This is a std-only implementation. A batch is stored row-major as a `d x m`
//! matrix (features down rows, examples across columns), matching the column-batch
//! convention used in the chapter. Each layer computes `Z = W @ A_prev + b` and
//! then applies an elementwise activation. The sigmoid is evaluated in a
//! numerically stable, branch-free form so it never overflows.

/// A dense row-major matrix of `f64` with `rows` features and `cols` examples.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Builds a matrix from a slice of equal-length rows.
    pub fn from_rows(rows: &[Vec<f64>]) -> Result<Matrix, String> {
        if rows.is_empty() {
            return Err("matrix must have at least one row".to_string());
        }
        let cols = rows[0].len();
        if cols == 0 {
            return Err("matrix must have at least one column".to_string());
        }
        let mut data = Vec::with_capacity(rows.len() * cols);
        for r in rows {
            if r.len() != cols {
                return Err("all rows must have the same length".to_string());
            }
            data.extend_from_slice(r);
        }
        Ok(Matrix {
            rows: rows.len(),
            cols,
            data,
        })
    }

    /// Builds a single-column `d x 1` matrix from a length-`d` vector.
    pub fn from_vec(v: &[f64]) -> Result<Matrix, String> {
        if v.is_empty() {
            return Err("vector must be non-empty".to_string());
        }
        Ok(Matrix {
            rows: v.len(),
            cols: 1,
            data: v.to_vec(),
        })
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }
}

/// The supported elementwise activations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Identity,
}

impl Activation {
    /// Parses an activation name, mirroring the Python/Julia string API.
    pub fn from_name(name: &str) -> Result<Activation, String> {
        match name {
            "relu" => Ok(Activation::Relu),
            "sigmoid" => Ok(Activation::Sigmoid),
            "tanh" => Ok(Activation::Tanh),
            "identity" => Ok(Activation::Identity),
            other => Err(format!(
                "unknown activation {:?}; expected one of [\"relu\", \"sigmoid\", \"tanh\", \"identity\"]",
                other
            )),
        }
    }

    /// Applies the activation to a single scalar.
    #[inline]
    pub fn apply(&self, z: f64) -> f64 {
        match self {
            Activation::Relu => {
                if z > 0.0 {
                    z
                } else {
                    0.0
                }
            }
            // Numerically stable logistic sigmoid.
            Activation::Sigmoid => {
                if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let ez = z.exp();
                    ez / (1.0 + ez)
                }
            }
            Activation::Tanh => z.tanh(),
            Activation::Identity => z,
        }
    }
}

/// One dense layer: an affine map `W @ a + b` plus an activation.
#[derive(Clone, Debug)]
pub struct Layer {
    /// Weight matrix of shape `(d_out, d_in)`.
    pub w: Matrix,
    /// Bias vector of length `d_out`.
    pub b: Vec<f64>,
    pub activation: Activation,
}

impl Layer {
    pub fn d_in(&self) -> usize {
        self.w.cols
    }
    pub fn d_out(&self) -> usize {
        self.w.rows
    }
}

/// Validates and constructs a `Layer` from raw weights, bias and activation name.
pub fn make_layer(w_rows: &[Vec<f64>], b: &[f64], activation: &str) -> Result<Layer, String> {
    let w = Matrix::from_rows(w_rows)?;
    let act = Activation::from_name(activation)?;
    if b.len() != w.rows {
        return Err(format!(
            "bias length {} does not match number of units {}",
            b.len(),
            w.rows
        ));
    }
    if w.data.iter().any(|v| !v.is_finite()) {
        return Err("W contains non-finite values (nan or inf)".to_string());
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err("b contains non-finite values (nan or inf)".to_string());
    }
    Ok(Layer {
        w,
        b: b.to_vec(),
        activation: act,
    })
}

/// Propagates one batched activation through a single layer.
///
/// Returns `(Z, A)` where `Z = W @ A_prev + b` (bias broadcast across columns) and
/// `A` is the activation of `Z`; both have shape `(d_out, m)`. The pre-activation
/// `Z` is returned alongside `A` because it is what a backward pass would cache.
pub fn forward_layer(layer: &Layer, a_prev: &Matrix) -> Result<(Matrix, Matrix), String> {
    if a_prev.rows != layer.d_in() {
        return Err(format!(
            "layer expects {} input features but received {}",
            layer.d_in(),
            a_prev.rows
        ));
    }
    let d_out = layer.d_out();
    let m = a_prev.cols;
    let mut z = vec![0.0f64; d_out * m];
    let mut a = vec![0.0f64; d_out * m];
    for i in 0..d_out {
        for col in 0..m {
            let mut acc = layer.b[i];
            for k in 0..layer.d_in() {
                acc += layer.w.get(i, k) * a_prev.get(k, col);
            }
            z[i * m + col] = acc;
            a[i * m + col] = layer.activation.apply(acc);
        }
    }
    Ok((
        Matrix {
            rows: d_out,
            cols: m,
            data: z,
        },
        Matrix {
            rows: d_out,
            cols: m,
            data: a,
        },
    ))
}

/// Runs the full forward sweep through every layer in order.
///
/// `x` is a `d0 x m` batch with examples as columns. Returns the network output
/// `A^[L]` of shape `(d_L, m)`.
pub fn forward(layers: &[Layer], x: &Matrix) -> Result<Matrix, String> {
    if layers.is_empty() {
        return Err("network must have at least one layer".to_string());
    }
    for i in 0..layers.len() - 1 {
        if layers[i].d_out() != layers[i + 1].d_in() {
            return Err(format!(
                "layer {} outputs {} features but layer {} expects {}",
                i,
                layers[i].d_out(),
                i + 1,
                layers[i + 1].d_in()
            ));
        }
    }
    if x.rows != layers[0].d_in() {
        return Err(format!(
            "input has {} features but first layer expects {}",
            x.rows,
            layers[0].d_in()
        ));
    }
    if x.data.iter().any(|v| !v.is_finite()) {
        return Err("input contains non-finite values (nan or inf)".to_string());
    }
    let mut a = x.clone();
    for layer in layers {
        let (_z, next) = forward_layer(layer, &a)?;
        a = next;
    }
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn net() -> Vec<Layer> {
        let w1: [[f64; 2]; 2] = [[0.5, -0.2], [0.1, 0.4]];
        let b1: [f64; 2] = [0.1, -0.3];
        let w2: [[f64; 2]; 1] = [[0.7, -0.6]];
        let b2: [f64; 1] = [0.2];
        vec![
            make_layer(&[w1[0].to_vec(), w1[1].to_vec()], &b1, "relu").unwrap(),
            make_layer(&[w2[0].to_vec()], &b2, "sigmoid").unwrap(),
        ]
    }

    fn batch() -> Matrix {
        // d0 = 2 features, m = 2 examples as columns.
        Matrix::from_rows(&[vec![1.0, 0.0], vec![2.0, 1.0]]).unwrap()
    }

    #[test]
    fn forward_layer_preactivation_matches_fixture() {
        let layer = &net()[0];
        let (z1, _a1) = forward_layer(layer, &batch()).unwrap();
        let expected: [f64; 4] = [0.2, -0.1, 0.6, 0.1];
        for (k, &e) in expected.iter().enumerate() {
            assert!((z1.data[k] - e).abs() < TOL);
        }
    }

    #[test]
    fn forward_layer_activation_matches_fixture() {
        let layer = &net()[0];
        let (_z1, a1) = forward_layer(layer, &batch()).unwrap();
        let expected: [f64; 4] = [0.2, 0.0, 0.6, 0.1];
        for (k, &e) in expected.iter().enumerate() {
            assert!((a1.data[k] - e).abs() < TOL);
        }
    }

    #[test]
    fn second_layer_preactivation_matches_fixture() {
        let layers = net();
        let (_z1, a1) = forward_layer(&layers[0], &batch()).unwrap();
        let (z2, _a2) = forward_layer(&layers[1], &a1).unwrap();
        let expected: [f64; 2] = [-0.02, 0.14];
        for (k, &e) in expected.iter().enumerate() {
            assert!((z2.data[k] - e).abs() < TOL);
        }
    }

    #[test]
    fn forward_output_matches_fixture() {
        let out = forward(&net(), &batch()).unwrap();
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, 2);
        let expected: [f64; 2] = [0.49500016666000024, 0.5349429451582145];
        for (k, &e) in expected.iter().enumerate() {
            assert!((out.data[k] - e).abs() < TOL);
        }
    }

    #[test]
    fn single_example_matches_first_column() {
        let single = Matrix::from_vec(&[1.0, 2.0]).unwrap();
        let out = forward(&net(), &single).unwrap();
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, 1);
        assert!((out.data[0] - 0.49500016666000024).abs() < TOL);
    }

    #[test]
    fn tanh_activation_matches_fixture() {
        let w3: [[f64; 3]; 1] = [[1.0, 2.0, -1.0]];
        let b3: [f64; 1] = [0.5];
        let layers = vec![make_layer(&[w3[0].to_vec()], &b3, "tanh").unwrap()];
        let x = Matrix::from_vec(&[0.5, -0.5, 1.0]).unwrap();
        let out = forward(&layers, &x).unwrap();
        assert!((out.data[0] - (-0.7615941559557649)).abs() < TOL);
    }

    #[test]
    fn relu_clips_negatives() {
        let inputs: [f64; 5] = [-2.0, -0.5, 0.0, 0.5, 2.0];
        let expected: [f64; 5] = [0.0, 0.0, 0.0, 0.5, 2.0];
        for (k, &z) in inputs.iter().enumerate() {
            assert!((Activation::Relu.apply(z) - expected[k]).abs() < TOL);
        }
    }

    #[test]
    fn sigmoid_is_numerically_stable() {
        assert!((Activation::Sigmoid.apply(-1000.0) - 0.0).abs() < TOL);
        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < TOL);
        assert!((Activation::Sigmoid.apply(1000.0) - 1.0).abs() < TOL);
        assert!(Activation::Sigmoid.apply(-1000.0).is_finite());
    }

    #[test]
    fn bias_length_mismatch_errors() {
        assert!(make_layer(&[vec![1.0, 2.0]], &[0.0, 0.0], "relu").is_err());
    }

    #[test]
    fn unknown_activation_errors() {
        assert!(make_layer(&[vec![1.0]], &[0.0], "softmax").is_err());
    }

    #[test]
    fn non_finite_weight_errors() {
        assert!(make_layer(&[vec![1.0, f64::NAN]], &[0.0], "relu").is_err());
    }

    #[test]
    fn empty_network_errors() {
        let x = Matrix::from_vec(&[1.0]).unwrap();
        assert!(forward(&[], &x).is_err());
    }

    #[test]
    fn incompatible_layers_error() {
        let bad = vec![
            make_layer(&[vec![1.0, 1.0]], &[0.0], "relu").unwrap(),
            make_layer(&[vec![1.0, 1.0]], &[0.0], "relu").unwrap(),
        ];
        let x = Matrix::from_vec(&[1.0, 1.0]).unwrap();
        assert!(forward(&bad, &x).is_err());
    }

    #[test]
    fn input_feature_mismatch_errors() {
        let x = Matrix::from_vec(&[1.0, 2.0, 3.0]).unwrap();
        assert!(forward(&net(), &x).is_err());
    }
}
