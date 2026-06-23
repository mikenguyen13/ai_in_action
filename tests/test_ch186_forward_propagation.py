"""Tests for aiinaction.ch186_forward_propagation, including shared fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. The network is the worked example from the chapter: a 2-2-1 net with a
ReLU hidden layer and a sigmoid output, evaluated on a two-example batch.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch186_forward_propagation as fp

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
W1 = [[0.5, -0.2], [0.1, 0.4]]
B1 = [0.1, -0.3]
W2 = [[0.7, -0.6]]
B2 = [0.2]
X = [[1.0, 0.0], [2.0, 1.0]]  # two examples (d0=2) stacked as columns (m=2)

EXPECTED = {
    "Z1": [0.2, -0.1, 0.6, 0.1],
    "A1": [0.2, 0.0, 0.6, 0.1],
    "Z2": [-0.02, 0.14],
    "Yhat": [0.49500016666000024, 0.5349429451582145],
    "single": 0.49500016666000024,
    "tanh_out": -0.7615941559557649,
}


def _net():
    return [
        fp.make_layer(W1, B1, "relu"),
        fp.make_layer(W2, B2, "sigmoid"),
    ]


def test_forward_layer_preactivation_matches_fixture():
    layer = fp.make_layer(W1, B1, "relu")
    Z, _ = fp.forward_layer(layer, np.asarray(X, dtype=float))
    assert Z.ravel().tolist() == pytest.approx(EXPECTED["Z1"])


def test_forward_layer_activation_matches_fixture():
    layer = fp.make_layer(W1, B1, "relu")
    _, A = fp.forward_layer(layer, np.asarray(X, dtype=float))
    assert A.ravel().tolist() == pytest.approx(EXPECTED["A1"])


def test_second_layer_preactivation_matches_fixture():
    net = _net()
    _, A1 = fp.forward_layer(net[0], np.asarray(X, dtype=float))
    Z2, _ = fp.forward_layer(net[1], A1)
    assert Z2.ravel().tolist() == pytest.approx(EXPECTED["Z2"])


def test_forward_output_matches_fixture():
    out = fp.forward(_net(), X)
    assert out.shape == (1, 2)
    assert out.ravel().tolist() == pytest.approx(EXPECTED["Yhat"])


def test_single_example_matches_first_column():
    out = fp.forward(_net(), [1.0, 2.0])
    assert out.shape == (1, 1)
    assert float(out[0, 0]) == pytest.approx(EXPECTED["single"])


def test_batch_first_column_equals_single_example():
    batch = fp.forward(_net(), X)
    single = fp.forward(_net(), [1.0, 2.0])
    assert float(batch[0, 0]) == pytest.approx(float(single[0, 0]))


def test_tanh_activation_matches_fixture():
    net = [fp.make_layer([[1.0, 2.0, -1.0]], [0.5], "tanh")]
    out = fp.forward(net, [0.5, -0.5, 1.0])
    assert float(out[0, 0]) == pytest.approx(EXPECTED["tanh_out"])


def test_relu_clips_negatives():
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert fp.relu(z).tolist() == [0.0, 0.0, 0.0, 0.5, 2.0]


def test_sigmoid_is_numerically_stable():
    z = np.array([-1000.0, 0.0, 1000.0])
    out = fp.sigmoid(z)
    assert out.tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert np.all(np.isfinite(out))


def test_identity_returns_input():
    z = np.array([[-1.0, 2.0], [3.0, -4.0]])
    assert fp.identity(z).tolist() == z.tolist()


def test_apply_activation_unknown_raises():
    with pytest.raises(ValueError, match="unknown activation"):
        fp.apply_activation("gelu", np.array([0.0]))


def test_identity_layers_compose_to_affine():
    # Two identity layers must equal one affine map W2 @ (W1 x + b1) + b2.
    layer1 = fp.make_layer([[1.0, 2.0], [0.0, 1.0]], [1.0, -1.0], "identity")
    layer2 = fp.make_layer([[2.0, 0.0]], [0.5], "identity")
    x = np.array([3.0, 4.0])
    out = fp.forward([layer1, layer2], x)
    W1m = np.array([[1.0, 2.0], [0.0, 1.0]])
    b1v = np.array([1.0, -1.0])
    W2m = np.array([[2.0, 0.0]])
    b2v = np.array([0.5])
    expected = W2m @ (W1m @ x + b1v) + b2v
    assert out.ravel().tolist() == pytest.approx(expected.tolist())


def test_make_layer_bias_length_mismatch_raises():
    with pytest.raises(ValueError, match="bias length"):
        fp.make_layer([[1.0, 2.0]], [0.0, 0.0], "relu")


def test_make_layer_non_2d_weight_raises():
    with pytest.raises(ValueError, match="2-D matrix"):
        fp.make_layer([1.0, 2.0], [0.0], "relu")


def test_make_layer_unknown_activation_raises():
    with pytest.raises(ValueError, match="unknown activation"):
        fp.make_layer([[1.0]], [0.0], "softmax")


def test_make_layer_non_finite_raises():
    with pytest.raises(ValueError, match="non-finite"):
        fp.make_layer([[1.0, float("nan")]], [0.0], "relu")


def test_forward_empty_network_raises():
    with pytest.raises(ValueError, match="at least one layer"):
        fp.forward([], [1.0])


def test_forward_layer_shape_mismatch_raises():
    net = _net()
    with pytest.raises(ValueError, match="incompatible|expects"):
        # second layer expects 2 inputs; feed it a 3-feature activation
        fp.forward_layer(net[1], np.zeros((3, 1)))


def test_forward_incompatible_layers_raise():
    bad = [fp.make_layer([[1.0, 1.0]], [0.0], "relu"), fp.make_layer([[1.0, 1.0]], [0.0], "relu")]
    with pytest.raises(ValueError, match="expects"):
        fp.forward(bad, [1.0, 1.0])


def test_forward_input_feature_mismatch_raises():
    with pytest.raises(ValueError, match="first layer expects"):
        fp.forward(_net(), [1.0, 2.0, 3.0])


def test_forward_non_finite_input_raises():
    with pytest.raises(ValueError, match="non-finite"):
        fp.forward(_net(), [1.0, float("inf")])
