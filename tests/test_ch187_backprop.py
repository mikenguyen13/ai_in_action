"""Tests for aiinaction.ch187_backprop, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. The network is a fixed two-layer MLP (sigmoid hidden layer, linear output)
with the squared-error loss, evaluated on one explicit example.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch187_backprop as bp

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
W1 = [[0.10, 0.20, -0.30], [0.40, -0.50, 0.60]]
B1 = [0.10, -0.20]
W2 = [[0.70, -0.80], [-0.10, 0.30]]
B2 = [0.05, -0.05]
X = [1.0, -2.0, 0.5]
Y = [0.3, -0.7]

EXPECTED = {
    "z1": [-0.3500000000000001, 1.5],
    "a1": [0.4133824210826699, 0.8175744761936437],
    "output": [-0.3146918861970461, 0.15393410074982605],
    "loss": 0.5535247816899481,
    "grad_W0": [
        -0.12505050629624692,
        0.25010101259249384,
        -0.06252525314812346,
        0.11155166358278021,
        -0.22310332716556042,
        0.055775831791390104,
    ],
    "grad_b0": [-0.12505050629624692, 0.11155166358278021],
    "grad_W1": [
        -0.25410282013600793,
        -0.5025563968780328,
        0.35300134601301564,
        0.6981547251244291,
    ],
    "grad_b1": [-0.6146918861970461, 0.853934100749826],
}


def _net() -> bp.MLP:
    return bp.make_mlp([W1, W2], [B1, B2])


def test_forward_hidden_preactivation():
    zs, _ = bp.forward(_net(), X)
    assert zs[0].tolist() == pytest.approx(EXPECTED["z1"])


def test_forward_hidden_activation():
    _, acts = bp.forward(_net(), X)
    assert acts[1].tolist() == pytest.approx(EXPECTED["a1"])


def test_forward_output_is_linear():
    zs, acts = bp.forward(_net(), X)
    # Output layer is the identity, so a^L == z^L.
    assert acts[-1].tolist() == pytest.approx(EXPECTED["output"])
    assert acts[-1].tolist() == pytest.approx(zs[-1].tolist())


def test_loss_matches_fixture():
    assert bp.squared_error_loss(_net(), X, Y) == pytest.approx(EXPECTED["loss"])


def test_grad_W_layer0_matches_fixture():
    gW, _ = bp.backprop(_net(), X, Y)
    assert gW[0].ravel().tolist() == pytest.approx(EXPECTED["grad_W0"])


def test_grad_b_layer0_matches_fixture():
    _, gb = bp.backprop(_net(), X, Y)
    assert gb[0].tolist() == pytest.approx(EXPECTED["grad_b0"])


def test_grad_W_layer1_matches_fixture():
    gW, _ = bp.backprop(_net(), X, Y)
    assert gW[1].ravel().tolist() == pytest.approx(EXPECTED["grad_W1"])


def test_grad_b_layer1_matches_fixture():
    _, gb = bp.backprop(_net(), X, Y)
    assert gb[1].tolist() == pytest.approx(EXPECTED["grad_b1"])


def test_output_delta_is_residual():
    # BP1 with a linear output: delta^L == a^L - y, which is exactly grad_b at L.
    _, gb = bp.backprop(_net(), X, Y)
    resid = (np.asarray(EXPECTED["output"]) - np.asarray(Y)).tolist()
    assert gb[-1].tolist() == pytest.approx(resid)


def test_analytic_matches_numerical_gradient():
    net = _net()
    gW, gb = bp.backprop(net, X, Y)
    ngW, ngb = bp.numerical_gradient(net, X, Y)
    for a, b in zip(gW, ngW):
        assert a.ravel().tolist() == pytest.approx(b.ravel().tolist(), abs=1e-7)
    for a, b in zip(gb, ngb):
        assert a.tolist() == pytest.approx(b.tolist(), abs=1e-7)


def test_zero_residual_gives_zero_gradient():
    # If the target equals the network output, every gradient is zero.
    net = _net()
    out = bp.forward(net, X)[1][-1]
    gW, gb = bp.backprop(net, X, out)
    for g in gW + gb:
        assert g.ravel().tolist() == pytest.approx([0.0] * g.size, abs=1e-12)


def test_sigmoid_and_prime_values():
    assert float(bp.sigmoid(np.array([0.0]))[0]) == pytest.approx(0.5)
    assert float(bp.sigmoid_prime(np.array([0.0]))[0]) == pytest.approx(0.25)
    # Stable on large-magnitude inputs.
    assert float(bp.sigmoid(np.array([1000.0]))[0]) == pytest.approx(1.0)
    assert float(bp.sigmoid(np.array([-1000.0]))[0]) == pytest.approx(0.0)


def test_mismatched_weights_biases_raises():
    with pytest.raises(ValueError, match="equal length"):
        bp.make_mlp([W1, W2], [B1])


def test_layer_dim_mismatch_raises():
    with pytest.raises(ValueError, match="expects .* inputs"):
        bp.make_mlp([[[1.0, 2.0]], [[1.0, 2.0, 3.0]]], [[0.0], [0.0]])


def test_bias_row_mismatch_raises():
    with pytest.raises(ValueError, match="rows but bias"):
        bp.make_mlp([[[1.0, 2.0], [3.0, 4.0]]], [[0.0]])


def test_bad_input_length_raises():
    with pytest.raises(ValueError, match="x has length"):
        bp.forward(_net(), [1.0, 2.0])


def test_bad_target_length_raises():
    with pytest.raises(ValueError, match="y has length"):
        bp.backprop(_net(), X, [0.0, 0.0, 0.0])


def test_non_finite_param_raises():
    with pytest.raises(ValueError, match="non-finite"):
        bp.make_mlp([[[float("nan"), 0.0]]], [[0.0]])


def test_eps_must_be_positive():
    with pytest.raises(ValueError, match="eps must be positive"):
        bp.numerical_gradient(_net(), X, Y, eps=0.0)
