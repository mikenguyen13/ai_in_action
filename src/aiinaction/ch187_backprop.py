"""Backpropagation from scratch for a feedforward neural network.

A small, well-validated reference implementation of the backpropagation algorithm
derived in the accompanying chapter. The public API mirrors the Julia
(``AIInAction.Ch187Backprop``) and Rust (``aiinaction::ch187_backprop``)
implementations one-to-one; the cross-language parity tests assert that all three
agree to within floating-point tolerance on shared fixtures.

The network is a plain multilayer perceptron. Each hidden layer applies the
logistic sigmoid ``sigma(z) = 1 / (1 + exp(-z))`` elementwise; the output layer is
*linear* (identity activation) and the loss is one-half squared error
``C = 1/2 * ||a^L - y||^2``. This is the cleanest setting in which to exhibit the
four backpropagation equations (BP1 through BP4) without the softmax/cross-entropy
cancellation muddying the activation-derivative factor.

Conventions (identical to the chapter):

* Layer ``l`` has weight matrix ``W^l`` of shape ``(n_l, n_{l-1})`` and bias
  ``b^l`` of length ``n_l``. The forward pass is ``z^l = W^l a^{l-1} + b^l`` and
  ``a^l = sigma(z^l)`` for hidden layers, ``a^L = z^L`` for the (linear) output.
* ``delta^l = dC/dz^l`` is the per-unit error signal.
* BP1: ``delta^L = (a^L - y)`` (linear output, squared-error loss).
* BP2: ``delta^l = (W^{l+1})^T delta^{l+1}) .* sigma'(z^l)``.
* BP3: ``dC/db^l = delta^l``.
* BP4: ``dC/dW^l = delta^l (a^{l-1})^T``.

Weights and biases are stored as plain (nested) Python lists wrapped in a frozen
:class:`MLP` dataclass so the structure is trivial to mirror in Rust and Julia.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MLP",
    "make_mlp",
    "sigmoid",
    "sigmoid_prime",
    "forward",
    "squared_error_loss",
    "backprop",
    "numerical_gradient",
]

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


@dataclass(frozen=True)
class MLP:
    """A feedforward network with sigmoid hidden layers and a linear output.

    Attributes
    ----------
    weights:
        One weight matrix per layer. ``weights[l]`` has shape ``(n_{l+1}, n_l)``
        mapping layer ``l`` activations to layer ``l+1`` pre-activations.
    biases:
        One bias vector per layer; ``biases[l]`` has length ``n_{l+1}``.
    """

    weights: list[NDArray[np.float64]]
    biases: list[NDArray[np.float64]]

    @property
    def num_layers(self) -> int:
        """Number of weight layers ``L`` (input layer not counted)."""
        return len(self.weights)

    @property
    def n_input(self) -> int:
        """Dimensionality of the input feature vector."""
        return int(self.weights[0].shape[1])

    @property
    def n_output(self) -> int:
        """Dimensionality of the network output."""
        return int(self.weights[-1].shape[0])


def _build(weights: Sequence[Matrix], biases: Sequence[Vector]) -> MLP:
    if len(weights) != len(biases):
        raise ValueError(
            f"weights and biases must have equal length, got {len(weights)} and {len(biases)}"
        )
    if not weights:
        raise ValueError("network must have at least one layer")

    W: list[NDArray[np.float64]] = []
    b: list[NDArray[np.float64]] = []
    prev_cols: int | None = None
    for l, (wl, bl) in enumerate(zip(weights, biases)):
        wm = np.asarray(wl, dtype=np.float64)
        bv = np.asarray(bl, dtype=np.float64)
        if wm.ndim != 2:
            raise ValueError(f"weights[{l}] must be a 2-D matrix, got {wm.ndim} dimension(s)")
        if bv.ndim != 1:
            raise ValueError(f"biases[{l}] must be a 1-D vector, got {bv.ndim} dimension(s)")
        if wm.shape[0] != bv.shape[0]:
            raise ValueError(
                f"layer {l}: weights has {wm.shape[0]} rows but bias has length {bv.shape[0]}"
            )
        if prev_cols is not None and wm.shape[1] != prev_cols:
            raise ValueError(
                f"layer {l}: weights expects {wm.shape[1]} inputs but previous layer emits {prev_cols}"
            )
        if not np.all(np.isfinite(wm)) or not np.all(np.isfinite(bv)):
            raise ValueError(f"layer {l}: parameters contain non-finite values (nan or inf)")
        prev_cols = wm.shape[0]
        W.append(wm)
        b.append(bv)
    return MLP(weights=W, biases=b)


def make_mlp(weights: Sequence[Matrix], biases: Sequence[Vector]) -> MLP:
    """Validate and construct an :class:`MLP` from raw weight/bias lists.

    Examples
    --------
    >>> net = make_mlp([[[1.0, 0.0], [0.0, 1.0]]], [[0.0, 0.0]])
    >>> net.n_input, net.n_output
    (2, 2)
    """
    return _build(weights, biases)


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable logistic sigmoid applied elementwise."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_prime(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Derivative ``sigma'(z) = sigma(z) (1 - sigma(z))``."""
    s = sigmoid(z)
    return s * (1.0 - s)


def _as_vector(x: Vector, name: str) -> NDArray[np.float64]:
    v = np.asarray(x, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got {v.ndim} dimension(s)")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} contains non-finite values (nan or inf)")
    return v


def forward(net: MLP, x: Vector) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Run the forward pass, caching every pre-activation and activation.

    Returns ``(zs, activations)`` where ``zs[l]`` is the pre-activation ``z^{l+1}``
    and ``activations[l]`` the activation ``a^l``. ``activations[0]`` is the input
    ``x`` and ``activations[-1]`` the (linearly activated) network output. Hidden
    layers use the sigmoid; the final layer is the identity.
    """
    a = _as_vector(x, "x")
    if a.shape[0] != net.n_input:
        raise ValueError(f"x has length {a.shape[0]} but network expects {net.n_input}")
    activations: list[NDArray[np.float64]] = [a]
    zs: list[NDArray[np.float64]] = []
    L = net.num_layers
    for l in range(L):
        z = net.weights[l] @ a + net.biases[l]
        zs.append(z)
        a = z if l == L - 1 else sigmoid(z)
        activations.append(a)
    return zs, activations


def squared_error_loss(net: MLP, x: Vector, y: Vector) -> float:
    """One-half squared-error loss ``1/2 ||a^L - y||^2`` for one example."""
    yv = _as_vector(y, "y")
    if yv.shape[0] != net.n_output:
        raise ValueError(f"y has length {yv.shape[0]} but network outputs {net.n_output}")
    _, activations = forward(net, x)
    diff = activations[-1] - yv
    return 0.5 * float(diff @ diff)


def backprop(
    net: MLP, x: Vector, y: Vector
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Compute parameter gradients for one example via backpropagation.

    Returns ``(grad_W, grad_b)`` with the same shapes as ``net.weights`` and
    ``net.biases``: ``grad_W[l] = dC/dW^{l+1}`` and ``grad_b[l] = dC/db^{l+1}``.

    Implements the four backpropagation equations directly. The output layer is
    linear so ``sigma'(z^L) = 1`` and BP1 reduces to ``delta^L = a^L - y``.

    Examples
    --------
    >>> net = make_mlp([[[1.0, 0.0], [0.0, 1.0]]], [[0.0, 0.0]])
    >>> gW, gb = backprop(net, [1.0, 2.0], [0.0, 0.0])
    >>> gb[0].tolist()
    [1.0, 2.0]
    """
    yv = _as_vector(y, "y")
    if yv.shape[0] != net.n_output:
        raise ValueError(f"y has length {yv.shape[0]} but network outputs {net.n_output}")

    zs, activations = forward(net, x)
    L = net.num_layers
    grad_W: list[NDArray[np.float64]] = [np.zeros_like(w) for w in net.weights]
    grad_b: list[NDArray[np.float64]] = [np.zeros_like(b) for b in net.biases]

    # BP1: output layer is linear, so delta^L = a^L - y.
    delta = activations[-1] - yv
    grad_b[L - 1] = delta                                  # BP3
    grad_W[L - 1] = np.outer(delta, activations[L - 1])    # BP4

    # BP2: propagate backward through the hidden (sigmoid) layers.
    for l in range(L - 2, -1, -1):
        sp = sigmoid_prime(zs[l])
        delta = (net.weights[l + 1].T @ delta) * sp
        grad_b[l] = delta                                  # BP3
        grad_W[l] = np.outer(delta, activations[l])        # BP4

    return grad_W, grad_b


def numerical_gradient(
    net: MLP, x: Vector, y: Vector, eps: float = 1e-6
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Central-difference estimate of the loss gradient, for gradient checking.

    Perturbs each parameter by ``+/- eps`` and forms ``(C+ - C-) / (2 eps)``.
    Returns ``(grad_W, grad_b)`` matching the shapes of :func:`backprop`. This is
    ``O(P)`` forward passes for ``P`` parameters and is intended only to validate
    the analytic gradients, never for training.
    """
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    grad_W: list[NDArray[np.float64]] = [np.zeros_like(w) for w in net.weights]
    grad_b: list[NDArray[np.float64]] = [np.zeros_like(b) for b in net.biases]

    for l in range(net.num_layers):
        w = net.weights[l]
        for j in range(w.shape[0]):
            for k in range(w.shape[1]):
                orig = w[j, k]
                w[j, k] = orig + eps
                cp = squared_error_loss(net, x, y)
                w[j, k] = orig - eps
                cm = squared_error_loss(net, x, y)
                w[j, k] = orig
                grad_W[l][j, k] = (cp - cm) / (2.0 * eps)
        b = net.biases[l]
        for j in range(b.shape[0]):
            orig = b[j]
            b[j] = orig + eps
            cp = squared_error_loss(net, x, y)
            b[j] = orig - eps
            cm = squared_error_loss(net, x, y)
            b[j] = orig
            grad_b[l][j] = (cp - cm) / (2.0 * eps)

    return grad_W, grad_b
