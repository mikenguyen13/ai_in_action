"""Feedforward neural network forward propagation from scratch.

A small, well-validated reference implementation of the forward sweep through a
fully connected (dense) feedforward network. The public API mirrors the Julia
(``AIInAction.Ch186ForwardPropagation``) and Rust
(``aiinaction::ch186_forward_propagation``) implementations one-to-one; the
cross-language parity tests assert that all three agree to within floating-point
tolerance on shared fixtures.

The computation follows the standard layer recurrence. With ``a^[0] = X`` (a
``d0 x m`` batch of ``m`` examples stacked as columns), for each layer
``l = 1, ..., L``:

    Z^[l] = W^[l] @ A^[l-1] + b^[l]          (affine map, bias broadcast)
    A^[l] = g^[l](Z^[l])                      (elementwise activation)

and the prediction is ``Y_hat = A^[L]``. Supported activations are ``"relu"``,
``"sigmoid"``, ``"tanh"``, and ``"identity"`` (linear). The sigmoid is evaluated
in a numerically stable, branch-free form so it does not overflow for large
negative pre-activations.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Layer",
    "relu",
    "sigmoid",
    "tanh",
    "identity",
    "apply_activation",
    "ACTIVATIONS",
    "forward_layer",
    "forward",
]

Matrix = Sequence[Sequence[float]]
Vector = Sequence[float]

ACTIVATIONS = ("relu", "sigmoid", "tanh", "identity")


def relu(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rectified linear unit, ``max(0, z)``, applied elementwise."""
    return np.maximum(z, 0.0)


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Logistic sigmoid ``1 / (1 + e^{-z})``, numerically stable and elementwise.

    Uses the identity ``sigmoid(z) = e^{z} / (1 + e^{z})`` for ``z < 0`` so the
    exponential never overflows on large-magnitude negative inputs.
    """
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def tanh(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hyperbolic tangent, applied elementwise."""
    return np.tanh(z)


def identity(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Identity (linear) activation, returns ``z`` unchanged."""
    return z


_FUNCS = {"relu": relu, "sigmoid": sigmoid, "tanh": tanh, "identity": identity}


def apply_activation(name: str, z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dispatch to the named activation, raising ``ValueError`` on unknown names."""
    fn = _FUNCS.get(name)
    if fn is None:
        raise ValueError(
            f"unknown activation {name!r}; expected one of {ACTIVATIONS}"
        )
    return fn(z)


@dataclass(frozen=True)
class Layer:
    """One dense layer: an affine map ``W @ a + b`` plus an activation.

    Attributes
    ----------
    W:
        Weight matrix of shape ``(d_out, d_in)``.
    b:
        Bias vector of shape ``(d_out,)``.
    activation:
        Name of the elementwise activation, one of :data:`ACTIVATIONS`.
    """

    W: NDArray[np.float64]
    b: NDArray[np.float64]
    activation: str

    @property
    def d_in(self) -> int:
        """Input dimensionality of the layer."""
        return int(self.W.shape[1])

    @property
    def d_out(self) -> int:
        """Output dimensionality (number of units) of the layer."""
        return int(self.W.shape[0])


def make_layer(W: Matrix, b: Vector, activation: str) -> Layer:
    """Validate and construct a :class:`Layer` from raw weights, bias and activation.

    Parameters
    ----------
    W:
        Weight matrix, coerced to shape ``(d_out, d_in)``. Must be 2-D and finite.
    b:
        Bias vector of length ``d_out``. Must be 1-D and finite.
    activation:
        One of :data:`ACTIVATIONS`.

    Raises
    ------
    ValueError
        If ``W`` is not 2-D, ``b`` is not 1-D, their shapes are inconsistent, the
        activation is unknown, or any value is non-finite.
    """
    Wa = np.asarray(W, dtype=np.float64)
    ba = np.asarray(b, dtype=np.float64)
    if Wa.ndim != 2:
        raise ValueError(f"W must be a 2-D matrix, got array with {Wa.ndim} dimension(s)")
    if Wa.shape[0] < 1 or Wa.shape[1] < 1:
        raise ValueError(f"W must have at least one row and column, got shape {Wa.shape}")
    if ba.ndim != 1:
        raise ValueError(f"b must be a 1-D vector, got array with {ba.ndim} dimension(s)")
    if ba.shape[0] != Wa.shape[0]:
        raise ValueError(
            f"bias length {ba.shape[0]} does not match number of units {Wa.shape[0]}"
        )
    if activation not in _FUNCS:
        raise ValueError(
            f"unknown activation {activation!r}; expected one of {ACTIVATIONS}"
        )
    if not np.all(np.isfinite(Wa)):
        raise ValueError("W contains non-finite values (nan or inf)")
    if not np.all(np.isfinite(ba)):
        raise ValueError("b contains non-finite values (nan or inf)")
    return Layer(W=Wa, b=ba, activation=activation)


def _as_batch(X: Matrix | Vector) -> NDArray[np.float64]:
    """Coerce input to a 2-D ``(d, m)`` batch; a 1-D input becomes one column."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"input must be 1-D (single example) or 2-D (d x m batch), "
            f"got array with {arr.ndim} dimension(s)"
        )
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"input must be non-empty, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("input contains non-finite values (nan or inf)")
    return arr


def forward_layer(layer: Layer, A_prev: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagate one batched activation through a single layer.

    Computes ``Z = W @ A_prev + b`` (bias broadcast across columns) followed by the
    layer's activation, returning ``(Z, A)`` where both have shape ``(d_out, m)``.
    The pre-activation ``Z`` is returned alongside ``A`` because it is exactly the
    quantity a backward pass needs to cache.
    """
    if A_prev.shape[0] != layer.d_in:
        raise ValueError(
            f"layer expects {layer.d_in} input features but received {A_prev.shape[0]}"
        )
    Z = layer.W @ A_prev + layer.b.reshape(-1, 1)
    A = apply_activation(layer.activation, Z)
    return Z, A


def forward(layers: Sequence[Layer], X: Matrix | Vector) -> NDArray[np.float64]:
    """Run the full forward sweep through every layer in order.

    Parameters
    ----------
    layers:
        Non-empty sequence of :class:`Layer`. Consecutive layers must be shape
        compatible: ``layers[i].d_out == layers[i + 1].d_in``.
    X:
        Input batch of shape ``(d0, m)`` with examples as columns. A 1-D input of
        length ``d0`` is treated as a single example (``m = 1``).

    Returns
    -------
    numpy.ndarray
        The network output ``A^[L]`` of shape ``(d_L, m)``.

    Examples
    --------
    >>> W1 = [[0.5, -0.2], [0.1, 0.4]]
    >>> b1 = [0.1, -0.3]
    >>> W2 = [[0.7, -0.6]]
    >>> b2 = [0.2]
    >>> net = [make_layer(W1, b1, "relu"), make_layer(W2, b2, "sigmoid")]
    >>> yhat = forward(net, [1.0, 2.0])
    >>> round(float(yhat[0, 0]), 6)
    0.495
    """
    if len(layers) == 0:
        raise ValueError("network must have at least one layer")
    for i in range(len(layers) - 1):
        if layers[i].d_out != layers[i + 1].d_in:
            raise ValueError(
                f"layer {i} outputs {layers[i].d_out} features but layer {i + 1} "
                f"expects {layers[i + 1].d_in}"
            )
    A = _as_batch(X)
    if A.shape[0] != layers[0].d_in:
        raise ValueError(
            f"input has {A.shape[0]} features but first layer expects {layers[0].d_in}"
        )
    for layer in layers:
        _, A = forward_layer(layer, A)
    return A
