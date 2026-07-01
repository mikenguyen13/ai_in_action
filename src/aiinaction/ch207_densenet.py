"""Dense connectivity (DenseNet) mechanics, from scratch.

A small, well-validated reference implementation of the arithmetic and forward
pass that make a *dense block* work: the concatenative connectivity pattern
where layer :math:`\\ell` consumes the feature maps of every preceding layer,
:math:`x_\\ell = H_\\ell([x_0, x_1, \\ldots, x_{\\ell-1}])`.

The public API mirrors the Julia (``AIInAction.Ch207Densenet``) and Rust
(``aiinaction::ch207_densenet``) implementations one-to-one; the cross-language
parity tests assert that all three agree exactly (channel/parameter counts,
which are integer arithmetic) or to within floating-point tolerance (the toy
forward pass) on shared fixtures.

Real dense blocks use batch normalization, a nonlinearity, and a convolution
for each :math:`H_\\ell`. This module keeps the connectivity pattern exact but
stands in for the convolution with a single seeded linear-plus-ReLU layer
acting on a flattened feature vector, so the forward pass is honest about
*how features accumulate through concatenation* without requiring a full
convolution/tensor stack. Weight initialization uses the same tiny 64-bit
linear congruential generator (LCG) used elsewhere in this book, so the
Python, Julia, and Rust implementations produce bit-identical layers given the
same seed.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Lcg",
    "dense_block_channel_sizes",
    "dense_block_param_count",
    "transition_output_channels",
    "plain_block_param_count",
    "init_layer_weights",
    "dense_layer_forward",
    "dense_block_forward",
    "DENSENET_VARIANTS",
    "densenet_dense_param_total",
]

# Numerical Recipes LCG constants for a full-period 64-bit generator.
_LCG_A = 6364136223846793005
_LCG_C = 1442695040888963407
_LCG_MOD = 1 << 64
_UNIT = float(1 << 53)


class Lcg:
    """A minimal, fully reproducible 64-bit linear congruential generator.

    The recurrence is ``state <- (a * state + c) mod 2**64`` with the Numerical
    Recipes constants. Each :meth:`next_uniform` call advances the state once
    and returns a double in ``[0, 1)`` formed from the top 53 bits, matching
    the Rust and Julia implementations bit for bit.
    """

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        if not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be an integer, got {type(seed).__name__}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        self.state = int(seed) % _LCG_MOD

    def next_uniform(self) -> float:
        """Advance the generator and return the next uniform draw in ``[0, 1)``."""
        self.state = (_LCG_A * self.state + _LCG_C) % _LCG_MOD
        top = self.state >> 11
        return top / _UNIT


# --------------------------------------------------------------------------
# Channel and parameter arithmetic
# --------------------------------------------------------------------------

def dense_block_channel_sizes(c0: int, growth_rate: int, num_layers: int) -> list[int]:
    """Return the input-channel count seen by each layer of a dense block.

    Layer ``l`` (0-indexed) receives the concatenation of the block input and
    every earlier layer's output, so it sees ``c0 + growth_rate * l`` channels.
    The final entry, appended after the loop, is the block's total output
    width ``c0 + growth_rate * num_layers``.

    Parameters
    ----------
    c0:
        Number of channels entering the block, must be ``>= 0``.
    growth_rate:
        Number of new feature maps ``k`` each layer contributes, must be ``>= 1``.
    num_layers:
        Number of layers in the block, must be ``>= 1``.

    Returns
    -------
    list[int]
        Length ``num_layers + 1``: input width of each layer followed by the
        block's total output width.

    Examples
    --------
    >>> dense_block_channel_sizes(c0=4, growth_rate=4, num_layers=3)
    [4, 8, 12, 16]
    """
    if c0 < 0:
        raise ValueError(f"c0 must be >= 0, got {c0}")
    if growth_rate < 1:
        raise ValueError(f"growth_rate must be >= 1, got {growth_rate}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    return [c0 + growth_rate * l for l in range(num_layers + 1)]


def dense_block_param_count(
    c0: int, growth_rate: int, num_layers: int, bn_size: int = 4
) -> int:
    """Count parameters in a DenseNet-BC style dense block.

    Each layer applies a bottleneck: a ``1x1`` convolution mapping the
    accumulated input to ``bn_size * growth_rate`` channels, followed by a
    ``3x3`` convolution mapping that down to ``growth_rate`` new channels
    (bias terms omitted, matching convolutions immediately followed by batch
    normalization). For layer ``l`` with input width ``c_l = c0 + growth_rate*l``,

    .. math::

        \\text{params}_l = \\underbrace{c_l \\cdot (b k)}_{1\\times 1}
        + \\underbrace{(bk) \\cdot k \\cdot 9}_{3\\times 3}

    where ``b = bn_size`` and ``k = growth_rate``. The block total is the sum
    over ``l = 0, ..., num_layers - 1``.

    Parameters
    ----------
    c0, growth_rate, num_layers:
        As in :func:`dense_block_channel_sizes`.
    bn_size:
        Bottleneck width multiplier ``b``, must be ``>= 1``. DenseNet-BC uses ``4``.

    Returns
    -------
    int
        Total weight count (excluding biases and batch-norm parameters).

    Examples
    --------
    >>> dense_block_param_count(c0=4, growth_rate=4, num_layers=3, bn_size=4)
    2112
    """
    if bn_size < 1:
        raise ValueError(f"bn_size must be >= 1, got {bn_size}")
    sizes = dense_block_channel_sizes(c0, growth_rate, num_layers)
    k = growth_rate
    total = 0
    for c_l in sizes[:-1]:
        bottleneck_width = bn_size * k
        total += c_l * bottleneck_width          # 1x1 conv
        total += bottleneck_width * k * 9         # 3x3 conv
    return total


def transition_output_channels(c_in: int, theta: float) -> int:
    """Return the compressed channel count after a transition layer.

    A DenseNet transition layer applies a ``1x1`` convolution that reduces the
    channel count by a compression factor ``theta`` in ``(0, 1]``:
    ``floor(theta * c_in)``.

    Examples
    --------
    >>> transition_output_channels(64, 0.5)
    32
    """
    if not (0.0 < theta <= 1.0):
        raise ValueError(f"theta must satisfy 0 < theta <= 1, got {theta}")
    if c_in < 1:
        raise ValueError(f"c_in must be >= 1, got {c_in}")
    return int(math.floor(theta * c_in))


def plain_block_param_count(c0: int, width: int, num_layers: int) -> int:
    """Count parameters of a plain (non-dense) stack matched on output width.

    The first layer is an ordinary ``3x3`` convolution mapping the block's
    ``c0`` input channels to a fixed ``width``; every subsequent layer maps
    ``width`` to ``width``. This is the additive, ResNet-style alternative
    that does not grow its channel count through concatenation, and it gives
    the parameter-cost baseline used to quantify dense connectivity's
    efficiency in Section 4.2.

    Examples
    --------
    >>> plain_block_param_count(c0=16, width=16, num_layers=3)
    6912
    """
    if c0 < 1:
        raise ValueError(f"c0 must be >= 1, got {c0}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    return c0 * width * 9 + (num_layers - 1) * (width * width * 9)


# --------------------------------------------------------------------------
# Toy forward pass demonstrating the concatenation mechanic
# --------------------------------------------------------------------------

def init_layer_weights(
    c_in: int, growth_rate: int, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Deterministically initialize one dense layer's weight matrix and bias.

    Draws every entry from :class:`Lcg` seeded with ``seed``, mapped to a
    symmetric interval ``[-lim, lim]`` with ``lim = sqrt(6 / (c_in + k))``
    (a Glorot-style bound), so results are reproducible bit for bit across
    languages given the same seed.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Weight matrix of shape ``(growth_rate, c_in)`` and bias of shape
        ``(growth_rate,)``.
    """
    if c_in < 1:
        raise ValueError(f"c_in must be >= 1, got {c_in}")
    if growth_rate < 1:
        raise ValueError(f"growth_rate must be >= 1, got {growth_rate}")
    lim = math.sqrt(6.0 / (c_in + growth_rate))
    rng = Lcg(seed)
    w = np.empty((growth_rate, c_in), dtype=np.float64)
    for i in range(growth_rate):
        for j in range(c_in):
            u = rng.next_uniform()
            w[i, j] = (2.0 * u - 1.0) * lim
    b = np.zeros(growth_rate, dtype=np.float64)
    return w, b


def dense_layer_forward(
    x: NDArray[np.float64], w: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply one dense-block layer ``H_l``: ``relu(W @ x + b)``.

    Stands in for the batch-norm/ReLU/convolution composite that a real dense
    block applies at each step.
    """
    z = w @ x + b
    return np.maximum(z, 0.0)


def dense_block_forward(
    x0: Sequence[float], growth_rate: int, num_layers: int, seed: int
) -> tuple[NDArray[np.float64], list[int]]:
    """Run a toy dense block forward pass, concatenating each layer's output.

    Implements :math:`x_\\ell = H_\\ell([x_0, \\ldots, x_{\\ell-1}])` literally:
    each layer sees the running concatenation of the block input and every
    prior layer's output, and contributes exactly ``growth_rate`` new entries.
    Layer ``l``'s weights come from :func:`init_layer_weights` seeded with
    ``seed + l``, so distinct layers get distinct, reproducible weights.

    Parameters
    ----------
    x0:
        Block input vector, standing in for ``c0`` globally-pooled channels.
    growth_rate, num_layers:
        As in :func:`dense_block_channel_sizes`.
    seed:
        Base seed; layer ``l`` uses ``seed + l``.

    Returns
    -------
    (numpy.ndarray, list[int])
        The final concatenated feature vector (length
        ``len(x0) + growth_rate * num_layers``) and the channel-size trace
        from :func:`dense_block_channel_sizes`, which must match the observed
        vector length at every step.

    Examples
    --------
    >>> out, sizes = dense_block_forward([1.0, -1.0, 0.5, 0.5], growth_rate=2, num_layers=2, seed=0)
    >>> sizes
    [4, 6, 8]
    >>> len(out) == sizes[-1]
    True
    """
    x = np.asarray(x0, dtype=np.float64)
    if x.ndim != 1 or x.shape[0] < 1:
        raise ValueError("x0 must be a non-empty 1-D vector")
    if not np.all(np.isfinite(x)):
        raise ValueError("x0 contains non-finite values (nan or inf)")
    c0 = x.shape[0]
    sizes = dense_block_channel_sizes(c0, growth_rate, num_layers)
    features = x
    for l in range(num_layers):
        w, b = init_layer_weights(features.shape[0], growth_rate, seed + l)
        new_features = dense_layer_forward(features, w, b)
        features = np.concatenate([features, new_features])
        assert features.shape[0] == sizes[l + 1]
    return features, sizes


# --------------------------------------------------------------------------
# DenseNet-BC architecture variants (Huang et al. 2017, Table 1)
# --------------------------------------------------------------------------

# Block configs (number of dense layers per block) for the four ImageNet
# variants, growth rate k=32, initial channels k0=64, compression theta=0.5.
DENSENET_VARIANTS: dict[str, tuple[int, int, int, int]] = {
    "121": (6, 12, 24, 16),
    "169": (6, 12, 32, 32),
    "201": (6, 12, 48, 32),
    "264": (6, 12, 64, 48),
}


def densenet_dense_param_total(
    variant: str, growth_rate: int = 32, k0: int = 64, theta: float = 0.5, bn_size: int = 4
) -> int:
    """Sum dense-block and transition-layer parameters for a DenseNet-BC variant.

    This totals only the dense blocks (via :func:`dense_block_param_count`) and
    the ``1x1`` transition convolutions between them (via
    :func:`transition_output_channels`); it deliberately excludes the initial
    ``7x7`` stem convolution, the final batch norm, and the classifier head,
    so it will be somewhat smaller than the full parameter counts Huang et al.
    report in their Table 1. It isolates the part of the network whose size
    is governed by the dense-connectivity arithmetic this module develops.

    Parameters
    ----------
    variant:
        One of ``"121"``, ``"169"``, ``"201"``, ``"264"`` (keys of
        :data:`DENSENET_VARIANTS`).

    Examples
    --------
    >>> densenet_dense_param_total("121") > 0
    True
    """
    if variant not in DENSENET_VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {sorted(DENSENET_VARIANTS)}"
        )
    blocks = DENSENET_VARIANTS[variant]
    c = k0
    total = 0
    for i, num_layers in enumerate(blocks):
        total += dense_block_param_count(c, growth_rate, num_layers, bn_size=bn_size)
        c = c + growth_rate * num_layers
        if i < len(blocks) - 1:  # no transition after the last block
            c_out = transition_output_channels(c, theta)
            total += c * c_out  # 1x1 transition conv
            c = c_out
    return total
