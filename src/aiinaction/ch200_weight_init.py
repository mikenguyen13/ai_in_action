"""Xavier (Glorot) and He (Kaiming) weight initialization from scratch.

A small, well-validated reference implementation of the two canonical
variance-preserving weight-initialization schemes. The public API mirrors the
Julia (`AIInAction.Ch200WeightInit`) and Rust (`aiinaction::ch200_weight_init`)
implementations one-to-one; the cross-language parity tests assert that all three
agree to within floating-point tolerance on shared fixtures.

The core idea (derived in the chapter) is that a layer's weight variance must
scale inversely with its fan to keep the variance of activations stable on the
forward pass and the variance of gradients stable on the backward pass:

* **Xavier / Glorot** targets symmetric activations (``tanh``) and compromises
  between fan-in and fan-out with ``Var(W) = gain^2 * 2 / (fan_in + fan_out)``.
* **He / Kaiming** targets the rectifier family and uses
  ``Var(W) = gain^2 / fan`` with ``gain = sqrt(2)`` for ReLU, compensating for
  the half of the signal that ReLU discards.

Two helpers return the theoretical scale (gaussian standard deviation and the
half-width of the matching uniform support) for any fan/gain. Two more *sample*
an actual weight matrix. Sampling uses a self-contained, deterministic SplitMix64
PRNG plus the Box-Muller transform so that, for a fixed seed, the Python, Julia,
and Rust implementations emit the *identical* weight matrix bit-for-bit (to
floating-point tolerance), not merely matching summary statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "InitScale",
    "calculate_gain",
    "xavier_scale",
    "he_scale",
    "xavier_normal",
    "xavier_uniform",
    "he_normal",
    "he_uniform",
]

# 64-bit masks for the SplitMix64 generator.
_MASK64 = (1 << 64) - 1
_INV_2_53 = 1.0 / float(1 << 53)


@dataclass(frozen=True)
class InitScale:
    """The theoretical spread of an initialization scheme.

    Attributes
    ----------
    std:
        Standard deviation of the matching zero-mean Gaussian, ``sqrt(Var(W))``.
    bound:
        Half-width ``r`` of the matching uniform support ``U(-r, r)``, which has
        the same variance because ``Var(U(-r, r)) = r^2 / 3``; hence
        ``r = std * sqrt(3)``.
    """

    std: float
    bound: float


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    """Recommended gain ``g`` for a nonlinearity, matching PyTorch conventions.

    The weight variance is ``g^2 / fan`` (He) or ``g^2 * 2 / (fan_in + fan_out)``
    (Xavier). Supported names: ``linear``, ``sigmoid``, ``tanh``, ``relu``,
    ``leaky_relu`` (uses ``param`` as the negative slope, default ``0.01``) and
    ``selu``.

    >>> calculate_gain("relu")
    1.4142135623730951
    >>> calculate_gain("tanh")
    1.6666666666666667
    """
    name = nonlinearity.lower()
    if name in ("linear", "conv1d", "conv2d", "conv3d", "sigmoid"):
        return 1.0
    if name == "tanh":
        return 5.0 / 3.0
    if name == "relu":
        return math.sqrt(2.0)
    if name == "leaky_relu":
        slope = 0.01 if param is None else float(param)
        if slope <= -1.0:
            raise ValueError(f"leaky_relu negative slope must be > -1, got {slope}")
        return math.sqrt(2.0 / (1.0 + slope * slope))
    if name == "selu":
        return 3.0 / 4.0
    raise ValueError(f"unsupported nonlinearity: {nonlinearity!r}")


def _check_fan(fan_in: int, fan_out: int) -> None:
    if not isinstance(fan_in, (int, np.integer)) or not isinstance(fan_out, (int, np.integer)):
        raise ValueError("fan_in and fan_out must be integers")
    if fan_in < 1:
        raise ValueError(f"fan_in must be a positive integer, got {fan_in}")
    if fan_out < 1:
        raise ValueError(f"fan_out must be a positive integer, got {fan_out}")


def xavier_scale(fan_in: int, fan_out: int, gain: float = 1.0) -> InitScale:
    """Xavier/Glorot scale: ``std = gain * sqrt(2 / (fan_in + fan_out))``.

    >>> s = xavier_scale(4, 6)
    >>> round(s.std, 6), round(s.bound, 6)
    (0.447214, 0.774597)
    """
    _check_fan(fan_in, fan_out)
    if gain <= 0.0:
        raise ValueError(f"gain must be positive, got {gain}")
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return InitScale(std=std, bound=std * math.sqrt(3.0))


def he_scale(fan: int, gain: float = math.sqrt(2.0)) -> InitScale:
    """He/Kaiming scale: ``std = gain / sqrt(fan)``.

    ``fan`` is the chosen mode: ``fan_in`` (forward, the default in practice) or
    ``fan_out`` (backward). The default ``gain = sqrt(2)`` is the ReLU value.

    >>> s = he_scale(8)
    >>> round(s.std, 6), round(s.bound, 6)
    (0.5, 0.866025)
    """
    if not isinstance(fan, (int, np.integer)):
        raise ValueError("fan must be an integer")
    if fan < 1:
        raise ValueError(f"fan must be a positive integer, got {fan}")
    if gain <= 0.0:
        raise ValueError(f"gain must be positive, got {gain}")
    std = gain / math.sqrt(fan)
    return InitScale(std=std, bound=std * math.sqrt(3.0))


class _SplitMix64:
    """Deterministic 64-bit SplitMix64 generator producing doubles in [0, 1).

    Reproduced identically in Julia and Rust so seeded weight matrices match
    bit-for-bit across languages. ``next_u64`` is the standard SplitMix64 step;
    ``next_double`` takes the top 53 bits to form a uniform double in [0, 1).
    """

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & _MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (z ^ (z >> 31)) & _MASK64

    def next_double(self) -> float:
        # Top 53 bits give a uniform double in [0, 1).
        return (self.next_u64() >> 11) * _INV_2_53


def _next_normal(rng: _SplitMix64) -> float:
    """One standard normal via the basic Box-Muller transform.

    Guards against ``u1 == 0`` (which would make ``log`` diverge) by nudging it to
    the smallest representable positive double, matching the other languages.
    """
    u1 = rng.next_double()
    u2 = rng.next_double()
    if u1 <= 0.0:
        u1 = 5e-324  # smallest positive subnormal double
    r = math.sqrt(-2.0 * math.log(u1))
    return r * math.cos(2.0 * math.pi * u2)


def _shape(fan_in: int, fan_out: int) -> tuple[int, int]:
    """Weight matrices are returned ``(fan_out, fan_in)`` (rows = output units)."""
    return (fan_out, fan_in)


def _fill_normal(fan_in: int, fan_out: int, std: float, seed: int) -> NDArray[np.float64]:
    rng = _SplitMix64(seed)
    rows, cols = _shape(fan_in, fan_out)
    out = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = _next_normal(rng) * std
    return out


def _fill_uniform(fan_in: int, fan_out: int, bound: float, seed: int) -> NDArray[np.float64]:
    rng = _SplitMix64(seed)
    rows, cols = _shape(fan_in, fan_out)
    out = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            # Map [0, 1) to [-bound, bound).
            out[i, j] = (rng.next_double() * 2.0 - 1.0) * bound
    return out


def xavier_normal(fan_in: int, fan_out: int, *, gain: float = 1.0, seed: int = 0) -> NDArray[np.float64]:
    """Sample a ``(fan_out, fan_in)`` weight matrix from Xavier normal.

    Entries are i.i.d. ``N(0, std^2)`` with ``std = gain * sqrt(2/(fan_in+fan_out))``.
    The ``seed`` drives the deterministic cross-language PRNG.
    """
    _check_fan(fan_in, fan_out)
    if gain <= 0.0:
        raise ValueError(f"gain must be positive, got {gain}")
    return _fill_normal(fan_in, fan_out, xavier_scale(fan_in, fan_out, gain).std, seed)


def xavier_uniform(fan_in: int, fan_out: int, *, gain: float = 1.0, seed: int = 0) -> NDArray[np.float64]:
    """Sample a ``(fan_out, fan_in)`` weight matrix from Xavier uniform ``U(-r, r)``.

    ``r = gain * sqrt(6/(fan_in+fan_out))``.
    """
    _check_fan(fan_in, fan_out)
    if gain <= 0.0:
        raise ValueError(f"gain must be positive, got {gain}")
    return _fill_uniform(fan_in, fan_out, xavier_scale(fan_in, fan_out, gain).bound, seed)


def he_normal(fan_in: int, fan_out: int, *, gain: float = math.sqrt(2.0), mode: str = "fan_in", seed: int = 0) -> NDArray[np.float64]:
    """Sample a ``(fan_out, fan_in)`` weight matrix from He normal.

    ``std = gain / sqrt(fan)`` where ``fan`` is ``fan_in`` (``mode='fan_in'``) or
    ``fan_out`` (``mode='fan_out'``). Default gain is the ReLU value ``sqrt(2)``.
    """
    _check_fan(fan_in, fan_out)
    if gain <= 0.0:
        raise ValueError(f"gain must be positive, got {gain}")
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")
    fan = fan_in if mode == "fan_in" else fan_out
    return _fill_normal(fan_in, fan_out, he_scale(fan, gain).std, seed)


def he_uniform(fan_in: int, fan_out: int, *, gain: float = math.sqrt(2.0), mode: str = "fan_in", seed: int = 0) -> NDArray[np.float64]:
    """Sample a ``(fan_out, fan_in)`` weight matrix from He uniform ``U(-r, r)``.

    ``r = gain * sqrt(3 / fan)``.
    """
    _check_fan(fan_in, fan_out)
    if gain <= 0.0:
        raise ValueError(f"gain must be positive, got {gain}")
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")
    fan = fan_in if mode == "fan_in" else fan_out
    return _fill_uniform(fan_in, fan_out, he_scale(fan, gain).bound, seed)
