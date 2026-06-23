"""Inverted dropout with Bernoulli masking, from scratch.

A small, well-validated reference implementation of *inverted dropout*: the
training-time regularizer that multiplies each activation by an independent
Bernoulli(p) mask and rescales the survivors by ``1/p`` so the expectation is
preserved and the test-time forward pass needs no correction.

The public API mirrors the Julia (``AIInAction.Ch204Dropout``) and Rust
(``aiinaction::ch204_dropout``) implementations one-to-one; the cross-language
parity tests assert that all three agree to within floating-point tolerance on
shared fixtures.

The randomness is supplied by a tiny self-contained 64-bit linear congruential
generator (the Numerical Recipes constants). Fixing this generator in all three
languages is what makes the masks reproducible and the parity tests meaningful:
given the same seed and length, the same units are dropped everywhere.

Sign/scale convention: ``p`` is the *retention* probability, so ``1 - p`` is the
drop probability. A retained unit is scaled by ``1/p``; a dropped unit becomes
exactly ``0``. With this convention ``E[mask_i] = 1`` and ``E[mask_i * h_i] =
h_i``, which is the defining property of inverted dropout.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Lcg",
    "bernoulli_mask",
    "inverted_dropout",
    "expected_scale",
]

# Numerical Recipes LCG constants for a full-period 64-bit generator.
_LCG_A = 6364136223846793005
_LCG_C = 1442695040888963407
_LCG_MOD = 1 << 64
# Divisor mapping a 53-bit mantissa to a uniform double in [0, 1).
_UNIT = float(1 << 53)


class Lcg:
    """A minimal, fully reproducible 64-bit linear congruential generator.

    The recurrence is ``state <- (a * state + c) mod 2**64`` with the Numerical
    Recipes constants. Each :meth:`next_uniform` call advances the state once and
    returns a double in ``[0, 1)`` formed from the top 53 bits, matching the Rust
    and Julia implementations bit for bit.
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
        # Use the top 53 bits so every output is an exact multiple of 2**-53.
        top = self.state >> 11
        return top / _UNIT


def expected_scale(p: float) -> float:
    """Return ``1 / p``, the survivor scaling that keeps the mask mean at 1.

    Raises ``ValueError`` unless ``0 < p <= 1``.
    """
    p = float(p)
    if not (0.0 < p <= 1.0):
        raise ValueError(f"retention probability p must satisfy 0 < p <= 1, got {p}")
    return 1.0 / p


def bernoulli_mask(n: int, p: float, seed: int) -> NDArray[np.float64]:
    """Build a length-``n`` inverted-dropout mask with retention probability ``p``.

    Unit ``i`` is retained when ``u_i < p`` (where ``u_i`` is the ``i``-th uniform
    draw from :class:`Lcg` seeded with ``seed``) and assigned the value ``1/p``;
    otherwise it is dropped and assigned ``0.0``.

    Parameters
    ----------
    n:
        Mask length, must be ``>= 1``.
    p:
        Retention probability, must satisfy ``0 < p <= 1``. ``p == 1`` retains
        every unit and yields an all-ones mask.
    seed:
        Non-negative integer seed for the generator.

    Returns
    -------
    numpy.ndarray
        The mask of shape ``(n,)`` with entries in ``{0, 1/p}``.

    Examples
    --------
    >>> m = bernoulli_mask(4, 1.0, seed=0)
    >>> m.tolist()
    [1.0, 1.0, 1.0, 1.0]
    """
    if not isinstance(n, (int, np.integer)):
        raise ValueError(f"n must be an integer, got {type(n).__name__}")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    scale = expected_scale(p)
    rng = Lcg(seed)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        u = rng.next_uniform()
        out[i] = scale if u < p else 0.0
    return out


def inverted_dropout(
    h: Sequence[float], p: float, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply inverted dropout to the activation vector ``h``.

    Computes ``mask = bernoulli_mask(len(h), p, seed)`` and returns the masked
    activations ``mask * h`` together with the mask itself. Dropped units are
    forced to exactly ``0``; survivors are scaled by ``1/p`` so that, in
    expectation over the mask, the layer transmits the unaltered activation.

    Parameters
    ----------
    h:
        Activation vector of finite values, length ``>= 1``.
    p:
        Retention probability in ``(0, 1]``.
    seed:
        Non-negative integer seed for the generator.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(masked, mask)``, both of shape ``(len(h),)``.

    Examples
    --------
    >>> out, mask = inverted_dropout([2.0, 4.0, 6.0], 1.0, seed=0)
    >>> out.tolist()
    [2.0, 4.0, 6.0]
    """
    arr = np.asarray(h, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"h must be a 1-D vector, got array with {arr.ndim} dimension(s)")
    if arr.shape[0] < 1:
        raise ValueError("h must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("h contains non-finite values (nan or inf)")
    mask = bernoulli_mask(arr.shape[0], p, seed)
    return mask * arr, mask
