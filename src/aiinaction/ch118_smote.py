"""SMOTE: Synthetic Minority Over-sampling Technique (Chapter 118).

A small, dependency-light reference implementation of SMOTE with explicit input
validation. SMOTE rebalances an imbalanced training set by synthesizing new
minority-class examples through linear interpolation between a minority point and
one of its ``k`` nearest minority neighbors:

    x_new = x_i + lambda * (x_nn - x_i),   lambda ~ Uniform(0, 1).

This module mirrors the Julia (``AIInAction.Ch118Smote``) and Rust
(``aiinaction::ch118_smote``) implementations one-to-one. To make the three agree
numerically on the shared fixtures, all randomness flows through a fixed
linear-congruential generator (the Numerical Recipes constants) rather than each
language's native RNG. Given the same seed, the three libraries emit the identical
synthetic points to floating-point tolerance.
"""
from __future__ import annotations

from collections.abc import Sequence

__all__ = ["LCG", "euclidean", "k_nearest", "smote_sample", "smote"]

# Numerical Recipes LCG constants (modulus 2**32).
_LCG_A = 1664525
_LCG_C = 1013904223
_LCG_M = 2 ** 32


class LCG:
    """Deterministic linear-congruential generator shared across all three languages.

    The recurrence is ``state = (A * state + C) mod 2**32`` with the Numerical
    Recipes constants. ``next_float`` returns a value in ``[0, 1)`` and
    ``next_index`` returns a uniform integer in ``[0, n)``. Using one explicit RNG
    keeps Python, Julia, and Rust bit-for-bit reproducible.
    """

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        self.state = seed % _LCG_M

    def next_uint(self) -> int:
        """Advance the generator and return the raw 32-bit state."""
        self.state = (_LCG_A * self.state + _LCG_C) % _LCG_M
        return self.state

    def next_float(self) -> float:
        """Return the next pseudo-random float in ``[0, 1)``."""
        return self.next_uint() / _LCG_M

    def next_index(self, n: int) -> int:
        """Return a pseudo-random integer in ``[0, n)``."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        return self.next_uint() % n


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal-length feature vectors.

    >>> euclidean([0.0, 0.0], [3.0, 4.0])
    5.0
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    if not a:
        raise ValueError("vectors must be non-empty")
    total = 0.0
    for ai, bi in zip(a, b):
        d = float(ai) - float(bi)
        total += d * d
    return total ** 0.5


def k_nearest(points: Sequence[Sequence[float]], idx: int, k: int) -> list[int]:
    """Indices of the ``k`` nearest neighbors of ``points[idx]``, excluding itself.

    Ties are broken by the lower point index, so the result is fully deterministic.
    """
    n = len(points)
    if n == 0:
        raise ValueError("points must be non-empty")
    if not (0 <= idx < n):
        raise ValueError(f"idx {idx} out of range for {n} points")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > n - 1:
        raise ValueError(f"k={k} exceeds available neighbors {n - 1}")
    dists = [
        (euclidean(points[idx], points[j]), j)
        for j in range(n)
        if j != idx
    ]
    dists.sort(key=lambda dj: (dj[0], dj[1]))
    return [j for _, j in dists[:k]]


def smote_sample(
    x_i: Sequence[float],
    x_nn: Sequence[float],
    lam: float,
) -> list[float]:
    """One synthetic point on the segment from ``x_i`` toward ``x_nn``.

    Computes ``x_i + lam * (x_nn - x_i)`` componentwise.
    """
    if len(x_i) != len(x_nn):
        raise ValueError(f"dimension mismatch: {len(x_i)} != {len(x_nn)}")
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    return [float(xi) + lam * (float(xn) - float(xi)) for xi, xn in zip(x_i, x_nn)]


def smote(
    minority: Sequence[Sequence[float]],
    n_synthetic: int,
    k: int = 5,
    seed: int = 0,
) -> list[list[float]]:
    """Generate ``n_synthetic`` synthetic minority examples via SMOTE.

    Parameters
    ----------
    minority:
        The minority-class feature matrix as a sequence of equal-length rows.
    n_synthetic:
        Number of synthetic points to produce (``>= 0``).
    k:
        Number of nearest minority neighbors to interpolate toward.
    seed:
        Seed for the shared LCG; the same seed yields identical output across the
        Python, Julia, and Rust implementations.

    For each synthetic point we (1) draw a base minority index in round-robin order,
    (2) draw one of its ``k`` nearest neighbors via the LCG, and (3) draw the
    interpolation coefficient ``lambda`` via the LCG. The draw order
    (neighbor first, then lambda) is fixed and shared across languages.

    >>> pts = smote([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], 2, k=2, seed=42)
    >>> len(pts)
    2
    """
    if n_synthetic < 0:
        raise ValueError(f"n_synthetic must be non-negative, got {n_synthetic}")
    n = len(minority)
    if n == 0:
        raise ValueError("minority set must be non-empty")
    dim = len(minority[0])
    if dim == 0:
        raise ValueError("feature vectors must be non-empty")
    for row in minority:
        if len(row) != dim:
            raise ValueError("all minority rows must have the same dimension")
    if n_synthetic == 0:
        return []
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > n - 1:
        raise ValueError(
            f"k={k} exceeds available neighbors {n - 1}; need at least k+1 minority points"
        )

    # Precompute neighbor lists once; they do not depend on the RNG.
    neighbors = [k_nearest(minority, i, k) for i in range(n)]

    rng = LCG(seed)
    synthetic: list[list[float]] = []
    for s in range(n_synthetic):
        base = s % n  # deterministic round-robin over minority points
        nn_choice = rng.next_index(k)
        nn = neighbors[base][nn_choice]
        lam = rng.next_float()
        synthetic.append(smote_sample(minority[base], minority[nn], lam))
    return synthetic
