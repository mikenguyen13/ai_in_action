"""Non-Negative Matrix Factorization (NMF) from scratch.

A small, well-validated reference implementation of NMF using the Lee-Seung
multiplicative update rules for the squared Frobenius objective. The public API
mirrors the Julia (`AIInAction.Ch139Nmf`) and Rust (`aiinaction::ch139_nmf`)
implementations one-to-one; the cross-language parity tests assert that all three
agree to within floating-point tolerance on shared fixtures.

Given a non-negative matrix ``V`` of shape ``(n, m)``, NMF seeks non-negative
factors ``W`` of shape ``(n, r)`` and ``H`` of shape ``(r, m)`` such that
``V ~= W H``. The factors are refined by the multiplicative updates

    H <- H * (W^T V) / (W^T W H + eps)
    W <- W * (V H^T) / (W H H^T + eps)

which preserve non-negativity automatically and monotonically decrease the cost.

Determinism: to make the three language implementations agree exactly on the
shared fixtures, the factors are seeded by a small self-contained linear
congruential generator (LCG) seeded from a user-supplied integer, not by any
language's built-in RNG. The same LCG, fill order, and fixed iteration count are
reproduced bit-for-bit in the Julia and Rust ports.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["NMFResult", "fit_nmf", "transform", "reconstruct", "reconstruction_error"]

Matrix = Sequence[Sequence[float]]

# Numerical floor added to denominators to avoid division by zero. Shared across
# languages so the iterates match bit-for-bit.
_EPS = 1e-10


@dataclass(frozen=True)
class NMFResult:
    """The fitted state of an NMF model.

    Attributes
    ----------
    W:
        Basis matrix of shape ``(n, r)``; column ``k`` is the ``k``-th latent
        component (dictionary atom) over the ``n`` features.
    H:
        Coefficient matrix of shape ``(r, m)``; column ``j`` holds the
        non-negative activations that reconstruct data column ``j``.
    n_iter:
        Number of multiplicative-update iterations actually run.
    error:
        Final Frobenius reconstruction error ``||V - W H||_F`` (not squared, not
        halved), recorded after the last iteration.
    """

    W: NDArray[np.float64]
    H: NDArray[np.float64]
    n_iter: int
    error: float

    @property
    def n_components(self) -> int:
        """Rank ``r`` of the factorization."""
        return int(self.W.shape[1])

    @property
    def n_features(self) -> int:
        """Number of features ``n`` (rows of ``V``)."""
        return int(self.W.shape[0])


def _as_matrix(V: Matrix) -> NDArray[np.float64]:
    arr = np.asarray(V, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"V must be a 2-D matrix, got array with {arr.ndim} dimension(s)")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"V must be non-empty, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("V contains non-finite values (nan or inf)")
    if np.any(arr < 0.0):
        raise ValueError("V must be non-negative (all entries >= 0)")
    return arr


def _seeded_uniform(rows: int, cols: int, seed: int) -> NDArray[np.float64]:
    """Fill a ``rows x cols`` matrix with deterministic values in ``(0, 1]``.

    Uses a 32-bit linear congruential generator (the Numerical Recipes constants)
    so that the exact same sequence and fill order can be reproduced in Julia and
    Rust. Row-major fill: entry ``(i, j)`` is the ``(i * cols + j)``-th draw.
    """
    out = np.empty((rows, cols), dtype=np.float64)
    state = seed & 0xFFFFFFFF
    for i in range(rows):
        for j in range(cols):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            # Map to (0, 1]; +1 avoids an exact zero entry that would freeze an
            # entire row/column of a multiplicative factor at zero forever.
            out[i, j] = (state + 1) / 4294967297.0
    return out


def fit_nmf(
    V: Matrix,
    n_components: int,
    *,
    max_iter: int = 200,
    seed: int = 0,
) -> NMFResult:
    """Factor a non-negative matrix ``V`` as ``W H`` via multiplicative updates.

    Parameters
    ----------
    V:
        Non-negative data matrix of shape ``(n, m)`` with finite entries.
    n_components:
        Rank ``r`` of the factorization. Must satisfy ``1 <= r <= min(n, m)``.
    max_iter:
        Number of multiplicative-update sweeps (each sweep updates ``H`` then
        ``W``). Must be ``>= 1``.
    seed:
        Non-negative integer seed for the deterministic LCG initializer.

    Returns
    -------
    NMFResult
        The fitted factors and final reconstruction error.

    Examples
    --------
    >>> V = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]]
    >>> r = fit_nmf(V, n_components=2, max_iter=300, seed=1)
    >>> r.error < 0.05
    True
    """
    arr = _as_matrix(V)
    n, m = arr.shape
    max_components = min(n, m)
    if not isinstance(n_components, (int, np.integer)):
        raise ValueError(f"n_components must be an integer, got {type(n_components).__name__}")
    r = int(n_components)
    if r < 1 or r > max_components:
        raise ValueError(
            f"n_components must be in [1, {max_components}] for a {n}x{m} matrix, got {r}"
        )
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    max_iter = int(max_iter)
    if int(seed) < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")

    # Deterministic non-negative initialization. W is filled before H so the LCG
    # stream order is fixed and reproducible across languages.
    W = _seeded_uniform(n, r, int(seed))
    H = _seeded_uniform(r, m, int(seed) + 1)

    for _ in range(max_iter):
        # Update H with W fixed: H *= (W^T V) / (W^T W H).
        WtV = W.T @ arr
        WtWH = W.T @ W @ H
        H = H * (WtV / (WtWH + _EPS))
        # Update W with H fixed: W *= (V H^T) / (W H H^T).
        VHt = arr @ H.T
        WHHt = W @ (H @ H.T)
        W = W * (VHt / (WHHt + _EPS))

    error = float(np.sqrt(np.sum((arr - W @ H) ** 2)))
    return NMFResult(W=W, H=H, n_iter=max_iter, error=error)


def transform(model: NMFResult, V: Matrix, *, max_iter: int = 200) -> NDArray[np.float64]:
    """Compute non-negative encodings of new data under a fixed basis ``W``.

    Holds ``model.W`` fixed and iterates the ``H`` multiplicative update only,
    starting from a deterministic seed, returning the coefficient matrix of shape
    ``(n_components, m)`` such that ``V ~= model.W @ H_new``.
    """
    arr = _as_matrix(V)
    if arr.shape[0] != model.n_features:
        raise ValueError(
            f"V has {arr.shape[0]} features but model basis has {model.n_features}"
        )
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    max_iter = int(max_iter)

    r = model.n_components
    m = arr.shape[1]
    W = model.W
    H = _seeded_uniform(r, m, 1)
    WtV = W.T @ arr
    WtW = W.T @ W
    for _ in range(max_iter):
        H = H * (WtV / (WtW @ H + _EPS))
    return H


def reconstruct(model: NMFResult) -> NDArray[np.float64]:
    """Return the low-rank reconstruction ``W H`` of the fitted model."""
    return model.W @ model.H


def reconstruction_error(V: Matrix, W: Matrix, H: Matrix) -> float:
    """Frobenius reconstruction error ``||V - W H||_F`` for given factors.

    Validates shape compatibility and non-negativity of ``V``, then returns the
    (non-squared) Frobenius norm of the residual.
    """
    Va = _as_matrix(V)
    Wa = np.asarray(W, dtype=np.float64)
    Ha = np.asarray(H, dtype=np.float64)
    if Wa.ndim != 2 or Ha.ndim != 2:
        raise ValueError("W and H must be 2-D matrices")
    if Wa.shape[0] != Va.shape[0]:
        raise ValueError(f"W has {Wa.shape[0]} rows but V has {Va.shape[0]}")
    if Ha.shape[1] != Va.shape[1]:
        raise ValueError(f"H has {Ha.shape[1]} columns but V has {Va.shape[1]}")
    if Wa.shape[1] != Ha.shape[0]:
        raise ValueError(
            f"inner dimensions disagree: W is {Wa.shape}, H is {Ha.shape}"
        )
    diff = Va - Wa @ Ha
    return float(np.sqrt(np.sum(diff**2)))
