"""FastICA for Independent Component Analysis (from scratch).

A small, deterministic reference implementation of the FastICA fixed-point
algorithm of Hyvarinen and Oja. The public API mirrors the Julia
(``AIInAction.Ch138Fastica``) and Rust (``aiinaction::ch138_fastica``)
implementations one-to-one; the cross-language parity tests assert that all three
agree to within floating-point tolerance on shared fixtures.

The pipeline is the standard three-stage recipe:

1. Center each observed signal by subtracting its column mean.
2. Whiten with the eigendecomposition of the covariance matrix so the whitened
   data has identity covariance; after whitening the remaining mixing matrix is
   orthogonal.
3. Run the FastICA fixed-point iteration with **symmetric** orthogonalization to
   find the unmixing rotation ``W`` that maximizes the non-Gaussianity of the
   recovered components.

Contrast function: this module uses the robust ``logcosh`` contrast, with first
derivative ``g(u) = tanh(u)`` and second derivative ``g'(u) = 1 - tanh(u)^2``.

Determinism for cross-language parity. FastICA is usually seeded with a random
weight matrix, which makes runs irreproducible. To keep Python, Julia and Rust
bit-comparable we instead:

* initialize the unmixing matrix to the identity,
* run a *fixed* number of iterations (``max_iter``) rather than stopping at a
  tolerance, and
* implement symmetric orthogonalization ``W <- (W W^T)^{-1/2} W`` with a
  self-contained cyclic Jacobi eigensolver (the same routine used by the PCA
  chapter), so no library-specific decomposition can perturb the result.

Sign/order convention. Like all ICA, the recovered sources are defined only up to
sign, scale and permutation. We fix the permutation deterministically by ordering
components by descending sample variance of the recovered sources, and fix the
sign so the largest-magnitude entry of each unmixing row is positive.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["ICAResult", "fit_ica", "transform"]

Matrix = Sequence[Sequence[float]]


@dataclass(frozen=True)
class ICAResult:
    """The fitted state of a FastICA model.

    Attributes
    ----------
    mean:
        Per-signal means estimated on the training data, shape ``(d,)``.
    whitening:
        Whitening matrix ``K`` of shape ``(n_components, d)`` mapping centered
        data to whitened data: ``z = K x_c``.
    unmixing:
        Orthogonal unmixing rotation ``W`` of shape ``(n_components,
        n_components)`` acting in the whitened space.
    components:
        The full unmixing operator ``W K`` of shape ``(n_components, d)``. The
        recovered sources are ``S = components @ X_c^T``, i.e. row ``j`` recovers
        source ``j`` from centered data.
    mixing:
        Estimated mixing matrix of shape ``(d, n_components)``, the pseudoinverse
        of ``components``, so ``X_c ~= S @ mixing^T``.
    n_iter:
        Number of fixed-point iterations actually run.
    """

    mean: NDArray[np.float64]
    whitening: NDArray[np.float64]
    unmixing: NDArray[np.float64]
    components: NDArray[np.float64]
    mixing: NDArray[np.float64]
    n_iter: int

    @property
    def n_components(self) -> int:
        """Number of recovered independent components."""
        return int(self.components.shape[0])

    @property
    def n_features(self) -> int:
        """Dimensionality of the observed signal space."""
        return int(self.components.shape[1])


def _as_matrix(X: Matrix) -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be a 2-D matrix, got array with {arr.ndim} dimension(s)")
    if arr.shape[0] < 2:
        raise ValueError(f"need at least 2 samples, got {arr.shape[0]}")
    if arr.shape[1] < 1:
        raise ValueError("X must have at least one feature")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X contains non-finite values (nan or inf)")
    return arr


def _jacobi_eigen(a: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Symmetric eigendecomposition by the cyclic Jacobi method.

    Returns ``(eigenvalues, V)`` sorted by descending eigenvalue, with
    eigenvector ``k`` in column ``k`` of ``V``. Matching the Rust/Julia routine
    keeps the whitening and orthogonalization bit-comparable across languages.
    """
    n = a.shape[0]
    A = a.copy()
    V = np.eye(n, dtype=np.float64)
    for _ in range(100):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += A[p, q] * A[p, q]
        if off < 1e-30:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = A[p, q]
                if abs(apq) < 1e-300:
                    continue
                theta = (A[q, q] - A[p, p]) / (2.0 * apq)
                if theta == 0.0:
                    t = 1.0
                else:
                    t = np.sign(theta) / (abs(theta) + np.sqrt(theta * theta + 1.0))
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp = A[k, p]
                    akq = A[k, q]
                    A[k, p] = c * akp - s * akq
                    A[k, q] = s * akp + c * akq
                for k in range(n):
                    apk = A[p, k]
                    aqk = A[q, k]
                    A[p, k] = c * apk - s * aqk
                    A[q, k] = s * apk + c * aqk
                for k in range(n):
                    vkp = V[k, p]
                    vkq = V[k, q]
                    V[k, p] = c * vkp - s * vkq
                    V[k, q] = s * vkp + c * vkq
    eigvals = np.array([A[i, i] for i in range(n)], dtype=np.float64)
    order = np.argsort(-eigvals, kind="stable")
    return eigvals[order], V[:, order]


def _symmetric_decorrelation(W: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetric orthogonalization ``W <- (W W^T)^{-1/2} W`` via Jacobi eigen."""
    eigvals, V = _jacobi_eigen(W @ W.T)
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(eigvals)) @ V.T
    return inv_sqrt @ W


def _fix_signs(W: NDArray[np.float64]) -> NDArray[np.float64]:
    """Force the largest-magnitude entry of each unmixing row to be positive."""
    out = W.copy()
    for j in range(out.shape[0]):
        k = int(np.argmax(np.abs(out[j])))
        if out[j, k] < 0.0:
            out[j] = -out[j]
    return out


def fit_ica(
    X: Matrix,
    n_components: int | None = None,
    *,
    max_iter: int = 200,
) -> ICAResult:
    """Fit a FastICA model to ``X`` (samples in rows, signals in columns).

    Parameters
    ----------
    X:
        Observed mixtures of shape ``(n, d)`` with ``n >= 2`` samples and
        ``d >= 1`` signals. Must contain only finite values.
    n_components:
        Number of independent components to recover. Defaults to ``d`` when
        ``None``. Must satisfy ``1 <= n_components <= d``.
    max_iter:
        Fixed number of symmetric fixed-point iterations to run. A fixed count
        (rather than a tolerance-based stop) keeps the result deterministic and
        identical across the Python, Julia and Rust implementations. Must be
        ``>= 1``.

    Returns
    -------
    ICAResult
        The fitted model.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 200)
    >>> S = np.c_[np.sign(np.sin(20 * t)), np.mod(5 * t, 1.0) - 0.5]
    >>> A = np.array([[1.0, 1.0], [0.5, 2.0]])
    >>> X = S @ A.T
    >>> r = fit_ica(X, n_components=2)
    >>> r.n_components
    2
    """
    arr = _as_matrix(X)
    n, d = arr.shape

    if n_components is None:
        n_components = d
    if not isinstance(n_components, (int, np.integer)):
        raise ValueError(f"n_components must be an integer, got {type(n_components).__name__}")
    n_components = int(n_components)
    if n_components < 1 or n_components > d:
        raise ValueError(
            f"n_components must be in [1, {d}] for a {n}x{d} matrix, got {n_components}"
        )
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    max_iter = int(max_iter)

    # 1. Center.
    mean = arr.mean(axis=0)
    Xc = arr - mean  # (n, d)

    # 2. Whiten via covariance eigendecomposition. Cov = Xc^T Xc / (n - 1).
    cov = (Xc.T @ Xc) / (n - 1)
    eigvals, E = _jacobi_eigen(cov)
    eigvals_k = eigvals[:n_components]
    if np.any(eigvals_k <= 0.0):
        raise ValueError(
            "data is rank-deficient: a retained whitening direction has zero variance"
        )
    Ek = E[:, :n_components]  # (d, k)
    # K maps centered data to unit-variance whitened data: z = K x_c.
    K = (Ek / np.sqrt(eigvals_k)).T  # (k, d)
    Z = Xc @ K.T  # (n, k) whitened, unit variance columns

    # 3. FastICA fixed-point iteration with symmetric orthogonalization.
    W = _symmetric_decorrelation(np.eye(n_components, dtype=np.float64))
    n_iter = 0
    for _ in range(max_iter):
        n_iter += 1
        WZ = Z @ W.T  # (n, k): each column is w_j^T z
        gwz = np.tanh(WZ)
        g_prime = 1.0 - gwz**2
        # w_j^+ = E[z g(w^T z)] - E[g'(w^T z)] w_j
        W_new = (gwz.T @ Z) / n - (g_prime.mean(axis=0)[:, None] * W)
        W = _symmetric_decorrelation(W_new)

    W = _fix_signs(W)
    components = W @ K  # (k, d): full unmixing operator
    # Order components by descending recovered-source variance for a stable
    # deterministic permutation.
    S = Xc @ components.T  # (n, k)
    var = S.var(axis=0, ddof=1)
    order = np.argsort(-var, kind="stable")
    W = W[order]
    components = components[order]

    # Estimated mixing matrix is the pseudoinverse of the unmixing operator.
    mixing = np.linalg.pinv(components)  # (d, k)

    return ICAResult(
        mean=mean,
        whitening=K,
        unmixing=W,
        components=components,
        mixing=mixing,
        n_iter=n_iter,
    )


def transform(model: ICAResult, X: Matrix) -> NDArray[np.float64]:
    """Recover the independent sources from observed mixtures ``X``.

    Applies the training mean, then projects via the full unmixing operator:
    ``S = (X - mean) @ components^T``. Returns an array of shape
    ``(n, n_components)`` whose columns are the recovered sources.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be a 2-D matrix, got array with {arr.ndim} dimension(s)")
    if arr.shape[1] != model.n_features:
        raise ValueError(
            f"X has {arr.shape[1]} features but model was fit on {model.n_features}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("X contains non-finite values (nan or inf)")

    Xc = arr - model.mean
    return Xc @ model.components.T
