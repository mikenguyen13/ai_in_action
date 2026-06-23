"""Kernel PCA (kernel principal component analysis) from scratch.

A small, well-validated reference implementation of Kernel PCA, the nonlinear
generalization of ordinary PCA introduced by Scholkopf, Smola, and Muller (1998).
Linear PCA is run implicitly in a high-dimensional feature space induced by a
positive-semidefinite kernel, never forming the feature vectors explicitly. The
public API mirrors the Julia (``AIInAction.Ch137KernelPca``) and Rust
(``aiinaction::ch137_kernel_pca``) implementations one-to-one; the cross-language
parity tests assert that all three agree to within floating-point tolerance on
shared fixtures.

The estimator follows the standard recipe:

1. Build the ``n x n`` Gram matrix ``K_ij = k(x_i, x_j)`` for the chosen kernel.
2. Center it implicitly in feature space via the centering matrix ``H = I - 1_n``,
   giving ``K_tilde = H K H``.
3. Diagonalize ``K_tilde`` to obtain eigenvalues ``mu_k`` (descending) and unit
   eigenvectors. Each coefficient vector ``alpha^k`` is the unit eigenvector scaled
   by ``1 / sqrt(mu_k)`` so the feature-space component has unit norm.
4. Project a training point onto component ``k`` via ``beta_i^k = mu_k alpha_i^k``,
   and a new point ``z`` via ``(alpha^k)^T k_tilde_z`` using the *training* row and
   column means for centering.

Supported kernels (passed as a ``(name, params)`` tuple):

* ``("linear", {})``: ``k(x, y) = x . y``.
* ``("poly", {"degree": p, "coef0": c, "gamma": g})``: ``(g x.y + c)^p``.
* ``("rbf", {"gamma": g})``: ``exp(-g ||x - y||^2)``.

Sign convention: each coefficient vector's sign is fixed deterministically so that
its largest-magnitude entry is positive. For eigenvalues with tied-magnitude
entries the overall sign can be numerically arbitrary, so the cross-language tests
compare such components up to sign.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "KernelSpec",
    "KernelPCAResult",
    "kernel_matrix",
    "fit_kernel_pca",
    "transform",
]

Matrix = Sequence[Sequence[float]]
# A kernel is a (name, params) pair, e.g. ("rbf", {"gamma": 0.5}).
KernelSpec = tuple[str, dict[str, float]]

_VALID_KERNELS = ("linear", "poly", "rbf")


@dataclass(frozen=True)
class KernelPCAResult:
    """The fitted state of a Kernel PCA model.

    Attributes
    ----------
    x_fit:
        The training inputs, shape ``(n, d)``. Retained because out-of-sample
        projection requires kernel evaluations against the training points.
    kernel:
        The ``(name, params)`` kernel specification used for fitting.
    alphas:
        Normalized coefficient vectors as columns, shape ``(n, n_components)``.
        Column ``k`` is ``alpha^k`` with ``(alpha^k)^T alpha^k = 1 / mu_k``.
    eigenvalues:
        Kernel-matrix eigenvalues ``mu_k`` of the retained components, descending,
        shape ``(n_components,)``. These equal ``n`` times the feature-space PCA
        variances.
    explained_variance_ratio:
        Fraction of the total feature-space variance captured by each component,
        ``mu_k / sum_j mu_j`` over all nonnegative eigenvalues, shape
        ``(n_components,)``.
    row_means:
        Per-row means of the training Gram matrix, shape ``(n,)``. Used to center
        out-of-sample kernel vectors with the same offsets seen in training.
    total_mean:
        The grand mean of the training Gram matrix (a scalar).
    """

    x_fit: NDArray[np.float64]
    kernel: KernelSpec
    alphas: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    row_means: NDArray[np.float64]
    total_mean: float

    @property
    def n_components(self) -> int:
        """Number of retained kernel principal components."""
        return int(self.alphas.shape[1])

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return int(self.x_fit.shape[0])


def _as_matrix(X: Matrix, name: str = "X") -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got array with {arr.ndim} dimension(s)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (nan or inf)")
    return arr


def _check_kernel(kernel: KernelSpec) -> tuple[str, dict[str, float]]:
    if not (isinstance(kernel, tuple) and len(kernel) == 2):
        raise ValueError("kernel must be a (name, params) tuple")
    name, params = kernel
    if name not in _VALID_KERNELS:
        raise ValueError(f"unknown kernel {name!r}; valid kernels are {_VALID_KERNELS}")
    if not isinstance(params, dict):
        raise ValueError("kernel params must be a dict")
    return name, params


def kernel_matrix(A: Matrix, B: Matrix, kernel: KernelSpec) -> NDArray[np.float64]:
    """Compute the cross-kernel matrix ``K_ij = k(a_i, b_j)`` for two point sets.

    Parameters
    ----------
    A, B:
        Point sets of shapes ``(n_a, d)`` and ``(n_b, d)``; the feature
        dimension ``d`` must match.
    kernel:
        A ``(name, params)`` specification (see module docstring).

    Returns
    -------
    numpy.ndarray
        The ``(n_a, n_b)`` kernel matrix.
    """
    a = _as_matrix(A, "A")
    b = _as_matrix(B, "B")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"A has {a.shape[1]} features but B has {b.shape[1]}")
    name, params = _check_kernel(kernel)

    if name == "linear":
        return a @ b.T
    if name == "poly":
        gamma = float(params.get("gamma", 1.0))
        coef0 = float(params.get("coef0", 1.0))
        degree = float(params.get("degree", 2.0))
        return (gamma * (a @ b.T) + coef0) ** degree
    # rbf
    gamma = float(params.get("gamma", 1.0))
    if gamma <= 0.0:
        raise ValueError(f"rbf gamma must be positive, got {gamma}")
    aa = np.sum(a**2, axis=1)[:, None]
    bb = np.sum(b**2, axis=1)[None, :]
    sq = aa + bb - 2.0 * (a @ b.T)
    np.maximum(sq, 0.0, out=sq)  # guard tiny negatives from rounding
    return np.exp(-gamma * sq)


def _fix_signs(alphas: NDArray[np.float64]) -> NDArray[np.float64]:
    """Force the largest-magnitude entry of each coefficient column to be positive."""
    out = alphas.copy()
    for k in range(out.shape[1]):
        col = out[:, k]
        i = int(np.argmax(np.abs(col)))
        if col[i] < 0.0:
            out[:, k] = -col
    return out


def fit_kernel_pca(
    X: Matrix,
    n_components: int | None = None,
    *,
    kernel: KernelSpec = ("rbf", {"gamma": 1.0}),
    eigenvalue_tol: float = 1e-12,
) -> KernelPCAResult:
    """Fit a Kernel PCA model to ``X``.

    Parameters
    ----------
    X:
        Data matrix of shape ``(n, d)`` with ``n >= 2`` samples and ``d >= 1``
        features. Must contain only finite values.
    n_components:
        Number of components to retain. When ``None``, retains every component
        with a strictly positive eigenvalue, capped at ``n - 1`` (centering
        removes one degree of freedom). An explicit value must satisfy
        ``1 <= n_components <= n - 1`` and each retained eigenvalue must be
        positive.
    kernel:
        A ``(name, params)`` kernel specification (see module docstring).
    eigenvalue_tol:
        Eigenvalues at or below this value are treated as numerically zero; a
        retained component whose eigenvalue is non-positive raises ``ValueError``.

    Returns
    -------
    KernelPCAResult
        The fitted model.

    Examples
    --------
    >>> X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    >>> m = fit_kernel_pca(X, n_components=1, kernel=("linear", {}))
    >>> m.n_components
    1
    """
    arr = _as_matrix(X)
    n = arr.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 samples for Kernel PCA, got {n}")
    if arr.shape[1] < 1:
        raise ValueError("X must have at least one feature")
    name, params = _check_kernel(kernel)

    max_components = n - 1

    K = kernel_matrix(arr, arr, (name, params))

    # Implicit feature-space centering: K_tilde = H K H, H = I - 1_n.
    row_means = K.mean(axis=1)
    total_mean = float(K.mean())
    K_tilde = K - row_means[:, None] - row_means[None, :] + total_mean
    # Symmetrize to kill asymmetric rounding before the symmetric eigensolver.
    K_tilde = (K_tilde + K_tilde.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(K_tilde)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    total_var = float(np.sum(np.maximum(eigvals, 0.0)))
    if total_var <= 0.0:
        raise ValueError("kernel matrix has no positive variance; Kernel PCA is undefined")

    # Number of usable (strictly positive) components, capped by centering rank.
    n_positive = int(np.sum(eigvals[:max_components] > eigenvalue_tol))

    if n_components is None:
        if n_positive < 1:
            raise ValueError("kernel matrix has no positive variance; Kernel PCA is undefined")
        n_components = n_positive
    else:
        if not isinstance(n_components, (int, np.integer)):
            raise ValueError(f"n_components must be an integer, got {type(n_components).__name__}")
        n_components = int(n_components)
        if n_components < 1 or n_components > max_components:
            raise ValueError(
                f"n_components must be in [1, {max_components}] for {n} samples, got {n_components}"
            )

    mu = eigvals[:n_components]
    if np.any(mu <= eigenvalue_tol):
        bad = int(np.argmin(mu))
        raise ValueError(
            f"component {bad} has non-positive eigenvalue {mu[bad]:.3e}; "
            f"request fewer components or a different kernel"
        )

    # Normalize so the feature-space eigenvectors have unit norm:
    # alpha^k = (unit eigenvector) / sqrt(mu_k).
    alphas = eigvecs[:, :n_components] / np.sqrt(mu)
    alphas = _fix_signs(alphas)

    explained_variance_ratio = mu / total_var

    return KernelPCAResult(
        x_fit=arr,
        kernel=(name, params),
        alphas=alphas,
        eigenvalues=mu,
        explained_variance_ratio=explained_variance_ratio,
        row_means=row_means,
        total_mean=total_mean,
    )


def transform(model: KernelPCAResult, Z: Matrix) -> NDArray[np.float64]:
    """Project new points ``Z`` onto the fitted kernel principal components.

    Centering of each test point reuses the *training* row and grand means, so the
    same affine offset is applied to train and test. Returns an array of shape
    ``(n_z, n_components)``.

    Passing the training data reproduces the in-sample projections
    ``beta_i^k = mu_k alpha_i^k``.
    """
    z = _as_matrix(Z, "Z")
    d = model.x_fit.shape[1]
    if z.shape[1] != d:
        raise ValueError(f"Z has {z.shape[1]} features but model was fit on {d}")

    # K_z[i, t] = k(x_i, z_t): kernel of training points against test points.
    K_z = kernel_matrix(model.x_fit, z, model.kernel)
    # Center against training statistics: subtract per-train-row mean and the
    # per-test-column mean, add back the training grand mean.
    col_means = K_z.mean(axis=0)  # length n_z
    K_z_centered = K_z - model.row_means[:, None] - col_means[None, :] + model.total_mean
    # Projection: (alpha^k)^T k_tilde_z for each component and test point.
    return K_z_centered.T @ model.alphas
