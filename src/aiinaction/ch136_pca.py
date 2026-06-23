"""Principal Component Analysis (PCA) from scratch.

A small, well-validated reference implementation of PCA built on the singular
value decomposition (SVD) of the centered data matrix. The public API mirrors the
Julia (`AIInAction.Ch136Pca`) and Rust (`aiinaction::ch136_pca`) implementations
one-to-one; the cross-language parity tests assert that all three agree to within
floating-point tolerance on shared fixtures.

The estimator follows the standard recipe:

1. Center each feature by subtracting its column mean (always).
2. Optionally scale each feature to unit standard deviation (correlation PCA).
3. Compute the thin SVD ``X_c = U S V^T``; the columns of ``V`` are the principal
   directions and ``lambda_j = s_j^2 / (n - 1)`` are the explained variances.
4. Project data via scores ``T = X_c V`` and optionally whiten so each retained
   component has unit variance.

Sign convention: each component's sign is fixed deterministically so that the
entry of largest absolute value in every loading vector is positive. This removes
the eigenvector sign ambiguity and makes results reproducible across languages.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["PCAResult", "fit_pca", "transform", "inverse_transform", "reconstruction_error"]

Matrix = Sequence[Sequence[float]]


@dataclass(frozen=True)
class PCAResult:
    """The fitted state of a PCA model.

    Attributes
    ----------
    mean:
        Per-feature means estimated on the training data, shape ``(d,)``.
    scale:
        Per-feature scales applied after centering, shape ``(d,)``. All ones when
        ``scale=False`` was requested.
    components:
        Principal directions as rows, shape ``(n_components, d)``. Row ``j`` is the
        unit-norm loading vector of component ``j``.
    explained_variance:
        Variance ``lambda_j = s_j^2 / (n - 1)`` captured by each retained
        component, shape ``(n_components,)``.
    explained_variance_ratio:
        Fraction of the *total* variance captured by each component, shape
        ``(n_components,)``. The total is computed over all ``d`` directions so the
        ratios of a truncated model sum to at most one.
    whiten:
        Whether :func:`transform` divides scores by ``sqrt(explained_variance)``.
    """

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    components: NDArray[np.float64]
    explained_variance: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    whiten: bool

    @property
    def n_components(self) -> int:
        """Number of retained principal components."""
        return int(self.components.shape[0])

    @property
    def n_features(self) -> int:
        """Dimensionality of the input feature space."""
        return int(self.components.shape[1])


def _as_matrix(X: Matrix) -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be a 2-D matrix, got array with {arr.ndim} dimension(s)")
    if arr.shape[0] < 2:
        raise ValueError(f"need at least 2 samples to estimate variance, got {arr.shape[0]}")
    if arr.shape[1] < 1:
        raise ValueError("X must have at least one feature")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X contains non-finite values (nan or inf)")
    return arr


def _fix_signs(components: NDArray[np.float64]) -> NDArray[np.float64]:
    """Force the largest-magnitude entry of each loading row to be positive."""
    out = components.copy()
    for j in range(out.shape[0]):
        row = out[j]
        k = int(np.argmax(np.abs(row)))
        if row[k] < 0.0:
            out[j] = -row
    return out


def fit_pca(X: Matrix, n_components: int | None = None, *, scale: bool = False, whiten: bool = False) -> PCAResult:
    """Fit a PCA model to ``X`` via the SVD of the centered (and optionally scaled) data.

    Parameters
    ----------
    X:
        Data matrix of shape ``(n, d)`` with ``n >= 2`` samples and ``d >= 1``
        features. Must contain only finite values.
    n_components:
        Number of components to retain. Defaults to ``min(n, d)`` when ``None``.
        Must satisfy ``1 <= n_components <= min(n, d)``.
    scale:
        If ``True``, divide each centered feature by its (population-corrected,
        ``ddof=1``) standard deviation before decomposing, yielding correlation
        PCA. A feature with zero variance raises ``ValueError``.
    whiten:
        If ``True``, :func:`transform` rescales each score column to unit variance.

    Returns
    -------
    PCAResult
        The fitted model.

    Examples
    --------
    >>> X = [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]
    >>> r = fit_pca(X, n_components=1)
    >>> round(float(r.explained_variance_ratio[0]), 6)
    1.0
    """
    arr = _as_matrix(X)
    n, d = arr.shape
    max_components = min(n, d)

    if n_components is None:
        n_components = max_components
    if not isinstance(n_components, (int, np.integer)):
        raise ValueError(f"n_components must be an integer, got {type(n_components).__name__}")
    n_components = int(n_components)
    if n_components < 1 or n_components > max_components:
        raise ValueError(
            f"n_components must be in [1, {max_components}] for a {n}x{d} matrix, got {n_components}"
        )

    mean = arr.mean(axis=0)
    Xc = arr - mean

    if scale:
        std = arr.std(axis=0, ddof=1)
        if np.any(std == 0.0):
            bad = int(np.argmin(std))
            raise ValueError(f"cannot scale: feature {bad} has zero variance")
        scale_vec = std
        Xc = Xc / scale_vec
    else:
        scale_vec = np.ones(d, dtype=np.float64)

    # Thin SVD of the centered data. Vt rows are the principal directions.
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = float(np.sum(s**2) / (n - 1))

    components = _fix_signs(Vt[:n_components])
    explained_variance = (s[:n_components] ** 2) / (n - 1)
    if total_var == 0.0:
        raise ValueError("X has zero total variance after centering; PCA is undefined")
    explained_variance_ratio = explained_variance / total_var

    return PCAResult(
        mean=mean,
        scale=scale_vec,
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        whiten=whiten,
    )


def transform(model: PCAResult, X: Matrix) -> NDArray[np.float64]:
    """Project ``X`` onto the fitted principal components.

    Applies the *training* mean and scale, then ``T = X_c V``. When the model was
    fitted with ``whiten=True`` each score column is divided by
    ``sqrt(explained_variance)`` so the transformed features have unit variance.

    Returns an array of shape ``(n, n_components)``.
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

    Xc = (arr - model.mean) / model.scale
    scores = Xc @ model.components.T
    if model.whiten:
        std = np.sqrt(model.explained_variance)
        if np.any(std == 0.0):
            raise ValueError("cannot whiten: a retained component has zero variance")
        scores = scores / std
    return scores


def inverse_transform(model: PCAResult, scores: Matrix) -> NDArray[np.float64]:
    """Map scores back to the original feature space (best rank-m reconstruction).

    Inverts whitening, the projection, the scaling, and the centering in turn,
    returning an array of shape ``(n, d)``.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"scores must be a 2-D matrix, got array with {arr.ndim} dimension(s)")
    if arr.shape[1] != model.n_components:
        raise ValueError(
            f"scores has {arr.shape[1]} columns but model has {model.n_components} components"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores contains non-finite values (nan or inf)")

    t = arr
    if model.whiten:
        t = t * np.sqrt(model.explained_variance)
    Xc = t @ model.components
    return Xc * model.scale + model.mean


def reconstruction_error(model: PCAResult, X: Matrix) -> float:
    """Mean squared reconstruction error of ``X`` under the truncated model.

    Computes ``mean_i || x_i - inverse_transform(transform(x_i)) ||^2`` averaged
    over samples, summed over features.
    """
    arr = _as_matrix(X)
    recon = inverse_transform(model, transform(model, arr))
    diff = arr - recon
    return float(np.sum(diff**2) / arr.shape[0])
