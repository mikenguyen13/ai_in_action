"""Spectral clustering from scratch (chapter 131).

A small, dependency-free reference implementation of spectral clustering built
directly on the normalized-cut relaxation developed in the chapter. The pipeline
is the classic one:

1. Build a Gaussian (RBF) similarity matrix ``W`` from the data.
2. Form the degree matrix ``D`` and the symmetric normalized Laplacian
   ``L_sym = I - D^{-1/2} W D^{-1/2}``.
3. Take the ``k`` eigenvectors of ``L_sym`` with the smallest eigenvalues.
4. Row-normalize that ``n x k`` embedding (the Ng-Jordan-Weiss step).
5. Cluster the embedded rows with k-means.

Every numerical primitive here, including the eigensolver (a cyclic Jacobi
rotation sweep for symmetric matrices) and k-means, is implemented from scratch
and made fully deterministic so that the Python, Julia, and Rust ports agree to
within ``1e-9`` on the shared fixtures. ``numpy`` is used only as a convenient
array container and for elementary linear-algebra products; no high-level
clustering or eigen routine is called.

This module mirrors the Julia (``AIInAction.Ch131SpectralClustering``) and Rust
(``aiinaction::ch131_spectral_clustering``) implementations one-to-one.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "rbf_affinity",
    "normalized_laplacian",
    "jacobi_eigh",
    "spectral_embedding",
    "kmeans",
    "spectral_clustering",
]


def _as_matrix(x: Sequence[Sequence[float]]) -> np.ndarray:
    """Coerce a 2-D sequence into a float ``(n, d)`` array with validation."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"X must be a 2-D array of shape (n_samples, n_features); got ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X must contain only finite values")
    return arr


def rbf_affinity(X: Sequence[Sequence[float]], sigma: float) -> np.ndarray:
    """Gaussian (RBF) similarity matrix with a zero diagonal.

    The off-diagonal entries are ``w_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))``
    and ``w_ii = 0`` so that vertices carry no self-loop, matching the chapter's
    convention ``w_ii = 0``.

    Args:
        X: Data of shape ``(n_samples, n_features)``.
        sigma: Positive kernel bandwidth.

    Returns:
        A symmetric ``(n, n)`` affinity matrix with zero diagonal.

    Raises:
        ValueError: If ``X`` is malformed or ``sigma`` is not positive.
    """
    arr = _as_matrix(X)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be a positive finite number; got {sigma}")
    n = arr.shape[0]
    denom = 2.0 * sigma * sigma
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            diff = arr[i] - arr[j]
            sq = float(diff @ diff)
            w = math.exp(-sq / denom)
            W[i, j] = w
            W[j, i] = w
    return W


def normalized_laplacian(W: Sequence[Sequence[float]]) -> np.ndarray:
    """Symmetric normalized Laplacian ``L_sym = I - D^{-1/2} W D^{-1/2}``.

    Args:
        W: A square symmetric non-negative affinity matrix.

    Returns:
        The symmetric normalized Laplacian as an ``(n, n)`` array.

    Raises:
        ValueError: If ``W`` is not square, not symmetric, has negative
            entries, or has an isolated vertex (zero degree).
    """
    A = np.asarray(W, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("W must be a square 2-D matrix")
    n = A.shape[0]
    if n == 0:
        raise ValueError("W must be non-empty")
    if not np.all(np.isfinite(A)):
        raise ValueError("W must contain only finite values")
    if np.any(A < 0.0):
        raise ValueError("W must have non-negative entries")
    if not np.allclose(A, A.T, atol=1e-12, rtol=0.0):
        raise ValueError("W must be symmetric")
    deg = A.sum(axis=1)
    if np.any(deg <= 0.0):
        raise ValueError("every vertex must have positive degree (no isolated vertices)")
    dinv_sqrt = 1.0 / np.sqrt(deg)
    L = np.eye(n) - (dinv_sqrt[:, None] * A * dinv_sqrt[None, :])
    # Symmetrize to remove any floating-point asymmetry.
    return 0.5 * (L + L.T)


def jacobi_eigh(A: Sequence[Sequence[float]], max_sweeps: int = 100, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of a real symmetric matrix via cyclic Jacobi rotations.

    This is a deterministic, dependency-free eigensolver (no ``numpy.linalg``)
    so that all three language ports produce bitwise-comparable results. It
    returns eigenvalues in ascending order together with the matching
    orthonormal eigenvectors as columns, with each eigenvector sign-fixed so its
    first nonzero entry is positive (removing the inherent sign ambiguity).

    Args:
        A: A square symmetric matrix.
        max_sweeps: Maximum number of full off-diagonal sweeps.
        tol: Convergence threshold on the off-diagonal Frobenius norm.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` with eigenvalues ascending and
        eigenvectors as columns of an orthogonal matrix.

    Raises:
        ValueError: If ``A`` is not square or not symmetric.
    """
    M = np.array(A, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("A must be a square 2-D matrix")
    n = M.shape[0]
    if n == 0:
        raise ValueError("A must be non-empty")
    if not np.allclose(M, M.T, atol=1e-12, rtol=0.0):
        raise ValueError("A must be symmetric")

    V = np.eye(n)
    for _ in range(max_sweeps):
        off = math.sqrt(sum(M[p, q] ** 2 for p in range(n) for q in range(p + 1, n)))
        if off <= tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = M[p, q]
                if abs(apq) <= 1e-300:
                    continue
                app = M[p, p]
                aqq = M[q, q]
                # Jacobi rotation angle that zeroes M[p, q].
                phi = 0.5 * math.atan2(2.0 * apq, aqq - app)
                c = math.cos(phi)
                s = math.sin(phi)
                for i in range(n):
                    mip = M[i, p]
                    miq = M[i, q]
                    M[i, p] = c * mip - s * miq
                    M[i, q] = s * mip + c * miq
                for i in range(n):
                    mpi = M[p, i]
                    mqi = M[q, i]
                    M[p, i] = c * mpi - s * mqi
                    M[q, i] = s * mpi + c * mqi
                for i in range(n):
                    vip = V[i, p]
                    viq = V[i, q]
                    V[i, p] = c * vip - s * viq
                    V[i, q] = s * vip + c * viq

    eigvals = np.array([M[i, i] for i in range(n)], dtype=float)
    order = sorted(range(n), key=lambda i: eigvals[i])
    eigvals = eigvals[order]
    V = V[:, order]
    # Sign-fix each eigenvector: first nonzero entry positive.
    for j in range(n):
        for i in range(n):
            if abs(V[i, j]) > 1e-12:
                if V[i, j] < 0.0:
                    V[:, j] = -V[:, j]
                break
    return eigvals, V


def spectral_embedding(W: Sequence[Sequence[float]], k: int) -> np.ndarray:
    """Row-normalized spectral embedding from the normalized Laplacian.

    Builds ``L_sym``, takes the ``k`` smallest eigenvectors, and normalizes each
    row to unit length (the Ng-Jordan-Weiss embedding). Rows that are exactly
    zero are left as zero.

    Args:
        W: Affinity matrix of shape ``(n, n)``.
        k: Number of eigenvectors / target clusters, ``1 <= k <= n``.

    Returns:
        An ``(n, k)`` embedding whose rows are clustered by k-means downstream.

    Raises:
        ValueError: If ``k`` is out of range.
    """
    L = normalized_laplacian(W)
    n = L.shape[0]
    if not isinstance(k, int) or k < 1 or k > n:
        raise ValueError(f"k must be an integer in [1, n]={n}; got {k}")
    _, vecs = jacobi_eigh(L)
    U = vecs[:, :k].copy()
    norms = np.sqrt((U * U).sum(axis=1))
    nonzero = norms > 1e-12
    U[nonzero] = U[nonzero] / norms[nonzero, None]
    return U


def kmeans(
    X: Sequence[Sequence[float]],
    k: int,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[list[int], np.ndarray]:
    """Lloyd's k-means with deterministic furthest-point seeding.

    The initialization is fully deterministic: the first center is the data row
    with the smallest (lexicographic) coordinates, and each subsequent center is
    the point furthest from the set of chosen centers, ties broken by lowest
    index. This removes the random seeding that would otherwise prevent
    cross-language parity.

    Args:
        X: Points of shape ``(n_samples, n_features)``.
        k: Number of clusters, ``1 <= k <= n``.
        max_iter: Maximum Lloyd iterations.
        tol: Convergence threshold on the total center shift.

    Returns:
        Tuple ``(labels, centers)`` where ``labels[i]`` is the cluster index of
        point ``i`` and ``centers`` has shape ``(k, n_features)``.

    Raises:
        ValueError: If ``X`` is malformed or ``k`` is out of range.
    """
    arr = _as_matrix(X)
    n = arr.shape[0]
    if not isinstance(k, int) or k < 1 or k > n:
        raise ValueError(f"k must be an integer in [1, n]={n}; got {k}")

    # Deterministic seed 0: lexicographically smallest point.
    first = min(range(n), key=lambda i: tuple(arr[i]))
    center_idx = [first]
    while len(center_idx) < k:
        best_i = -1
        best_d = -1.0
        for i in range(n):
            if i in center_idx:
                continue
            d = min(float((arr[i] - arr[c]) @ (arr[i] - arr[c])) for c in center_idx)
            if d > best_d:
                best_d = d
                best_i = i
        center_idx.append(best_i)
    centers = arr[center_idx].astype(float).copy()

    labels = [0] * n
    for _ in range(max_iter):
        # Assignment step (ties broken by lowest cluster index).
        for i in range(n):
            best_c = 0
            best_d = float("inf")
            for c in range(k):
                diff = arr[i] - centers[c]
                d = float(diff @ diff)
                if d < best_d:
                    best_d = d
                    best_c = c
            labels[i] = best_c
        # Update step. Empty clusters keep their previous center.
        new_centers = centers.copy()
        for c in range(k):
            members = [i for i in range(n) if labels[i] == c]
            if members:
                new_centers[c] = arr[members].mean(axis=0)
        shift = float(np.sqrt(((new_centers - centers) ** 2).sum()))
        centers = new_centers
        if shift <= tol:
            break
    return labels, centers


def spectral_clustering(
    X: Sequence[Sequence[float]],
    k: int,
    sigma: float = 1.0,
) -> list[int]:
    """End-to-end spectral clustering (Ng-Jordan-Weiss).

    Builds an RBF affinity matrix, computes the row-normalized spectral
    embedding from the symmetric normalized Laplacian, and clusters the rows
    with deterministic k-means.

    Args:
        X: Data of shape ``(n_samples, n_features)``.
        k: Number of clusters, ``1 <= k <= n_samples``.
        sigma: Positive RBF bandwidth.

    Returns:
        A list of integer cluster labels, one per sample.

    Raises:
        ValueError: If any argument is malformed.
    """
    W = rbf_affinity(X, sigma)
    embedding = spectral_embedding(W, k)
    labels, _ = kmeans(embedding, k)
    return labels
