"""Isomap (isometric feature mapping) from scratch.

A small, well-validated reference implementation of Isomap, the global manifold
learning method of Tenenbaum, de Silva and Langford (2000). The public API mirrors
the Julia (`AIInAction.Ch143Isomap`) and Rust (`aiinaction::ch143_isomap`)
implementations one-to-one; the cross-language parity tests assert that all three
agree to within floating-point tolerance on shared fixtures.

The algorithm has three stages:

1. **Neighborhood graph.** Connect each point to its ``k`` nearest neighbors by
   Euclidean distance. The graph is symmetrized (an edge is kept if either endpoint
   lists the other as a neighbor), so the resulting weighted graph is undirected.
2. **Geodesics.** Approximate the geodesic distance between every pair of points by
   the shortest path through the graph, computed with the Floyd-Warshall all-pairs
   algorithm. If the graph is disconnected, some distances are infinite and the
   embedding is undefined; this raises ``ValueError``.
3. **Embedding.** Apply classical multidimensional scaling (MDS) to the geodesic
   distance matrix: double-center the squared distances to form the Gram matrix
   ``B = -1/2 H D2 H``, take its top ``d`` eigenpairs, and set ``Y = U_d Lambda_d^{1/2}``.

Sign convention: each embedding coordinate (column of ``Y``) is flipped so its
largest-magnitude entry is positive. This removes the eigenvector sign ambiguity
and makes results reproducible across languages.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "IsomapResult",
    "pairwise_distances",
    "knn_graph",
    "graph_shortest_paths",
    "fit_isomap",
]

Matrix = Sequence[Sequence[float]]

INF = float("inf")


@dataclass(frozen=True)
class IsomapResult:
    """The fitted state of an Isomap embedding.

    Attributes
    ----------
    embedding:
        Low-dimensional coordinates, shape ``(n, n_components)``. Row ``i`` is the
        embedded image of input sample ``i``.
    eigenvalues:
        The top ``n_components`` eigenvalues of the centered Gram matrix ``B``,
        in descending order, shape ``(n_components,)``. Negative eigenvalues are
        clamped to zero (they signal that the geodesic distances are not exactly
        Euclidean-realizable).
    geodesic_distances:
        The ``(n, n)`` matrix of graph shortest-path distances used for the
        embedding.
    n_neighbors:
        The ``k`` used to build the neighborhood graph.
    """

    embedding: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    geodesic_distances: NDArray[np.float64]
    n_neighbors: int

    @property
    def n_samples(self) -> int:
        """Number of embedded samples."""
        return int(self.embedding.shape[0])

    @property
    def n_components(self) -> int:
        """Embedding dimensionality."""
        return int(self.embedding.shape[1])


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


def pairwise_distances(X: Matrix) -> NDArray[np.float64]:
    """Euclidean distance matrix of the rows of ``X``, shape ``(n, n)``.

    The returned matrix is symmetric with a zero diagonal.

    Examples
    --------
    >>> pairwise_distances([[0.0, 0.0], [3.0, 4.0]]).tolist()
    [[0.0, 5.0], [5.0, 0.0]]
    """
    arr = _as_matrix(X)
    diff = arr[:, None, :] - arr[None, :, :]
    sq = np.sum(diff * diff, axis=2)
    # Clamp tiny negatives from round-off before the square root.
    np.maximum(sq, 0.0, out=sq)
    dist = np.sqrt(sq)
    np.fill_diagonal(dist, 0.0)
    return dist


def knn_graph(X: Matrix, n_neighbors: int) -> NDArray[np.float64]:
    """Symmetric weighted k-nearest-neighbor graph as an adjacency matrix.

    Entry ``(i, j)`` is the Euclidean distance ``d(x_i, x_j)`` if ``j`` is among the
    ``n_neighbors`` nearest neighbors of ``i`` *or* ``i`` is among those of ``j``;
    otherwise it is ``inf`` (no edge). The diagonal is ``0``.

    Parameters
    ----------
    X:
        Data matrix of shape ``(n, d)``.
    n_neighbors:
        Number of nearest neighbors per point. Must satisfy
        ``1 <= n_neighbors <= n - 1``.
    """
    arr = _as_matrix(X)
    n = arr.shape[0]
    if not isinstance(n_neighbors, (int, np.integer)):
        raise ValueError(f"n_neighbors must be an integer, got {type(n_neighbors).__name__}")
    n_neighbors = int(n_neighbors)
    if n_neighbors < 1 or n_neighbors > n - 1:
        raise ValueError(
            f"n_neighbors must be in [1, {n - 1}] for {n} samples, got {n_neighbors}"
        )

    dist = pairwise_distances(arr)
    adj = np.full((n, n), INF, dtype=np.float64)
    for i in range(n):
        order = np.argsort(dist[i], kind="stable")
        # order[0] is i itself (distance 0); take the next n_neighbors.
        count = 0
        for j in order:
            if j == i:
                continue
            adj[i, j] = dist[i, j]
            count += 1
            if count == n_neighbors:
                break
    # Symmetrize: keep an edge if it exists in either direction (use the min weight).
    sym = np.minimum(adj, adj.T)
    np.fill_diagonal(sym, 0.0)
    return sym


def graph_shortest_paths(adj: Matrix) -> NDArray[np.float64]:
    """All-pairs shortest-path distances via Floyd-Warshall.

    Parameters
    ----------
    adj:
        A square ``(n, n)`` weighted adjacency matrix. Non-edges are ``inf`` and the
        diagonal is ``0``. The matrix should be symmetric (undirected graph).

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` matrix of shortest-path distances. Entries remain ``inf``
        for pairs in different connected components.
    """
    d = np.array(adj, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"adjacency must be a square matrix, got shape {d.shape}")
    n = d.shape[0]
    for k in range(n):
        dk = d[k]
        # d[i, j] = min(d[i, j], d[i, k] + d[k, j]); vectorized over i, j.
        through_k = d[:, k][:, None] + dk[None, :]
        np.minimum(d, through_k, out=d)
    return d


def _fix_signs(Y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flip each embedding column so its largest-magnitude entry is positive."""
    out = Y.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        k = int(np.argmax(np.abs(col)))
        if col[k] < 0.0:
            out[:, j] = -col
    return out


def classical_mds(distances: Matrix, n_components: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Classical (Torgerson) MDS of a distance matrix.

    Double-centers the squared distances to form ``B = -1/2 H D2 H``, then returns
    the embedding ``Y = U_d Lambda_d^{1/2}`` from the top ``n_components`` eigenpairs
    of ``B`` together with those (clamped, non-negative) eigenvalues.

    Returns
    -------
    (embedding, eigenvalues):
        ``embedding`` has shape ``(n, n_components)``; ``eigenvalues`` has shape
        ``(n_components,)``, in descending order.
    """
    D = np.asarray(distances, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"distances must be a square matrix, got shape {D.shape}")
    n = D.shape[0]
    if not np.all(np.isfinite(D)):
        raise ValueError(
            "distance matrix contains non-finite values; the neighborhood graph is "
            "likely disconnected (increase n_neighbors)"
        )
    if n_components < 1 or n_components > n:
        raise ValueError(f"n_components must be in [1, {n}], got {n_components}")

    D2 = D * D
    # Double centering: B = -1/2 H D2 H, with H = I - 1/n 11^T.
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    total_mean = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + total_mean)
    # Symmetrize to kill round-off asymmetry before the symmetric eigensolver.
    B = 0.5 * (B + B.T)

    eigvals, eigvecs = np.linalg.eigh(B)
    # eigh returns ascending order; reverse to descending.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    top_vals = eigvals[:n_components]
    top_vecs = eigvecs[:, :n_components]
    clamped = np.maximum(top_vals, 0.0)
    Y = top_vecs * np.sqrt(clamped)[None, :]
    Y = _fix_signs(Y)
    return Y, clamped


def fit_isomap(X: Matrix, n_components: int = 2, *, n_neighbors: int = 5) -> IsomapResult:
    """Fit an Isomap embedding of ``X`` into ``n_components`` dimensions.

    Parameters
    ----------
    X:
        Data matrix of shape ``(n, d)`` with ``n >= 2`` samples and ``d >= 1``
        features. Must contain only finite values.
    n_components:
        Target embedding dimensionality. Must satisfy ``1 <= n_components <= n``.
    n_neighbors:
        Number of nearest neighbors used to build the graph, in ``[1, n - 1]``.

    Returns
    -------
    IsomapResult
        The fitted embedding, top eigenvalues, and geodesic distance matrix.

    Raises
    ------
    ValueError
        If inputs are malformed, the parameters are out of range, or the
        neighborhood graph is disconnected (so some geodesic distances are
        infinite).

    Examples
    --------
    >>> # Points on a 1-D path embedded in 2-D recover a line.
    >>> X = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    >>> r = fit_isomap(X, n_components=1, n_neighbors=2)
    >>> r.embedding.shape
    (4, 1)
    """
    arr = _as_matrix(X)
    n = arr.shape[0]
    if not isinstance(n_components, (int, np.integer)):
        raise ValueError(f"n_components must be an integer, got {type(n_components).__name__}")
    n_components = int(n_components)
    if n_components < 1 or n_components > n:
        raise ValueError(f"n_components must be in [1, {n}], got {n_components}")

    adj = knn_graph(arr, n_neighbors)
    geo = graph_shortest_paths(adj)
    if not np.all(np.isfinite(geo)):
        raise ValueError(
            "neighborhood graph is disconnected; some geodesic distances are "
            "infinite. Increase n_neighbors."
        )

    embedding, eigenvalues = classical_mds(geo, n_components)
    return IsomapResult(
        embedding=embedding,
        eigenvalues=eigenvalues,
        geodesic_distances=geo,
        n_neighbors=int(n_neighbors),
    )
