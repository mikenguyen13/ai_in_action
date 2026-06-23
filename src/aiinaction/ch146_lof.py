"""Local Outlier Factor (LOF) from scratch.

A small, well-validated reference implementation of the Local Outlier Factor of
Breunig, Kriegel, Ng, and Sander (2000). The public API mirrors the Julia
(`AIInAction.Ch146Lof`) and Rust (`aiinaction::ch146_lof`) implementations
one-to-one; the cross-language parity tests assert that all three agree to within
floating-point tolerance on shared fixtures.

The algorithm scores each point by how much sparser its local neighborhood is than
the neighborhoods of its own ``k`` nearest neighbors:

1. For every point compute the Euclidean distances to all others and take the
   ``k`` nearest as its neighborhood ``N_k(x)``. The ``k``-distance ``kdist(x)`` is
   the distance to the ``k``-th nearest neighbor.
2. The reachability distance of ``x`` from ``y`` is
   ``reach_dist(x, y) = max(kdist(y), dist(x, y))``, a flooring that stabilizes
   density estimates for very close points.
3. The local reachability density ``lrd(x)`` is the inverse mean reachability
   distance from ``x`` to its neighbors.
4. ``LOF(x)`` is the mean ratio ``lrd(y) / lrd(x)`` over ``y in N_k(x)``. A value
   near ``1`` is normal; ``LOF(x) >> 1`` marks a local outlier.

Ties in distance are broken by ascending point index, so the neighborhood of every
point is deterministic and identical across the three languages.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "euclidean",
    "knn_distances",
    "k_distance",
    "lrd",
    "lof_scores",
    "top_anomalies",
]

Matrix = Sequence[Sequence[float]]


def _as_matrix(X: Matrix) -> list[list[float]]:
    rows = [[float(v) for v in row] for row in X]
    if not rows:
        raise ValueError("X must have at least one row")
    d = len(rows[0])
    if d < 1:
        raise ValueError("X must have at least one feature")
    for r in rows:
        if len(r) != d:
            raise ValueError("all rows must have the same number of features")
        for v in r:
            if not math.isfinite(v):
                raise ValueError("X contains non-finite values (nan or inf)")
    return rows


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean (L2) distance between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} != {len(b)}")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _pairwise(rows: list[list[float]]) -> list[list[float]]:
    n = len(rows)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dij = euclidean(rows[i], rows[j])
            dist[i][j] = dij
            dist[j][i] = dij
    return dist


def _neighbors(dist_row: list[float], i: int, k: int, n: int) -> list[int]:
    """Indices of the ``k`` nearest neighbors of point ``i`` (excluding itself).

    Ties are broken by ascending index, giving a deterministic neighborhood.
    """
    order = sorted((j for j in range(n) if j != i), key=lambda j: (dist_row[j], j))
    return order[:k]


def _check_k(k: int, n: int) -> int:
    if not isinstance(k, int):
        raise ValueError(f"k must be an integer, got {type(k).__name__}")
    if k < 1 or k > n - 1:
        raise ValueError(f"k must be in [1, {n - 1}] for {n} points, got {k}")
    return k


def knn_distances(X: Matrix, k: int) -> list[list[int]]:
    """Return, for each point, the indices of its ``k`` nearest neighbors.

    Examples
    --------
    >>> knn_distances([[0.0], [1.0], [5.0]], k=1)
    [[1], [0], [1]]
    """
    rows = _as_matrix(X)
    n = len(rows)
    _check_k(k, n)
    dist = _pairwise(rows)
    return [_neighbors(dist[i], i, k, n) for i in range(n)]


def k_distance(X: Matrix, k: int) -> list[float]:
    """Return the ``k``-distance (distance to the ``k``-th nearest neighbor) of each point."""
    rows = _as_matrix(X)
    n = len(rows)
    _check_k(k, n)
    dist = _pairwise(rows)
    out = []
    for i in range(n):
        nbrs = _neighbors(dist[i], i, k, n)
        out.append(dist[i][nbrs[-1]])
    return out


def _lrd_from(dist: list[list[float]], neighbors: list[list[int]], kdist: list[float]) -> list[float]:
    n = len(dist)
    lrd_vals = [0.0] * n
    for i in range(n):
        nbrs = neighbors[i]
        total = 0.0
        for y in nbrs:
            total += max(kdist[y], dist[i][y])
        mean_reach = total / len(nbrs)
        if mean_reach == 0.0:
            lrd_vals[i] = math.inf
        else:
            lrd_vals[i] = 1.0 / mean_reach
    return lrd_vals


def lrd(X: Matrix, k: int) -> list[float]:
    """Local reachability density of each point.

    ``lrd(x) = 1 / mean_{y in N_k(x)} reach_dist(x, y)`` where
    ``reach_dist(x, y) = max(kdist(y), dist(x, y))``.
    """
    rows = _as_matrix(X)
    n = len(rows)
    _check_k(k, n)
    dist = _pairwise(rows)
    neighbors = [_neighbors(dist[i], i, k, n) for i in range(n)]
    kdist = [dist[i][neighbors[i][-1]] for i in range(n)]
    return _lrd_from(dist, neighbors, kdist)


def lof_scores(X: Matrix, k: int) -> list[float]:
    """Local Outlier Factor score of every point in ``X``.

    Parameters
    ----------
    X:
        Data matrix of shape ``(n, d)`` with ``n >= 2`` points and ``d >= 1``
        features. Must contain only finite values.
    k:
        Neighborhood size, ``1 <= k <= n - 1``.

    Returns
    -------
    list[float]
        ``LOF(x_i)`` for each point. Values near ``1`` are inliers; values much
        greater than ``1`` are local outliers.

    Examples
    --------
    >>> s = lof_scores([[0.0], [0.5], [1.0], [10.0]], k=2)
    >>> max(range(len(s)), key=lambda i: s[i])
    3
    """
    rows = _as_matrix(X)
    n = len(rows)
    _check_k(k, n)
    dist = _pairwise(rows)
    neighbors = [_neighbors(dist[i], i, k, n) for i in range(n)]
    kdist = [dist[i][neighbors[i][-1]] for i in range(n)]
    lrd_vals = _lrd_from(dist, neighbors, kdist)

    scores = [0.0] * n
    for i in range(n):
        nbrs = neighbors[i]
        li = lrd_vals[i]
        if math.isinf(li):
            # x sits on top of its neighbors: density ratio collapses to the
            # average of neighbor densities over an infinite own-density.
            ratios = [0.0 if math.isinf(lrd_vals[y]) else 0.0 for y in nbrs]
            scores[i] = sum(ratios) / len(nbrs)
        else:
            total = 0.0
            for y in nbrs:
                ly = lrd_vals[y]
                total += math.inf if math.isinf(ly) else ly / li
            scores[i] = total / len(nbrs)
    return scores


def top_anomalies(X: Matrix, k: int, m: int) -> list[int]:
    """Indices of the ``m`` highest-LOF points, most anomalous first.

    Ties in score are broken by ascending point index.
    """
    if not isinstance(m, int) or m < 1:
        raise ValueError(f"m must be a positive integer, got {m}")
    scores = lof_scores(X, k)
    n = len(scores)
    if m > n:
        raise ValueError(f"m must be in [1, {n}] for {n} points, got {m}")
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    return order[:m]
