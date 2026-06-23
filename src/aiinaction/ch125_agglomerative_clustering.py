"""Agglomerative hierarchical clustering from scratch.

A small, dependency-free reference implementation of bottom-up hierarchical
clustering. It supports the four Lance-Williams linkages (single, complete,
average, Ward) and returns a linkage matrix in the same row format as
``scipy.cluster.hierarchy.linkage``: each of the ``n - 1`` rows is
``[node_a, node_b, height, size]``, where node ids ``0 .. n-1`` are the original
points and ids ``n .. 2n-2`` are the merges in the order they were formed.

This mirrors the Julia (`AIInAction.Ch125AgglomerativeClustering`) and Rust
(`aiinaction::ch125_agglomerative_clustering`) implementations one-to-one; the
cross-language parity tests assert that all three agree to within floating-point
tolerance on shared fixtures.

Algorithm. We materialize the full pairwise distance matrix and apply the
Lance-Williams recurrence to update cluster distances after each merge. For the
distance-based linkages (single, complete, average) the input distances are used
directly; for Ward linkage the input must be raw coordinates, from which we form
the squared-Euclidean distances the closed-form Ward update requires. The cost is
``O(n^2)`` memory and ``O(n^3)`` time, which keeps the code transparent and is
appropriate for the small didactic datasets in this chapter.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "LINKAGES",
    "linkage_matrix",
    "fcluster",
    "cophenetic_distances",
]

LINKAGES = ("single", "complete", "average", "ward")


def _lance_williams(
    d_ak: float,
    d_bk: float,
    d_ab: float,
    n_a: int,
    n_b: int,
    n_k: int,
    linkage: str,
) -> float:
    """Distance from a merged cluster ``a+b`` to another cluster ``k``.

    Implements the Lance-Williams recurrence

        D(a+b, k) = alpha_a D(a,k) + alpha_b D(b,k) + beta D(a,b)
                    + gamma |D(a,k) - D(b,k)|

    with the coefficient table for each supported linkage. For Ward the inputs
    ``d_ak``, ``d_bk`` and ``d_ab`` are squared Euclidean distances.
    """
    if linkage == "single":
        return 0.5 * d_ak + 0.5 * d_bk - 0.5 * abs(d_ak - d_bk)
    if linkage == "complete":
        return 0.5 * d_ak + 0.5 * d_bk + 0.5 * abs(d_ak - d_bk)
    if linkage == "average":
        total = n_a + n_b
        return (n_a / total) * d_ak + (n_b / total) * d_bk
    if linkage == "ward":
        total = n_a + n_b + n_k
        return (
            (n_a + n_k) / total * d_ak
            + (n_b + n_k) / total * d_bk
            - n_k / total * d_ab
        )
    raise ValueError(f"unknown linkage {linkage!r}; expected one of {LINKAGES}")


def _as_matrix(points: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = [[float(v) for v in row] for row in points]
    if not rows:
        raise ValueError("inputs must be non-empty")
    width = len(rows[0])
    if width == 0:
        raise ValueError("points must have at least one feature")
    for r in rows:
        if len(r) != width:
            raise ValueError("all points must have the same number of features")
    return rows


def _pairwise_sq_euclidean(points: list[list[float]]) -> list[list[float]]:
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = sum((a - b) ** 2 for a, b in zip(points[i], points[j]))
            dist[i][j] = s
            dist[j][i] = s
    return dist


def _pairwise_euclidean(points: list[list[float]]) -> list[list[float]]:
    sq = _pairwise_sq_euclidean(points)
    return [[math.sqrt(v) for v in row] for row in sq]


def linkage_matrix(
    points: Sequence[Sequence[float]],
    linkage: str = "ward",
) -> list[list[float]]:
    """Agglomerative clustering of ``points``; returns a SciPy-style linkage matrix.

    Parameters
    ----------
    points:
        ``n`` rows of coordinates, each with the same number of features.
    linkage:
        One of ``"single"``, ``"complete"``, ``"average"``, ``"ward"``.

    Returns
    -------
    A list of ``n - 1`` rows ``[node_a, node_b, height, size]`` with
    ``node_a < node_b``. Original points have ids ``0 .. n-1``; the merge formed
    at step ``t`` (0-indexed) has id ``n + t``. For Ward the reported height is
    the Euclidean (non-squared) merge distance, matching SciPy.

    >>> m = linkage_matrix([[0.0], [0.0], [5.0]], "single")
    >>> [round(x, 6) for x in m[0]]
    [0.0, 1.0, 0.0, 2.0]
    >>> [round(x, 6) for x in m[1]]
    [2.0, 3.0, 5.0, 3.0]
    """
    if linkage not in LINKAGES:
        raise ValueError(f"unknown linkage {linkage!r}; expected one of {LINKAGES}")
    pts = _as_matrix(points)
    n = len(pts)
    if n < 2:
        raise ValueError("need at least 2 points to cluster")

    # Ward works in squared-Euclidean space; the others use the raw distances.
    if linkage == "ward":
        dist = _pairwise_sq_euclidean(pts)
    else:
        dist = _pairwise_euclidean(pts)

    active = list(range(n))          # current cluster ids
    sizes = {i: 1 for i in range(n)}
    next_id = n
    result: list[list[float]] = []

    for _ in range(n - 1):
        # Find the closest active pair.
        best = math.inf
        bi = bj = -1
        for a_idx in range(len(active)):
            for b_idx in range(a_idx + 1, len(active)):
                d = dist[active[a_idx]][active[b_idx]]
                if d < best:
                    best = d
                    bi, bj = a_idx, b_idx
        ca, cb = active[bi], active[bj]
        n_a, n_b = sizes[ca], sizes[cb]

        height = math.sqrt(best) if linkage == "ward" else best
        node_a, node_b = (ca, cb) if ca < cb else (cb, ca)
        result.append([float(node_a), float(node_b), height, float(n_a + n_b)])

        # Lance-Williams update against every remaining cluster. Compute the new
        # distances before growing the matrix, since the formula reads dist[ca][.]
        # and dist[cb][.].
        new_id = next_id
        next_id += 1
        updates = {}
        for ck in active:
            if ck == ca or ck == cb:
                continue
            updates[ck] = _lance_williams(
                dist[ca][ck], dist[cb][ck], best, n_a, n_b, sizes[ck], linkage
            )
        # Grow the matrix by one row/column for new_id and fill it in.
        for row in dist:
            row.append(0.0)
        dist.append([0.0] * (new_id + 1))
        for ck, d_new in updates.items():
            dist[new_id][ck] = d_new
            dist[ck][new_id] = d_new

        active = [c for c in active if c != ca and c != cb]
        active.append(new_id)
        sizes[new_id] = n_a + n_b

    return result


def cophenetic_distances(linkage_mat: Sequence[Sequence[float]]) -> list[list[float]]:
    """Cophenetic distance matrix induced by a linkage matrix.

    The cophenetic distance between two original points is the height of the merge
    at which they first share a cluster. Returns an ``n x n`` symmetric matrix with
    zero diagonal, where ``n = len(linkage_mat) + 1``.
    """
    rows = [list(r) for r in linkage_mat]
    n = len(rows) + 1
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    coph = [[0.0] * n for _ in range(n)]
    for t, row in enumerate(rows):
        a, b, height = int(row[0]), int(row[1]), float(row[2])
        ma, mb = members[a], members[b]
        for x in ma:
            for y in mb:
                coph[x][y] = height
                coph[y][x] = height
        members[n + t] = ma + mb
    return coph


def fcluster(
    linkage_mat: Sequence[Sequence[float]],
    n_clusters: int,
) -> list[int]:
    """Cut the tree to obtain exactly ``n_clusters`` flat clusters.

    Stops the agglomeration after ``n - n_clusters`` merges and labels each
    original point with a cluster id in ``0 .. n_clusters - 1``. Labels are
    assigned in increasing order of the smallest original-point id in each
    cluster, so the output is deterministic across languages.
    """
    rows = [list(r) for r in linkage_mat]
    n = len(rows) + 1
    if not (1 <= n_clusters <= n):
        raise ValueError(f"n_clusters must be in 1..{n}, got {n_clusters}")

    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    merges_to_apply = n - n_clusters
    for t in range(merges_to_apply):
        a, b = int(rows[t][0]), int(rows[t][1])
        members[n + t] = members[a] + members[b]
        del members[a]
        del members[b]

    clusters = sorted(members.values(), key=min)
    labels = [0] * n
    for cid, group in enumerate(clusters):
        for point in group:
            labels[point] = cid
    return labels
