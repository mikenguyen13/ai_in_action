"""K-Means variants and extensions (Chapter 124).

Production reference implementations of the centroid-based clustering family that
relaxes the assumptions of Lloyd's algorithm: mini-batch K-Means, k-medoids,
k-medians, kernel K-Means, fuzzy c-means, and bisecting K-Means.

The Python implementation here is the executed reference. The Julia module
(``AIInAction.Ch124KMeansVariants``) and the Rust module
(``aiinaction::ch124_kmeans_variants``) mirror the same public API, and the
cross-language parity tests assert that all three agree on the shared fixtures in
``tests/test_ch124_kmeans_variants.py``.

Every routine here is deterministic given its inputs: initialization is supplied by
the caller as explicit centroid indices or centroid coordinates, so the three
language implementations produce bit-comparable results without sharing a random
number generator. This is what makes the parity tests meaningful.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "lloyd_step",
    "mini_batch_update",
    "kmedians_centroid",
    "pam_assign_cost",
    "kernel_assignment_distances",
    "fuzzy_memberships",
    "fuzzy_centroids",
    "bisecting_split",
    "inertia",
    "rbf_kernel_matrix",
]

Matrix = Sequence[Sequence[float]]
Vector = Sequence[float]


def _as_matrix(x: Matrix, name: str = "X") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array of shape (n_samples, n_features), got ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one feature")
    return arr


def _as_centroids(c: Matrix, n_features: int) -> np.ndarray:
    arr = np.asarray(c, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"centroids must be a 2-D array of shape (k, n_features), got ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError("centroids must contain at least one centroid")
    if arr.shape[1] != n_features:
        raise ValueError(
            f"centroid dimension {arr.shape[1]} does not match data dimension {n_features}"
        )
    return arr


def _pairwise_sq_dist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, shape (n, k)."""
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, computed stably.
    diff = X[:, None, :] - C[None, :, :]
    return np.einsum("nkd,nkd->nk", diff, diff)


def lloyd_step(X: Matrix, centroids: Matrix) -> tuple[list[int], list[list[float]]]:
    """One iteration of Lloyd's algorithm: assign then update.

    Assigns each point to its nearest centroid by squared Euclidean distance
    (ties broken toward the lowest index), then recomputes each centroid as the
    mean of its assigned points. A centroid that captures no points is left
    unchanged.

    Returns a tuple ``(labels, new_centroids)`` where ``labels[i]`` is the cluster
    index of point ``i`` and ``new_centroids`` is the updated centroid list.

    >>> labels, c = lloyd_step([[0.0], [0.2], [5.0], [5.2]], [[0.0], [5.0]])
    >>> labels
    [0, 0, 1, 1]
    """
    Xa = _as_matrix(X)
    C = _as_centroids(centroids, Xa.shape[1])
    d2 = _pairwise_sq_dist(Xa, C)
    labels = np.argmin(d2, axis=1)
    new_c = C.copy()
    for j in range(C.shape[0]):
        members = Xa[labels == j]
        if members.shape[0] > 0:
            new_c[j] = members.mean(axis=0)
    return labels.tolist(), new_c.tolist()


def inertia(X: Matrix, centroids: Matrix) -> float:
    """Within-cluster sum of squared distances to the nearest centroid.

    This is the K-Means objective ``J(C) = sum_i min_j ||x_i - c_j||^2``.
    """
    Xa = _as_matrix(X)
    C = _as_centroids(centroids, Xa.shape[1])
    d2 = _pairwise_sq_dist(Xa, C)
    return float(d2.min(axis=1).sum())


def mini_batch_update(
    centroids: Matrix,
    counts: Sequence[float],
    batch: Matrix,
) -> tuple[list[list[float]], list[float]]:
    """One mini-batch K-Means update (Sculley, 2010).

    Each point in ``batch`` is assigned to its nearest current centroid; the per
    centroid count is incremented and the centroid is moved toward the point with
    learning rate ``eta = 1 / count``. Points are processed in order, so a centroid
    moves as later points in the same batch are absorbed (the running-mean form of
    SGD on the K-Means objective).

    Returns ``(new_centroids, new_counts)``.
    """
    C = np.asarray(centroids, dtype=float)
    if C.ndim != 2:
        raise ValueError("centroids must be a 2-D array of shape (k, n_features)")
    cnt = np.asarray(counts, dtype=float)
    if cnt.shape[0] != C.shape[0]:
        raise ValueError(f"counts length {cnt.shape[0]} does not match number of centroids {C.shape[0]}")
    if np.any(cnt < 0):
        raise ValueError("counts must be non-negative")
    B = _as_matrix(batch, name="batch")
    if B.shape[1] != C.shape[1]:
        raise ValueError(f"batch dimension {B.shape[1]} does not match centroid dimension {C.shape[1]}")
    C = C.copy()
    cnt = cnt.copy()
    for x in B:
        d2 = np.sum((C - x) ** 2, axis=1)
        j = int(np.argmin(d2))
        cnt[j] += 1.0
        eta = 1.0 / cnt[j]
        C[j] = (1.0 - eta) * C[j] + eta * x
    return C.tolist(), cnt.tolist()


def kmedians_centroid(members: Matrix) -> list[float]:
    """Coordinatewise median, the L1-optimal cluster representative.

    The minimizer of total L1 deviation along each axis is the per-dimension
    median, so this is the k-medians update step. For an even number of points the
    lower of the two central order statistics is returned, matching the convention
    used by the Julia and Rust implementations so the three agree exactly.
    """
    M = _as_matrix(members, name="members")
    n = M.shape[0]
    out = []
    for d in range(M.shape[1]):
        col = np.sort(M[:, d])
        # Lower median for even n keeps all three languages bit-identical.
        idx = (n - 1) // 2
        out.append(float(col[idx]))
    return out


def pam_assign_cost(distances: Matrix, medoid_indices: Sequence[int]) -> tuple[list[int], float]:
    """Assign every point to its nearest medoid given a dissimilarity matrix.

    ``distances`` is an ``n x n`` matrix of arbitrary (possibly non-metric)
    dissimilarities; ``medoid_indices`` selects ``k`` rows/columns as the current
    medoids. Returns ``(labels, total_cost)`` where ``labels[i]`` is the position
    within ``medoid_indices`` of the medoid nearest to point ``i`` and
    ``total_cost`` is the sum of those nearest dissimilarities (the k-medoids
    objective). This is the assignment core shared by PAM and FastPAM.
    """
    D = np.asarray(distances, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distances must be a square n x n dissimilarity matrix")
    n = D.shape[0]
    med = list(medoid_indices)
    if not med:
        raise ValueError("medoid_indices must be non-empty")
    for m in med:
        if not 0 <= m < n:
            raise ValueError(f"medoid index {m} out of range [0, {n})")
    labels = []
    total = 0.0
    for i in range(n):
        best_pos, best_cost = 0, math.inf
        for pos, m in enumerate(med):
            c = D[i, m]
            if c < best_cost:
                best_cost, best_pos = c, pos
        labels.append(best_pos)
        total += best_cost
    return labels, float(total)


def rbf_kernel_matrix(X: Matrix, gamma: float) -> list[list[float]]:
    """Gaussian (RBF) kernel matrix ``K[i,j] = exp(-gamma ||x_i - x_j||^2)``."""
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    Xa = _as_matrix(X)
    d2 = _pairwise_sq_dist(Xa, Xa)
    return np.exp(-gamma * d2).tolist()


def kernel_assignment_distances(kernel: Matrix, labels: Sequence[int], n_clusters: int) -> list[list[float]]:
    """Feature-space squared distances from each point to every cluster mean.

    Uses the kernel trick: with a precomputed kernel ``K`` and a hard assignment,
    the squared distance from ``phi(x_i)`` to the mean of cluster ``j`` is

        K[i,i] - (2/|S_j|) sum_{l in S_j} K[i,l] + (1/|S_j|^2) sum_{l,m in S_j} K[l,m].

    Returns an ``n x n_clusters`` matrix of these distances; an empty cluster
    yields ``+inf`` for every point (so it is never chosen by argmin).
    """
    K = np.asarray(kernel, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("kernel must be a square n x n matrix")
    n = K.shape[0]
    lab = list(labels)
    if len(lab) != n:
        raise ValueError(f"labels length {len(lab)} does not match kernel size {n}")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    for c in lab:
        if not 0 <= c < n_clusters:
            raise ValueError(f"label {c} out of range [0, {n_clusters})")

    members = [[i for i in range(n) if lab[i] == j] for j in range(n_clusters)]
    third = []  # constant per-cluster term
    for S in members:
        if S:
            sub = K[np.ix_(S, S)]
            third.append(float(sub.sum()) / (len(S) ** 2))
        else:
            third.append(math.inf)

    out = []
    for i in range(n):
        row = []
        for j in range(n_clusters):
            S = members[j]
            if not S:
                row.append(math.inf)
                continue
            cross = float(K[i, S].sum())
            d = K[i, i] - 2.0 * cross / len(S) + third[j]
            row.append(d)
        out.append(row)
    return out


def fuzzy_centroids(X: Matrix, memberships: Matrix, m: float) -> list[list[float]]:
    """Membership-weighted cluster centers for fuzzy c-means.

    ``c_j = sum_i u_ij^m x_i / sum_i u_ij^m`` with fuzziness exponent ``m > 1``.
    """
    if m <= 1.0:
        raise ValueError(f"fuzziness exponent m must be greater than 1, got {m}")
    Xa = _as_matrix(X)
    U = np.asarray(memberships, dtype=float)
    if U.ndim != 2 or U.shape[0] != Xa.shape[0]:
        raise ValueError("memberships must have shape (n_samples, n_clusters)")
    w = U ** m
    denom = w.sum(axis=0)
    if np.any(denom == 0.0):
        raise ValueError("a cluster has zero total membership; cannot form its centroid")
    return ((w.T @ Xa) / denom[:, None]).tolist()


def fuzzy_memberships(X: Matrix, centroids: Matrix, m: float) -> list[list[float]]:
    """Update fuzzy memberships from distances to all centroids (Bezdek FCM).

    ``u_ij = 1 / sum_l (||x_i - c_j|| / ||x_i - c_l||)^{2/(m-1)}``. A point that
    coincides exactly with one or more centroids is assigned membership split
    uniformly across those zero-distance centroids and zero elsewhere.
    """
    if m <= 1.0:
        raise ValueError(f"fuzziness exponent m must be greater than 1, got {m}")
    Xa = _as_matrix(X)
    C = _as_centroids(centroids, Xa.shape[1])
    dist = np.sqrt(_pairwise_sq_dist(Xa, C))  # (n, k)
    p = 2.0 / (m - 1.0)
    n, k = dist.shape
    out = np.zeros((n, k), dtype=float)
    for i in range(n):
        zero = np.where(dist[i] == 0.0)[0]
        if zero.size > 0:
            out[i, zero] = 1.0 / zero.size
            continue
        ratios = (dist[i][:, None] / dist[i][None, :]) ** p  # (k, k)
        out[i] = 1.0 / ratios.sum(axis=1)
    return out.tolist()


def bisecting_split(X: Matrix, init_two_centroids: Matrix) -> tuple[list[int], list[list[float]], float]:
    """Run 2-means to convergence to bisect a cluster.

    Given the points of a cluster and two initial centroids, alternates Lloyd
    steps until the assignment stops changing, returning
    ``(labels, two_centroids, sse)`` where ``labels`` is in ``{0, 1}`` and ``sse``
    is the resulting within-cluster sum of squares. This is the inner split that
    bisecting K-Means repeats top-down to build a divisive hierarchy.
    """
    Xa = _as_matrix(X)
    C = _as_centroids(init_two_centroids, Xa.shape[1])
    if C.shape[0] != 2:
        raise ValueError(f"bisecting requires exactly 2 initial centroids, got {C.shape[0]}")
    prev = None
    centroids = C.tolist()
    labels: list[int] = []
    for _ in range(1000):
        labels, centroids = lloyd_step(Xa.tolist(), centroids)
        if prev is not None and labels == prev:
            break
        prev = labels
    return labels, centroids, inertia(Xa.tolist(), centroids)
