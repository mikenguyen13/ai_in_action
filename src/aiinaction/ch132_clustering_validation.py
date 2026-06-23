"""Clustering validation metrics (Chapter 132).

From-scratch reference implementations of the four workhorse clustering
validation indices:

- ``silhouette_score`` -- mean per-point silhouette (internal, higher is better).
- ``davies_bouldin_index`` -- average worst-case cluster similarity (internal,
  lower is better).
- ``calinski_harabasz_index`` -- variance ratio criterion (internal, higher is
  better).
- ``adjusted_rand_index`` -- chance-corrected pair-counting agreement against a
  reference labeling (external, 1.0 perfect, ~0 for chance, can go negative).

These mirror the Julia (`AIInAction.Ch132ClusteringValidation`) and Rust
(`aiinaction::ch132_clustering_validation`) implementations one-to-one; the
cross-language parity tests assert all three agree to floating-point tolerance on
the shared fixtures in ``tests/test_ch132_clustering_validation.py``.

Points are rows of a 2-D array ``X`` of shape ``(n_samples, n_features)``. Labels
are an integer vector of length ``n_samples``; their actual values are arbitrary
(only the induced partition matters).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "silhouette_score",
    "davies_bouldin_index",
    "calinski_harabasz_index",
    "adjusted_rand_index",
]


def _as_matrix(X: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    return arr


def _as_labels(labels: Sequence[int], n: int) -> np.ndarray:
    lab = np.asarray(labels)
    if lab.ndim != 1:
        raise ValueError(f"labels must be 1-D, got ndim={lab.ndim}")
    if lab.shape[0] != n:
        raise ValueError(f"length mismatch: len(labels)={lab.shape[0]} != n_samples={n}")
    return lab.astype(np.int64)


def _check_n_clusters(unique: np.ndarray, n: int) -> None:
    k = unique.shape[0]
    if k < 2:
        raise ValueError(f"need at least 2 clusters, got {k}")
    if k > n:
        raise ValueError(f"number of clusters ({k}) cannot exceed number of samples ({n})")


def silhouette_score(X: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Mean silhouette coefficient over all samples (Euclidean distance).

    For point ``i`` in cluster ``C``, ``a(i)`` is the mean distance to the other
    points of ``C`` and ``b(i)`` is the minimum, over other clusters, of the mean
    distance to that cluster. The point silhouette is
    ``s(i) = (b(i) - a(i)) / max(a(i), b(i))`` in ``[-1, 1]``; points alone in
    their cluster get ``s(i) = 0`` by convention. The score is the mean of
    ``s(i)``.

    Requires at least 2 clusters.

    >>> round(silhouette_score([[0.0], [0.1], [10.0], [10.1]], [0, 0, 1, 1]), 6)
    0.989796
    """
    arr = _as_matrix(X)
    n = arr.shape[0]
    lab = _as_labels(labels, n)
    unique = np.unique(lab)
    _check_n_clusters(unique, n)

    # Full pairwise Euclidean distance matrix.
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    members = {int(c): np.where(lab == c)[0] for c in unique}
    sizes = {c: idx.shape[0] for c, idx in members.items()}

    s = np.zeros(n, dtype=float)
    for i in range(n):
        own = int(lab[i])
        if sizes[own] <= 1:
            s[i] = 0.0
            continue
        own_idx = members[own]
        a_i = float(np.sum(dist[i, own_idx])) / (sizes[own] - 1)  # excludes self (d=0)
        b_i = math.inf
        for c, idx in members.items():
            if c == own:
                continue
            mean_to_c = float(np.mean(dist[i, idx]))
            if mean_to_c < b_i:
                b_i = mean_to_c
        denom = max(a_i, b_i)
        s[i] = 0.0 if denom == 0.0 else (b_i - a_i) / denom
    return float(np.mean(s))


def davies_bouldin_index(X: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Davies-Bouldin index (lower is better; 0 is the ideal).

    Cluster scatter ``S_j`` is the mean Euclidean distance of cluster members to
    the centroid; separation ``M_{jm}`` is the distance between centroids. The
    index averages, over clusters ``j``, the worst-case ratio
    ``max_{m != j} (S_j + S_m) / M_{jm}``.

    Requires at least 2 clusters. Raises ``ValueError`` if two clusters share a
    centroid (separation zero).
    """
    arr = _as_matrix(X)
    n = arr.shape[0]
    lab = _as_labels(labels, n)
    unique = np.unique(lab)
    _check_n_clusters(unique, n)

    k = unique.shape[0]
    centroids = np.zeros((k, arr.shape[1]), dtype=float)
    scatter = np.zeros(k, dtype=float)
    for j, c in enumerate(unique):
        idx = np.where(lab == c)[0]
        pts = arr[idx]
        centroids[j] = pts.mean(axis=0)
        d = np.sqrt(np.sum((pts - centroids[j]) ** 2, axis=1))
        scatter[j] = float(np.mean(d))

    total = 0.0
    for j in range(k):
        worst = 0.0
        for m in range(k):
            if m == j:
                continue
            sep = float(np.sqrt(np.sum((centroids[j] - centroids[m]) ** 2)))
            if sep == 0.0:
                raise ValueError(
                    f"clusters {int(unique[j])} and {int(unique[m])} have identical "
                    "centroids; Davies-Bouldin is undefined (zero separation)"
                )
            ratio = (scatter[j] + scatter[m]) / sep
            if ratio > worst:
                worst = ratio
        total += worst
    return total / k


def calinski_harabasz_index(X: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Calinski-Harabasz index, a.k.a. variance ratio criterion (higher is better).

    With between-cluster scatter ``B = sum_j |C_j| * ||mu_j - mu||^2`` and
    within-cluster scatter ``W = sum_j sum_{i in C_j} ||x_i - mu_j||^2``, the index
    is ``(B / W) * (n - k) / (k - 1)``.

    Requires at least 2 clusters. Raises ``ValueError`` if ``W`` is zero (every
    cluster collapsed to a single point), which makes the ratio undefined.
    """
    arr = _as_matrix(X)
    n = arr.shape[0]
    lab = _as_labels(labels, n)
    unique = np.unique(lab)
    _check_n_clusters(unique, n)

    k = unique.shape[0]
    grand = arr.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in unique:
        idx = np.where(lab == c)[0]
        pts = arr[idx]
        mu = pts.mean(axis=0)
        between += idx.shape[0] * float(np.sum((mu - grand) ** 2))
        within += float(np.sum((pts - mu) ** 2))
    if within == 0.0:
        raise ValueError(
            "within-cluster scatter is zero; Calinski-Harabasz is undefined"
        )
    return (between / within) * (n - k) / (k - 1)


def adjusted_rand_index(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    """Adjusted Rand Index between two labelings (external, chance-corrected).

    Invariant to relabeling. Returns 1.0 for perfect agreement, ~0.0 for chance,
    and may be negative when agreement is worse than expected by chance. Computed
    from the contingency table via

        ARI = (sum_ij C(n_ij, 2) - E) / (0.5 (sum_i C(a_i, 2) + sum_j C(b_j, 2)) - E)

    where ``E = [sum_i C(a_i, 2)][sum_j C(b_j, 2)] / C(n, 2)`` and ``C(x, 2)`` is
    the binomial coefficient ``x (x - 1) / 2``.

    When both labelings are trivial (one cluster each, or all singletons) the ARI
    is defined to be 1.0 (perfect agreement of degenerate partitions).

    >>> adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0])
    1.0
    """
    lt = np.asarray(labels_true)
    lp = np.asarray(labels_pred)
    if lt.ndim != 1 or lp.ndim != 1:
        raise ValueError("labels_true and labels_pred must be 1-D")
    if lt.shape[0] != lp.shape[0]:
        raise ValueError(
            f"length mismatch: len(labels_true)={lt.shape[0]} != "
            f"len(labels_pred)={lp.shape[0]}"
        )
    n = lt.shape[0]
    if n == 0:
        raise ValueError("inputs must be non-empty")

    true_vals = {v: i for i, v in enumerate(np.unique(lt))}
    pred_vals = {v: i for i, v in enumerate(np.unique(lp))}
    table = np.zeros((len(true_vals), len(pred_vals)), dtype=np.int64)
    for a, b in zip(lt.tolist(), lp.tolist()):
        table[true_vals[a], pred_vals[b]] += 1

    def comb2(x: np.ndarray | int) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(x * (x - 1.0) / 2.0))

    sum_ij = comb2(table)
    sum_a = comb2(table.sum(axis=1))
    sum_b = comb2(table.sum(axis=0))
    total_pairs = n * (n - 1) / 2.0

    expected = (sum_a * sum_b) / total_pairs if total_pairs > 0 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0.0:
        # Both partitions degenerate (e.g. all-in-one or all-singletons): perfect.
        return 1.0
    return (sum_ij - expected) / denom
