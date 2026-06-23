"""Clustering at scale: mini-batch k-means, BIRCH features, canopy, k-means||.

Small, dependency-light reference implementations with explicit input validation.
These mirror the Julia (`AIInAction.Ch133ClusteringAtScale`) and Rust
(`aiinaction::ch133_clustering_at_scale`) implementations one-to-one; the
cross-language parity tests assert that all three agree to within floating-point
tolerance on shared fixtures.

The algorithms here trade exactness for scalability:

* :class:`ClusteringFeature` is the additive ``(N, LS, SS)`` sufficient statistic
  at the heart of BIRCH. It is mergeable, so partition summaries compose.
* :func:`mini_batch_kmeans` performs Sculley's stochastic centroid update with a
  per-center ``1/count`` learning rate.
* :func:`canopy_clustering` is the McCallum-Nigam-Ungar blocking step with two
  thresholds ``T1 > T2``.
* :func:`kmeans_parallel_init` is the oversampling seed of Bahmani et al.

Determinism: every routine that samples takes an explicit integer ``seed`` and
uses a tiny self-contained linear congruential generator so that Python, Julia,
and Rust produce byte-identical random streams on the shared fixtures.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "ClusteringFeature",
    "Lcg",
    "squared_distance",
    "nearest_centroid",
    "inertia",
    "mini_batch_kmeans",
    "canopy_clustering",
    "kmeans_parallel_init",
]

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


# --------------------------------------------------------------------------- #
# Deterministic RNG (32-bit linear congruential generator, Numerical Recipes). #
# --------------------------------------------------------------------------- #
class Lcg:
    """Minimal 32-bit linear congruential generator for reproducible sampling.

    Uses the Numerical Recipes constants ``a = 1664525``, ``c = 1013904223`` with
    modulus ``2**32``. The exact integer recurrence is mirrored in the Julia and
    Rust ports so all three languages draw the identical sequence.

    >>> rng = Lcg(7)
    >>> rng.next_below(10)
    8
    """

    _A = 1664525
    _C = 1013904223
    _M = 1 << 32

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & 0xFFFFFFFF

    def next_u32(self) -> int:
        """Advance the state and return the new 32-bit value."""
        self.state = (self._A * self.state + self._C) % self._M
        return self.state

    def next_below(self, bound: int) -> int:
        """Return an integer in ``[0, bound)`` via modulo reduction."""
        if bound <= 0:
            raise ValueError(f"bound must be positive, got {bound}")
        return self.next_u32() % bound


# --------------------------------------------------------------------------- #
# Geometry helpers.                                                           #
# --------------------------------------------------------------------------- #
def _as_matrix(points: Matrix, name: str = "points") -> list[list[float]]:
    rows = [[float(v) for v in row] for row in points]
    if not rows:
        raise ValueError(f"{name} must be non-empty")
    dim = len(rows[0])
    if dim == 0:
        raise ValueError(f"{name} rows must have at least one dimension")
    for i, row in enumerate(rows):
        if len(row) != dim:
            raise ValueError(
                f"{name} are ragged: row 0 has {dim} dims but row {i} has {len(row)}"
            )
    return rows


def squared_distance(a: Vector, b: Vector) -> float:
    """Squared Euclidean distance between two equal-length vectors.

    >>> squared_distance([0.0, 0.0], [3.0, 4.0])
    25.0
    """
    av = [float(v) for v in a]
    bv = [float(v) for v in b]
    if len(av) != len(bv):
        raise ValueError(f"dimension mismatch: {len(av)} != {len(bv)}")
    if not av:
        raise ValueError("vectors must be non-empty")
    return sum((x - y) ** 2 for x, y in zip(av, bv))


def nearest_centroid(point: Vector, centroids: Matrix) -> tuple[int, float]:
    """Return ``(index, squared_distance)`` of the nearest centroid to ``point``.

    Ties resolve to the lowest index.
    """
    cs = _as_matrix(centroids, "centroids")
    best_j = 0
    best_d = squared_distance(point, cs[0])
    for j in range(1, len(cs)):
        d = squared_distance(point, cs[j])
        if d < best_d:
            best_d = d
            best_j = j
    return best_j, best_d


def inertia(points: Matrix, centroids: Matrix) -> float:
    """Total within-cluster sum of squared distances (the k-means objective)."""
    pts = _as_matrix(points, "points")
    cs = _as_matrix(centroids, "centroids")
    return sum(nearest_centroid(p, cs)[1] for p in pts)


# --------------------------------------------------------------------------- #
# BIRCH clustering feature: the additive (N, LS, SS) sufficient statistic.     #
# --------------------------------------------------------------------------- #
class ClusteringFeature:
    """BIRCH clustering feature ``CF = (N, LS, SS)``.

    ``N`` is the point count, ``LS`` the linear sum (a vector), and ``SS`` the
    sum of squared L2 norms (a scalar). From these three numbers alone the
    centroid and radius of a subcluster are recoverable, and two CFs add to
    summarize the union of their points -- the property that lets BIRCH cluster a
    huge dataset in one pass with bounded memory.

    >>> cf = ClusteringFeature.from_points([[1.0, 1.0], [3.0, 3.0]])
    >>> cf.centroid()
    [2.0, 2.0]
    >>> round(cf.radius(), 6)
    1.414214
    """

    def __init__(self, n: int, ls: Vector, ss: float) -> None:
        if n < 0:
            raise ValueError(f"count N must be non-negative, got {n}")
        self.n = int(n)
        self.ls = [float(v) for v in ls]
        if not self.ls:
            raise ValueError("linear sum LS must be non-empty")
        self.ss = float(ss)

    @property
    def dim(self) -> int:
        return len(self.ls)

    @classmethod
    def from_points(cls, points: Matrix) -> "ClusteringFeature":
        """Build a CF from raw points."""
        pts = _as_matrix(points, "points")
        dim = len(pts[0])
        ls = [0.0] * dim
        ss = 0.0
        for p in pts:
            for d in range(dim):
                ls[d] += p[d]
            ss += sum(v * v for v in p)
        return cls(len(pts), ls, ss)

    def merge(self, other: "ClusteringFeature") -> "ClusteringFeature":
        """Return the additive union ``self + other``."""
        if self.dim != other.dim:
            raise ValueError(f"dimension mismatch: {self.dim} != {other.dim}")
        ls = [a + b for a, b in zip(self.ls, other.ls)]
        return ClusteringFeature(self.n + other.n, ls, self.ss + other.ss)

    def centroid(self) -> list[float]:
        """Centroid ``LS / N``."""
        if self.n == 0:
            raise ValueError("centroid is undefined for an empty CF (N=0)")
        return [v / self.n for v in self.ls]

    def radius(self) -> float:
        """Root-mean-square distance of members to the centroid.

        Derived purely from ``(N, LS, SS)``:
        ``radius^2 = SS/N - ||LS/N||^2``. Small negatives from floating-point
        cancellation are clamped to zero.
        """
        if self.n == 0:
            raise ValueError("radius is undefined for an empty CF (N=0)")
        mean_ss = self.ss / self.n
        centroid_norm_sq = sum(v * v for v in self.ls) / (self.n * self.n)
        var = mean_ss - centroid_norm_sq
        return math.sqrt(var) if var > 0.0 else 0.0


# --------------------------------------------------------------------------- #
# Mini-batch k-means (Sculley 2010).                                          #
# --------------------------------------------------------------------------- #
def mini_batch_kmeans(
    points: Matrix,
    centroids: Matrix,
    *,
    batch_size: int,
    n_iter: int,
    seed: int,
) -> list[list[float]]:
    """Run mini-batch k-means and return the final centroids.

    Each iteration draws ``batch_size`` points (with replacement, via the shared
    LCG), assigns each to its nearest current centroid, then moves that centroid
    toward the point with a per-center learning rate ``eta = 1 / count``. This is
    online stochastic averaging of the points absorbed by each center.

    Args:
        points: ``n`` rows of ``d`` coordinates.
        centroids: ``k`` initial centers, each ``d`` coordinates.
        batch_size: number of points sampled per iteration (must be positive).
        n_iter: number of mini-batch iterations (must be non-negative).
        seed: integer seed for the deterministic sampler.

    Returns:
        The updated ``k`` centroids.
    """
    pts = _as_matrix(points, "points")
    cs = [list(row) for row in _as_matrix(centroids, "centroids")]
    dim = len(pts[0])
    if len(cs[0]) != dim:
        raise ValueError(
            f"centroid dim {len(cs[0])} does not match point dim {dim}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n_iter < 0:
        raise ValueError(f"n_iter must be non-negative, got {n_iter}")

    counts = [0] * len(cs)
    rng = Lcg(seed)
    n = len(pts)
    for _ in range(n_iter):
        batch = [pts[rng.next_below(n)] for _ in range(batch_size)]
        # Cache assignments against the centroids as they were at batch start.
        assignments = [nearest_centroid(x, cs)[0] for x in batch]
        for x, j in zip(batch, assignments):
            counts[j] += 1
            eta = 1.0 / counts[j]
            cs[j] = [(1.0 - eta) * cj + eta * xi for cj, xi in zip(cs[j], x)]
    return cs


# --------------------------------------------------------------------------- #
# Canopy clustering (McCallum, Nigam, Ungar 2000).                            #
# --------------------------------------------------------------------------- #
def canopy_clustering(
    points: Matrix,
    *,
    t1: float,
    t2: float,
    seed: int,
) -> list[list[int]]:
    """Partition points into overlapping canopies using two distance thresholds.

    A random un-removed point becomes a canopy center. Every point within squared
    distance ``t1`` joins that canopy; every point within the tighter ``t2`` is
    removed from the pool so it cannot seed another canopy. Thresholds are on
    squared Euclidean distance to match :func:`squared_distance`.

    Args:
        points: ``n`` rows of ``d`` coordinates.
        t1: loose squared-distance threshold (membership).
        t2: tight squared-distance threshold (removal); must satisfy ``t2 <= t1``.
        seed: integer seed selecting canopy centers deterministically.

    Returns:
        A list of canopies, each a sorted list of point indices. Canopies may
        overlap; every point belongs to at least one.
    """
    pts = _as_matrix(points, "points")
    if t1 < 0 or t2 < 0:
        raise ValueError(f"thresholds must be non-negative, got t1={t1}, t2={t2}")
    if t2 > t1:
        raise ValueError(f"require t2 <= t1, got t1={t1}, t2={t2}")

    n = len(pts)
    pool = list(range(n))
    rng = Lcg(seed)
    canopies: list[list[int]] = []
    while pool:
        center_idx = pool[rng.next_below(len(pool))]
        center = pts[center_idx]
        members: list[int] = []
        survivors: list[int] = []
        for idx in pool:
            d = squared_distance(center, pts[idx])
            if d <= t1:
                members.append(idx)
            if d > t2:
                survivors.append(idx)
        canopies.append(sorted(members))
        pool = survivors
    return canopies


# --------------------------------------------------------------------------- #
# k-means|| oversampling initialization (Bahmani et al. 2012).                #
# --------------------------------------------------------------------------- #
def kmeans_parallel_init(
    points: Matrix,
    k: int,
    *,
    oversampling: float,
    n_rounds: int,
    seed: int,
) -> list[list[float]]:
    """Seed ``k`` centers with the scalable k-means|| oversampling scheme.

    One initial center is chosen uniformly at random. Over ``n_rounds`` rounds,
    each remaining point is sampled independently with probability
    ``min(1, oversampling * d^2(x) / phi)`` where ``d^2(x)`` is the point's
    squared distance to the current center set and ``phi`` is the current total
    cost. The oversampled candidate set is then reduced to exactly ``k`` centers
    by greedily picking the candidate that most reduces total cost (a weighted
    k-means++ style reclustering done deterministically here).

    Args:
        points: ``n`` rows of ``d`` coordinates.
        k: number of seeds to return (``1 <= k <= n``).
        oversampling: expected points sampled per round (must be positive).
        n_rounds: number of oversampling rounds (must be non-negative).
        seed: integer seed for the deterministic sampler.

    Returns:
        Exactly ``k`` centers drawn from ``points``.
    """
    pts = _as_matrix(points, "points")
    n = len(pts)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > n:
        raise ValueError(f"k={k} exceeds number of points n={n}")
    if oversampling <= 0:
        raise ValueError(f"oversampling must be positive, got {oversampling}")
    if n_rounds < 0:
        raise ValueError(f"n_rounds must be non-negative, got {n_rounds}")

    rng = Lcg(seed)
    chosen = [rng.next_below(n)]

    def min_sq_to_chosen(idx: int) -> float:
        return min(squared_distance(pts[idx], pts[c]) for c in chosen)

    for _ in range(n_rounds):
        dists = [min_sq_to_chosen(i) for i in range(n)]
        phi = sum(dists)
        if phi <= 0.0:
            break
        for i in range(n):
            if i in chosen:
                continue
            prob = oversampling * dists[i] / phi
            if prob > 1.0:
                prob = 1.0
            # Uniform draw in [0, 1) from the shared 32-bit stream.
            u = rng.next_u32() / Lcg._M
            if u < prob:
                chosen.append(i)

    # De-duplicate while preserving order.
    candidates: list[int] = []
    for c in chosen:
        if c not in candidates:
            candidates.append(c)

    # Reduce candidates to exactly k by farthest-point (greedy cost) selection,
    # always starting from the first chosen center for determinism.
    seeds = [candidates[0]]
    while len(seeds) < k:
        best_idx = -1
        best_gain = -1.0
        for c in candidates:
            if c in seeds:
                continue
            d = min(squared_distance(pts[c], pts[s]) for s in seeds)
            if d > best_gain:
                best_gain = d
                best_idx = c
        if best_idx < 0:
            # Not enough distinct candidates; fall back to any unused point.
            for i in range(n):
                if i not in seeds:
                    best_idx = i
                    break
        seeds.append(best_idx)

    return [list(pts[s]) for s in seeds[:k]]
