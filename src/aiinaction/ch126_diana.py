"""DIANA: divisive (top-down) hierarchical clustering.

A small, NumPy-backed reference implementation of the DIANA (Divisive Analysis)
algorithm of Kaufman and Rousseeuw, with the Macnaughton-Smith splinter-group
heuristic at its core. The implementation mirrors the Julia (`AIInAction.Ch126Diana`)
and Rust (`aiinaction::ch126_diana`) versions one-to-one; the cross-language parity
tests assert that all three agree to within floating-point tolerance on shared
fixtures.

The public API is deliberately small:

* :func:`macnaughton_smith_split` -- split one cluster into a splinter group and
  the remainder using averaged dissimilarities.
* :func:`diana` -- run the full top-down algorithm, returning the ordered list of
  splits (a flat dendrogram record).
* :func:`diana_labels` -- cut the resulting hierarchy into ``k`` flat clusters.
* :func:`diameter` -- maximum pairwise dissimilarity inside a cluster.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Split",
    "diameter",
    "macnaughton_smith_split",
    "diana",
    "diana_labels",
]


@dataclass(frozen=True)
class Split:
    """A single recorded division produced by DIANA.

    Attributes
    ----------
    parent:
        Sorted indices of the cluster that was split.
    splinter:
        Sorted indices of the breakaway (splinter) group.
    remainder:
        Sorted indices of the objects that stayed behind.
    diameter:
        The diameter (largest pairwise dissimilarity) of ``parent`` at the time
        of the split. This is the height at which the split occurs.
    """

    parent: tuple[int, ...]
    splinter: tuple[int, ...]
    remainder: tuple[int, ...]
    diameter: float


def _as_distance_matrix(d: Sequence[Sequence[float]]) -> np.ndarray:
    """Validate and coerce ``d`` into a square, symmetric, hollow distance matrix."""
    arr = np.asarray(d, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"distance matrix must be square 2-D; got shape {arr.shape}"
        )
    n = arr.shape[0]
    if n == 0:
        raise ValueError("distance matrix must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("distance matrix must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError("distance matrix must be non-negative")
    if not np.allclose(arr, arr.T, atol=1e-12):
        raise ValueError("distance matrix must be symmetric")
    if np.any(np.abs(np.diag(arr)) > 1e-12):
        raise ValueError("distance matrix must have a zero diagonal")
    return arr


def diameter(dist: Sequence[Sequence[float]], members: Sequence[int]) -> float:
    """Return the diameter of ``members``: the largest pairwise dissimilarity.

    A cluster with fewer than two members has diameter ``0.0``.

    >>> diameter([[0.0, 1.0, 4.0], [1.0, 0.0, 2.0], [4.0, 2.0, 0.0]], [0, 1, 2])
    4.0
    """
    arr = _as_distance_matrix(dist)
    idx = list(members)
    _check_members(idx, arr.shape[0])
    if len(idx) < 2:
        return 0.0
    sub = arr[np.ix_(idx, idx)]
    return float(sub.max())


def _check_members(members: Sequence[int], n: int) -> None:
    if len(members) == 0:
        raise ValueError("member list must be non-empty")
    seen = set()
    for i in members:
        if not (0 <= i < n):
            raise ValueError(f"member index {i} out of range [0, {n})")
        if i in seen:
            raise ValueError(f"duplicate member index {i}")
        seen.add(i)


def macnaughton_smith_split(
    dist: Sequence[Sequence[float]], members: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split one cluster into ``(splinter, remainder)`` via Macnaughton-Smith.

    The splinter group is seeded with the object whose average dissimilarity to
    the rest of the cluster is largest, then grown one object at a time: at each
    step the object maximizing

    .. math::

        D(i) = \\operatorname{avg\\_to\\_remainder}(i) - \\operatorname{avg\\_to\\_splinter}(i)

    is moved into the splinter if that maximum is strictly positive. The process
    stops when no remaining object prefers the splinter.

    Both returned tuples are sorted ascending. A singleton cluster splits into an
    empty splinter and itself.

    Raises
    ------
    ValueError
        If ``members`` is empty, contains duplicates, or indexes outside the
        distance matrix.
    """
    arr = _as_distance_matrix(dist)
    idx = list(members)
    _check_members(idx, arr.shape[0])

    if len(idx) < 2:
        return (tuple(), tuple(sorted(idx)))

    remainder = list(idx)
    splinter: list[int] = []

    # Seed: object with largest average dissimilarity to the rest of the cluster.
    best_seed = None
    best_avg = -np.inf
    for i in remainder:
        others = [j for j in remainder if j != i]
        avg = float(np.mean([arr[i, j] for j in others]))
        if avg > best_avg:
            best_avg = avg
            best_seed = i
    assert best_seed is not None
    remainder.remove(best_seed)
    splinter.append(best_seed)

    # Grow the splinter while some object prefers it.
    while len(remainder) > 1:
        best_obj = None
        best_d = 0.0
        for i in remainder:
            others = [j for j in remainder if j != i]
            avg_rem = float(np.mean([arr[i, j] for j in others]))
            avg_spl = float(np.mean([arr[i, j] for j in splinter]))
            d_i = avg_rem - avg_spl
            if d_i > best_d:
                best_d = d_i
                best_obj = i
        if best_obj is None:
            break
        remainder.remove(best_obj)
        splinter.append(best_obj)

    return (tuple(sorted(splinter)), tuple(sorted(remainder)))


def diana(dist: Sequence[Sequence[float]]) -> list[Split]:
    """Run DIANA to completion, returning the ordered list of splits.

    Starting from the single cluster containing all ``n`` objects, repeatedly
    select the leaf cluster of largest diameter and split it with
    :func:`macnaughton_smith_split`. The returned list has exactly ``n - 1``
    entries (one per internal node of the dendrogram), in the order the splits
    were performed (largest diameter first).

    Raises
    ------
    ValueError
        If the distance matrix is invalid (see :func:`_as_distance_matrix`).
    """
    arr = _as_distance_matrix(dist)
    n = arr.shape[0]
    if n == 1:
        return []

    # Active leaf clusters that still have more than one member.
    clusters: list[tuple[int, ...]] = [tuple(range(n))]
    splits: list[Split] = []

    while clusters:
        # Pick the cluster with the largest diameter; ties broken by first found.
        best_k = 0
        best_diam = -1.0
        for k, c in enumerate(clusters):
            dm = diameter(arr, c)
            if dm > best_diam:
                best_diam = dm
                best_k = k
        target = clusters.pop(best_k)
        splinter, remainder = macnaughton_smith_split(arr, target)
        splits.append(
            Split(
                parent=tuple(sorted(target)),
                splinter=splinter,
                remainder=remainder,
                diameter=best_diam,
            )
        )
        for part in (splinter, remainder):
            if len(part) > 1:
                clusters.append(part)

    return splits


def diana_labels(dist: Sequence[Sequence[float]], k: int) -> list[int]:
    """Cut the DIANA hierarchy into exactly ``k`` flat clusters.

    Performs the first ``k - 1`` splits (in DIANA's largest-diameter-first order)
    and labels each object ``0 .. k-1`` by the cluster it lands in. Labels are
    assigned in order of each cluster's smallest member index, so the result is
    deterministic.

    Raises
    ------
    ValueError
        If ``k`` is not in ``1 .. n``.
    """
    arr = _as_distance_matrix(dist)
    n = arr.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"k must be in [1, {n}]; got {k}")

    clusters: list[tuple[int, ...]] = [tuple(range(n))]
    while len(clusters) < k:
        # Pick the largest-diameter cluster among those with > 1 member.
        best_k = -1
        best_diam = -1.0
        for i, c in enumerate(clusters):
            if len(c) < 2:
                continue
            dm = diameter(arr, c)
            if dm > best_diam:
                best_diam = dm
                best_k = i
        if best_k < 0:
            break  # all clusters are singletons; cannot split further
        target = clusters.pop(best_k)
        splinter, remainder = macnaughton_smith_split(arr, target)
        clusters.append(splinter)
        clusters.append(remainder)

    clusters.sort(key=lambda c: min(c))
    labels = [0] * n
    for label, c in enumerate(clusters):
        for obj in c:
            labels[obj] = label
    return labels
