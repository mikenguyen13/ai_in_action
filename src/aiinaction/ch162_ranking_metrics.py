"""Ranking metrics from scratch: MRR, MAP, and NDCG.

Small, well-validated reference implementations of the three workhorse ranking
metrics used in search, recommendation, and retrieval evaluation. The public API
mirrors the Julia (`AIInAction.Ch162RankingMetrics`) and Rust
(`aiinaction::ch162_ranking_metrics`) implementations one-to-one; the
cross-language parity tests assert that all three agree to within floating-point
tolerance on shared fixtures.

Conventions
-----------
* A *relevance list* for a query is the per-position relevance of the items in
  ranked order, position 1 being the top. Binary relevance uses values in
  ``{0, 1}``; graded relevance uses non-negative integers.
* The optional cutoff ``k`` evaluates only the top ``k`` positions. ``None``
  means no cutoff (use the whole list).
* Reciprocal rank, average precision, and NDCG are all ``0`` for a query with no
  relevant item (within the cutoff).

The query-set aggregators (:func:`mean_reciprocal_rank`,
:func:`mean_average_precision`, :func:`mean_ndcg`) average the per-query score
over a non-empty collection of queries, matching ``Metric = (1/|Q|) sum_q
score(q)``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "precision_at_k",
    "average_precision",
    "mean_average_precision",
    "dcg",
    "ndcg",
    "mean_ndcg",
]

RelList = Sequence[float]


def _check_cutoff(k: int | None) -> None:
    if k is not None and k < 1:
        raise ValueError(f"k must be a positive integer or None, got {k}")


def _as_rel(relevances: RelList) -> list[float]:
    rel = [float(r) for r in relevances]
    if any(r < 0.0 for r in rel):
        raise ValueError("relevances must be non-negative")
    if any(not math.isfinite(r) for r in rel):
        raise ValueError("relevances must be finite")
    return rel


def reciprocal_rank(relevances: RelList, k: int | None = None) -> float:
    """Reciprocal rank of the first relevant item.

    A relevance is treated as a hit when it is strictly positive. Returns
    ``1 / rank`` of the first hit within the top ``k``, or ``0.0`` if none.

    >>> reciprocal_rank([0, 0, 1, 0])
    0.3333333333333333
    >>> reciprocal_rank([0, 0, 1, 0], k=2)
    0.0
    """
    _check_cutoff(k)
    rel = _as_rel(relevances)
    cut = len(rel) if k is None else min(k, len(rel))
    for i in range(cut):
        if rel[i] > 0.0:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(queries: Sequence[RelList], k: int | None = None) -> float:
    """Mean reciprocal rank over a non-empty set of queries.

    Each element of ``queries`` is a per-query relevance list in ranked order.

    >>> round(mean_reciprocal_rank([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), 6)
    0.611111
    """
    _check_cutoff(k)
    if len(queries) == 0:
        raise ValueError("queries must be non-empty")
    return sum(reciprocal_rank(q, k) for q in queries) / len(queries)


def precision_at_k(relevances: RelList, k: int) -> float:
    """Precision at cutoff ``k``: fraction of the top ``k`` items that are relevant.

    An item counts as relevant when its relevance is strictly positive. ``k`` may
    exceed the list length, in which case the denominator is still ``k`` (missing
    positions count as non-relevant), matching the standard P@k definition.

    >>> precision_at_k([1, 0, 1, 0], 2)
    0.5
    """
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    rel = _as_rel(relevances)
    cut = min(k, len(rel))
    hits = sum(1 for i in range(cut) if rel[i] > 0.0)
    return hits / k


def average_precision(relevances: RelList, n_relevant: int | None = None, k: int | None = None) -> float:
    """Average precision for a single query under binary relevance.

    Averages the precision computed at each rank holding a relevant item:
    ``AP = (1/R) sum_k P@k * rel(i_k)``, where ``R`` is the total number of
    relevant items for the query.

    Parameters
    ----------
    relevances:
        Per-position relevance in ranked order; positive means relevant.
    n_relevant:
        Total number of relevant items ``R`` for the query. Defaults to the count
        of positive entries in ``relevances`` (i.e. assumes all relevant items are
        present in the list). Must be ``>= 0`` and at least the number of hits
        observed within the cutoff.
    k:
        Optional cutoff; only positions ``1..k`` contribute.

    Returns ``0.0`` when ``R == 0``.

    >>> round(average_precision([1, 0, 1, 0, 0, 1]), 6)
    0.722222
    """
    _check_cutoff(k)
    rel = _as_rel(relevances)
    cut = len(rel) if k is None else min(k, len(rel))

    observed_hits = sum(1 for i in range(cut) if rel[i] > 0.0)
    if n_relevant is None:
        R = sum(1 for r in rel if r > 0.0)
    else:
        if n_relevant < 0:
            raise ValueError(f"n_relevant must be non-negative, got {n_relevant}")
        if n_relevant < observed_hits:
            raise ValueError(
                f"n_relevant={n_relevant} is smaller than the {observed_hits} relevant items observed"
            )
        R = int(n_relevant)

    if R == 0:
        return 0.0

    hits = 0
    score = 0.0
    for i in range(cut):
        if rel[i] > 0.0:
            hits += 1
            score += hits / (i + 1)
    return score / R


def mean_average_precision(
    queries: Sequence[RelList],
    n_relevant: Sequence[int] | None = None,
    k: int | None = None,
) -> float:
    """Mean average precision over a non-empty set of queries.

    ``n_relevant`` optionally supplies the true relevant count ``R`` per query; if
    omitted, each query's count is inferred from its own positive entries.

    >>> round(mean_average_precision([[1, 0, 1, 0, 0, 1], [0, 1, 1, 0]]), 6)
    0.652778
    """
    _check_cutoff(k)
    if len(queries) == 0:
        raise ValueError("queries must be non-empty")
    if n_relevant is not None and len(n_relevant) != len(queries):
        raise ValueError(
            f"n_relevant has length {len(n_relevant)} but there are {len(queries)} queries"
        )
    total = 0.0
    for idx, q in enumerate(queries):
        R = None if n_relevant is None else n_relevant[idx]
        total += average_precision(q, n_relevant=R, k=k)
    return total / len(queries)


def dcg(relevances: RelList, k: int | None = None, *, exponential: bool = True) -> float:
    """Discounted cumulative gain at cutoff ``k``.

    With ``exponential=True`` (default) the gain of a position is
    ``2**rel - 1``; with ``exponential=False`` it is the raw ``rel``. The discount
    at rank ``j`` (1-based) is ``1 / log2(j + 1)``.

    >>> round(dcg([3, 2, 0, 1, 2]), 6)
    10.484024
    """
    _check_cutoff(k)
    rel = _as_rel(relevances)
    cut = len(rel) if k is None else min(k, len(rel))
    total = 0.0
    for j in range(cut):
        gain = (2.0 ** rel[j] - 1.0) if exponential else rel[j]
        total += gain / math.log2(j + 2)
    return total


def ndcg(relevances: RelList, k: int | None = None, *, exponential: bool = True) -> float:
    """Normalized discounted cumulative gain at cutoff ``k``.

    Divides :func:`dcg` by the ideal DCG (the DCG of the relevances sorted in
    descending order). Returns ``0.0`` when the ideal DCG is ``0`` (no relevant
    item within the cutoff). The result lies in ``[0, 1]``.

    >>> round(ndcg([3, 2, 0, 1, 2]), 6)
    0.968638
    """
    _check_cutoff(k)
    rel = _as_rel(relevances)
    actual = dcg(rel, k, exponential=exponential)
    ideal_rel = sorted(rel, reverse=True)
    ideal = dcg(ideal_rel, k, exponential=exponential)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def mean_ndcg(queries: Sequence[RelList], k: int | None = None, *, exponential: bool = True) -> float:
    """Mean NDCG over a non-empty set of queries.

    >>> round(mean_ndcg([[3, 2, 0, 1, 2], [0, 0, 2, 1]]), 6)
    0.750184
    """
    _check_cutoff(k)
    if len(queries) == 0:
        raise ValueError("queries must be non-empty")
    return sum(ndcg(q, k, exponential=exponential) for q in queries) / len(queries)
