"""Information retrieval (IR) ranking metrics from scratch.

Reference implementations of the standard offline metrics used to evaluate
ranked retrieval systems: precision@k, recall@k, average precision (AP) and its
mean (MAP), discounted cumulative gain (DCG / NDCG@k), and reciprocal rank (RR /
MRR). The public API mirrors the Julia (``AIInAction.Ch163IrMetrics``) and Rust
(``aiinaction::ch163_ir_metrics``) implementations one-to-one; the cross-language
parity tests assert that all three agree to within floating-point tolerance on
shared fixtures.

Two input conventions are used, matching how the metrics are defined:

* **Binary, ranked.** A ranking is a sequence of ``0/1`` relevance labels in rank
  order (label at index 0 is the top-ranked document). ``precision_at_k``,
  ``recall_at_k``, ``average_precision``, ``reciprocal_rank`` and the means
  ``mean_average_precision`` / ``mean_reciprocal_rank`` use this convention.
* **Graded.** ``dcg_at_k`` and ``ndcg_at_k`` take a sequence of non-negative
  integer relevance grades in rank order.

Recall and MAP additionally need the total number of relevant documents
``num_relevant = |R_q|`` for the query, since relevant documents may sit beyond
the truncated list and still count against recall.

This module is intentionally dependency-light: only :func:`ndcg_at_k` and the
graded DCG use ``math.log2``; everything else is plain arithmetic so the three
language ports stay bit-for-bit comparable.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "average_precision",
    "mean_average_precision",
    "dcg_at_k",
    "ndcg_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
]


def _check_binary(relevances: Sequence[int], name: str = "relevances") -> list[int]:
    rel = list(relevances)
    for i, r in enumerate(rel):
        if r not in (0, 1):
            raise ValueError(f"{name} must be binary 0/1 labels, got {r!r} at index {i}")
    return rel


def _check_grades(grades: Sequence[int], name: str = "grades") -> list[int]:
    out: list[int] = []
    for i, g in enumerate(grades):
        gi = int(g)
        if gi != g:
            raise ValueError(f"{name} must be integers, got {g!r} at index {i}")
        if gi < 0:
            raise ValueError(f"{name} must be non-negative, got {gi} at index {i}")
        out.append(gi)
    return out


def _check_k(k: int, n: int) -> int:
    if not isinstance(k, int):
        raise ValueError(f"k must be an integer, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    return min(k, n)


def precision_at_k(relevances: Sequence[int], k: int) -> float:
    """Precision@k: fraction of the top ``k`` ranked items that are relevant.

    ``relevances`` is a rank-ordered sequence of binary labels. If ``k`` exceeds
    the list length it is clamped to the length, so the denominator never exceeds
    the number of available items.

    >>> round(precision_at_k([1, 0, 1, 1, 0], 5), 2)
    0.6
    """
    rel = _check_binary(relevances)
    if not rel:
        raise ValueError("relevances must be non-empty")
    kk = _check_k(k, len(rel))
    return sum(rel[:kk]) / kk


def recall_at_k(relevances: Sequence[int], k: int, num_relevant: int) -> float:
    """Recall@k: fraction of all relevant documents found in the top ``k``.

    ``num_relevant`` is ``|R_q|``, the total number of documents relevant to the
    query (which may exceed the number present in ``relevances``). It must be at
    least the count of relevant labels in the list.

    >>> recall_at_k([1, 0, 1, 1, 0], 5, num_relevant=4)
    0.75
    """
    rel = _check_binary(relevances)
    if not rel:
        raise ValueError("relevances must be non-empty")
    if num_relevant <= 0:
        raise ValueError(f"num_relevant must be a positive integer, got {num_relevant}")
    found_total = sum(rel)
    if num_relevant < found_total:
        raise ValueError(
            f"num_relevant={num_relevant} is smaller than the {found_total} relevant labels present"
        )
    kk = _check_k(k, len(rel))
    return sum(rel[:kk]) / num_relevant


def average_precision(relevances: Sequence[int], num_relevant: int | None = None) -> float:
    """Average Precision (AP) for a single query.

    Averages ``P@i`` over the ranks ``i`` where a relevant document appears,
    dividing by ``num_relevant`` (defaults to the number of relevant labels in
    ``relevances``). Relevant documents that are never retrieved contribute zero,
    so AP penalizes missing recall when ``num_relevant`` exceeds the number found.

    >>> round(average_precision([1, 0, 1, 0, 0, 1]), 4)
    0.7222
    """
    rel = _check_binary(relevances)
    if not rel:
        raise ValueError("relevances must be non-empty")
    found_total = sum(rel)
    if num_relevant is None:
        num_relevant = found_total
    if num_relevant <= 0:
        raise ValueError(f"num_relevant must be a positive integer, got {num_relevant}")
    if num_relevant < found_total:
        raise ValueError(
            f"num_relevant={num_relevant} is smaller than the {found_total} relevant labels present"
        )
    hits = 0
    precision_sum = 0.0
    for i, r in enumerate(rel, start=1):
        if r == 1:
            hits += 1
            precision_sum += hits / i
    return precision_sum / num_relevant


def mean_average_precision(
    rankings: Sequence[Sequence[int]],
    num_relevant: Sequence[int] | None = None,
) -> float:
    """Mean Average Precision (MAP): AP averaged over a set of queries.

    ``rankings`` is one binary ranking per query. ``num_relevant``, if given, is
    the matching ``|R_q|`` per query; otherwise each query's relevant count is
    inferred from its own labels.

    >>> round(mean_average_precision([[1, 0, 1], [0, 1, 0]]), 4)
    0.6667
    """
    queries = list(rankings)
    if not queries:
        raise ValueError("rankings must contain at least one query")
    if num_relevant is None:
        aps = [average_precision(r) for r in queries]
    else:
        nrel = list(num_relevant)
        if len(nrel) != len(queries):
            raise ValueError(
                f"length mismatch: {len(queries)} rankings != {len(nrel)} num_relevant entries"
            )
        aps = [average_precision(r, n) for r, n in zip(queries, nrel)]
    return sum(aps) / len(aps)


def dcg_at_k(grades: Sequence[int], k: int) -> float:
    """Discounted Cumulative Gain at cutoff ``k`` (exponential-gain form).

    Uses gain ``2^rel - 1`` and discount ``log2(i + 1)`` for rank ``i`` (1-based).
    ``grades`` is a rank-ordered sequence of non-negative integer relevance
    grades. ``k`` is clamped to the list length.

    >>> round(dcg_at_k([3, 2, 3], 3), 4)
    12.3928
    """
    grd = _check_grades(grades)
    if not grd:
        raise ValueError("grades must be non-empty")
    kk = _check_k(k, len(grd))
    total = 0.0
    for i in range(1, kk + 1):
        gain = (2 ** grd[i - 1]) - 1
        total += gain / math.log2(i + 1)
    return total


def ndcg_at_k(grades: Sequence[int], k: int, ideal_grades: Sequence[int] | None = None) -> float:
    """Normalized DCG at cutoff ``k``, in ``[0, 1]``.

    ``NDCG@k = DCG@k / IDCG@k`` where the ideal DCG is the DCG of the best
    achievable ordering. By default the ideal ordering is ``grades`` sorted
    descending; pass ``ideal_grades`` (the full pool of judged grades for the
    query) when relevant documents exist beyond the returned list.

    Raises ``ValueError`` if the ideal DCG is zero (no relevant documents), since
    NDCG is then undefined.

    >>> round(ndcg_at_k([3, 2, 3], 3), 4)
    0.9595
    """
    grd = _check_grades(grades)
    if not grd:
        raise ValueError("grades must be non-empty")
    if ideal_grades is None:
        ideal = sorted(grd, reverse=True)
    else:
        ideal = sorted(_check_grades(ideal_grades, "ideal_grades"), reverse=True)
        if not ideal:
            raise ValueError("ideal_grades must be non-empty")
    dcg = dcg_at_k(grd, k)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        raise ValueError("IDCG@k is zero (no relevant documents); NDCG is undefined")
    return dcg / idcg


def reciprocal_rank(relevances: Sequence[int]) -> float:
    """Reciprocal Rank: ``1 / rank`` of the first relevant document.

    Returns ``0.0`` when no relevant document appears in the list.

    >>> reciprocal_rank([0, 0, 1, 0])
    0.3333333333333333
    """
    rel = _check_binary(relevances)
    if not rel:
        raise ValueError("relevances must be non-empty")
    for i, r in enumerate(rel, start=1):
        if r == 1:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(rankings: Sequence[Sequence[int]]) -> float:
    """Mean Reciprocal Rank (MRR): reciprocal rank averaged over queries.

    >>> round(mean_reciprocal_rank([[0, 1, 0], [1, 0, 0], [0, 0, 0]]), 4)
    0.5
    """
    queries = list(rankings)
    if not queries:
        raise ValueError("rankings must contain at least one query")
    return sum(reciprocal_rank(r) for r in queries) / len(queries)
