"""Association rule and frequent-pattern mining from scratch.

Reference implementations of the two canonical frequent-itemset miners, **Apriori**
(Agrawal and Srikant, 1994) and **FP-Growth** (Han, Pei and Yin, 2000), plus
association-rule extraction with the standard interestingness measures (support,
confidence, lift, leverage, conviction).

The public API mirrors the Julia (`AIInAction.Ch150AssociationRules`) and Rust
(`aiinaction::ch150_association_rules`) implementations one-to-one; the
cross-language parity tests assert that all three agree exactly on the shared
fixtures.

Data model
----------
A *transaction* is a set of integer item ids; a *dataset* is a list of
transactions. Items are integers so the three languages agree byte-for-byte on
ordering; a string catalog maps to this by assigning each label an id.

Determinism
-----------
Every itemset is represented as a sorted ``tuple[int, ...]`` and results are
returned in a canonical order (ascending by length, then lexicographically by the
sorted item tuple). Apriori and FP-Growth therefore return *identical* frequent
sets and supports, which the parity tests rely on.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

__all__ = [
    "Rule",
    "apriori",
    "fpgrowth",
    "association_rules",
    "support",
]

Transaction = Iterable[int]
Itemset = tuple[int, ...]


@dataclass(frozen=True)
class Rule:
    """A single association rule ``antecedent => consequent`` with its metrics.

    Attributes
    ----------
    antecedent, consequent:
        Disjoint, sorted item tuples. The rule reads "transactions containing
        ``antecedent`` tend to also contain ``consequent``".
    support:
        ``supp(antecedent union consequent)``, the fraction of transactions that
        contain every item of the rule.
    confidence:
        ``supp(union) / supp(antecedent)`` ~= ``P(consequent | antecedent)``.
    lift:
        ``confidence / supp(consequent)``. 1.0 means independence, >1 positive
        association, <1 substitution.
    leverage:
        ``supp(union) - supp(antecedent) * supp(consequent)``, the absolute excess
        co-occurrence over independence (in [-0.25, 0.25]).
    conviction:
        ``(1 - supp(consequent)) / (1 - confidence)``; ``inf`` for a perfectly
        confident rule. Higher means the antecedent more strongly implies the
        consequent.
    """

    antecedent: Itemset
    consequent: Itemset
    support: float
    confidence: float
    lift: float
    leverage: float
    conviction: float


def _normalize(transactions: Sequence[Transaction]) -> list[frozenset[int]]:
    """Validate and convert raw transactions to a list of ``frozenset``s of ints."""
    if not isinstance(transactions, Sequence) or isinstance(transactions, (str, bytes)):
        raise ValueError("transactions must be a sequence of item collections")
    if len(transactions) == 0:
        raise ValueError("transactions must be non-empty")
    out: list[frozenset[int]] = []
    for k, t in enumerate(transactions):
        if isinstance(t, (str, bytes)):
            raise ValueError(f"transaction {k} must be a collection of integer items, not a string")
        items: set[int] = set()
        for it in t:
            if isinstance(it, bool) or not isinstance(it, int):
                raise ValueError(f"transaction {k} contains a non-integer item: {it!r}")
            items.add(int(it))
        out.append(frozenset(items))
    return out


def _check_min_support(min_support: float) -> None:
    if not isinstance(min_support, (int, float)) or isinstance(min_support, bool):
        raise ValueError("min_support must be a real number")
    if not (0.0 < float(min_support) <= 1.0):
        raise ValueError(f"min_support must be in (0, 1], got {min_support}")


def support(transactions: Sequence[Transaction], itemset: Iterable[int]) -> float:
    """Fraction of transactions that contain every item of ``itemset``.

    >>> support([[1, 2], [1], [1, 2, 3]], [1, 2])
    0.6666666666666666
    """
    data = _normalize(transactions)
    target = frozenset(int(i) for i in itemset)
    if not target:
        raise ValueError("itemset must be non-empty")
    cover = sum(1 for t in data if target <= t)
    return cover / len(data)


def _frequent_singletons(
    data: list[frozenset[int]], min_count: int
) -> dict[Itemset, int]:
    counts: dict[int, int] = {}
    for t in data:
        for it in t:
            counts[it] = counts.get(it, 0) + 1
    return {(it,): c for it, c in counts.items() if c >= min_count}


def _apriori_gen(prev: list[Itemset]) -> list[Itemset]:
    """Join + prune step. ``prev`` is a sorted list of frequent (k-1)-itemsets."""
    prev_set = set(prev)
    k = len(prev[0]) + 1 if prev else 0
    candidates: list[Itemset] = []
    for i in range(len(prev)):
        for j in range(i + 1, len(prev)):
            a, b = prev[i], prev[j]
            # Join: share the first k-2 items, differ only in the last.
            if a[:-1] == b[:-1] and a[-1] < b[-1]:
                cand = a + (b[-1],)
                # Prune: every (k-1)-subset must be frequent.
                if all(
                    tuple(sub) in prev_set
                    for sub in combinations(cand, k - 1)
                ):
                    candidates.append(cand)
            else:
                # prev is sorted; once prefixes diverge no further j can join with i.
                if a[:-1] != b[:-1]:
                    break
    candidates.sort()
    return candidates


def apriori(
    transactions: Sequence[Transaction], min_support: float
) -> dict[Itemset, float]:
    """Mine all frequent itemsets with the Apriori level-wise algorithm.

    Parameters
    ----------
    transactions:
        Non-empty sequence of item collections; each item is an ``int``.
    min_support:
        Minimum support threshold in ``(0, 1]``. An itemset is frequent when its
        support is ``>= min_support`` (compared on integer counts to avoid
        floating-point edge effects: ``count >= ceil(min_support * n)``).

    Returns
    -------
    dict
        Maps each frequent itemset (a sorted ``tuple[int, ...]``) to its support.

    Examples
    --------
    >>> data = [[1, 2, 3], [1, 2], [1, 3], [2, 3], [1]]
    >>> sorted(apriori(data, 0.4))
    [(1,), (1, 2), (1, 3), (2,), (2, 3), (3,)]
    """
    data = _normalize(transactions)
    _check_min_support(min_support)
    n = len(data)
    min_count = _min_count(min_support, n)

    frequent: dict[Itemset, int] = {}
    current = _frequent_singletons(data, min_count)
    frequent.update(current)

    level = sorted(current)
    while level:
        candidates = _apriori_gen(level)
        if not candidates:
            break
        counts: dict[Itemset, int] = {c: 0 for c in candidates}
        for t in data:
            for c in candidates:
                if all(item in t for item in c):
                    counts[c] += 1
        level = sorted(c for c, cnt in counts.items() if cnt >= min_count)
        for c in level:
            frequent[c] = counts[c]

    return {k: v / n for k, v in frequent.items()}


def _min_count(min_support: float, n: int) -> int:
    """Smallest integer count meeting the support threshold (ceil with epsilon)."""
    import math

    return max(1, math.ceil(float(min_support) * n - 1e-9))


# --------------------------------------------------------------------------- #
# FP-Growth
# --------------------------------------------------------------------------- #


class _FPNode:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item: int | None, parent: "_FPNode | None") -> None:
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict[int, _FPNode] = {}
        self.link: _FPNode | None = None


def _order_key(item: int, counts: dict[int, int]) -> tuple[int, int]:
    # Descending support, ties broken by ascending item id for determinism.
    return (-counts[item], item)


def _build_tree(
    data: list[frozenset[int]], counts: dict[int, int], min_count: int
) -> tuple[_FPNode, dict[int, _FPNode]]:
    frequent_items = {it for it, c in counts.items() if c >= min_count}
    root = _FPNode(None, None)
    header: dict[int, _FPNode] = {}

    def link(node: _FPNode) -> None:
        assert node.item is not None
        head = header.get(node.item)
        if head is None:
            header[node.item] = node
        else:
            while head.link is not None:
                head = head.link
            head.link = node

    for t in data:
        ordered = sorted(
            (it for it in t if it in frequent_items),
            key=lambda it: _order_key(it, counts),
        )
        node = root
        for it in ordered:
            child = node.children.get(it)
            if child is None:
                child = _FPNode(it, node)
                node.children[it] = child
                link(child)
            child.count += 1
            node = child
    return root, header


def _ascend(node: _FPNode) -> list[int]:
    path: list[int] = []
    cur = node.parent
    while cur is not None and cur.item is not None:
        path.append(cur.item)
        cur = cur.parent
    return path


def _mine_tree(
    header: dict[int, _FPNode],
    counts: dict[int, int],
    min_count: int,
    suffix: tuple[int, ...],
    frequent: dict[Itemset, int],
) -> None:
    # Process items in ascending support order (least frequent first) for a
    # deterministic, correct pattern-growth recursion.
    items = sorted(header.keys(), key=lambda it: (counts[it], it))
    for item in items:
        new_suffix = tuple(sorted(suffix + (item,)))
        frequent[new_suffix] = counts[item]

        # Build the conditional pattern base for `item`.
        cond_patterns: list[tuple[list[int], int]] = []
        node: _FPNode | None = header[item]
        while node is not None:
            prefix = _ascend(node)
            if prefix:
                cond_patterns.append((prefix, node.count))
            node = node.link

        # Count items within the conditional pattern base.
        cond_counts: dict[int, int] = {}
        for prefix, cnt in cond_patterns:
            for it in prefix:
                cond_counts[it] = cond_counts.get(it, 0) + cnt

        cond_frequent = {it: c for it, c in cond_counts.items() if c >= min_count}
        if not cond_frequent:
            continue

        # Build the conditional FP-tree from the (filtered) pattern base.
        cond_data: list[frozenset[int]] = []
        cond_multiplicity: list[int] = []
        for prefix, cnt in cond_patterns:
            kept = frozenset(it for it in prefix if it in cond_frequent)
            if kept:
                cond_data.append(kept)
                cond_multiplicity.append(cnt)

        cond_root, cond_header = _build_weighted_tree(
            cond_data, cond_multiplicity, cond_frequent
        )
        if cond_header:
            _mine_tree(cond_header, cond_frequent, min_count, new_suffix, frequent)


def _build_weighted_tree(
    data: list[frozenset[int]],
    multiplicity: list[int],
    counts: dict[int, int],
) -> tuple[_FPNode, dict[int, _FPNode]]:
    root = _FPNode(None, None)
    header: dict[int, _FPNode] = {}

    def link(node: _FPNode) -> None:
        assert node.item is not None
        head = header.get(node.item)
        if head is None:
            header[node.item] = node
        else:
            while head.link is not None:
                head = head.link
            head.link = node

    for t, w in zip(data, multiplicity):
        ordered = sorted(t, key=lambda it: _order_key(it, counts))
        node = root
        for it in ordered:
            child = node.children.get(it)
            if child is None:
                child = _FPNode(it, node)
                node.children[it] = child
                link(child)
            child.count += w
            node = child
    return root, header


def fpgrowth(
    transactions: Sequence[Transaction], min_support: float
) -> dict[Itemset, float]:
    """Mine all frequent itemsets with FP-Growth (no candidate generation).

    Returns exactly the same mapping as :func:`apriori` for the same inputs; the
    two are cross-checked in the test suite.

    Examples
    --------
    >>> data = [[1, 2, 3], [1, 2], [1, 3], [2, 3], [1]]
    >>> sorted(fpgrowth(data, 0.6)) == sorted(apriori(data, 0.6))
    True
    """
    data = _normalize(transactions)
    _check_min_support(min_support)
    n = len(data)
    min_count = _min_count(min_support, n)

    counts: dict[int, int] = {}
    for t in data:
        for it in t:
            counts[it] = counts.get(it, 0) + 1

    _, header = _build_tree(data, counts, min_count)
    frequent: dict[Itemset, int] = {}
    _mine_tree(header, counts, min_count, (), frequent)
    return {k: v / n for k, v in frequent.items()}


# --------------------------------------------------------------------------- #
# Rule generation
# --------------------------------------------------------------------------- #


def association_rules(
    frequent_itemsets: dict[Itemset, float],
    min_confidence: float,
) -> list[Rule]:
    """Extract association rules from a mapping of frequent itemsets to supports.

    For every frequent itemset ``Z`` with ``|Z| >= 2`` and every non-empty proper
    subset ``A`` (antecedent), emits ``A => Z\\A`` when its confidence is at least
    ``min_confidence``. Single-item itemsets generate no rules.

    Parameters
    ----------
    frequent_itemsets:
        Output of :func:`apriori` or :func:`fpgrowth` (itemset -> support).
    min_confidence:
        Minimum confidence in ``[0, 1]``.

    Returns
    -------
    list[Rule]
        Rules sorted deterministically by ``(-confidence, antecedent, consequent)``.
    """
    if not isinstance(min_confidence, (int, float)) or isinstance(min_confidence, bool):
        raise ValueError("min_confidence must be a real number")
    if not (0.0 <= float(min_confidence) <= 1.0):
        raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
    supp = {tuple(sorted(k)): float(v) for k, v in frequent_itemsets.items()}

    rules: list[Rule] = []
    for z, supp_z in supp.items():
        if len(z) < 2:
            continue
        for r in range(1, len(z)):
            for antecedent in combinations(z, r):
                consequent = tuple(item for item in z if item not in antecedent)
                supp_a = supp.get(antecedent)
                supp_c = supp.get(consequent)
                if supp_a is None or supp_c is None:
                    # Antecedent/consequent should be frequent too (downward closure),
                    # but guard in case a partial map was passed in.
                    continue
                conf = supp_z / supp_a
                if conf < float(min_confidence):
                    continue
                lift = conf / supp_c
                leverage = supp_z - supp_a * supp_c
                if conf >= 1.0:
                    conviction = float("inf")
                else:
                    conviction = (1.0 - supp_c) / (1.0 - conf)
                rules.append(
                    Rule(
                        antecedent=antecedent,
                        consequent=consequent,
                        support=supp_z,
                        confidence=conf,
                        lift=lift,
                        leverage=leverage,
                        conviction=conviction,
                    )
                )

    rules.sort(key=lambda rr: (-rr.confidence, rr.antecedent, rr.consequent))
    return rules
