"""External clustering-comparison metrics (Chapter 164).

From-scratch reference implementations of the information-theoretic and
pair-counting indices that compare a clustering against a reference labeling:

- ``mutual_information`` -- raw mutual information ``I(U; V)`` in nats (external).
- ``normalized_mutual_information`` -- ``I(U; V)`` divided by an average of the
  marginal entropies, in ``[0, 1]`` (external).
- ``homogeneity`` -- ``1 - H(V | U) / H(V)``; each cluster is pure (external).
- ``completeness`` -- ``1 - H(U | V) / H(U)``; each class is intact (external).
- ``v_measure`` -- weighted harmonic mean of homogeneity and completeness.
- ``fowlkes_mallows_index`` -- geometric mean of pairwise precision and recall.

The silhouette coefficient and the adjusted Rand index are already provided by
:mod:`aiinaction.ch132_clustering_validation`; this module re-exports them
(:func:`silhouette_score`, :func:`adjusted_rand_index`) rather than duplicating
them, so a single ``from aiinaction.ch164_clustering_metrics import ...`` pulls in
the full external/internal toolkit named by the chapter.

These mirror the Julia (`AIInAction.Ch164ClusteringMetrics`) and Rust
(`aiinaction::ch164_clustering_metrics`) implementations one-to-one; the
cross-language parity tests assert all three agree to floating-point tolerance on
the shared fixtures in ``tests/test_ch164_clustering_metrics.py``.

Labels are sequences of length ``n`` whose actual values are arbitrary: only the
induced partition matters. All logarithms are natural (nats); since every metric
here is either a ratio of logs or a difference normalized by a sum of logs, the
choice of base cancels and the reported numbers are base-independent.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from aiinaction.ch132_clustering_validation import adjusted_rand_index, silhouette_score

__all__ = [
    "contingency_matrix",
    "mutual_information",
    "entropy",
    "normalized_mutual_information",
    "homogeneity",
    "completeness",
    "v_measure",
    "fowlkes_mallows_index",
    # Re-exported from ch132 so the chapter's full API lives in one import.
    "silhouette_score",
    "adjusted_rand_index",
]


def _as_label_pair(labels_true: Sequence[int], labels_pred: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    lt = np.asarray(labels_true)
    lp = np.asarray(labels_pred)
    if lt.ndim != 1 or lp.ndim != 1:
        raise ValueError("labels_true and labels_pred must be 1-D")
    if lt.shape[0] != lp.shape[0]:
        raise ValueError(
            f"length mismatch: len(labels_true)={lt.shape[0]} != "
            f"len(labels_pred)={lp.shape[0]}"
        )
    if lt.shape[0] == 0:
        raise ValueError("inputs must be non-empty")
    return lt, lp


def contingency_matrix(labels_true: Sequence[int], labels_pred: Sequence[int]) -> np.ndarray:
    """Contingency table ``n_ij = |U_i intersect V_j|`` of the two labelings.

    Rows index the distinct values of ``labels_true`` (sorted), columns the
    distinct values of ``labels_pred`` (sorted). Entry ``[i, j]`` counts samples
    in true class ``i`` and predicted cluster ``j``. The row sums recover the true
    class sizes and the column sums the predicted cluster sizes.

    >>> contingency_matrix([0, 0, 1, 1], [0, 0, 1, 1]).tolist()
    [[2, 0], [0, 2]]
    """
    lt, lp = _as_label_pair(labels_true, labels_pred)
    true_vals = {v: i for i, v in enumerate(np.unique(lt))}
    pred_vals = {v: i for i, v in enumerate(np.unique(lp))}
    table = np.zeros((len(true_vals), len(pred_vals)), dtype=np.int64)
    for a, b in zip(lt.tolist(), lp.tolist()):
        table[true_vals[a], pred_vals[b]] += 1
    return table


def entropy(labels: Sequence[int]) -> float:
    """Shannon entropy (in nats) of the partition induced by ``labels``.

    ``H = -sum_k (n_k / n) log(n_k / n)`` over the distinct label values. A
    partition with a single cluster has entropy 0.

    >>> round(entropy([0, 0, 1, 1]), 6)
    0.693147
    """
    lab = np.asarray(labels)
    if lab.ndim != 1:
        raise ValueError("labels must be 1-D")
    n = lab.shape[0]
    if n == 0:
        raise ValueError("inputs must be non-empty")
    _, counts = np.unique(lab, return_counts=True)
    p = counts.astype(float) / n
    return float(-np.sum(p * np.log(p)))


def mutual_information(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    """Mutual information ``I(U; V)`` (in nats) between two labelings.

    With contingency counts ``n_ij``, marginals ``a_i`` and ``b_j``, and total
    ``n``,

        I = sum_ij (n_ij / n) log( n / n_ij * (a_i b_j) ... )

    written compactly as ``sum_ij (n_ij/n) log( (n n_ij) / (a_i b_j) )`` over the
    cells with ``n_ij > 0``. It is symmetric, nonnegative, and zero exactly when
    the two partitions are statistically independent.

    >>> round(mutual_information([0, 0, 1, 1], [0, 0, 1, 1]), 6)
    0.693147
    """
    lt, lp = _as_label_pair(labels_true, labels_pred)
    table = contingency_matrix(lt, lp)
    n = float(lt.shape[0])
    a = table.sum(axis=1).astype(float)  # true class sizes
    b = table.sum(axis=0).astype(float)  # predicted cluster sizes

    mi = 0.0
    rows, cols = table.shape
    for i in range(rows):
        for j in range(cols):
            nij = float(table[i, j])
            if nij == 0.0:
                continue
            mi += (nij / n) * math.log((n * nij) / (a[i] * b[j]))
    # Numerical floor: MI is provably nonnegative.
    return max(mi, 0.0)


def normalized_mutual_information(
    labels_true: Sequence[int],
    labels_pred: Sequence[int],
    *,
    average_method: str = "arithmetic",
) -> float:
    """Normalized mutual information ``I(U; V) / mean(H(U), H(V))`` in ``[0, 1]``.

    ``average_method`` selects how the two marginal entropies are combined:
    ``"arithmetic"`` (default), ``"geometric"``, ``"min"``, or ``"max"``.

    When both partitions are trivial (a single cluster each) both entropies are 0;
    the NMI is then defined to be 1.0 (two degenerate partitions agree perfectly).
    """
    lt, lp = _as_label_pair(labels_true, labels_pred)
    h_true = entropy(lt)
    h_pred = entropy(lp)
    mi = mutual_information(lt, lp)

    if average_method == "arithmetic":
        denom = (h_true + h_pred) / 2.0
    elif average_method == "geometric":
        denom = math.sqrt(h_true * h_pred)
    elif average_method == "min":
        denom = min(h_true, h_pred)
    elif average_method == "max":
        denom = max(h_true, h_pred)
    else:
        raise ValueError(
            f"average_method must be one of 'arithmetic', 'geometric', 'min', 'max'; "
            f"got {average_method!r}"
        )

    if denom == 0.0:
        # Both labelings have a single cluster: degenerate but in perfect agreement.
        return 1.0
    return mi / denom


def _homogeneity_completeness(
    labels_true: Sequence[int],
    labels_pred: Sequence[int],
) -> tuple[float, float, float, float]:
    """Returns ``(homogeneity, completeness, H(true), H(pred))``.

    Uses ``H(V|U) = H(V) - I(U;V)`` so that homogeneity ``= I / H(V)`` and
    completeness ``= I / H(U)`` directly, which is numerically stable and exactly
    matches the conditional-entropy definitions.
    """
    lt, lp = _as_label_pair(labels_true, labels_pred)
    h_true = entropy(lt)  # H(U): the reference/class partition
    h_pred = entropy(lp)  # H(V): the cluster partition
    mi = mutual_information(lt, lp)

    # Homogeneity: each cluster contains a single class -> 1 - H(class | cluster).
    #   H(U | V) = H(U) - I  ==> homogeneity = 1 - H(U|V)/H(U) = I / H(U).
    homog = 1.0 if h_true == 0.0 else mi / h_true
    # Completeness: each class lands in a single cluster -> 1 - H(cluster | class).
    #   H(V | U) = H(V) - I  ==> completeness = 1 - H(V|U)/H(V) = I / H(V).
    compl = 1.0 if h_pred == 0.0 else mi / h_pred
    return homog, compl, h_true, h_pred


def homogeneity(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    """Homogeneity: every cluster contains members of a single true class.

    ``h = 1 - H(true | pred) / H(true) = I(U; V) / H(true)`` in ``[0, 1]``. Equals
    1.0 when the true labeling is trivial (one class).
    """
    return _homogeneity_completeness(labels_true, labels_pred)[0]


def completeness(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    """Completeness: all members of a true class are in the same cluster.

    ``c = 1 - H(pred | true) / H(pred) = I(U; V) / H(pred)`` in ``[0, 1]``. Equals
    1.0 when the predicted labeling is trivial (one cluster).
    """
    return _homogeneity_completeness(labels_true, labels_pred)[1]


def v_measure(
    labels_true: Sequence[int],
    labels_pred: Sequence[int],
    *,
    beta: float = 1.0,
) -> float:
    """V-measure: weighted harmonic mean of homogeneity ``h`` and completeness ``c``.

    ``V_beta = (1 + beta) h c / (beta h + c)``. With ``beta = 1`` this is the plain
    harmonic mean; ``beta > 1`` weights completeness more, ``beta < 1`` homogeneity
    more. Returns 0.0 when both components are 0.

    >>> round(v_measure([0, 0, 1, 1], [0, 0, 1, 1]), 6)
    1.0
    """
    if beta < 0.0:
        raise ValueError(f"beta must be nonnegative, got {beta}")
    h, c, _, _ = _homogeneity_completeness(labels_true, labels_pred)
    denom = beta * h + c
    if denom == 0.0:
        return 0.0
    return (1.0 + beta) * h * c / denom


def fowlkes_mallows_index(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    """Fowlkes-Mallows index: geometric mean of pairwise precision and recall.

    With ``a`` the number of point pairs together in *both* partitions, ``a + b``
    the pairs together in the prediction, and ``a + c`` the pairs together in the
    reference,

        FM = a / sqrt((a + b)(a + c)).

    Returns a value in ``[0, 1]``; 1.0 means the partitions agree on every pair.
    When no pair is co-clustered in either partition (all singletons) the index is
    defined to be 0.0.

    >>> round(fowlkes_mallows_index([0, 0, 1, 1], [1, 1, 0, 0]), 6)
    1.0
    """
    lt, lp = _as_label_pair(labels_true, labels_pred)
    table = contingency_matrix(lt, lp)

    def comb2(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(x * (x - 1.0) / 2.0))

    a = comb2(table)               # pairs together in both
    a_plus_b = comb2(table.sum(axis=0))  # pairs together in prediction (cluster sizes)
    a_plus_c = comb2(table.sum(axis=1))  # pairs together in reference (class sizes)

    denom = a_plus_b * a_plus_c
    if denom == 0.0:
        return 0.0
    return a / math.sqrt(denom)
