"""Tests for aiinaction.ch132_clustering_validation and the shared fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. The expected values were cross-checked against scikit-learn.
"""
from __future__ import annotations

import pytest

from aiinaction.ch132_clustering_validation import (
    adjusted_rand_index,
    calinski_harabasz_index,
    davies_bouldin_index,
    silhouette_score,
)

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
# Two well-separated blobs in the plane.
X = [
    [1.0, 1.0],
    [1.5, 2.0],
    [1.0, 0.5],
    [8.0, 8.0],
    [8.5, 7.5],
    [7.5, 8.5],
]
LABELS = [0, 0, 0, 1, 1, 1]

# External-index fixture: a reference partition and a finer clustering.
LABELS_TRUE = [0, 0, 0, 1, 1, 1]
LABELS_PRED = [0, 0, 1, 1, 2, 2]

EXPECTED = {
    "silhouette": 0.8954900167230767,
    "davies_bouldin": 0.11157205284841143,
    "calinski_harabasz": 240.14285714285714,
    "ari": 0.24242424242424246,
}


def test_silhouette_matches_fixture():
    assert silhouette_score(X, LABELS) == pytest.approx(EXPECTED["silhouette"])


def test_davies_bouldin_matches_fixture():
    assert davies_bouldin_index(X, LABELS) == pytest.approx(EXPECTED["davies_bouldin"])


def test_calinski_harabasz_matches_fixture():
    assert calinski_harabasz_index(X, LABELS) == pytest.approx(
        EXPECTED["calinski_harabasz"]
    )


def test_ari_matches_fixture():
    assert adjusted_rand_index(LABELS_TRUE, LABELS_PRED) == pytest.approx(
        EXPECTED["ari"]
    )


def test_silhouette_one_d_fixture():
    # Two tight, far-apart clusters on the line: near-perfect silhouette.
    assert silhouette_score([[0.0], [0.1], [10.0], [10.1]], [0, 0, 1, 1]) == pytest.approx(
        0.9899997499937498
    )


def test_silhouette_bounds():
    s = silhouette_score(X, LABELS)
    assert -1.0 <= s <= 1.0


def test_ari_perfect_is_one():
    assert adjusted_rand_index(LABELS_TRUE, LABELS_TRUE) == pytest.approx(1.0)


def test_ari_invariant_to_relabeling():
    # Swapping label names must not change the ARI.
    assert adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_ari_can_be_negative_or_chance():
    # An unrelated clustering should score near or below zero.
    val = adjusted_rand_index([0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1])
    assert val <= 1e-9


def test_ari_degenerate_partitions():
    # All points in one cluster on both sides: degenerate but in agreement.
    assert adjusted_rand_index([0, 0, 0], [5, 5, 5]) == pytest.approx(1.0)


# --- validation / edge cases -------------------------------------------------


@pytest.mark.parametrize(
    "fn", [silhouette_score, davies_bouldin_index, calinski_harabasz_index]
)
def test_single_cluster_raises(fn):
    with pytest.raises(ValueError, match="at least 2 clusters"):
        fn([[0.0], [1.0], [2.0]], [0, 0, 0])


@pytest.mark.parametrize(
    "fn", [silhouette_score, davies_bouldin_index, calinski_harabasz_index]
)
def test_label_length_mismatch_raises(fn):
    with pytest.raises(ValueError, match="length mismatch"):
        fn([[0.0], [1.0], [2.0]], [0, 1])


def test_ari_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        adjusted_rand_index([0, 0, 1], [0, 1])


def test_davies_bouldin_identical_centroids_raises():
    # Two clusters with the same centroid -> zero separation.
    with pytest.raises(ValueError, match="zero separation"):
        davies_bouldin_index([[0.0], [0.0], [0.0], [0.0]], [0, 1, 0, 1])


def test_calinski_harabasz_zero_within_raises():
    # Every cluster collapses to a single point -> within-scatter zero.
    with pytest.raises(ValueError, match="undefined"):
        calinski_harabasz_index([[0.0], [5.0]], [0, 1])


def test_non_2d_X_raises():
    with pytest.raises(ValueError, match="2-D"):
        silhouette_score([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1])
