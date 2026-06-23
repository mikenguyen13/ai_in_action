"""Tests for aiinaction.ch146_lof, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. The data is four points forming a unit square (a dense cluster) plus one
far outlier, so the LOF score of the outlier is large and the others are exactly 1.
"""
from __future__ import annotations

import math

import pytest

from aiinaction import ch146_lof

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
X = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [8.0, 8.0],
]
K = 2

EXPECTED = {
    "knn": [[1, 2], [0, 3], [0, 3], [1, 2], [3, 1]],
    "k_distance": [1.0, 1.0, 1.0, 1.0, 10.63014581273465],
    "lrd": [1.0, 1.0, 1.0, 1.0, 0.09742011681639788],
    "lof": [1.0, 1.0, 1.0, 1.0, 10.264820374673157],
}


def test_euclidean_basic():
    assert ch146_lof.euclidean([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)


def test_knn_indices_match_fixture():
    assert ch146_lof.knn_distances(X, K) == EXPECTED["knn"]


def test_k_distance_matches_fixture():
    assert ch146_lof.k_distance(X, K) == pytest.approx(EXPECTED["k_distance"])


def test_lrd_matches_fixture():
    assert ch146_lof.lrd(X, K) == pytest.approx(EXPECTED["lrd"])


def test_lof_scores_match_fixture():
    assert ch146_lof.lof_scores(X, K) == pytest.approx(EXPECTED["lof"])


def test_inliers_have_lof_near_one():
    scores = ch146_lof.lof_scores(X, K)
    for s in scores[:4]:
        assert s == pytest.approx(1.0, abs=1e-9)


def test_outlier_has_largest_lof():
    scores = ch146_lof.lof_scores(X, K)
    assert max(range(len(scores)), key=lambda i: scores[i]) == 4


def test_top_anomalies_orders_by_score():
    assert ch146_lof.top_anomalies(X, K, 1) == [4]
    assert ch146_lof.top_anomalies(X, K, 2)[0] == 4


def test_top_anomalies_returns_m_indices():
    out = ch146_lof.top_anomalies(X, K, 3)
    assert len(out) == 3
    assert out[0] == 4


def test_uniform_grid_lof_finite_and_modest():
    # A perfectly regular line of equally spaced points has no outlier, so every
    # LOF score stays close to 1 (no large spikes) and all scores are finite.
    grid = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    scores = ch146_lof.lof_scores(grid, k=2)
    assert all(math.isfinite(s) for s in scores)
    assert all(0.5 <= s <= 1.5 for s in scores)


def test_duplicate_point_handled():
    # Two coincident points plus a cluster: the coincident pair has zero k-distance
    # to each other, but lof_scores must not raise.
    dup = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    scores = ch146_lof.lof_scores(dup, k=2)
    assert len(scores) == 4
    assert all(math.isfinite(s) for s in scores)


def test_empty_raises():
    with pytest.raises(ValueError, match="at least one row"):
        ch146_lof.lof_scores([], k=1)


def test_bad_k_too_large_raises():
    with pytest.raises(ValueError, match="k must be in"):
        ch146_lof.lof_scores(X, k=5)


def test_bad_k_zero_raises():
    with pytest.raises(ValueError, match="k must be in"):
        ch146_lof.lof_scores(X, k=0)


def test_non_finite_raises():
    with pytest.raises(ValueError, match="non-finite"):
        ch146_lof.lof_scores([[0.0, 0.0], [float("nan"), 1.0], [1.0, 1.0]], k=1)


def test_ragged_rows_raise():
    with pytest.raises(ValueError, match="same number of features"):
        ch146_lof.lof_scores([[0.0, 0.0], [1.0]], k=1)


def test_euclidean_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        ch146_lof.euclidean([1.0, 2.0], [1.0])


def test_bad_m_raises():
    with pytest.raises(ValueError, match="m must be a positive integer"):
        ch146_lof.top_anomalies(X, K, 0)


def test_m_too_large_raises():
    with pytest.raises(ValueError, match="m must be in"):
        ch146_lof.top_anomalies(X, K, 99)
