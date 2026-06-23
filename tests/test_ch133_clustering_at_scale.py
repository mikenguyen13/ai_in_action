"""Tests for aiinaction.ch133_clustering_at_scale and the shared fixtures.

The constants below are the single source of truth: the Julia and Rust test
suites assert against the same numbers (1e-9 tolerance), which is what keeps the
three libraries at parity. The deterministic LCG makes every sampled result
reproducible across languages.
"""
from __future__ import annotations

import pytest

from aiinaction import ch133_clustering_at_scale as cs

# --------------------------------------------------------------------------- #
# Shared fixtures: identical to julia/AIInAction/test and rust inline tests.   #
# --------------------------------------------------------------------------- #
POINTS = [
    [0.0, 0.0], [0.2, 0.1], [0.1, 0.2],
    [5.0, 5.0], [5.2, 4.9], [4.9, 5.1],
    [0.0, 5.0], [0.1, 4.8], [-0.1, 5.2],
]
INIT = [[1.0, 1.0], [4.0, 4.0], [1.0, 4.0]]

EXPECTED = {
    "lcg_next_below": 8,  # Lcg(7).next_below(10)
    "sqdist": 25.0,  # [0,0] -> [3,4]
    "cf_ls": [15.4, 30.299999999999997],
    "cf_ss": 226.27000000000004,
    "cf_centroid": [1.7111111111111112, 3.3666666666666663],
    "cf_radius": 3.29829735349904,
    "mbkm": [
        [0.09999999999999999, 0.10769230769230771],
        [5.042307692307693, 5.000000000000001],
        [-0.0035714285714285696, 5.007142857142858],
    ],
    "mbkm_inertia": 0.20727712534718032,
    "canopies": [[3, 4, 5], [6, 7, 8], [0, 1, 2]],
    "seeds": [[0.2, 0.1], [5.2, 4.9], [-0.1, 5.2]],
}


# --------------------------------------------------------------------------- #
# RNG / geometry.                                                             #
# --------------------------------------------------------------------------- #
def test_lcg_deterministic():
    assert cs.Lcg(7).next_below(10) == EXPECTED["lcg_next_below"]
    # Two generators with the same seed agree.
    a, b = cs.Lcg(99), cs.Lcg(99)
    assert [a.next_u32() for _ in range(5)] == [b.next_u32() for _ in range(5)]


def test_squared_distance():
    assert cs.squared_distance([0.0, 0.0], [3.0, 4.0]) == EXPECTED["sqdist"]


def test_squared_distance_dim_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        cs.squared_distance([1.0], [1.0, 2.0])


def test_nearest_centroid():
    j, d = cs.nearest_centroid([0.1, 0.1], INIT)
    assert j == 0
    assert d == pytest.approx(1.62)


def test_nearest_centroid_ties_lowest_index():
    j, _ = cs.nearest_centroid([0.0, 0.0], [[1.0, 0.0], [-1.0, 0.0]])
    assert j == 0


# --------------------------------------------------------------------------- #
# Clustering feature (BIRCH).                                                 #
# --------------------------------------------------------------------------- #
def test_cf_from_points():
    cf = cs.ClusteringFeature.from_points(POINTS)
    assert cf.n == 9
    assert cf.ls == pytest.approx(EXPECTED["cf_ls"])
    assert cf.ss == pytest.approx(EXPECTED["cf_ss"])
    assert cf.centroid() == pytest.approx(EXPECTED["cf_centroid"])
    assert cf.radius() == pytest.approx(EXPECTED["cf_radius"])


def test_cf_additivity():
    whole = cs.ClusteringFeature.from_points(POINTS)
    left = cs.ClusteringFeature.from_points(POINTS[:3])
    right = cs.ClusteringFeature.from_points(POINTS[3:])
    merged = left.merge(right)
    assert merged.n == whole.n
    assert merged.ls == pytest.approx(whole.ls)
    assert merged.ss == pytest.approx(whole.ss)


def test_cf_radius_single_point_is_zero():
    cf = cs.ClusteringFeature.from_points([[2.0, 3.0]])
    assert cf.radius() == 0.0


def test_cf_empty_centroid_raises():
    cf = cs.ClusteringFeature(0, [0.0, 0.0], 0.0)
    with pytest.raises(ValueError, match="undefined"):
        cf.centroid()


def test_cf_merge_dim_mismatch_raises():
    a = cs.ClusteringFeature.from_points([[1.0, 1.0]])
    b = cs.ClusteringFeature.from_points([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="dimension mismatch"):
        a.merge(b)


# --------------------------------------------------------------------------- #
# Mini-batch k-means.                                                         #
# --------------------------------------------------------------------------- #
def test_mini_batch_kmeans_matches_fixture():
    out = cs.mini_batch_kmeans(POINTS, INIT, batch_size=4, n_iter=20, seed=42)
    for row, exp in zip(out, EXPECTED["mbkm"]):
        assert row == pytest.approx(exp)


def test_mini_batch_kmeans_inertia():
    out = cs.mini_batch_kmeans(POINTS, INIT, batch_size=4, n_iter=20, seed=42)
    assert cs.inertia(POINTS, out) == pytest.approx(EXPECTED["mbkm_inertia"])


def test_mini_batch_kmeans_zero_iter_returns_init():
    out = cs.mini_batch_kmeans(POINTS, INIT, batch_size=4, n_iter=0, seed=1)
    assert out == [list(r) for r in INIT]


def test_mini_batch_kmeans_bad_batch_raises():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        cs.mini_batch_kmeans(POINTS, INIT, batch_size=0, n_iter=1, seed=1)


# --------------------------------------------------------------------------- #
# Canopy clustering.                                                          #
# --------------------------------------------------------------------------- #
def test_canopy_matches_fixture():
    out = cs.canopy_clustering(POINTS, t1=2.0, t2=1.0, seed=7)
    assert out == EXPECTED["canopies"]


def test_canopy_covers_every_point():
    out = cs.canopy_clustering(POINTS, t1=2.0, t2=1.0, seed=7)
    covered = set()
    for canopy in out:
        covered.update(canopy)
    assert covered == set(range(len(POINTS)))


def test_canopy_threshold_order_raises():
    with pytest.raises(ValueError, match="t2 <= t1"):
        cs.canopy_clustering(POINTS, t1=1.0, t2=2.0, seed=7)


# --------------------------------------------------------------------------- #
# k-means|| seeding.                                                         #
# --------------------------------------------------------------------------- #
def test_kmeans_parallel_init_matches_fixture():
    seeds = cs.kmeans_parallel_init(POINTS, 3, oversampling=2.0, n_rounds=3, seed=123)
    assert seeds == [pytest.approx(s) for s in EXPECTED["seeds"]]


def test_kmeans_parallel_init_returns_k_distinct():
    seeds = cs.kmeans_parallel_init(POINTS, 3, oversampling=2.0, n_rounds=3, seed=123)
    assert len(seeds) == 3
    # Seeds are drawn from the data and one per well-separated cluster here.
    assert all(s in [list(p) for p in POINTS] for s in seeds)


def test_kmeans_parallel_init_k_too_large_raises():
    with pytest.raises(ValueError, match="exceeds number of points"):
        cs.kmeans_parallel_init(POINTS, 99, oversampling=2.0, n_rounds=1, seed=1)
