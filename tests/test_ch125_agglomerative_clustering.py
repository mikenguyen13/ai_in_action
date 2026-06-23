"""Tests for aiinaction.ch125_agglomerative_clustering and shared cross-language fixtures.

The PTS fixture and the EXPECTED merge heights are the single source of truth: the
Julia and Rust test suites assert against the same numbers, which keeps the three
libraries at parity. The expected heights were cross-checked against
``scipy.cluster.hierarchy.linkage``.
"""
from __future__ import annotations

import math

import pytest

from aiinaction import ch125_agglomerative_clustering as hc

# Shared fixture: three points near the origin and two points near (10, 10).
PTS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [10.0, 10.0],
    [10.0, 11.0],
]

# Merge structure (node_a, node_b, size) is identical across linkages; only the
# final two heights differ. Index 0 and 1 always merge first at height 1, and
# 3 and 4 merge at height 1; the third merge joins point 2 with cluster {0,1}.
EXPECTED_FINAL_HEIGHTS = {
    "single": (1.0, 13.45362404707371),
    "complete": (1.414213562373095, 14.866068747318508),
    "average": (1.2071067811865475, 14.045043082079953),
    "ward": (1.2909944487358056, 21.73323108360405),
}


@pytest.mark.parametrize("linkage", hc.LINKAGES)
def test_first_merge_is_nearest_pair(linkage):
    m = hc.linkage_matrix(PTS, linkage)
    assert int(m[0][0]) == 0 and int(m[0][1]) == 1
    assert m[0][2] == pytest.approx(1.0)
    assert m[0][3] == pytest.approx(2.0)


@pytest.mark.parametrize("linkage", hc.LINKAGES)
def test_final_two_heights_match_fixture(linkage):
    m = hc.linkage_matrix(PTS, linkage)
    h_third, h_root = EXPECTED_FINAL_HEIGHTS[linkage]
    assert m[2][2] == pytest.approx(h_third)
    assert m[3][2] == pytest.approx(h_root)
    # Root merge joins the two top-level subtrees and contains all 5 points.
    assert m[3][3] == pytest.approx(5.0)


@pytest.mark.parametrize("linkage", hc.LINKAGES)
def test_linkage_matrix_shape_and_monotonic(linkage):
    m = hc.linkage_matrix(PTS, linkage)
    assert len(m) == len(PTS) - 1
    heights = [row[2] for row in m]
    # Monotone non-decreasing for all four linkages, up to float noise.
    for earlier, later in zip(heights, heights[1:]):
        assert later >= earlier - 1e-9
    for row in m:
        assert row[0] < row[1]


@pytest.mark.parametrize("linkage", hc.LINKAGES)
def test_fcluster_two_groups(linkage):
    m = hc.linkage_matrix(PTS, linkage)
    labels = hc.fcluster(m, 2)
    assert labels == [0, 0, 0, 1, 1]
    # k = n gives every point its own cluster.
    assert hc.fcluster(m, len(PTS)) == [0, 1, 2, 3, 4]
    # k = 1 collapses everything.
    assert hc.fcluster(m, 1) == [0, 0, 0, 0, 0]


def test_cophenetic_distances_single():
    m = hc.linkage_matrix(PTS, "single")
    coph = hc.cophenetic_distances(m)
    root = EXPECTED_FINAL_HEIGHTS["single"][1]
    assert coph[0][1] == pytest.approx(1.0)
    assert coph[3][4] == pytest.approx(1.0)
    assert coph[0][3] == pytest.approx(root)
    assert coph[0][0] == 0.0
    # Symmetric.
    for i in range(len(PTS)):
        for j in range(len(PTS)):
            assert coph[i][j] == coph[j][i]


def test_collinear_example_matches_docstring():
    m = hc.linkage_matrix([[0.0], [0.0], [5.0]], "single")
    assert [round(x, 6) for x in m[0]] == [0.0, 1.0, 0.0, 2.0]
    assert [round(x, 6) for x in m[1]] == [2.0, 3.0, 5.0, 3.0]


def test_unknown_linkage_raises():
    with pytest.raises(ValueError, match="unknown linkage"):
        hc.linkage_matrix(PTS, "centroid")


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="at least 2 points"):
        hc.linkage_matrix([[1.0, 2.0]], "single")


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        hc.linkage_matrix([], "single")


def test_ragged_points_raise():
    with pytest.raises(ValueError, match="same number of features"):
        hc.linkage_matrix([[0.0, 0.0], [1.0]], "single")


def test_fcluster_out_of_range_raises():
    m = hc.linkage_matrix(PTS, "ward")
    with pytest.raises(ValueError, match="n_clusters must be"):
        hc.fcluster(m, 0)
    with pytest.raises(ValueError, match="n_clusters must be"):
        hc.fcluster(m, len(PTS) + 1)
