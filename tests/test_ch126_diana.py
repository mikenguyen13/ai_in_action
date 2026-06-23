"""Tests for aiinaction.ch126_diana (DIANA divisive clustering).

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same distance matrix and the same expected splits/labels, which
is what keeps the three implementations at parity.
"""
from __future__ import annotations

import pytest

from aiinaction import ch126_diana as diana_mod
from aiinaction.ch126_diana import (
    Split,
    diameter,
    diana,
    diana_labels,
    macnaughton_smith_split,
)

# Shared fixture: 1-D points {1,2,3,20,21,22} with d = absolute difference.
# Mirrored exactly in julia/AIInAction/test and rust/aiinaction/src/ch126_diana.rs.
POINTS = [1.0, 2.0, 3.0, 20.0, 21.0, 22.0]
N = len(POINTS)
DIST = [[abs(POINTS[i] - POINTS[j]) for j in range(N)] for i in range(N)]

EXPECTED_DIAMETER = 21.0
EXPECTED_SPLINTER = (0, 1, 2)
EXPECTED_REMAINDER = (3, 4, 5)
EXPECTED_LABELS_K2 = [0, 0, 0, 1, 1, 1]
EXPECTED_LABELS_K3 = [0, 1, 1, 2, 2, 2]
EXPECTED_LABELS_K6 = [0, 1, 2, 3, 4, 5]


def test_diameter_matches_fixture():
    assert diameter(DIST, list(range(N))) == pytest.approx(EXPECTED_DIAMETER)


def test_diameter_singleton_is_zero():
    assert diameter(DIST, [3]) == 0.0


def test_first_split_matches_fixture():
    splinter, remainder = macnaughton_smith_split(DIST, list(range(N)))
    assert splinter == EXPECTED_SPLINTER
    assert remainder == EXPECTED_REMAINDER


def test_split_partitions_cluster():
    splinter, remainder = macnaughton_smith_split(DIST, list(range(N)))
    assert sorted(splinter + remainder) == list(range(N))
    assert set(splinter).isdisjoint(remainder)


def test_singleton_split():
    splinter, remainder = macnaughton_smith_split(DIST, [2])
    assert splinter == ()
    assert remainder == (2,)


def test_full_diana_split_count_and_order():
    splits = diana(DIST)
    # n - 1 internal nodes.
    assert len(splits) == N - 1
    # The first (largest-diameter) split is the macro cut.
    first = splits[0]
    assert isinstance(first, Split)
    assert first.parent == (0, 1, 2, 3, 4, 5)
    assert first.splinter == EXPECTED_SPLINTER
    assert first.remainder == EXPECTED_REMAINDER
    assert first.diameter == pytest.approx(EXPECTED_DIAMETER)
    # Diameters are non-increasing in split order (widest-first selection).
    diams = [s.diameter for s in splits]
    assert diams == sorted(diams, reverse=True)


def test_labels_k2():
    assert diana_labels(DIST, 2) == EXPECTED_LABELS_K2


def test_labels_k3():
    assert diana_labels(DIST, 3) == EXPECTED_LABELS_K3


def test_labels_k_equals_n_is_singletons():
    assert diana_labels(DIST, N) == EXPECTED_LABELS_K6


def test_labels_k1_is_all_zero():
    assert diana_labels(DIST, 1) == [0] * N


def test_single_object_dataset():
    assert diana([[0.0]]) == []
    assert diana_labels([[0.0]], 1) == [0]


# --- validation / edge cases ---


def test_non_square_matrix_raises():
    with pytest.raises(ValueError, match="square"):
        diameter([[0.0, 1.0]], [0, 1])


def test_asymmetric_matrix_raises():
    with pytest.raises(ValueError, match="symmetric"):
        diana([[0.0, 1.0], [2.0, 0.0]])


def test_nonzero_diagonal_raises():
    with pytest.raises(ValueError, match="zero diagonal"):
        diana([[1.0, 1.0], [1.0, 1.0]])


def test_negative_distance_raises():
    with pytest.raises(ValueError, match="non-negative"):
        diana([[0.0, -1.0], [-1.0, 0.0]])


def test_empty_matrix_raises():
    # An empty 2-D matrix is rejected before any member processing.
    with pytest.raises(ValueError):
        diameter([], [])


def test_bad_member_index_raises():
    with pytest.raises(ValueError, match="out of range"):
        macnaughton_smith_split(DIST, [0, 99])


def test_duplicate_member_raises():
    with pytest.raises(ValueError, match="duplicate"):
        macnaughton_smith_split(DIST, [0, 0])


@pytest.mark.parametrize("k", [0, N + 1, -1])
def test_bad_k_raises(k):
    with pytest.raises(ValueError, match="k must be"):
        diana_labels(DIST, k)
