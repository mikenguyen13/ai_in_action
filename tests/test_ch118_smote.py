"""Tests for aiinaction.ch118_smote, including the shared cross-language fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the identical minority set, seed, neighbor lists, and synthetic
points (to 1e-9 tolerance), which is what keeps the three SMOTE implementations at
parity.
"""
from __future__ import annotations

import pytest

from aiinaction import ch118_smote as sm

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
MINORITY = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]]
SEED = 42
K = 2

# Neighbor lists (k=2), deterministic with ties broken by lower index.
EXPECTED_KNN = {
    0: [1, 2],
    1: [0, 2],
    2: [1, 3],
    3: [2, 1],
}

# Synthetic points from smote(MINORITY, 4, k=2, seed=42).
EXPECTED_SMOTE = [
    [0.17625009082257748, 0.0],
    [1.222554265987128, 0.777445734012872],
    [2.0256639048457146, 0.02566390484571457],
    [2.763079992495477, 1.0],
]

# LCG(42).next_float() sequence.
EXPECTED_LCG = [
    0.252345174784,
    0.088125045411,
    0.577281198232,
    0.222554265987,
]


def test_euclidean_3_4_5():
    assert sm.euclidean([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)


def test_euclidean_zero():
    assert sm.euclidean([1.5, -2.0], [1.5, -2.0]) == 0.0


def test_lcg_sequence_matches_fixture():
    rng = sm.LCG(SEED)
    for expected in EXPECTED_LCG:
        assert rng.next_float() == pytest.approx(expected, abs=1e-9)


def test_lcg_floats_in_unit_interval():
    rng = sm.LCG(123)
    for _ in range(100):
        v = rng.next_float()
        assert 0.0 <= v < 1.0


def test_lcg_index_in_range():
    rng = sm.LCG(7)
    for _ in range(50):
        assert 0 <= rng.next_index(3) < 3


@pytest.mark.parametrize("idx,expected", list(EXPECTED_KNN.items()))
def test_k_nearest_matches_fixture(idx, expected):
    assert sm.k_nearest(MINORITY, idx, K) == expected


def test_smote_sample_midpoint():
    assert sm.smote_sample([0.0, 0.0], [2.0, 4.0], 0.5) == [1.0, 2.0]


def test_smote_sample_endpoints():
    assert sm.smote_sample([1.0, 1.0], [3.0, 5.0], 0.0) == [1.0, 1.0]
    assert sm.smote_sample([1.0, 1.0], [3.0, 5.0], 1.0) == [3.0, 5.0]


def test_smote_matches_fixture():
    pts = sm.smote(MINORITY, 4, k=K, seed=SEED)
    assert len(pts) == 4
    for got, expected in zip(pts, EXPECTED_SMOTE):
        assert got == pytest.approx(expected, abs=1e-9)


def test_smote_count():
    pts = sm.smote(MINORITY, 10, k=K, seed=SEED)
    assert len(pts) == 10


def test_smote_zero_synthetic_empty():
    assert sm.smote(MINORITY, 0, k=K, seed=SEED) == []


def test_smote_points_lie_in_minority_hull_dim():
    pts = sm.smote(MINORITY, 5, k=K, seed=SEED)
    for p in pts:
        assert len(p) == 2


# --- validation / edge cases ---

def test_euclidean_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="dimension mismatch"):
        sm.euclidean([1.0, 2.0], [1.0])


def test_lcg_negative_seed_raises():
    with pytest.raises(ValueError, match="non-negative"):
        sm.LCG(-1)


def test_k_nearest_k_too_large_raises():
    with pytest.raises(ValueError, match="exceeds available neighbors"):
        sm.k_nearest(MINORITY, 0, 4)


def test_k_nearest_bad_idx_raises():
    with pytest.raises(ValueError, match="out of range"):
        sm.k_nearest(MINORITY, 9, 2)


def test_smote_negative_count_raises():
    with pytest.raises(ValueError, match="non-negative"):
        sm.smote(MINORITY, -1, k=K, seed=SEED)


def test_smote_k_too_large_raises():
    with pytest.raises(ValueError, match="exceeds available neighbors"):
        sm.smote([[0.0, 0.0], [1.0, 1.0]], 3, k=5, seed=SEED)


def test_smote_empty_minority_raises():
    with pytest.raises(ValueError, match="non-empty"):
        sm.smote([], 3, k=2, seed=SEED)


def test_smote_sample_lambda_out_of_range_raises():
    with pytest.raises(ValueError, match=r"lam must be in"):
        sm.smote_sample([0.0], [1.0], 1.5)
