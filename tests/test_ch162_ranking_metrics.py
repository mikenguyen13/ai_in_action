"""Tests for aiinaction.ch162_ranking_metrics, including shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity.
"""
from __future__ import annotations

import pytest

from aiinaction import ch162_ranking_metrics as rm

# --- Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src ---

# MRR: three queries with a single relevant item each, at ranks 1, 3, 2.
MRR_QUERIES = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]

# AP / MAP: two queries under binary relevance.
AP_Q1 = [1, 0, 1, 0, 0, 1]
AP_Q2 = [0, 1, 1, 0]

# NDCG: graded relevance lists.
NDCG_Q1 = [3, 2, 0, 1, 2]
NDCG_Q2 = [0, 0, 2, 1]

EXPECTED = {
    "mrr": 0.6111111111111112,
    "rr_q1": 1.0,
    "rr_q2": 0.3333333333333333,
    "rr_q3": 0.5,
    "ap_q1": 0.7222222222222222,
    "ap_q2": 0.5833333333333333,
    "map": 0.6527777777777778,
    "p_at_3_q1": 0.6666666666666666,
    "dcg_q1": 10.484024240491392,
    "idcg_q1": 10.823465818787767,
    "ndcg_q1": 0.9686383655679718,
    "ndcg_q2": 0.531730627995306,
    "mean_ndcg": 0.7501844967816389,
}


def test_reciprocal_rank_per_query():
    assert rm.reciprocal_rank(MRR_QUERIES[0]) == pytest.approx(EXPECTED["rr_q1"])
    assert rm.reciprocal_rank(MRR_QUERIES[1]) == pytest.approx(EXPECTED["rr_q2"])
    assert rm.reciprocal_rank(MRR_QUERIES[2]) == pytest.approx(EXPECTED["rr_q3"])


def test_mean_reciprocal_rank_matches_fixture():
    assert rm.mean_reciprocal_rank(MRR_QUERIES) == pytest.approx(EXPECTED["mrr"])


def test_reciprocal_rank_no_hit_is_zero():
    assert rm.reciprocal_rank([0, 0, 0]) == 0.0


def test_reciprocal_rank_cutoff_excludes_late_hit():
    assert rm.reciprocal_rank([0, 0, 1, 0], k=2) == 0.0
    assert rm.reciprocal_rank([0, 0, 1, 0], k=3) == pytest.approx(1.0 / 3.0)


def test_average_precision_matches_fixture():
    assert rm.average_precision(AP_Q1) == pytest.approx(EXPECTED["ap_q1"])
    assert rm.average_precision(AP_Q2) == pytest.approx(EXPECTED["ap_q2"])


def test_mean_average_precision_matches_fixture():
    assert rm.mean_average_precision([AP_Q1, AP_Q2]) == pytest.approx(EXPECTED["map"])


def test_average_precision_perfect_is_one():
    assert rm.average_precision([1, 1, 1, 0, 0]) == pytest.approx(1.0)


def test_average_precision_no_relevant_is_zero():
    assert rm.average_precision([0, 0, 0]) == 0.0


def test_average_precision_with_explicit_n_relevant():
    # Three positives present but R=5 total relevant in the pool: AP penalized.
    assert rm.average_precision(AP_Q1, n_relevant=5) == pytest.approx(0.43333333333333335)


def test_average_precision_cutoff():
    # k=3 keeps the hits at ranks 1 and 3 only: (1/1 + 2/3) / 3.
    assert rm.average_precision(AP_Q1, k=3) == pytest.approx(5.0 / 9.0)


def test_precision_at_k_matches_fixture():
    assert rm.precision_at_k(AP_Q1, 3) == pytest.approx(EXPECTED["p_at_3_q1"])


def test_precision_at_k_beyond_length_uses_k_denominator():
    # Two relevant items, k=10: precision = 2/10.
    assert rm.precision_at_k([1, 0, 1], 10) == pytest.approx(0.2)


def test_dcg_matches_fixture():
    assert rm.dcg(NDCG_Q1) == pytest.approx(EXPECTED["dcg_q1"])


def test_idcg_matches_fixture():
    ideal = sorted(NDCG_Q1, reverse=True)
    assert rm.dcg(ideal) == pytest.approx(EXPECTED["idcg_q1"])


def test_ndcg_matches_fixture():
    assert rm.ndcg(NDCG_Q1) == pytest.approx(EXPECTED["ndcg_q1"])
    assert rm.ndcg(NDCG_Q2) == pytest.approx(EXPECTED["ndcg_q2"])


def test_mean_ndcg_matches_fixture():
    assert rm.mean_ndcg([NDCG_Q1, NDCG_Q2]) == pytest.approx(EXPECTED["mean_ndcg"])


def test_ndcg_ideal_ordering_is_one():
    assert rm.ndcg([3, 2, 2, 1, 0]) == pytest.approx(1.0)


def test_ndcg_all_zero_is_zero():
    assert rm.ndcg([0, 0, 0]) == 0.0


def test_ndcg_linear_gain_differs():
    exp = rm.ndcg(NDCG_Q1, exponential=True)
    lin = rm.ndcg(NDCG_Q1, exponential=False)
    assert exp != pytest.approx(lin)


def test_dcg_linear_gain():
    # Linear gain, single relevant item at rank 1: DCG = rel / log2(2) = 3.
    assert rm.dcg([3, 0, 0], exponential=False) == pytest.approx(3.0)


# --- Edge cases and validation ---


def test_empty_queries_raise():
    with pytest.raises(ValueError, match="non-empty"):
        rm.mean_reciprocal_rank([])
    with pytest.raises(ValueError, match="non-empty"):
        rm.mean_average_precision([])
    with pytest.raises(ValueError, match="non-empty"):
        rm.mean_ndcg([])


def test_negative_relevance_raises():
    with pytest.raises(ValueError, match="non-negative"):
        rm.dcg([1, -1, 0])
    with pytest.raises(ValueError, match="non-negative"):
        rm.reciprocal_rank([0, -2])


def test_bad_cutoff_raises():
    with pytest.raises(ValueError, match="positive"):
        rm.reciprocal_rank([1, 0], k=0)
    with pytest.raises(ValueError, match="positive"):
        rm.precision_at_k([1, 0], 0)


def test_n_relevant_too_small_raises():
    with pytest.raises(ValueError, match="smaller than"):
        rm.average_precision([1, 1, 1], n_relevant=2)


def test_n_relevant_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        rm.mean_average_precision([AP_Q1, AP_Q2], n_relevant=[3])
