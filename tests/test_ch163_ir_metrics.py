"""Tests for aiinaction.ch163_ir_metrics, including shared cross-language fixtures.

The numbers asserted here are the single source of truth: the Julia and Rust test
suites assert against the same fixtures, which keeps the three libraries at parity.
"""
from __future__ import annotations

import math

import pytest

from aiinaction import ch163_ir_metrics as ir

# Shared fixtures, mirrored in julia/AIInAction/test and rust/aiinaction tests.
# A single ranked query with binary relevance labels (rank 1 first).
RANKING = [1, 0, 1, 1, 0, 1]          # relevant at ranks 1, 3, 4, 6
NUM_RELEVANT = 5                       # one relevant doc never retrieved

# Graded relevances for the DCG/NDCG fixtures.
GRADES = [3, 2, 3, 0, 1, 2]
IDEAL_GRADES = [3, 3, 2, 2, 1, 0]      # full judged pool, best ordering

# A small query set for MAP / MRR.
QUERY_SET = [
    [1, 0, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
]

EXPECTED = {
    "p_at_1": 1.0,
    "p_at_3": 2.0 / 3.0,
    "p_at_6": 4.0 / 6.0,
    "r_at_3": 2.0 / 5.0,
    "r_at_6": 4.0 / 5.0,
    # AP over RANKING with num_relevant=5: (1/1 + 2/3 + 3/4 + 4/6) / 5
    "ap": (1.0 + 2.0 / 3.0 + 3.0 / 4.0 + 4.0 / 6.0) / 5.0,
    "dcg_at_3": 7.0 / math.log2(2) + 3.0 / math.log2(3) + 7.0 / math.log2(4),
    "ndcg_at_3": (7.0 / math.log2(2) + 3.0 / math.log2(3) + 7.0 / math.log2(4))
    / (7.0 / math.log2(2) + 7.0 / math.log2(3) + 3.0 / math.log2(4)),
    "rr": 1.0,  # first hit at rank 1 in RANKING
}


def test_precision_at_k_matches_fixture():
    assert ir.precision_at_k(RANKING, 1) == pytest.approx(EXPECTED["p_at_1"])
    assert ir.precision_at_k(RANKING, 3) == pytest.approx(EXPECTED["p_at_3"])
    assert ir.precision_at_k(RANKING, 6) == pytest.approx(EXPECTED["p_at_6"])


def test_precision_k_is_clamped_to_length():
    assert ir.precision_at_k(RANKING, 100) == pytest.approx(EXPECTED["p_at_6"])


def test_recall_at_k_matches_fixture():
    assert ir.recall_at_k(RANKING, 3, NUM_RELEVANT) == pytest.approx(EXPECTED["r_at_3"])
    assert ir.recall_at_k(RANKING, 6, NUM_RELEVANT) == pytest.approx(EXPECTED["r_at_6"])


def test_average_precision_matches_fixture():
    assert ir.average_precision(RANKING, NUM_RELEVANT) == pytest.approx(EXPECTED["ap"])


def test_average_precision_default_num_relevant():
    # Without num_relevant, divisor is the 4 relevant labels present.
    expected = (1.0 + 2.0 / 3.0 + 3.0 / 4.0 + 4.0 / 6.0) / 4.0
    assert ir.average_precision(RANKING) == pytest.approx(expected)


def test_average_precision_perfect_ranking():
    assert ir.average_precision([1, 1, 1]) == pytest.approx(1.0)


def test_dcg_at_k_matches_fixture():
    assert ir.dcg_at_k(GRADES, 3) == pytest.approx(EXPECTED["dcg_at_3"])


def test_ndcg_at_k_self_ideal_matches_fixture():
    # Ideal derived from GRADES itself (top-3 view).
    assert ir.ndcg_at_k(GRADES, 3) == pytest.approx(EXPECTED["ndcg_at_3"])


def test_ndcg_ideal_ranking_is_one():
    assert ir.ndcg_at_k(IDEAL_GRADES, 6) == pytest.approx(1.0)


def test_ndcg_with_external_ideal_pool():
    val = ir.ndcg_at_k(GRADES, 6, ideal_grades=IDEAL_GRADES)
    assert 0.0 < val <= 1.0
    # DCG of GRADES@6 over IDCG of the ideal pool@6.
    dcg = ir.dcg_at_k(GRADES, 6)
    idcg = ir.dcg_at_k(sorted(IDEAL_GRADES, reverse=True), 6)
    assert val == pytest.approx(dcg / idcg)


def test_reciprocal_rank_matches_fixture():
    assert ir.reciprocal_rank(RANKING) == pytest.approx(EXPECTED["rr"])
    assert ir.reciprocal_rank([0, 0, 1, 0]) == pytest.approx(1.0 / 3.0)


def test_reciprocal_rank_no_hit_is_zero():
    assert ir.reciprocal_rank([0, 0, 0]) == 0.0


def test_mean_average_precision_over_query_set():
    aps = [ir.average_precision(r) for r in QUERY_SET]
    assert ir.mean_average_precision(QUERY_SET) == pytest.approx(sum(aps) / len(aps))


def test_mean_reciprocal_rank_over_query_set():
    # First hits at ranks 1, 2, 6 -> (1 + 1/2 + 1/6) / 3.
    expected = (1.0 + 1.0 / 2.0 + 1.0 / 6.0) / 3.0
    assert ir.mean_reciprocal_rank(QUERY_SET) == pytest.approx(expected)


# --- edge cases and validation ---


def test_precision_rejects_non_binary():
    with pytest.raises(ValueError, match="binary"):
        ir.precision_at_k([1, 2, 0], 3)


def test_precision_rejects_nonpositive_k():
    with pytest.raises(ValueError, match="positive"):
        ir.precision_at_k([1, 0, 1], 0)


def test_recall_rejects_num_relevant_too_small():
    with pytest.raises(ValueError, match="smaller"):
        ir.recall_at_k([1, 1, 1], 3, num_relevant=2)


def test_recall_rejects_nonpositive_num_relevant():
    with pytest.raises(ValueError, match="positive"):
        ir.recall_at_k([1, 0, 1], 3, num_relevant=0)


def test_grades_reject_negative():
    with pytest.raises(ValueError, match="non-negative"):
        ir.dcg_at_k([1, -1, 2], 3)


def test_ndcg_zero_idcg_raises():
    with pytest.raises(ValueError, match="undefined"):
        ir.ndcg_at_k([0, 0, 0], 3)


def test_empty_inputs_raise():
    with pytest.raises(ValueError, match="non-empty"):
        ir.precision_at_k([], 1)
    with pytest.raises(ValueError, match="non-empty"):
        ir.dcg_at_k([], 1)


def test_empty_query_set_raises():
    with pytest.raises(ValueError, match="at least one query"):
        ir.mean_average_precision([])
    with pytest.raises(ValueError, match="at least one query"):
        ir.mean_reciprocal_rank([])


def test_map_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        ir.mean_average_precision([[1, 0], [0, 1]], num_relevant=[1])
