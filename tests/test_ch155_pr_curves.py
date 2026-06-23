"""Tests for aiinaction.ch155_pr_curves, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. The AP fixture matches scikit-learn's ``average_precision_score`` on the
same data.
"""
from __future__ import annotations

import pytest

from aiinaction import ch155_pr_curves as prc

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
Y_TRUE = [1, 0, 1, 1, 0, 1, 0, 0]
SCORES = [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.51]

EXPECTED = {
    "thresholds": [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.51],
    "precision": [
        1.0,
        0.5,
        0.6666666666666666,
        0.75,
        0.6,
        0.6666666666666666,
        0.5714285714285714,
        0.5,
    ],
    "recall": [0.25, 0.25, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0],
    "average_precision": 0.7708333333333333,
    "auprc_trapezoid": 0.48125,
}


def test_pr_curve_thresholds_match_fixture():
    c = prc.pr_curve(Y_TRUE, SCORES)
    assert c.thresholds == pytest.approx(EXPECTED["thresholds"])


def test_pr_curve_precision_matches_fixture():
    c = prc.pr_curve(Y_TRUE, SCORES)
    assert c.precision == pytest.approx(EXPECTED["precision"])


def test_pr_curve_recall_matches_fixture():
    c = prc.pr_curve(Y_TRUE, SCORES)
    assert c.recall == pytest.approx(EXPECTED["recall"])


def test_average_precision_matches_fixture():
    ap = prc.average_precision(Y_TRUE, SCORES)
    assert ap == pytest.approx(EXPECTED["average_precision"])


def test_auprc_trapezoid_matches_fixture():
    area = prc.auprc_trapezoid(Y_TRUE, SCORES)
    assert area == pytest.approx(EXPECTED["auprc_trapezoid"])


def test_recall_is_non_decreasing():
    c = prc.pr_curve(Y_TRUE, SCORES)
    for a, b in zip(c.recall, c.recall[1:]):
        assert b >= a


def test_curve_length_is_distinct_thresholds():
    c = prc.pr_curve(Y_TRUE, SCORES)
    assert len(c) == len(set(SCORES))


def test_perfect_ranking_gives_ap_one():
    # All positives strictly outscore all negatives.
    y = [1, 1, 0, 0]
    s = [0.9, 0.8, 0.2, 0.1]
    assert prc.average_precision(y, s) == pytest.approx(1.0)


def test_perfect_ranking_trapezoid_over_realized_points():
    # The trapezoid covers only realized curve points (no anchor at recall 0). For
    # a perfect ranking the precision stays 1.0 from recall 0.5 to 1.0, so the area
    # is 1.0 * (1.0 - 0.5) = 0.5. AP, by contrast, is exactly 1.0 above.
    y = [1, 1, 0, 0]
    s = [0.9, 0.8, 0.2, 0.1]
    assert prc.auprc_trapezoid(y, s) == pytest.approx(0.5)


def test_worst_ranking_gives_low_ap():
    # All negatives outscore all positives.
    y = [1, 1, 0, 0]
    s = [0.2, 0.1, 0.9, 0.8]
    # Positives appear at ranks 3 and 4: (1/3 + 2/4) / 2 = 0.41666...
    assert prc.average_precision(y, s) == pytest.approx((1 / 3 + 2 / 4) / 2)


def test_ap_invariant_to_monotone_score_transform():
    # AP depends only on the ranking, so a strictly increasing map leaves it fixed.
    ap1 = prc.average_precision(Y_TRUE, SCORES)
    shifted = [10.0 * x + 3.0 for x in SCORES]
    ap2 = prc.average_precision(Y_TRUE, shifted)
    assert ap1 == pytest.approx(ap2)


def test_tie_handling_is_deterministic():
    # Tied scores across a positive and a negative: lower original index ranks first.
    y = [1, 0]
    s = [0.5, 0.5]
    # Positive (index 0) ranked first: Precision@1 = 1.0, AP = 1.0.
    assert prc.average_precision(y, s) == pytest.approx(1.0)


def test_single_threshold_when_all_scores_equal():
    y = [1, 0, 1]
    s = [0.5, 0.5, 0.5]
    c = prc.pr_curve(y, s)
    assert len(c) == 1
    assert c.recall == pytest.approx([1.0])
    assert c.precision == pytest.approx([2 / 3])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        prc.average_precision([1, 0], [0.5])


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        prc.average_precision([], [])


def test_non_binary_label_raises():
    with pytest.raises(ValueError, match="only 0/1 labels"):
        prc.pr_curve([1, 2, 0], [0.1, 0.2, 0.3])


def test_no_positive_raises():
    with pytest.raises(ValueError, match="at least one positive"):
        prc.average_precision([0, 0, 0], [0.1, 0.2, 0.3])


def test_non_finite_score_raises():
    with pytest.raises(ValueError, match="non-finite"):
        prc.pr_curve([1, 0], [float("nan"), 0.3])
