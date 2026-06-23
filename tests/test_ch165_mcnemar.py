"""Tests for aiinaction.ch165_mcnemar, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which keeps the three implementations at parity.
"""
from __future__ import annotations

import pytest

from aiinaction import ch165_mcnemar as mc

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
# A 12-example correctness pattern for the contingency-table check.
CORRECT_A = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1]
CORRECT_B = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]

EXPECTED_TABLE = {"a": 2, "b": 6, "c": 3, "d": 1, "n": 12, "n_discordant": 9}

# Discordant-count fixtures: (b, c) -> expected statistics.
# chi2_corr / p_corr use Edwards' continuity correction; chi2_raw / p_raw do not;
# exact_p is the two-sided exact binomial p-value on min(b, c).
EXPECTED = {
    (12, 5): {
        "chi2_corr": 2.1176470588235294,
        "p_corr": 0.14561009539686698,
        "chi2_raw": 2.8823529411764706,
        "p_raw": 0.08955507441364255,
        "exact_p": 0.1434631347656256,
    },
    (30, 15): {
        "chi2_corr": 4.355555555555555,
        "p_corr": 0.03688842570704986,
        "chi2_raw": 5.0,
        "p_raw": 0.025347318677468256,
        "exact_p": 0.03569780355519456,
    },
    (3, 1): {
        "chi2_corr": 0.25,
        "p_corr": 0.6170750774519738,
        "chi2_raw": 1.0,
        "p_raw": 0.31731050786291404,
        "exact_p": 0.6249999999999994,
    },
}


def test_contingency_table_matches_fixture():
    t = mc.contingency_table(CORRECT_A, CORRECT_B)
    assert (t.a, t.b, t.c, t.d) == (
        EXPECTED_TABLE["a"],
        EXPECTED_TABLE["b"],
        EXPECTED_TABLE["c"],
        EXPECTED_TABLE["d"],
    )
    assert t.n == EXPECTED_TABLE["n"]
    assert t.n_discordant == EXPECTED_TABLE["n_discordant"]


@pytest.mark.parametrize("bc", list(EXPECTED.keys()))
def test_chi2_with_correction_matches_fixture(bc):
    b, c = bc
    r = mc.mcnemar_test(b, c, exact=False, correction=True)
    assert r.method == "chi2"
    assert r.statistic == pytest.approx(EXPECTED[bc]["chi2_corr"])
    assert r.p_value == pytest.approx(EXPECTED[bc]["p_corr"])


@pytest.mark.parametrize("bc", list(EXPECTED.keys()))
def test_chi2_without_correction_matches_fixture(bc):
    b, c = bc
    r = mc.mcnemar_test(b, c, exact=False, correction=False)
    assert r.statistic == pytest.approx(EXPECTED[bc]["chi2_raw"])
    assert r.p_value == pytest.approx(EXPECTED[bc]["p_raw"])


@pytest.mark.parametrize("bc", list(EXPECTED.keys()))
def test_exact_matches_fixture(bc):
    b, c = bc
    r = mc.mcnemar_test(b, c, exact=True)
    assert r.method == "exact"
    assert r.statistic == pytest.approx(float(min(b, c)))
    assert r.p_value == pytest.approx(EXPECTED[bc]["exact_p"])


def test_auto_method_selection():
    # b + c = 17 < 25 -> exact; b + c = 45 >= 25 -> chi2.
    assert mc.mcnemar_test(12, 5).method == "exact"
    assert mc.mcnemar_test(30, 15).method == "chi2"
    assert mc.mcnemar_test(3, 1).method == "exact"


def test_symmetry_in_b_and_c():
    # Swapping b and c is a two-sided test, so the p-value is unchanged.
    r1 = mc.mcnemar_test(30, 15, exact=False)
    r2 = mc.mcnemar_test(15, 30, exact=False)
    assert r1.p_value == pytest.approx(r2.p_value)
    assert r1.statistic == pytest.approx(r2.statistic)
    e1 = mc.mcnemar_test(12, 5, exact=True)
    e2 = mc.mcnemar_test(5, 12, exact=True)
    assert e1.p_value == pytest.approx(e2.p_value)


def test_equal_discordant_counts_give_p_one():
    # b == c is the strongest evidence for H_0.
    r = mc.mcnemar_test(10, 10, exact=False)
    assert r.statistic == pytest.approx(0.0)
    assert r.p_value == pytest.approx(1.0)
    e = mc.mcnemar_test(10, 10, exact=True)
    assert e.p_value == pytest.approx(1.0)


def test_contingency_then_test_round_trip():
    t = mc.contingency_table(CORRECT_A, CORRECT_B)
    r = mc.mcnemar_test(t.b, t.c, exact=True)
    assert r.b == EXPECTED_TABLE["b"]
    assert r.c == EXPECTED_TABLE["c"]


def test_zero_discordant_raises():
    with pytest.raises(ValueError, match="undefined when b \\+ c = 0"):
        mc.mcnemar_test(0, 0)


def test_negative_count_raises():
    with pytest.raises(ValueError, match="non-negative"):
        mc.mcnemar_test(-1, 5)


def test_non_integer_count_raises():
    with pytest.raises(ValueError, match="non-negative integer"):
        mc.mcnemar_test(2.5, 5)  # type: ignore[arg-type]


def test_boolean_count_rejected():
    # bool is a subclass of int; counts must be true ints.
    with pytest.raises(ValueError, match="non-negative integer"):
        mc.mcnemar_test(True, 5)  # type: ignore[arg-type]


def test_table_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        mc.contingency_table([1, 0, 1], [1, 0])


def test_table_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        mc.contingency_table([], [])
