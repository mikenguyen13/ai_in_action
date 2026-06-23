"""Tests for aiinaction.ch150_association_rules and the shared parity fixtures.

The fixtures below are the single source of truth: the Julia and Rust suites assert
against the same numbers, which keeps the three implementations at parity. The data
set is the classic Han/Kamber market-basket example (9 transactions over items
1..5) with min_support = 2/9.
"""
from __future__ import annotations

import math

import pytest

from aiinaction import ch150_association_rules as ar

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
TRANSACTIONS = [
    [1, 2, 5],
    [2, 4],
    [2, 3],
    [1, 2, 4],
    [1, 3],
    [2, 3],
    [1, 3],
    [1, 2, 3, 5],
    [1, 2, 3],
]
MIN_SUPPORT = 2.0 / 9.0  # count threshold 2 out of 9 transactions

# All frequent itemsets (itemset -> support). 13 of them at this threshold.
EXPECTED_ITEMSETS = {
    (1,): 6 / 9,
    (2,): 7 / 9,
    (3,): 6 / 9,
    (4,): 2 / 9,
    (5,): 2 / 9,
    (1, 2): 4 / 9,
    (1, 3): 4 / 9,
    (1, 5): 2 / 9,
    (2, 3): 4 / 9,
    (2, 4): 2 / 9,
    (2, 5): 2 / 9,
    (1, 2, 3): 2 / 9,
    (1, 2, 5): 2 / 9,
}


def _sorted_keys(d):
    return sorted(d.keys(), key=lambda x: (len(x), x))


def test_apriori_finds_expected_itemsets():
    fis = ar.apriori(TRANSACTIONS, MIN_SUPPORT)
    assert _sorted_keys(fis) == _sorted_keys(EXPECTED_ITEMSETS)


def test_apriori_supports_match_fixture():
    fis = ar.apriori(TRANSACTIONS, MIN_SUPPORT)
    for k, v in EXPECTED_ITEMSETS.items():
        assert fis[k] == pytest.approx(v)


def test_fpgrowth_matches_apriori_keys():
    fis = ar.apriori(TRANSACTIONS, MIN_SUPPORT)
    fg = ar.fpgrowth(TRANSACTIONS, MIN_SUPPORT)
    assert _sorted_keys(fg) == _sorted_keys(fis)


def test_fpgrowth_supports_match_fixture():
    fg = ar.fpgrowth(TRANSACTIONS, MIN_SUPPORT)
    for k, v in EXPECTED_ITEMSETS.items():
        assert fg[k] == pytest.approx(v)


def test_support_helper():
    assert ar.support(TRANSACTIONS, [1, 2]) == pytest.approx(4 / 9)
    assert ar.support(TRANSACTIONS, [2]) == pytest.approx(7 / 9)
    assert ar.support(TRANSACTIONS, [4, 5]) == pytest.approx(0.0)


def test_association_rules_high_confidence():
    fis = ar.apriori(TRANSACTIONS, MIN_SUPPORT)
    rules = ar.association_rules(fis, 0.7)
    got = {(r.antecedent, r.consequent) for r in rules}
    expected = {
        ((1, 5), (2,)),
        ((2, 5), (1,)),
        ((4,), (2,)),
        ((5,), (1,)),
        ((5,), (1, 2)),
        ((5,), (2,)),
    }
    assert got == expected
    assert len(rules) == 6


def test_rule_metrics_match_fixture():
    fis = ar.apriori(TRANSACTIONS, MIN_SUPPORT)
    rules = ar.association_rules(fis, 0.7)
    by_key = {(r.antecedent, r.consequent): r for r in rules}

    r = by_key[((5,), (1, 2))]
    assert r.support == pytest.approx(2 / 9)
    assert r.confidence == pytest.approx(1.0)
    assert r.lift == pytest.approx(2.25)
    assert r.leverage == pytest.approx(0.12345679012345678)
    assert math.isinf(r.conviction)

    r2 = by_key[((2, 5), (1,))]
    assert r2.lift == pytest.approx(1.5)
    assert r2.leverage == pytest.approx(0.07407407407407407)


def test_rule_with_finite_conviction():
    fis = ar.apriori(TRANSACTIONS, MIN_SUPPORT)
    rules = ar.association_rules(fis, 0.5)
    by_key = {(r.antecedent, r.consequent): r for r in rules}
    r = by_key[((1,), (2,))]
    assert r.confidence == pytest.approx(2 / 3)
    assert r.lift == pytest.approx(0.8571428571428571)
    assert r.leverage == pytest.approx(-0.07407407407407407)
    assert r.conviction == pytest.approx(0.6666666666666665)


def test_rules_are_sorted_by_descending_confidence():
    fis = ar.apriori(TRANSACTIONS, MIN_SUPPORT)
    rules = ar.association_rules(fis, 0.0)
    confs = [r.confidence for r in rules]
    assert confs == sorted(confs, reverse=True)


def test_singletons_generate_no_rules():
    fis = {(1,): 0.5, (2,): 0.5}
    assert ar.association_rules(fis, 0.0) == []


def test_min_support_one_keeps_only_universal_items():
    # Only item present in every transaction would survive; here none is.
    data = [[1, 2], [2, 3], [2, 4]]
    fis = ar.apriori(data, 1.0)
    assert set(fis.keys()) == {(2,)}
    assert fis[(2,)] == pytest.approx(1.0)


def test_apriori_fpgrowth_agree_on_dense_data():
    data = [[1, 2, 3, 4], [1, 2, 3], [1, 2, 4], [1, 2], [2, 3, 4]]
    for ms in (0.2, 0.4, 0.6, 0.8):
        a = ar.apriori(data, ms)
        f = ar.fpgrowth(data, ms)
        assert _sorted_keys(a) == _sorted_keys(f)
        for k in a:
            assert a[k] == pytest.approx(f[k])


def test_empty_transactions_raises():
    with pytest.raises(ValueError, match="non-empty"):
        ar.apriori([], 0.5)


def test_bad_min_support_raises():
    with pytest.raises(ValueError, match="min_support must be in"):
        ar.apriori(TRANSACTIONS, 0.0)
    with pytest.raises(ValueError, match="min_support must be in"):
        ar.apriori(TRANSACTIONS, 1.5)


def test_non_integer_item_raises():
    with pytest.raises(ValueError, match="non-integer item"):
        ar.apriori([[1, 2], [1, 2.5]], 0.5)


def test_string_transaction_raises():
    with pytest.raises(ValueError, match="string"):
        ar.apriori(["ab", "cd"], 0.5)


def test_bad_min_confidence_raises():
    fis = ar.apriori(TRANSACTIONS, MIN_SUPPORT)
    with pytest.raises(ValueError, match="min_confidence must be in"):
        ar.association_rules(fis, 1.5)


def test_support_empty_itemset_raises():
    with pytest.raises(ValueError, match="non-empty"):
        ar.support(TRANSACTIONS, [])
