"""Tests for aiinaction.ch204_dropout, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. Because dropout is random, parity hinges on a shared, fully specified
pseudo-random generator (a 64-bit LCG with the Numerical Recipes constants); the
same seed therefore drops the same units in every language.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch204_dropout as dp

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
# First six uniform draws of Lcg(42), to full double precision.
EXPECTED_UNIFORMS = [
    0.5682303266439076,
    0.2254634289477513,
    0.41283831882951183,
    0.6303980498395979,
    0.6801478072421157,
    0.02622891069993838,
]

# bernoulli_mask(8, 0.5, seed=42): u < 0.5 retains (value 1/0.5 = 2.0), else 0.0.
EXPECTED_MASK_8 = [0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0]

# inverted_dropout([1..8], 0.5, seed=42) == EXPECTED_MASK_8 elementwise-times h.
EXPECTED_H = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
EXPECTED_OUT_8 = [0.0, 4.0, 6.0, 0.0, 0.0, 12.0, 14.0, 16.0]


def test_lcg_uniform_sequence_matches_fixture():
    rng = dp.Lcg(42)
    got = [rng.next_uniform() for _ in range(6)]
    assert got == pytest.approx(EXPECTED_UNIFORMS, abs=1e-12)


def test_lcg_uniforms_in_unit_interval():
    rng = dp.Lcg(123)
    for _ in range(1000):
        u = rng.next_uniform()
        assert 0.0 <= u < 1.0


def test_lcg_is_deterministic_for_same_seed():
    a = [dp.Lcg(7).next_uniform() for _ in range(5)]
    b = [dp.Lcg(7).next_uniform() for _ in range(5)]
    assert a == b


def test_expected_scale_is_reciprocal():
    assert dp.expected_scale(0.5) == pytest.approx(2.0)
    assert dp.expected_scale(0.8) == pytest.approx(1.25)
    assert dp.expected_scale(1.0) == pytest.approx(1.0)


def test_expected_scale_rejects_out_of_range():
    with pytest.raises(ValueError, match="0 < p <= 1"):
        dp.expected_scale(0.0)
    with pytest.raises(ValueError, match="0 < p <= 1"):
        dp.expected_scale(1.5)
    with pytest.raises(ValueError, match="0 < p <= 1"):
        dp.expected_scale(-0.2)


def test_bernoulli_mask_matches_fixture():
    m = dp.bernoulli_mask(8, 0.5, seed=42)
    assert m.tolist() == pytest.approx(EXPECTED_MASK_8)


def test_bernoulli_mask_p_one_is_all_ones():
    m = dp.bernoulli_mask(5, 1.0, seed=7)
    assert m.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])


def test_bernoulli_mask_entries_are_zero_or_inverse_p():
    p = 0.3
    m = dp.bernoulli_mask(50, p, seed=99)
    inv = 1.0 / p
    for v in m.tolist():
        assert v == pytest.approx(0.0) or v == pytest.approx(inv)


def test_bernoulli_mask_mean_approaches_one_in_expectation():
    # Over a long mask, the average survivor scaling pulls the mean toward 1.
    m = dp.bernoulli_mask(20000, 0.5, seed=2024)
    assert float(m.mean()) == pytest.approx(1.0, abs=0.05)


def test_inverted_dropout_matches_fixture():
    out, mask = dp.inverted_dropout(EXPECTED_H, 0.5, seed=42)
    assert out.tolist() == pytest.approx(EXPECTED_OUT_8)
    assert mask.tolist() == pytest.approx(EXPECTED_MASK_8)


def test_inverted_dropout_p_one_is_identity():
    out, mask = dp.inverted_dropout(EXPECTED_H, 1.0, seed=42)
    assert out.tolist() == pytest.approx(EXPECTED_H)
    assert mask.tolist() == pytest.approx([1.0] * len(EXPECTED_H))


def test_inverted_dropout_preserves_expectation_over_seeds():
    # Averaging masked outputs over many seeds recovers h (the inverted-dropout
    # guarantee E[mask * h] = h), since seeds give independent mask realizations.
    h = [3.0, -1.0, 4.0, 2.0]
    acc = np.zeros(len(h))
    trials = 6000
    for seed in range(trials):
        out, _ = dp.inverted_dropout(h, 0.5, seed=seed)
        acc += out
    avg = acc / trials
    assert avg.tolist() == pytest.approx(h, abs=0.15)


def test_dropped_units_are_exactly_zero():
    out, mask = dp.inverted_dropout(EXPECTED_H, 0.5, seed=42)
    for o, m in zip(out.tolist(), mask.tolist()):
        if m == 0.0:
            assert o == 0.0


def test_lcg_rejects_negative_seed():
    with pytest.raises(ValueError, match="non-negative"):
        dp.Lcg(-1)


def test_bernoulli_mask_rejects_nonpositive_n():
    with pytest.raises(ValueError, match="n must be >= 1"):
        dp.bernoulli_mask(0, 0.5, seed=1)


def test_inverted_dropout_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        dp.inverted_dropout([], 0.5, seed=1)


def test_inverted_dropout_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        dp.inverted_dropout([1.0, float("nan")], 0.5, seed=1)


def test_inverted_dropout_rejects_2d():
    with pytest.raises(ValueError, match="1-D vector"):
        dp.inverted_dropout([[1.0, 2.0]], 0.5, seed=1)


def test_inverted_dropout_rejects_bad_p():
    with pytest.raises(ValueError, match="0 < p <= 1"):
        dp.inverted_dropout([1.0, 2.0], 0.0, seed=1)
