"""Tests for aiinaction.ch161_calibration, including shared cross-language fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity.
"""
from __future__ import annotations

import pytest

from aiinaction import ch161_calibration as cal

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
CONF = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3]
CORRECT = [1, 1, 0, 1, 1, 0, 1, 0, 0, 1]

# Expected values with 5 equal-width bins.
EXPECTED_ECE_5 = 0.195
EXPECTED_MCE_5 = 0.7
EXPECTED_ECE_10 = 0.285
EXPECTED_BRIER = 0.23274999999999996


def test_ece_5_bins_matches_fixture():
    assert cal.expected_calibration_error(CONF, CORRECT, 5) == pytest.approx(EXPECTED_ECE_5)


def test_mce_5_bins_matches_fixture():
    assert cal.maximum_calibration_error(CONF, CORRECT, 5) == pytest.approx(EXPECTED_MCE_5)


def test_ece_10_bins_matches_fixture():
    assert cal.expected_calibration_error(CONF, CORRECT, 10) == pytest.approx(EXPECTED_ECE_10)


def test_brier_matches_fixture():
    assert cal.brier_score(CONF, CORRECT) == pytest.approx(EXPECTED_BRIER)


def test_reliability_curve_occupied_bins():
    rc = cal.reliability_curve(CONF, CORRECT, 5)
    assert rc.n_bins == 5
    assert rc.n_samples == 10
    occ = rc.occupied
    assert [b.count for b in occ] == [1, 3, 2, 4]
    # Bin (0.2, 0.4]: single example p=0.3, correct -> acc 1.0, conf 0.3.
    assert occ[0].accuracy == pytest.approx(1.0)
    assert occ[0].confidence == pytest.approx(0.3)
    assert occ[0].gap == pytest.approx(0.7)
    # Last bin [0.8, 1.0]: four examples, three correct.
    assert occ[3].count == 4
    assert occ[3].accuracy == pytest.approx(0.75)
    assert occ[3].confidence == pytest.approx(0.875)


def test_empty_bins_present_but_ignored():
    rc = cal.reliability_curve(CONF, CORRECT, 5)
    # First equal-width bin [0.0, 0.2) is empty for this fixture.
    assert rc.bins[0].count == 0
    assert rc.bins[0].accuracy == 0.0
    assert rc.bins[0].confidence == 0.0


def test_perfectly_calibrated_is_zero():
    # Two bins, each with accuracy equal to its average confidence.
    conf = [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75]
    correct = [1, 0, 0, 0, 1, 1, 1, 0]  # bin1 acc 0.25, bin2 acc 0.75
    assert cal.expected_calibration_error(conf, correct, 2) == pytest.approx(0.0)
    assert cal.maximum_calibration_error(conf, correct, 2) == pytest.approx(0.0)


def test_confidence_one_folds_into_last_bin():
    rc = cal.reliability_curve([1.0, 1.0], [1, 0], n_bins=4)
    assert rc.bins[-1].count == 2
    assert rc.bins[-1].accuracy == pytest.approx(0.5)
    assert rc.bins[-1].confidence == pytest.approx(1.0)


@pytest.mark.parametrize(
    "fn", [cal.expected_calibration_error, cal.maximum_calibration_error]
)
def test_length_mismatch_raises(fn):
    with pytest.raises(ValueError, match="length mismatch"):
        fn([0.5, 0.5], [1], 5)


def test_brier_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        cal.brier_score([0.5, 0.5], [1])


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        cal.expected_calibration_error([], [], 5)


def test_bad_n_bins_raises():
    with pytest.raises(ValueError, match="positive integer"):
        cal.expected_calibration_error(CONF, CORRECT, 0)


def test_confidence_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cal.reliability_curve([1.5], [1], 5)


def test_correct_not_binary_raises():
    with pytest.raises(ValueError, match="0 or 1"):
        cal.reliability_curve([0.5], [2], 5)
