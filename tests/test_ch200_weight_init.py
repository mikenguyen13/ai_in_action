"""Tests for aiinaction.ch200_weight_init (Xavier / He weight initialization).

The numeric fixtures here are the single source of truth: the Julia and Rust test
suites assert against the same numbers, which is what keeps the three libraries at
parity. The seeded weight matrices are produced by a shared SplitMix64 + Box-Muller
pipeline, so the actual sampled entries (not just summary statistics) must match
across all three languages to 1e-9.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aiinaction import ch200_weight_init as wi

# ---------------------------------------------------------------------------
# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
# ---------------------------------------------------------------------------

GAIN_RELU = 1.4142135623730951
GAIN_TANH = 1.6666666666666667
GAIN_LEAKY_02 = 1.3867504905630728  # leaky_relu with slope 0.2

# xavier_scale(4, 6)
XAVIER_46_STD = 0.4472135954999579
XAVIER_46_BOUND = 0.7745966692414833

# he_scale(8)  (default ReLU gain)
HE_8_STD = 0.5
HE_8_BOUND = 0.8660254037844386

# xavier_normal(fan_in=3, fan_out=2, seed=42) -> shape (2, 3)
XAVIER_NORMAL_3_2_SEED42 = [
    [0.262291800404047, -0.5640783697534434, 1.0938907166332181],
    [0.34508066325448, -0.683313150259559, -1.1250423158375467],
]

# he_uniform(fan_in=3, fan_out=2, seed=7) -> shape (2, 3)
HE_UNIFORM_3_2_SEED7 = [
    [-0.3116085279902403, -1.3667290947514303, 1.1335223795602536],
    [0.23456229026376593, -0.13451463415109, -0.7087146789818501],
]

# he_normal(fan_in=4, fan_out=3, seed=123, mode='fan_in') -> shape (3, 4)
HE_NORMAL_4_3_SEED123 = [
    [0.5830829313806632, -0.1503944856454256, -0.30548437167556053, -0.00772328473476067],
    [0.43836777367788293, 0.45770047377992695, 0.6982243250890897, -0.151260320705838],
    [-0.16215580034083707, -1.1099931389726774, 0.23040596666525975, -0.4700322147558535],
]

TOL = 1e-9


# ---------------------------------------------------------------------------
# Gain factors.
# ---------------------------------------------------------------------------

def test_gain_relu():
    assert wi.calculate_gain("relu") == pytest.approx(GAIN_RELU)


def test_gain_tanh():
    assert wi.calculate_gain("tanh") == pytest.approx(GAIN_TANH)


def test_gain_linear_and_sigmoid():
    assert wi.calculate_gain("linear") == 1.0
    assert wi.calculate_gain("sigmoid") == 1.0


def test_gain_leaky_relu_default_and_param():
    assert wi.calculate_gain("leaky_relu") == pytest.approx(math.sqrt(2.0 / 1.0001))
    assert wi.calculate_gain("leaky_relu", 0.2) == pytest.approx(GAIN_LEAKY_02)


def test_gain_leaky_alpha_one_is_linear():
    # alpha = 1 makes leaky ReLU the identity, recovering gain 1.
    assert wi.calculate_gain("leaky_relu", 1.0) == pytest.approx(1.0)


def test_gain_unsupported_raises():
    with pytest.raises(ValueError, match="unsupported nonlinearity"):
        wi.calculate_gain("gelu_bananas")


def test_gain_leaky_bad_slope_raises():
    with pytest.raises(ValueError, match="negative slope"):
        wi.calculate_gain("leaky_relu", -1.0)


# ---------------------------------------------------------------------------
# Theoretical scales.
# ---------------------------------------------------------------------------

def test_xavier_scale_matches_fixture():
    s = wi.xavier_scale(4, 6)
    assert s.std == pytest.approx(XAVIER_46_STD, abs=TOL)
    assert s.bound == pytest.approx(XAVIER_46_BOUND, abs=TOL)


def test_xavier_scale_bound_is_std_times_sqrt3():
    s = wi.xavier_scale(7, 11, gain=1.3)
    assert s.bound == pytest.approx(s.std * math.sqrt(3.0), abs=TOL)


def test_xavier_uniform_variance_matches_normal():
    # U(-r, r) has variance r^2/3, which must equal std^2 by construction.
    s = wi.xavier_scale(10, 20)
    assert s.bound**2 / 3.0 == pytest.approx(s.std**2, abs=TOL)


def test_he_scale_matches_fixture():
    s = wi.he_scale(8)
    assert s.std == pytest.approx(HE_8_STD, abs=TOL)
    assert s.bound == pytest.approx(HE_8_BOUND, abs=TOL)


def test_he_scale_relu_gain_default():
    # he_scale(2) with default ReLU gain sqrt(2) gives std exactly 1.
    assert wi.he_scale(2).std == pytest.approx(1.0, abs=TOL)


# ---------------------------------------------------------------------------
# Seeded sampling: exact entry-by-entry parity fixtures.
# ---------------------------------------------------------------------------

def test_xavier_normal_seed_fixture():
    w = wi.xavier_normal(3, 2, seed=42)
    assert w.shape == (2, 3)
    np.testing.assert_allclose(w, XAVIER_NORMAL_3_2_SEED42, atol=TOL)


def test_he_uniform_seed_fixture():
    w = wi.he_uniform(3, 2, seed=7)
    assert w.shape == (2, 3)
    np.testing.assert_allclose(w, HE_UNIFORM_3_2_SEED7, atol=TOL)


def test_he_normal_seed_fixture():
    w = wi.he_normal(4, 3, seed=123, mode="fan_in")
    assert w.shape == (3, 4)
    np.testing.assert_allclose(w, HE_NORMAL_4_3_SEED123, atol=TOL)


def test_same_seed_is_reproducible():
    a = wi.he_normal(5, 5, seed=99)
    b = wi.he_normal(5, 5, seed=99)
    np.testing.assert_array_equal(a, b)


def test_different_seed_differs():
    a = wi.he_normal(5, 5, seed=1)
    b = wi.he_normal(5, 5, seed=2)
    assert not np.allclose(a, b)


def test_he_uniform_respects_bound():
    bound = wi.he_scale(16).bound
    w = wi.he_uniform(16, 16, seed=5)
    assert np.all(np.abs(w) <= bound + TOL)


def test_he_normal_empirical_variance_is_in_ballpark():
    # Large draw: sample variance should approach the theoretical std^2.
    fan = 64
    w = wi.he_normal(fan, fan, seed=2024)
    target = wi.he_scale(fan).std ** 2
    assert w.var() == pytest.approx(target, rel=0.15)


# ---------------------------------------------------------------------------
# Validation / edge cases.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [wi.xavier_normal, wi.xavier_uniform])
def test_xavier_bad_fan_raises(fn):
    with pytest.raises(ValueError, match="positive integer"):
        fn(0, 4)
    with pytest.raises(ValueError, match="positive integer"):
        fn(4, -1)


@pytest.mark.parametrize("fn", [wi.he_normal, wi.he_uniform])
def test_he_bad_mode_raises(fn):
    with pytest.raises(ValueError, match="mode must be"):
        fn(4, 4, mode="fan_sideways")


@pytest.mark.parametrize("fn", [wi.xavier_normal, wi.he_normal])
def test_nonpositive_gain_raises(fn):
    with pytest.raises(ValueError, match="gain must be positive"):
        fn(4, 4, gain=0.0)


def test_he_scale_bad_fan_raises():
    with pytest.raises(ValueError, match="positive integer"):
        wi.he_scale(0)
