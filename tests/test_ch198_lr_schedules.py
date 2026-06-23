"""Tests for aiinaction.ch198_lr_schedules, including shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity to within 1e-9.
"""
from __future__ import annotations

import math

import pytest

from aiinaction import ch198_lr_schedules as lr

# --- Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src ---

# cosine_annealing(t, total_steps=8, eta_max=0.2, eta_min=0.01), t = 0..8
COSINE_T = 8
COSINE_MAX = 0.2
COSINE_MIN = 0.01
COSINE_EXPECTED = [
    0.2,
    0.19276855558857230,
    0.17217514421272200,
    0.14135492607468360,
    0.105,
    0.06864507392531650,
    0.03782485578727799,
    0.01723144441142780,
    0.01,
]

# warmup_cosine(t, warmup=3, total=10, eta_max=0.5, eta_min=0.0), t = 0..10
WC_WARMUP = 3
WC_TOTAL = 10
WC_MAX = 0.5
WC_EXPECTED = [
    0.0,
    0.16666666666666666,
    0.33333333333333330,
    0.5,
    0.47524221697560480,
    0.40587245046468340,
    0.30563023348907860,
    0.19436976651092142,
    0.09412754953531660,
    0.02475778302439520,
    0.0,
]

# one_cycle(t, total=10, eta_max=1.0, eta_min=0.04, pct_start=0.3), t = 0..10
OC_TOTAL = 10
OC_MAX = 1.0
OC_MIN = 0.04
OC_PCT = 0.3
OC_EXPECTED = [
    0.04,
    0.27999999999999990,
    0.75999999999999990,
    1.0,
    0.95246505659316120,
    0.81927510489219210,
    0.62681004829903090,
    0.41318995170096910,
    0.22072489510780793,
    0.08753494340683890,
    0.04,
]

# one_cycle_momentum(t, total=10, mom_max=0.95, mom_min=0.85, pct_start=0.3), t = 0..10
MOM_EXPECTED = [
    0.95,
    0.92499999999999990,
    0.875,
    0.85,
    0.85495155660487900,
    0.86882550990706330,
    0.88887395330218420,
    0.91112604669781570,
    0.93117449009293660,
    0.94504844339512100,
    0.95,
]

TOL = 1e-9


def test_cosine_annealing_matches_fixture():
    got = [lr.cosine_annealing(t, COSINE_T, COSINE_MAX, COSINE_MIN) for t in range(COSINE_T + 1)]
    assert got == pytest.approx(COSINE_EXPECTED, abs=TOL)


def test_warmup_cosine_matches_fixture():
    got = [lr.warmup_cosine(t, WC_WARMUP, WC_TOTAL, WC_MAX, 0.0) for t in range(WC_TOTAL + 1)]
    assert got == pytest.approx(WC_EXPECTED, abs=TOL)


def test_one_cycle_matches_fixture():
    got = [lr.one_cycle(t, OC_TOTAL, OC_MAX, OC_MIN, OC_PCT) for t in range(OC_TOTAL + 1)]
    assert got == pytest.approx(OC_EXPECTED, abs=TOL)


def test_one_cycle_momentum_matches_fixture():
    got = [lr.one_cycle_momentum(t, OC_TOTAL, 0.95, 0.85, OC_PCT) for t in range(OC_TOTAL + 1)]
    assert got == pytest.approx(MOM_EXPECTED, abs=TOL)


# --- Boundary and property tests ---


def test_cosine_endpoints():
    assert lr.cosine_annealing(0, 10, 0.1, 0.0) == pytest.approx(0.1)
    assert lr.cosine_annealing(10, 10, 0.1, 0.0) == pytest.approx(0.0)
    # midpoint is exactly the average of the bounds
    assert lr.cosine_annealing(5, 10, 0.1, 0.0) == pytest.approx(0.05)


def test_cosine_clamps_past_horizon():
    assert lr.cosine_annealing(50, 10, 0.1, 0.01) == pytest.approx(0.01)


def test_cosine_is_monotone_decreasing():
    vals = [lr.cosine_annealing(t, 20, 0.3, 0.0) for t in range(21)]
    assert all(vals[i] >= vals[i + 1] for i in range(20))


def test_linear_warmup_endpoints():
    assert lr.linear_warmup(0, 4, 0.1) == pytest.approx(0.0)
    assert lr.linear_warmup(2, 4, 0.1) == pytest.approx(0.05)
    assert lr.linear_warmup(4, 4, 0.1) == pytest.approx(0.1)
    assert lr.linear_warmup(7, 4, 0.1) == pytest.approx(0.1)  # clamps after warmup


def test_linear_warmup_nonzero_start():
    assert lr.linear_warmup(0, 4, 0.1, eta_start=0.02) == pytest.approx(0.02)
    assert lr.linear_warmup(2, 4, 0.1, eta_start=0.02) == pytest.approx(0.06)


def test_warmup_cosine_peak_at_warmup_boundary():
    assert lr.warmup_cosine(WC_WARMUP, WC_WARMUP, WC_TOTAL, WC_MAX) == pytest.approx(WC_MAX)


def test_warmup_cosine_continuous_at_handoff():
    # The two legs must agree at the warmup boundary (continuity).
    left = lr.linear_warmup(WC_WARMUP, WC_WARMUP, WC_MAX)
    right = lr.warmup_cosine(WC_WARMUP, WC_WARMUP, WC_TOTAL, WC_MAX)
    assert left == pytest.approx(right)


def test_one_cycle_peak_at_pct_start():
    # t1 = 0.3 * 10 = 3, an integer step, so the peak is hit exactly.
    assert lr.one_cycle(3, 10, 1.0, 0.04, 0.3) == pytest.approx(1.0)


def test_one_cycle_default_eta_min():
    # Default eta_min = eta_max / 25.
    assert lr.one_cycle(0, 10, 1.0) == pytest.approx(0.04)
    assert lr.one_cycle(0, 10, 0.5) == pytest.approx(0.02)


def test_one_cycle_momentum_is_antiphase():
    # Momentum is at its minimum exactly when the LR is at its maximum (t = t1).
    assert lr.one_cycle_momentum(3, 10, 0.95, 0.85, 0.3) == pytest.approx(0.85)
    assert lr.one_cycle_momentum(0, 10, 0.95, 0.85, 0.3) == pytest.approx(0.95)
    assert lr.one_cycle_momentum(10, 10, 0.95, 0.85, 0.3) == pytest.approx(0.95)


def test_schedule_curve_lengths_and_values():
    curve = lr.schedule_curve("cosine", COSINE_T, eta_max=COSINE_MAX, eta_min=COSINE_MIN)
    assert len(curve) == COSINE_T
    # curve covers t = 0..T-1 (not including the final clamp point)
    assert curve == pytest.approx(COSINE_EXPECTED[:-1], abs=TOL)


def test_schedule_curve_one_cycle():
    curve = lr.schedule_curve("one_cycle", OC_TOTAL, eta_max=OC_MAX, eta_min=OC_MIN, pct_start=OC_PCT)
    assert len(curve) == OC_TOTAL
    assert curve == pytest.approx(OC_EXPECTED[:-1], abs=TOL)


# --- Validation / edge cases ---


def test_negative_step_raises():
    with pytest.raises(ValueError, match="t must be >= 0"):
        lr.cosine_annealing(-1, 10, 0.1, 0.0)


def test_zero_horizon_raises():
    with pytest.raises(ValueError, match="must be >= 1"):
        lr.cosine_annealing(0, 0, 0.1, 0.0)


def test_eta_max_below_min_raises():
    with pytest.raises(ValueError, match="must be >= eta_min"):
        lr.cosine_annealing(0, 10, 0.01, 0.1)


def test_warmup_not_less_than_total_raises():
    with pytest.raises(ValueError, match="0 <= warmup_steps < total_steps"):
        lr.warmup_cosine(0, 10, 10, 0.1)


def test_one_cycle_bad_pct_start_raises():
    with pytest.raises(ValueError, match=r"pct_start must be in \(0, 1\)"):
        lr.one_cycle(0, 10, 1.0, 0.04, pct_start=1.0)


def test_momentum_bad_bounds_raise():
    with pytest.raises(ValueError, match="momenta must satisfy"):
        lr.one_cycle_momentum(0, 10, mom_max=0.85, mom_min=0.95)
    with pytest.raises(ValueError, match="momenta must satisfy"):
        lr.one_cycle_momentum(0, 10, mom_max=1.0, mom_min=0.85)


def test_schedule_curve_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown schedule name"):
        lr.schedule_curve("triangular", 10, eta_max=0.1)


def test_robbins_monro_intuition_cosine_floor_reachable():
    # Sanity: a zero floor schedule actually reaches (near) zero at the horizon.
    assert lr.cosine_annealing(10, 10, 0.1, 0.0) == pytest.approx(0.0, abs=1e-15)
    # And the cosine curve never exceeds eta_max nor drops below eta_min.
    for t in range(11):
        v = lr.cosine_annealing(t, 10, 0.1, 0.0)
        assert -1e-12 <= v <= 0.1 + 1e-12
        assert not math.isnan(v)
