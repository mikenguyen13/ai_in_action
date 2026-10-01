"""Tests for aiinaction.ch525_oversight, including shared cross-language fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which keeps the three implementations at parity.
"""
from __future__ import annotations

import math

import pytest

from aiinaction import ch525_oversight as o

# ---------------------------------------------------------------------------
# Shared fixtures (mirrored in julia/AIInAction/test and rust/aiinaction/src)
# ---------------------------------------------------------------------------
SDT_COUNTS = (18, 2, 5, 75)  # hits, misses, false alarms, correct rejections
SDT_LOGLINEAR_DPRIME = 2.6713670993913583
SDT_LOGLINEAR_CRITERION = 0.15592243208381895
SDT_RAW_DPRIME = 2.8156721098971467
SDT_RAW_CRITERION = 0.12628448940397252
SDT_PERFECT_DPRIME = 4.482858869990343  # (20, 0, 0, 80) with log-linear correction
SDT_PERFECT_CRITERION = 0.26067703834789313

WILSON_18_OF_20 = (0.6989663547715128, 0.9721335187862319)
ZERO_FAILURE_30 = 0.09503385285530419
CANARIES_09_07 = 20
CANARIES_095_085 = 44
CANARIES_09_08_POWER09 = 102

SLOPE_X = [1.0, 2.0, 3.0, 4.0, 5.0]
SLOPE_Y = [2.0, 4.1, 5.9, 8.2, 9.8]
SLOPE = 1.97

ROLLING_FLAGS = [0, 1, 0, 0, 1, 1, 0, 0]
ROLLING_3 = [1 / 3, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1 / 3]

CUSUM_VALUES = [5.0, 5.1, 4.9, 4.2, 4.0, 3.9, 4.1, 3.8]
CUSUM_PATH = [0.0, 0.0, 0.0, 0.55, 1.3, 2.15, 2.8, 3.75]
CUSUM_ALARM = 5  # 0-based (Julia: 6)

SKILL_AT_3 = 0.34869187738724783  # t=3, s0=0.8, a=0.9, q=0, eta=2, lambda=0.5
MIN_PRACTICE = 0.275  # target 0.6, a=0.9, eta=2, lambda=0.5
STEADY_A09_Q015 = 0.5

SCHEDULE_TTFT = [0.0, 0.6438561897747247, 1.931568569324174, 2.575424759098899, 3.863137138648348]

MAJORITY_3_085 = 0.93925
MAJORITY_5_07 = 0.83692
VIGILANCE_30 = 0.666939048044529  # t=30, d0=0.9, floor=0.6, tau=20
DETECT_REVIEW_2 = 0.6995827187900596  # r=2, d_max=0.95, r0=1.5


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


def test_probit_known_values():
    assert o.probit(0.975) == pytest.approx(1.959963984540054, abs=1e-12)
    assert o.probit(0.5) == pytest.approx(0.0, abs=1e-15)
    assert o.probit(0.02) == pytest.approx(-2.053748910631822, abs=1e-12)
    with pytest.raises(ValueError):
        o.probit(0.0)
    with pytest.raises(ValueError):
        o.probit(1.0)


def test_sdt_loglinear_fixture():
    r = o.sdt_measures(*SDT_COUNTS)
    assert r.d_prime == pytest.approx(SDT_LOGLINEAR_DPRIME, abs=1e-12)
    assert r.criterion == pytest.approx(SDT_LOGLINEAR_CRITERION, abs=1e-12)
    assert r.hit_rate == pytest.approx(18.5 / 21)
    assert r.false_alarm_rate == pytest.approx(5.5 / 81)


def test_sdt_raw_fixture():
    r = o.sdt_measures(*SDT_COUNTS, correction="none")
    assert r.d_prime == pytest.approx(SDT_RAW_DPRIME, abs=1e-12)
    assert r.criterion == pytest.approx(SDT_RAW_CRITERION, abs=1e-12)


def test_sdt_perfect_reviewer_stays_finite():
    r = o.sdt_measures(20, 0, 0, 80)
    assert math.isfinite(r.d_prime)
    assert r.d_prime == pytest.approx(SDT_PERFECT_DPRIME, abs=1e-12)
    assert r.criterion == pytest.approx(SDT_PERFECT_CRITERION, abs=1e-12)
    with pytest.raises(ValueError):
        o.sdt_measures(20, 0, 0, 80, correction="none")


def test_sdt_rubber_stamp_raises_criterion():
    careful = o.sdt_measures(18, 2, 8, 72)
    stamping = o.sdt_measures(10, 10, 1, 79)
    assert stamping.criterion > careful.criterion


def test_sdt_validation():
    with pytest.raises(ValueError):
        o.sdt_measures(0, 0, 1, 1)
    with pytest.raises(ValueError):
        o.sdt_measures(1, -1, 1, 1)
    with pytest.raises(ValueError):
        o.sdt_measures(1, 1, 1, 1, correction="bogus")


# ---------------------------------------------------------------------------
# Canaries
# ---------------------------------------------------------------------------


def test_wilson_fixture():
    lo, hi = o.wilson_interval(18, 20)
    assert lo == pytest.approx(WILSON_18_OF_20[0], abs=1e-12)
    assert hi == pytest.approx(WILSON_18_OF_20[1], abs=1e-12)


def test_wilson_edges_and_validation():
    lo, hi = o.wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.35
    lo, hi = o.wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and lo > 0.65
    with pytest.raises(ValueError):
        o.wilson_interval(3, 2)
    with pytest.raises(ValueError):
        o.wilson_interval(0, 0)


def test_zero_failure_bound_matches_rule_of_three():
    assert o.zero_failure_upper_bound(30) == pytest.approx(ZERO_FAILURE_30, abs=1e-12)
    for n in (50, 100, 1000):
        assert o.zero_failure_upper_bound(n) == pytest.approx(3 / n, rel=0.05)


def test_canary_sample_sizes():
    assert o.canaries_to_detect_decline(0.9, 0.7) == CANARIES_09_07
    assert o.canaries_to_detect_decline(0.95, 0.85) == CANARIES_095_085
    assert o.canaries_to_detect_decline(0.9, 0.8, 0.05, 0.9) == CANARIES_09_08_POWER09
    with pytest.raises(ValueError):
        o.canaries_to_detect_decline(0.7, 0.9)


# ---------------------------------------------------------------------------
# Behavioral signatures
# ---------------------------------------------------------------------------


def test_ols_slope_fixture():
    assert o.ols_slope(SLOPE_X, SLOPE_Y) == pytest.approx(SLOPE, abs=1e-12)
    with pytest.raises(ValueError):
        o.ols_slope([1.0, 1.0], [2.0, 3.0])
    with pytest.raises(ValueError):
        o.ols_slope([1.0], [2.0])


def test_rolling_rate_fixture():
    assert o.rolling_rate(ROLLING_FLAGS, 3) == pytest.approx(ROLLING_3)
    with pytest.raises(ValueError):
        o.rolling_rate(ROLLING_FLAGS, 0)
    with pytest.raises(ValueError):
        o.rolling_rate(ROLLING_FLAGS, 9)


def test_cusum_fixture():
    res = o.cusum_lower(CUSUM_VALUES, 5.0, 0.25, 1.5)
    assert res.statistic == pytest.approx(CUSUM_PATH, abs=1e-12)
    assert res.alarm_index == CUSUM_ALARM


def test_cusum_no_alarm_on_stable_series():
    res = o.cusum_lower([5.0, 5.2, 4.9, 5.1, 4.95], 5.0, 0.25, 1.5)
    assert res.alarm_index is None
    assert max(res.statistic) == 0.0


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


def test_gate_routes():
    assert o.gate_action(0.005, 100, 1, 0.9).route is o.Route.AUTO
    assert o.gate_action(0.02, 100, 1, 0.9).route is o.Route.REVIEW
    assert o.gate_action(0.2, 100, 1, 0.9, 5.0).route is o.Route.REVIEW
    assert o.gate_action(0.2, 100, 1, 0.2, 5.0).route is o.Route.BLOCK


def test_gate_losses_and_threshold():
    d = o.gate_action(0.02, 100, 1, 0.9)
    assert d.loss_auto == pytest.approx(2.0)
    assert d.loss_review == pytest.approx(1.2)
    assert d.threshold == pytest.approx(1 / 90)
    assert o.review_threshold(100, 1, 0.0) == math.inf


def test_gate_threshold_is_the_indifference_point():
    harm, cost, det = 250.0, 3.0, 0.8
    p_star = o.review_threshold(harm, cost, det)
    d = o.gate_action(p_star, harm, cost, det)
    assert d.loss_auto == pytest.approx(d.loss_review)


def test_vigilance_and_review_time_fixtures():
    assert o.vigilance(30, 0.9, 0.6, 20) == pytest.approx(VIGILANCE_30, abs=1e-12)
    assert o.vigilance(0, 0.9, 0.6, 20) == pytest.approx(0.9)
    assert o.detection_from_review_time(2, 0.95, 1.5) == pytest.approx(DETECT_REVIEW_2, abs=1e-12)


# ---------------------------------------------------------------------------
# Skill model
# ---------------------------------------------------------------------------


def test_skill_fixtures():
    assert o.skill_at(3.0, 0.8, 0.9, 0.0, 2.0, 0.5) == pytest.approx(SKILL_AT_3, abs=1e-12)
    assert o.min_practice_rate(0.6, 0.9, 2.0, 0.5) == pytest.approx(MIN_PRACTICE, abs=1e-12)
    assert o.steady_state_skill(0.9, 0.15, 2.0, 0.5) == pytest.approx(STEADY_A09_Q015, abs=1e-12)


def test_skill_converges_to_steady_state():
    s_star = o.steady_state_skill(0.7, 0.05, 1.5, 0.3)
    assert o.skill_at(200.0, 0.1, 0.7, 0.05, 1.5, 0.3) == pytest.approx(s_star, abs=1e-10)


def test_min_practice_rate_achieves_target():
    q = o.min_practice_rate(0.75, 0.95, 2.0, 0.5)
    assert o.steady_state_skill(0.95, q, 2.0, 0.5) == pytest.approx(0.75)
    assert o.min_practice_rate(0.1, 0.0, 2.0, 0.5) == 0.0


def test_full_automation_without_practice_erodes_skill_to_zero():
    assert o.steady_state_skill(1.0, 0.0, 2.0, 0.5) == 0.0


def test_detection_from_skill():
    assert o.detection_from_skill(0.5, 0.3, 0.95) == pytest.approx(0.625)
    with pytest.raises(ValueError):
        o.detection_from_skill(0.5, 0.9, 0.3)


# ---------------------------------------------------------------------------
# Spaced practice
# ---------------------------------------------------------------------------


def test_recall_and_interval():
    assert o.recall_probability(2.0, 2.0) == pytest.approx(0.5)
    h, r = 4.0, 0.8
    assert o.recall_probability(o.next_interval(h, r), h) == pytest.approx(r)


def test_schedule_fixture():
    assert o.practice_schedule([True, True, False, True]) == pytest.approx(SCHEDULE_TTFT, abs=1e-12)


def test_update_half_life_floor():
    assert o.update_half_life(0.6, False) == 0.5
    with pytest.raises(ValueError):
        o.update_half_life(1.0, True, growth=0.5)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def test_majority_vote_fixtures():
    assert o.majority_vote_accuracy(3, 0.85) == pytest.approx(MAJORITY_3_085, abs=1e-12)
    assert o.majority_vote_accuracy(5, 0.7) == pytest.approx(MAJORITY_5_07, abs=1e-12)
    # With ties split by a coin, an even panel is no better than the odd panel below it.
    assert o.majority_vote_accuracy(4, 0.85) == pytest.approx(o.majority_vote_accuracy(3, 0.85))
    assert o.majority_vote_accuracy(1, 0.85) == pytest.approx(0.85)


def test_layered_residual_error():
    assert o.layered_residual_error(0.1, [0.7, 0.6, 0.5]) == pytest.approx(0.006)
    assert o.layered_residual_error(0.1, []) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _action(i, p=0.05, harm=50.0, complexity=1.0, defective=None, canary=False):
    return o.AgentAction(f"a{i}", "demo", p, harm, complexity, defective, canary)


def test_gate_detection_estimate_tracks_canaries():
    gate = o.OversightGate(review_cost=1.0, prior_detection=0.9, prior_strength=10.0, seed=0)
    assert gate.estimated_detection() == pytest.approx(0.9)
    for i in range(20):
        a = _action(i, defective=True, canary=True)
        gate.log_review(a, o.Route.REVIEW, approved=True, review_seconds=5.0)  # misses every canary
    assert gate.estimated_detection() == pytest.approx(9.0 / 30.0)
    rep = gate.report()
    assert rep.canary_catch_rate == 0.0
    assert any("rotate" in a for a in rep.alerts)


def test_gate_canary_always_reviewed():
    gate = o.OversightGate(review_cost=100.0, seed=1)
    assert gate.route(_action(0, p=0.0, harm=0.0, canary=True, defective=True)).route is o.Route.REVIEW


def test_gate_report_signatures():
    gate = o.OversightGate(review_cost=1.0, baseline_log_seconds=math.log(60.0), seed=2)
    # Early: careful, time scales with complexity. Late: fast and flat.
    for i in range(10):
        c = 1.0 + (i % 3)
        gate.log_review(_action(i, complexity=c, defective=(i % 5 == 0)), o.Route.REVIEW,
                        approved=(i % 5 != 0), review_seconds=40.0 * c)
    for i in range(10, 40):
        c = 1.0 + (i % 3)
        gate.log_review(_action(i, complexity=c, defective=(i % 5 == 0)), o.Route.REVIEW,
                        approved=True, review_seconds=8.0)
    rep = gate.report()
    assert rep.n_reviews == 40
    assert rep.cusum_alarm_index is not None
    assert rep.sdt is not None and rep.sdt.criterion > 0.5
    assert any("break" in a for a in rep.alerts)


def test_should_plant_canary_is_seeded():
    g1 = o.OversightGate(review_cost=1.0, canary_rate=0.3, seed=42)
    g2 = o.OversightGate(review_cost=1.0, canary_rate=0.3, seed=42)
    assert [g1.should_plant_canary() for _ in range(50)] == [g2.should_plant_canary() for _ in range(50)]
