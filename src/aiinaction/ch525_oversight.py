"""Human-in-the-loop oversight: measuring, gating, and maintaining the overseer.

Reference implementations for Chapter 525 ("Human in the Loop"). The chapter
argues, following Mitchell, Ghosh, and Passi (2026), that agent deployments
quietly degrade the human oversight they rely on. This module turns the
remedies into code that can run next to a real agent:

* **Measuring the overseer.** Signal detection theory (``sdt_measures``) treats
  a defective agent action as the "signal". Sensitivity ``d'`` says how well a
  reviewer separates good actions from bad ones; the criterion ``c`` says how
  willing they are to approve. Rubber-stamping shows up as a rising ``c``,
  fatigue as a falling ``d'``.
* **Canaries.** Known-defective actions planted in the review stream give an
  unbiased estimate of the catch rate (``wilson_interval``,
  ``zero_failure_upper_bound``, ``canaries_to_detect_decline``).
* **Behavioral signatures.** Review time that stops scaling with complexity
  (``ols_slope``), a falling override rate (``rolling_rate``), and a downward
  shift in review time (``cusum_lower``).
* **Approval design.** ``gate_action`` routes an action to automatic
  execution, human review, or a block, by minimizing expected loss subject to
  a cap on residual harm. Review is worth its cost only when
  ``p_error * detection * harm > review_cost``, so the gate is only as good
  as the detection probability it is given.
* **Skill maintenance.** A first-order skill model (``skill_at``,
  ``steady_state_skill``, ``min_practice_rate``) formalizes Bainbridge's irony
  of automation: practice comes from doing the work, automation removes the
  work, so skill decays unless deliberate practice replaces it. A half-life
  scheduler (``practice_schedule``) spaces the practice sessions.
* **Independent panels.** ``majority_vote_accuracy`` and
  ``layered_residual_error`` quantify what independent, multi-layer human
  review buys.

All numeric functions mirror the Julia module ``AIInAction.Ch525Oversight``
and the Rust module ``aiinaction::ch525_oversight`` one-to-one, and the three
test suites share fixtures. ``OversightGate`` (a stateful harness) is Python
only.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist

__all__ = [
    "probit",
    "SDTResult",
    "sdt_measures",
    "wilson_interval",
    "zero_failure_upper_bound",
    "canaries_to_detect_decline",
    "ols_slope",
    "rolling_rate",
    "CusumResult",
    "cusum_lower",
    "Route",
    "GateDecision",
    "gate_action",
    "review_threshold",
    "detection_from_review_time",
    "vigilance",
    "steady_state_skill",
    "skill_at",
    "min_practice_rate",
    "detection_from_skill",
    "recall_probability",
    "next_interval",
    "update_half_life",
    "practice_schedule",
    "majority_vote_accuracy",
    "layered_residual_error",
    "AgentAction",
    "ReviewRecord",
    "OversightReport",
    "OversightGate",
]

_STD_NORMAL = NormalDist()
Z_95_TWO_SIDED = 1.959963984540054


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _check_prob(name: str, value: float, *, open_low: bool = False, open_high: bool = False) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    low_ok = value > 0.0 if open_low else value >= 0.0
    high_ok = value < 1.0 if open_high else value <= 1.0
    if not (low_ok and high_ok):
        lo = "(0" if open_low else "[0"
        hi = "1)" if open_high else "1]"
        raise ValueError(f"{name} must lie in {lo}, {hi}, got {value}")


def _check_positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be a positive finite number, got {value}")


def _check_nonneg(name: str, value: float) -> None:
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be a non-negative finite number, got {value}")


def _check_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


# --------------------------------------------------------------------------
# Signal detection theory for overseers
# --------------------------------------------------------------------------


def probit(p: float) -> float:
    """Inverse standard normal CDF, ``z(p) = Phi^{-1}(p)`` (Wichura AS241)."""
    _check_prob("p", p, open_low=True, open_high=True)
    return _STD_NORMAL.inv_cdf(p)


@dataclass(frozen=True)
class SDTResult:
    """Equal-variance Gaussian signal detection summary of one reviewer.

    The *signal* is a defective agent action. A hit is a rejected defective
    action, a false alarm is a rejected correct action.

    ``criterion`` is ``c = -(z(H) + z(F)) / 2``. Positive ``c`` means the
    reviewer needs strong evidence before rejecting, i.e. leans toward
    approving; a rising ``c`` over a shift is the signature of rubber-stamping.
    """

    hit_rate: float
    false_alarm_rate: float
    d_prime: float
    criterion: float


def sdt_measures(
    hits: int,
    misses: int,
    false_alarms: int,
    correct_rejections: int,
    correction: str = "loglinear",
) -> SDTResult:
    """Sensitivity ``d'`` and criterion ``c`` from a reviewer's confusion counts.

    ``correction="loglinear"`` (Hautus, 1995) adds 0.5 to every cell, which
    keeps ``z`` finite when a reviewer catches every canary or never rejects a
    correct action. ``correction="none"`` uses raw rates and raises if either
    rate is 0 or 1.
    """
    for name, v in (
        ("hits", hits),
        ("misses", misses),
        ("false_alarms", false_alarms),
        ("correct_rejections", correct_rejections),
    ):
        _check_count(name, v)
    n_signal = hits + misses
    n_noise = false_alarms + correct_rejections
    if n_signal == 0 or n_noise == 0:
        raise ValueError("need at least one defective and one correct action")
    if correction == "loglinear":
        h = (hits + 0.5) / (n_signal + 1.0)
        f = (false_alarms + 0.5) / (n_noise + 1.0)
    elif correction == "none":
        h = hits / n_signal
        f = false_alarms / n_noise
        if h in (0.0, 1.0) or f in (0.0, 1.0):
            raise ValueError("raw hit or false-alarm rate is 0 or 1; use correction='loglinear'")
    else:
        raise ValueError(f"unknown correction {correction!r}")
    zh, zf = probit(h), probit(f)
    return SDTResult(hit_rate=h, false_alarm_rate=f, d_prime=zh - zf, criterion=-(zh + zf) / 2.0)


# --------------------------------------------------------------------------
# Canary statistics
# --------------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, z: float = Z_95_TWO_SIDED) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion such as a canary catch rate."""
    _check_count("successes", successes)
    _check_count("trials", trials)
    if trials == 0:
        raise ValueError("trials must be positive")
    if successes > trials:
        raise ValueError("successes cannot exceed trials")
    _check_positive("z", z)
    n = float(trials)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def zero_failure_upper_bound(trials: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper bound on the miss rate after ``trials`` caught canaries.

    Solves ``(1 - p)^n = alpha`` for ``p``. For ``alpha = 0.05`` this is close
    to the "rule of three", ``3 / n``.
    """
    _check_count("trials", trials)
    if trials == 0:
        raise ValueError("trials must be positive")
    _check_prob("alpha", alpha, open_low=True, open_high=True)
    return 1.0 - alpha ** (1.0 / trials)


def canaries_to_detect_decline(p0: float, p1: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Canaries needed for a one-sided test to detect a catch-rate drop from ``p0`` to ``p1``.

    Normal-approximation sample size
    ``n = ((z_{1-alpha} sqrt(p0 q0) + z_{power} sqrt(p1 q1)) / (p0 - p1))^2``.
    """
    _check_prob("p0", p0, open_low=True, open_high=True)
    _check_prob("p1", p1, open_low=True, open_high=True)
    _check_prob("alpha", alpha, open_low=True, open_high=True)
    _check_prob("power", power, open_low=True, open_high=True)
    if not p1 < p0:
        raise ValueError("p1 must be smaller than p0 (we detect a decline)")
    za = probit(1.0 - alpha)
    zb = probit(power)
    num = za * math.sqrt(p0 * (1.0 - p0)) + zb * math.sqrt(p1 * (1.0 - p1))
    return math.ceil((num / (p0 - p1)) ** 2 - 1e-12)


# --------------------------------------------------------------------------
# Behavioral signatures
# --------------------------------------------------------------------------


def ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Least-squares slope of ``y`` on ``x``.

    Applied to review time against action complexity, a slope near zero means
    scrutiny no longer scales with what is at stake.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        raise ValueError("need at least two points")
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0.0:
        raise ValueError("x has zero variance")
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return sxy / sxx


def rolling_rate(flags: Sequence[float], window: int) -> list[float]:
    """Trailing-window mean of 0/1 flags (for example, override decisions).

    Returns ``len(flags) - window + 1`` values; value ``t`` averages
    ``flags[t : t + window]``.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    if window > len(flags):
        raise ValueError("window longer than the series")
    s = sum(flags[:window])
    out = [s / window]
    for t in range(window, len(flags)):
        s += flags[t] - flags[t - window]
        out.append(s / window)
    return out


@dataclass(frozen=True)
class CusumResult:
    """Lower one-sided CUSUM path and the first alarm (0-based), if any."""

    statistic: list[float]
    alarm_index: int | None


def cusum_lower(values: Sequence[float], target_mean: float, slack: float, threshold: float) -> CusumResult:
    """Page's one-sided CUSUM for a *downward* shift in the mean.

    ``S_t = max(0, S_{t-1} + (target_mean - x_t) - slack)`` with ``S_{-1} = 0``;
    alarm at the first ``t`` with ``S_t > threshold``. On log review times this
    flags the moment a reviewer starts spending less time per action.
    """
    _check_nonneg("slack", slack)
    _check_positive("threshold", threshold)
    if not math.isfinite(target_mean):
        raise ValueError("target_mean must be finite")
    s = 0.0
    path: list[float] = []
    alarm: int | None = None
    for t, x in enumerate(values):
        s = max(0.0, s + (target_mean - x) - slack)
        path.append(s)
        if alarm is None and s > threshold:
            alarm = t
    return CusumResult(statistic=path, alarm_index=alarm)


# --------------------------------------------------------------------------
# Approval design: expected-loss gating
# --------------------------------------------------------------------------


class Route(str, Enum):
    AUTO = "auto"
    REVIEW = "review"
    BLOCK = "block"


@dataclass(frozen=True)
class GateDecision:
    route: Route
    loss_auto: float
    loss_review: float
    residual_auto: float
    residual_review: float
    threshold: float


def review_threshold(harm: float, review_cost: float, detection_prob: float) -> float:
    """Error probability above which review beats automatic execution: ``c / (d H)``."""
    _check_nonneg("harm", harm)
    _check_nonneg("review_cost", review_cost)
    _check_prob("detection_prob", detection_prob)
    if detection_prob == 0.0 or harm == 0.0:
        return math.inf
    return review_cost / (detection_prob * harm)


def gate_action(
    p_error: float,
    harm: float,
    review_cost: float,
    detection_prob: float,
    max_residual_loss: float = math.inf,
) -> GateDecision:
    """Route one agent action to AUTO, REVIEW, or BLOCK.

    Expected losses: ``AUTO = p H`` and ``REVIEW = c + p (1 - d) H``. The gate
    picks the cheaper route among those whose residual expected harm (``p H``
    or ``p (1 - d) H``) stays within ``max_residual_loss``; if neither does,
    the action is blocked and escalated.
    """
    _check_prob("p_error", p_error)
    _check_nonneg("harm", harm)
    _check_nonneg("review_cost", review_cost)
    _check_prob("detection_prob", detection_prob)
    if not (max_residual_loss >= 0.0):
        raise ValueError("max_residual_loss must be non-negative")
    residual_auto = p_error * harm
    residual_review = p_error * (1.0 - detection_prob) * harm
    loss_auto = residual_auto
    loss_review = review_cost + residual_review
    feasible = []
    if residual_auto <= max_residual_loss:
        feasible.append((loss_auto, 0, Route.AUTO))
    if residual_review <= max_residual_loss:
        feasible.append((loss_review, 1, Route.REVIEW))
    route = min(feasible)[2] if feasible else Route.BLOCK
    return GateDecision(
        route=route,
        loss_auto=loss_auto,
        loss_review=loss_review,
        residual_auto=residual_auto,
        residual_review=residual_review,
        threshold=review_threshold(harm, review_cost, detection_prob),
    )


def detection_from_review_time(review_minutes: float, d_max: float, time_scale: float) -> float:
    """Saturating detection curve ``d(r) = d_max (1 - exp(-r / r0))``."""
    _check_nonneg("review_minutes", review_minutes)
    _check_prob("d_max", d_max)
    _check_positive("time_scale", time_scale)
    return d_max * (1.0 - math.exp(-review_minutes / time_scale))


def vigilance(time_on_task: float, d_start: float, d_floor: float, time_constant: float) -> float:
    """Vigilance decrement ``d(t) = d_floor + (d_start - d_floor) exp(-t / tau)``."""
    _check_nonneg("time_on_task", time_on_task)
    _check_prob("d_start", d_start)
    _check_prob("d_floor", d_floor)
    _check_positive("time_constant", time_constant)
    return d_floor + (d_start - d_floor) * math.exp(-time_on_task / time_constant)


# --------------------------------------------------------------------------
# Skill maintenance: the irony of automation as an ODE
# --------------------------------------------------------------------------


def _practice_intensity(automation: float, practice: float) -> float:
    _check_prob("automation", automation)
    _check_nonneg("practice", practice)
    return (1.0 - automation) + practice


def steady_state_skill(automation: float, practice: float, learn_rate: float, decay_rate: float) -> float:
    """Fixed point of ``ds/dt = eta u (1 - s) - lambda s`` with ``u = (1 - a) + q``."""
    _check_positive("learn_rate", learn_rate)
    _check_positive("decay_rate", decay_rate)
    gain = learn_rate * _practice_intensity(automation, practice)
    return gain / (gain + decay_rate)


def skill_at(
    t: float,
    s0: float,
    automation: float,
    practice: float,
    learn_rate: float,
    decay_rate: float,
) -> float:
    """Closed-form skill ``s(t) = s* + (s0 - s*) exp(-(eta u + lambda) t)``."""
    _check_nonneg("t", t)
    _check_prob("s0", s0)
    _check_positive("learn_rate", learn_rate)
    _check_positive("decay_rate", decay_rate)
    gain = learn_rate * _practice_intensity(automation, practice)
    s_star = gain / (gain + decay_rate)
    return s_star + (s0 - s_star) * math.exp(-(gain + decay_rate) * t)


def min_practice_rate(target_skill: float, automation: float, learn_rate: float, decay_rate: float) -> float:
    """Smallest deliberate-practice rate ``q`` with steady-state skill >= target.

    ``q_min = max(0, lambda s / (eta (1 - s)) - (1 - a))``.
    """
    _check_prob("target_skill", target_skill, open_high=True)
    _check_prob("automation", automation)
    _check_positive("learn_rate", learn_rate)
    _check_positive("decay_rate", decay_rate)
    required = decay_rate * target_skill / (learn_rate * (1.0 - target_skill))
    return max(0.0, required - (1.0 - automation))


def detection_from_skill(skill: float, d_floor: float, d_ceiling: float) -> float:
    """Linear map from skill in [0, 1] to detection probability in [d_floor, d_ceiling]."""
    _check_prob("skill", skill)
    _check_prob("d_floor", d_floor)
    _check_prob("d_ceiling", d_ceiling)
    if d_ceiling < d_floor:
        raise ValueError("d_ceiling must be >= d_floor")
    return d_floor + (d_ceiling - d_floor) * skill


# --------------------------------------------------------------------------
# Spaced practice scheduling (half-life model)
# --------------------------------------------------------------------------


def recall_probability(elapsed: float, half_life: float) -> float:
    """Exponential forgetting ``p = 2^(-elapsed / h)``."""
    _check_nonneg("elapsed", elapsed)
    _check_positive("half_life", half_life)
    return 2.0 ** (-elapsed / half_life)


def next_interval(half_life: float, target_retention: float) -> float:
    """Days until predicted recall falls to ``target_retention``: ``h log2(1 / R)``."""
    _check_positive("half_life", half_life)
    _check_prob("target_retention", target_retention, open_low=True, open_high=True)
    return half_life * math.log2(1.0 / target_retention)


def update_half_life(
    half_life: float,
    success: bool,
    growth: float = 2.0,
    shrink: float = 0.5,
    floor: float = 0.5,
) -> float:
    """Grow the half-life after a successful session, shrink it after a failure."""
    _check_positive("half_life", half_life)
    _check_positive("growth", growth)
    _check_positive("shrink", shrink)
    _check_positive("floor", floor)
    if growth < 1.0:
        raise ValueError("growth must be >= 1")
    if shrink > 1.0:
        raise ValueError("shrink must be <= 1")
    return half_life * growth if success else max(floor, half_life * shrink)


def practice_schedule(
    outcomes: Sequence[bool],
    initial_half_life: float = 1.0,
    target_retention: float = 0.8,
    growth: float = 2.0,
    shrink: float = 0.5,
    floor: float = 0.5,
) -> list[float]:
    """Day of every practice session given the outcomes of the sessions so far.

    Session 0 happens on day 0. After session ``i`` (outcome ``outcomes[i]``)
    the half-life is updated and session ``i + 1`` is scheduled
    ``next_interval(h, target_retention)`` days later. Returns
    ``len(outcomes) + 1`` days (the last one is the next planned session).
    """
    h = initial_half_life
    _check_positive("initial_half_life", h)
    days = [0.0]
    for ok in outcomes:
        h = update_half_life(h, bool(ok), growth, shrink, floor)
        days.append(days[-1] + next_interval(h, target_retention))
    return days


# --------------------------------------------------------------------------
# Independent review panels
# --------------------------------------------------------------------------


def majority_vote_accuracy(n_reviewers: int, accuracy: float) -> float:
    """Probability that a majority of independent reviewers is correct (ties split 50/50)."""
    if isinstance(n_reviewers, bool) or not isinstance(n_reviewers, int) or n_reviewers < 1:
        raise ValueError("n_reviewers must be a positive integer")
    _check_prob("accuracy", accuracy)
    n = n_reviewers
    total = 0.0
    for k in range(n + 1):
        pk = math.comb(n, k) * accuracy**k * (1.0 - accuracy) ** (n - k)
        if 2 * k > n:
            total += pk
        elif 2 * k == n:
            total += 0.5 * pk
    return total


def layered_residual_error(base_error: float, catch_probs: Sequence[float]) -> float:
    """Error rate left after independent QA layers: ``eps * prod(1 - p_i)``."""
    _check_prob("base_error", base_error)
    out = base_error
    for i, p in enumerate(catch_probs):
        _check_prob(f"catch_probs[{i}]", p)
        out *= 1.0 - p
    return out


# --------------------------------------------------------------------------
# A stateful oversight harness (Python only)
# --------------------------------------------------------------------------


@dataclass
class AgentAction:
    """One proposed agent action awaiting a routing decision."""

    action_id: str
    description: str
    p_error: float
    harm: float
    complexity: float = 1.0
    is_defective: bool | None = None
    is_canary: bool = False


@dataclass(frozen=True)
class ReviewRecord:
    action_id: str
    route: Route
    approved: bool
    review_seconds: float
    complexity: float
    is_canary: bool
    is_defective: bool | None
    pre_commitment: bool | None = None


@dataclass(frozen=True)
class OversightReport:
    n_reviews: int
    n_canaries: int
    canary_catch_rate: float | None
    canary_interval: tuple[float, float] | None
    sdt: SDTResult | None
    override_rate: float | None
    time_complexity_slope: float | None
    cusum_alarm_index: int | None
    estimated_detection: float
    alerts: tuple[str, ...]


@dataclass
class OversightGate:
    """Route agent actions, plant canaries, and watch the overseer's health.

    The gate's detection estimate is the posterior mean of the canary catch
    rate under a ``Beta(prior_detection * prior_strength,
    (1 - prior_detection) * prior_strength)`` prior, so routing reacts to what
    reviewers actually catch rather than to what they are assumed to catch.
    """

    review_cost: float
    max_residual_loss: float = math.inf
    prior_detection: float = 0.9
    prior_strength: float = 10.0
    canary_rate: float = 0.05
    min_detection: float = 0.7
    baseline_log_seconds: float | None = None
    cusum_slack: float = 0.25
    cusum_threshold: float = 3.0
    seed: int | None = None
    records: list[ReviewRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        _check_nonneg("review_cost", self.review_cost)
        _check_prob("prior_detection", self.prior_detection)
        _check_positive("prior_strength", self.prior_strength)
        _check_prob("canary_rate", self.canary_rate)
        _check_prob("min_detection", self.min_detection)
        self._rng = random.Random(self.seed)

    # -- canaries ---------------------------------------------------------
    def _canary_counts(self) -> tuple[int, int]:
        caught = sum(1 for r in self.records if r.is_canary and not r.approved)
        planted = sum(1 for r in self.records if r.is_canary)
        return caught, planted

    def estimated_detection(self) -> float:
        caught, planted = self._canary_counts()
        a = self.prior_detection * self.prior_strength + caught
        b = (1.0 - self.prior_detection) * self.prior_strength + (planted - caught)
        return a / (a + b)

    def should_plant_canary(self) -> bool:
        return self._rng.random() < self.canary_rate

    # -- routing ----------------------------------------------------------
    def route(self, action: AgentAction) -> GateDecision:
        if action.is_canary:
            # Canaries must look like ordinary review items to the reviewer.
            return GateDecision(Route.REVIEW, math.nan, math.nan, math.nan, math.nan, 0.0)
        return gate_action(
            action.p_error,
            action.harm,
            self.review_cost,
            self.estimated_detection(),
            self.max_residual_loss,
        )

    def log_review(
        self,
        action: AgentAction,
        route: Route,
        approved: bool,
        review_seconds: float,
        pre_commitment: bool | None = None,
    ) -> ReviewRecord:
        _check_nonneg("review_seconds", review_seconds)
        rec = ReviewRecord(
            action_id=action.action_id,
            route=route,
            approved=bool(approved),
            review_seconds=float(review_seconds),
            complexity=float(action.complexity),
            is_canary=action.is_canary,
            is_defective=action.is_defective,
            pre_commitment=pre_commitment,
        )
        self.records.append(rec)
        return rec

    # -- health -----------------------------------------------------------
    def report(self) -> OversightReport:
        reviewed = [r for r in self.records if r.route == Route.REVIEW]
        alerts: list[str] = []
        caught, planted = self._canary_counts()
        catch_rate = caught / planted if planted else None
        interval = wilson_interval(caught, planted) if planted else None

        labelled = [r for r in reviewed if r.is_defective is not None]
        sdt = None
        if any(r.is_defective for r in labelled) and any(not r.is_defective for r in labelled):
            hits = sum(1 for r in labelled if r.is_defective and not r.approved)
            misses = sum(1 for r in labelled if r.is_defective and r.approved)
            fas = sum(1 for r in labelled if not r.is_defective and not r.approved)
            crs = sum(1 for r in labelled if not r.is_defective and r.approved)
            sdt = sdt_measures(hits, misses, fas, crs)

        override = None
        if reviewed:
            override = sum(1 for r in reviewed if not r.approved) / len(reviewed)

        slope = None
        xs = [r.complexity for r in reviewed]
        if len(reviewed) >= 2 and len(set(xs)) > 1:
            slope = ols_slope(xs, [r.review_seconds for r in reviewed])

        alarm = None
        if reviewed and self.baseline_log_seconds is not None:
            logs = [math.log(max(r.review_seconds, 1e-9)) for r in reviewed]
            alarm = cusum_lower(logs, self.baseline_log_seconds, self.cusum_slack, self.cusum_threshold).alarm_index
            if alarm is not None:
                alerts.append(f"review time dropped (CUSUM alarm at review #{alarm}): enforce a break")

        d_hat = self.estimated_detection()
        if interval is not None and interval[1] < self.min_detection:
            alerts.append("canary catch rate is credibly below the floor: rotate the reviewer")
        if d_hat < self.min_detection:
            alerts.append(f"estimated detection {d_hat:.2f} < {self.min_detection:.2f}: routing now blocks more actions")
        if slope is not None and slope <= 0.0:
            alerts.append("review time no longer scales with complexity")

        return OversightReport(
            n_reviews=len(reviewed),
            n_canaries=planted,
            canary_catch_rate=catch_rate,
            canary_interval=interval,
            sdt=sdt,
            override_rate=override,
            time_complexity_slope=slope,
            cusum_alarm_index=alarm,
            estimated_detection=d_hat,
            alerts=tuple(alerts),
        )
