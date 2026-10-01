using Test
using AIInAction.Ch525Oversight

# Shared fixtures: identical to the Python (tests/test_ch525_oversight.py) and Rust
# suites. Names are prefixed with OV525_ so they cannot collide with other test files.
const OV525_SDT_COUNTS = (18, 2, 5, 75)  # hits, misses, false alarms, correct rejections
const OV525_SDT_LOGLINEAR_DPRIME = 2.6713670993913583
const OV525_SDT_LOGLINEAR_CRITERION = 0.15592243208381895
const OV525_SDT_RAW_DPRIME = 2.8156721098971467
const OV525_SDT_RAW_CRITERION = 0.12628448940397252
const OV525_SDT_PERFECT_DPRIME = 4.482858869990343  # (20, 0, 0, 80), log-linear
const OV525_SDT_PERFECT_CRITERION = 0.26067703834789313

const OV525_WILSON_18_OF_20 = (0.6989663547715128, 0.9721335187862319)
const OV525_ZERO_FAILURE_30 = 0.09503385285530419
const OV525_CANARIES_09_07 = 20
const OV525_CANARIES_095_085 = 44
const OV525_CANARIES_09_08_POWER09 = 102

const OV525_SLOPE_X = [1.0, 2.0, 3.0, 4.0, 5.0]
const OV525_SLOPE_Y = [2.0, 4.1, 5.9, 8.2, 9.8]
const OV525_SLOPE = 1.97

const OV525_ROLLING_FLAGS = [0, 1, 0, 0, 1, 1, 0, 0]
const OV525_ROLLING_3 = [1 / 3, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1 / 3]

const OV525_CUSUM_VALUES = [5.0, 5.1, 4.9, 4.2, 4.0, 3.9, 4.1, 3.8]
const OV525_CUSUM_PATH = [0.0, 0.0, 0.0, 0.55, 1.3, 2.15, 2.8, 3.75]
const OV525_CUSUM_ALARM = 6  # 1-based (Python: 5)

const OV525_SKILL_AT_3 = 0.34869187738724783  # t=3, s0=0.8, a=0.9, q=0, eta=2, lambda=0.5
const OV525_MIN_PRACTICE = 0.275  # target 0.6, a=0.9, eta=2, lambda=0.5
const OV525_STEADY_A09_Q015 = 0.5

const OV525_SCHEDULE_TTFT = [0.0, 0.6438561897747247, 1.931568569324174,
    2.575424759098899, 3.863137138648348]

const OV525_MAJORITY_3_085 = 0.93925
const OV525_MAJORITY_5_07 = 0.83692
const OV525_VIGILANCE_30 = 0.666939048044529  # t=30, d0=0.9, floor=0.6, tau=20
const OV525_DETECT_REVIEW_2 = 0.6995827187900596  # r=2, d_max=0.95, r0=1.5

# Extra probit values from Python's statistics.NormalDist().inv_cdf, one per AS241
# branch (central, intermediate tail, far tail).
const OV525_PROBIT_03 = -0.5244005127080407
const OV525_PROBIT_1EM10 = -6.361340902404056
const OV525_PROBIT_1EM30 = -11.464024688443617

@testset "Ch525 probit (AS241)" begin
    @test probit(0.975) ≈ 1.959963984540054 atol = 1e-12
    @test probit(0.5) ≈ 0.0 atol = 1e-15
    @test probit(0.02) ≈ -2.053748910631822 atol = 1e-12
    @test probit(0.3) ≈ OV525_PROBIT_03 atol = 1e-12
    @test probit(1e-10) ≈ OV525_PROBIT_1EM10 atol = 1e-12
    @test probit(1e-30) ≈ OV525_PROBIT_1EM30 atol = 1e-12
    @test probit(0.9) ≈ -probit(0.1) atol = 1e-15
    @test_throws ArgumentError probit(0.0)
    @test_throws ArgumentError probit(1.0)
    @test_throws ArgumentError probit(NaN)
end

@testset "Ch525 signal detection fixtures" begin
    r = sdt_measures(OV525_SDT_COUNTS...)
    @test r isa SDTResult
    @test r.d_prime ≈ OV525_SDT_LOGLINEAR_DPRIME atol = 1e-12
    @test r.criterion ≈ OV525_SDT_LOGLINEAR_CRITERION atol = 1e-12
    @test r.hit_rate ≈ 18.5 / 21
    @test r.false_alarm_rate ≈ 5.5 / 81

    raw = sdt_measures(OV525_SDT_COUNTS...; correction=:none)
    @test raw.d_prime ≈ OV525_SDT_RAW_DPRIME atol = 1e-12
    @test raw.criterion ≈ OV525_SDT_RAW_CRITERION atol = 1e-12

    perfect = sdt_measures(20, 0, 0, 80)
    @test isfinite(perfect.d_prime)
    @test perfect.d_prime ≈ OV525_SDT_PERFECT_DPRIME atol = 1e-12
    @test perfect.criterion ≈ OV525_SDT_PERFECT_CRITERION atol = 1e-12
    @test_throws ArgumentError sdt_measures(20, 0, 0, 80; correction=:none)
end

@testset "Ch525 signal detection properties and validation" begin
    careful = sdt_measures(18, 2, 8, 72)
    stamping = sdt_measures(10, 10, 1, 79)
    @test stamping.criterion > careful.criterion

    @test_throws ArgumentError sdt_measures(0, 0, 1, 1)
    @test_throws ArgumentError sdt_measures(1, -1, 1, 1)
    @test_throws ArgumentError sdt_measures(1, 1, 1, 1; correction=:bogus)
end

@testset "Ch525 canary statistics" begin
    lo, hi = wilson_interval(18, 20)
    @test lo ≈ OV525_WILSON_18_OF_20[1] atol = 1e-12
    @test hi ≈ OV525_WILSON_18_OF_20[2] atol = 1e-12
    @test wilson_interval(18, 20) isa Tuple{Float64,Float64}

    lo0, hi0 = wilson_interval(0, 10)
    @test lo0 == 0.0
    @test 0.0 < hi0 < 0.35
    lo10, hi10 = wilson_interval(10, 10)
    @test hi10 ≈ 1.0
    @test lo10 > 0.65
    @test_throws ArgumentError wilson_interval(3, 2)
    @test_throws ArgumentError wilson_interval(0, 0)

    @test zero_failure_upper_bound(30) ≈ OV525_ZERO_FAILURE_30 atol = 1e-12
    for n in (50, 100, 1000)
        # Rule of three: the exact bound is close to 3 / n.
        @test zero_failure_upper_bound(n) ≈ 3 / n rtol = 0.05
    end
    @test_throws ArgumentError zero_failure_upper_bound(0)

    @test canaries_to_detect_decline(0.9, 0.7) == OV525_CANARIES_09_07
    @test canaries_to_detect_decline(0.95, 0.85) == OV525_CANARIES_095_085
    @test canaries_to_detect_decline(0.9, 0.8, 0.05, 0.9) == OV525_CANARIES_09_08_POWER09
    @test canaries_to_detect_decline(0.9, 0.7) isa Int
    @test_throws ArgumentError canaries_to_detect_decline(0.7, 0.9)
end

@testset "Ch525 behavioral signatures" begin
    @test ols_slope(OV525_SLOPE_X, OV525_SLOPE_Y) ≈ OV525_SLOPE atol = 1e-12
    @test_throws ArgumentError ols_slope([1.0, 1.0], [2.0, 3.0])
    @test_throws ArgumentError ols_slope([1.0], [2.0])
    @test_throws ArgumentError ols_slope([1.0, 2.0], [2.0])

    rr = rolling_rate(OV525_ROLLING_FLAGS, 3)
    @test length(rr) == length(OV525_ROLLING_FLAGS) - 3 + 1
    @test rr ≈ OV525_ROLLING_3 atol = 1e-12
    @test_throws ArgumentError rolling_rate(OV525_ROLLING_FLAGS, 0)
    @test_throws ArgumentError rolling_rate(OV525_ROLLING_FLAGS, 9)

    res = cusum_lower(OV525_CUSUM_VALUES, 5.0, 0.25, 1.5)
    @test res isa CusumResult
    @test length(res.statistic) == length(OV525_CUSUM_PATH)
    @test res.statistic ≈ OV525_CUSUM_PATH atol = 1e-12
    @test res.alarm_index == OV525_CUSUM_ALARM

    stable = cusum_lower([5.0, 5.2, 4.9, 5.1, 4.95], 5.0, 0.25, 1.5)
    @test stable.alarm_index === nothing
    @test maximum(stable.statistic) == 0.0
    @test_throws ArgumentError cusum_lower(OV525_CUSUM_VALUES, 5.0, -0.1, 1.5)
    @test_throws ArgumentError cusum_lower(OV525_CUSUM_VALUES, 5.0, 0.25, 0.0)
    @test_throws ArgumentError cusum_lower(OV525_CUSUM_VALUES, Inf, 0.25, 1.5)
end

@testset "Ch525 gating" begin
    @test gate_action(0.005, 100, 1, 0.9).route == AUTO
    @test gate_action(0.02, 100, 1, 0.9).route == REVIEW
    @test gate_action(0.2, 100, 1, 0.9; max_residual_loss=5.0).route == REVIEW
    @test gate_action(0.2, 100, 1, 0.2; max_residual_loss=5.0).route == BLOCK

    d = gate_action(0.02, 100, 1, 0.9)
    @test d isa GateDecision
    @test d.loss_auto ≈ 2.0
    @test d.loss_review ≈ 1.2
    @test d.threshold ≈ 1 / 90
    @test review_threshold(100, 1, 0.0) == Inf

    # The review threshold is the indifference point between AUTO and REVIEW.
    harm, cost, det_p = 250.0, 3.0, 0.8
    p_star = review_threshold(harm, cost, det_p)
    di = gate_action(p_star, harm, cost, det_p)
    @test di.loss_auto ≈ di.loss_review

    @test_throws ArgumentError gate_action(1.5, 100, 1, 0.9)
    @test_throws ArgumentError gate_action(0.1, 100, 1, 0.9; max_residual_loss=-1.0)
    @test_throws ArgumentError review_threshold(-1.0, 1, 0.9)

    @test vigilance(30, 0.9, 0.6, 20) ≈ OV525_VIGILANCE_30 atol = 1e-12
    @test vigilance(0, 0.9, 0.6, 20) ≈ 0.9
    @test detection_from_review_time(2, 0.95, 1.5) ≈ OV525_DETECT_REVIEW_2 atol = 1e-12
    @test_throws ArgumentError detection_from_review_time(2, 0.95, 0.0)
end

@testset "Ch525 skill model" begin
    @test skill_at(3.0, 0.8, 0.9, 0.0, 2.0, 0.5) ≈ OV525_SKILL_AT_3 atol = 1e-12
    @test min_practice_rate(0.6, 0.9, 2.0, 0.5) ≈ OV525_MIN_PRACTICE atol = 1e-12
    @test steady_state_skill(0.9, 0.15, 2.0, 0.5) ≈ OV525_STEADY_A09_Q015 atol = 1e-12

    s_star = steady_state_skill(0.7, 0.05, 1.5, 0.3)
    @test skill_at(200.0, 0.1, 0.7, 0.05, 1.5, 0.3) ≈ s_star atol = 1e-10

    # The minimum practice rate achieves the target exactly.
    q = min_practice_rate(0.75, 0.95, 2.0, 0.5)
    @test steady_state_skill(0.95, q, 2.0, 0.5) ≈ 0.75
    @test min_practice_rate(0.1, 0.0, 2.0, 0.5) == 0.0

    # Full automation without practice erodes skill to zero.
    @test steady_state_skill(1.0, 0.0, 2.0, 0.5) == 0.0

    @test detection_from_skill(0.5, 0.3, 0.95) ≈ 0.625
    @test_throws ArgumentError detection_from_skill(0.5, 0.9, 0.3)
    @test_throws ArgumentError min_practice_rate(1.0, 0.9, 2.0, 0.5)
    @test_throws ArgumentError steady_state_skill(0.9, 0.1, 0.0, 0.5)
end

@testset "Ch525 spaced practice" begin
    @test recall_probability(2.0, 2.0) ≈ 0.5
    h, r = 4.0, 0.8
    @test recall_probability(next_interval(h, r), h) ≈ r

    sched = practice_schedule([true, true, false, true])
    @test length(sched) == 5
    @test sched ≈ OV525_SCHEDULE_TTFT atol = 1e-12
    @test practice_schedule(Bool[]) == [0.0]

    @test update_half_life(0.6, false) == 0.5
    @test update_half_life(1.0, true) == 2.0
    @test_throws ArgumentError update_half_life(1.0, true; growth=0.5)
    @test_throws ArgumentError update_half_life(1.0, false; shrink=1.5)
    @test_throws ArgumentError next_interval(1.0, 1.0)
end

@testset "Ch525 independent panels" begin
    @test majority_vote_accuracy(3, 0.85) ≈ OV525_MAJORITY_3_085 atol = 1e-12
    @test majority_vote_accuracy(5, 0.7) ≈ OV525_MAJORITY_5_07 atol = 1e-12
    # With ties split by a coin, an even panel is no better than the odd panel below it.
    @test majority_vote_accuracy(4, 0.85) ≈ majority_vote_accuracy(3, 0.85)
    for m in 1:5
        @test majority_vote_accuracy(2 * m, 0.7) ≈ majority_vote_accuracy(2 * m - 1, 0.7)
    end
    @test majority_vote_accuracy(1, 0.85) ≈ 0.85
    @test_throws ArgumentError majority_vote_accuracy(0, 0.85)
    @test_throws ArgumentError majority_vote_accuracy(3, 1.5)

    @test layered_residual_error(0.1, [0.7, 0.6, 0.5]) ≈ 0.006
    @test layered_residual_error(0.1, Float64[]) ≈ 0.1
    @test_throws ArgumentError layered_residual_error(0.1, [0.5, 1.5])
end
