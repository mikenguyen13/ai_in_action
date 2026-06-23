using Test
using AIInAction.Ch198LrSchedules

# Shared fixtures: identical to the Python and Rust test suites.

# cosine_annealing(t, 8, 0.2, 0.01), t = 0..8
const CH198_COSINE_EXPECTED = [
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

# warmup_cosine(t, 3, 10, 0.5, 0.0, 0.0), t = 0..10
const CH198_WC_EXPECTED = [
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

# one_cycle(t, 10, 1.0, 0.04, 0.3), t = 0..10
const CH198_OC_EXPECTED = [
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

# one_cycle_momentum(t, 10, 0.95, 0.85, 0.3), t = 0..10
const CH198_MOM_EXPECTED = [
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

const CH198_TOL = 1e-9

@testset "Ch198 LR schedule parity fixtures" begin
    for t in 0:8
        @test cosine_annealing(t, 8, 0.2, 0.01) ≈ CH198_COSINE_EXPECTED[t + 1] atol = CH198_TOL
    end
    for t in 0:10
        @test warmup_cosine(t, 3, 10, 0.5, 0.0, 0.0) ≈ CH198_WC_EXPECTED[t + 1] atol = CH198_TOL
    end
    for t in 0:10
        @test one_cycle(t, 10, 1.0, 0.04, 0.3) ≈ CH198_OC_EXPECTED[t + 1] atol = CH198_TOL
    end
    for t in 0:10
        @test one_cycle_momentum(t, 10, 0.95, 0.85, 0.3) ≈ CH198_MOM_EXPECTED[t + 1] atol = CH198_TOL
    end
end

@testset "Ch198 LR schedule properties and edge cases" begin
    # Cosine endpoints / midpoint.
    @test cosine_annealing(0, 10, 0.1, 0.0) ≈ 0.1
    @test cosine_annealing(5, 10, 0.1, 0.0) ≈ 0.05
    @test cosine_annealing(10, 10, 0.1, 0.0) ≈ 0.0 atol = 1e-15
    @test cosine_annealing(50, 10, 0.1, 0.01) ≈ 0.01  # clamps past horizon

    # Monotone decreasing.
    cvals = [cosine_annealing(t, 20, 0.3, 0.0) for t in 0:20]
    @test all(cvals[i] >= cvals[i + 1] - CH198_TOL for i in 1:20)

    # Linear warmup.
    @test linear_warmup(0, 4, 0.1) ≈ 0.0
    @test linear_warmup(2, 4, 0.1) ≈ 0.05
    @test linear_warmup(4, 4, 0.1) ≈ 0.1
    @test linear_warmup(7, 4, 0.1) ≈ 0.1
    @test linear_warmup(2, 4, 0.1, 0.02) ≈ 0.06

    # Continuity of warmup -> cosine handoff.
    @test linear_warmup(3, 3, 0.5) ≈ warmup_cosine(3, 3, 10, 0.5)
    @test warmup_cosine(3, 3, 10, 0.5) ≈ 0.5

    # One-cycle default eta_min and peak.
    @test one_cycle(0, 10, 1.0) ≈ 0.04
    @test one_cycle(0, 10, 0.5) ≈ 0.02
    @test one_cycle(3, 10, 1.0, 0.04, 0.3) ≈ 1.0

    # Momentum antiphase.
    @test one_cycle_momentum(3, 10, 0.95, 0.85, 0.3) ≈ 0.85
    @test one_cycle_momentum(0, 10, 0.95, 0.85, 0.3) ≈ 0.95
    @test one_cycle_momentum(10, 10, 0.95, 0.85, 0.3) ≈ 0.95

    # schedule_curve.
    curve = schedule_curve("cosine", 8; eta_max=0.2, eta_min=0.01)
    @test length(curve) == 8
    @test curve ≈ CH198_COSINE_EXPECTED[1:8] atol = CH198_TOL

    # Validation.
    @test_throws ArgumentError cosine_annealing(-1, 10, 0.1, 0.0)
    @test_throws ArgumentError cosine_annealing(0, 0, 0.1, 0.0)
    @test_throws ArgumentError cosine_annealing(0, 10, 0.01, 0.1)
    @test_throws ArgumentError warmup_cosine(0, 10, 10, 0.1)
    @test_throws ArgumentError one_cycle(0, 10, 1.0, 0.04, 1.0)
    @test_throws ArgumentError one_cycle_momentum(0, 10, 0.85, 0.95, 0.3)
    @test_throws ArgumentError one_cycle_momentum(0, 10, 1.0, 0.85, 0.3)
    @test_throws ArgumentError schedule_curve("triangular", 10; eta_max=0.1)
end
