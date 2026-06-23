using Test
using AIInAction.Ch155PrCurves

# Shared fixtures: identical to the Python and Rust test suites.
const Y155 = [1, 0, 1, 1, 0, 1, 0, 0]
const S155 = [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.51]

const PREC155 = [
    1.0,
    0.5,
    0.6666666666666666,
    0.75,
    0.6,
    0.6666666666666666,
    0.5714285714285714,
    0.5,
]
const REC155 = [0.25, 0.25, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0]
const THR155 = [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.51]

@testset "Ch155 PR-curve parity fixtures" begin
    c = pr_curve(Y155, S155)
    @test length(c) == 8
    @test c.thresholds ≈ THR155
    @test c.precision ≈ PREC155
    @test c.recall ≈ REC155
    @test average_precision(Y155, S155) ≈ 0.7708333333333333
    @test auprc_trapezoid(Y155, S155) ≈ 0.48125
end

@testset "Ch155 PR-curve properties and edge cases" begin
    c = pr_curve(Y155, S155)
    for k in 2:length(c.recall)
        @test c.recall[k] >= c.recall[k - 1]
    end

    # Perfect ranking: AP = 1.
    @test average_precision([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) ≈ 1.0

    # Worst ranking: positives at ranks 3 and 4.
    @test average_precision([1, 1, 0, 0], [0.2, 0.1, 0.9, 0.8]) ≈ (1 / 3 + 2 / 4) / 2

    # AP is invariant to a strictly increasing score transform.
    shifted = 10.0 .* S155 .+ 3.0
    @test average_precision(Y155, S155) ≈ average_precision(Y155, shifted)

    # Tie between a positive and a negative: lower index ranks first.
    @test average_precision([1, 0], [0.5, 0.5]) ≈ 1.0

    # All scores equal collapses to a single threshold.
    ceq = pr_curve([1, 0, 1], [0.5, 0.5, 0.5])
    @test length(ceq) == 1
    @test ceq.recall ≈ [1.0]
    @test ceq.precision ≈ [2 / 3]

    @test_throws ArgumentError average_precision([1, 0], [0.5])
    @test_throws ArgumentError average_precision(Int[], Float64[])
    @test_throws ArgumentError pr_curve([1, 2, 0], [0.1, 0.2, 0.3])
    @test_throws ArgumentError average_precision([0, 0, 0], [0.1, 0.2, 0.3])
    @test_throws ArgumentError pr_curve([1, 0], [NaN, 0.3])
end
