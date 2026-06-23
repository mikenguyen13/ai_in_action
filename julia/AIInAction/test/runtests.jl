using Test
using AIInAction.Metrics

# Shared fixtures: identical to the Python and Rust test suites.
const Y_TRUE = [3.0, -0.5, 2.0, 7.0]
const Y_PRED = [2.5, 0.0, 2.0, 8.0]

@testset "Metrics parity fixtures" begin
    @test rmse(Y_TRUE, Y_PRED) ≈ 0.6123724356957945
    @test mae(Y_TRUE, Y_PRED) ≈ 0.5
    @test r2_score(Y_TRUE, Y_PRED) ≈ 0.9486081370449679
end

@testset "Edge cases" begin
    @test rmse(Y_TRUE, Y_TRUE) == 0.0
    @test r2_score(Y_TRUE, Y_TRUE) ≈ 1.0
    @test accuracy([1, 0, 1, 1], [1, 1, 1, 0]) ≈ 0.5
    @test_throws ArgumentError rmse([1.0, 2.0], [1.0])
    @test_throws ArgumentError r2_score([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
end
