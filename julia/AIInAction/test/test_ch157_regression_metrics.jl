using Test
using AIInAction.Ch157RegressionMetrics

# Shared fixtures: identical to the Python and Rust test suites. Every top-level
# const is prefixed with RM157 so it cannot collide with other test files.
const RM157_Y_TRUE = [3.0, -0.5, 2.0, 7.0]
const RM157_Y_PRED = [2.5, 0.0, 2.0, 8.0]
const RM157_DELTA = 0.75

@testset "Ch157 regression-metrics parity fixtures" begin
    @test mse(RM157_Y_TRUE, RM157_Y_PRED) ≈ 0.375
    @test rmse(RM157_Y_TRUE, RM157_Y_PRED) ≈ 0.6123724356957945
    @test mae(RM157_Y_TRUE, RM157_Y_PRED) ≈ 0.5
    @test huber_loss(RM157_Y_TRUE, RM157_Y_PRED; delta=RM157_DELTA) ≈
          [0.125, 0.125, 0.0, 0.46875]
    @test huber_loss_mean(RM157_Y_TRUE, RM157_Y_PRED; delta=RM157_DELTA) ≈ 0.1796875
end

@testset "Ch157 regression-metrics properties and edge cases" begin
    @test rmse(RM157_Y_TRUE, RM157_Y_PRED) ≈ sqrt(mse(RM157_Y_TRUE, RM157_Y_PRED))

    # Continuity at the threshold: both branches give 0.5*delta^2.
    d = 1.5
    @test huber_loss([0.0], [d]; delta=d)[1] ≈ 0.5 * d * d
    @test huber_loss([0.0], [d + 1e-9]; delta=d)[1] ≈ 0.5 * d * d atol = 1e-6

    # Large delta: mean Huber -> 0.5 * MSE.
    @test huber_loss_mean(RM157_Y_TRUE, RM157_Y_PRED; delta=1000.0) ≈
          0.5 * mse(RM157_Y_TRUE, RM157_Y_PRED)

    # Perfect prediction is zero everywhere.
    @test mse(RM157_Y_TRUE, RM157_Y_TRUE) == 0.0
    @test rmse(RM157_Y_TRUE, RM157_Y_TRUE) == 0.0
    @test huber_loss_mean(RM157_Y_TRUE, RM157_Y_TRUE; delta=1.0) == 0.0

    @test_throws ArgumentError mse([1.0, 2.0], [1.0])
    @test_throws ArgumentError huber_loss([1.0, 2.0], [1.0]; delta=1.0)
    @test_throws ArgumentError mse(Float64[], Float64[])
    @test_throws ArgumentError huber_loss(RM157_Y_TRUE, RM157_Y_PRED; delta=0.0)
    @test_throws ArgumentError huber_loss(RM157_Y_TRUE, RM157_Y_PRED; delta=-1.0)
    @test_throws ArgumentError huber_loss(RM157_Y_TRUE, RM157_Y_PRED; delta=Inf)
end
