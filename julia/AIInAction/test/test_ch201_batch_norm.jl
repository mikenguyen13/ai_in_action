using Test
using AIInAction.Ch201BatchNorm

# Shared fixtures: identical to the Python and Rust test suites.
const X201 = [
    1.0 2.0 3.0
    4.0 5.0 6.0
    7.0 8.0 9.0
    2.0 0.0 1.0
]
const GAMMA201 = [1.0, 2.0, 0.5]
const BETA201 = [0.0, 1.0, -1.0]
const EPS201 = 1e-5

const DY201 = [
    0.1 -0.2 0.3
    0.4 0.5 -0.6
    -0.7 0.8 0.9
    1.0 -1.1 1.2
]

const RUNMEAN201 = [3.0, 4.0, 5.0]
const RUNVAR201 = [2.0, 3.0, 4.0]

@testset "Ch201 BatchNorm parity fixtures" begin
    Y, cache = batch_norm_forward(X201, GAMMA201, BETA201; eps=EPS201)
    @test cache.mean ≈ [3.5, 3.75, 4.75]
    @test cache.var ≈ [5.25, 9.1875, 9.1875]
    @test cache.x_hat[1, :] ≈ [-1.0910884120486357, -0.5773499549856541, -0.5773499549856541]
    @test Y[1, :] ≈ [-1.0910884120486357, -0.15469990997130822, -1.288674977492827]
    @test Y[4, :] ≈ [-0.6546530472291814, -1.4743569499385174, -1.6185892374846294]

    dX, dgamma, dbeta = batch_norm_backward(DY201, cache)
    @test dgamma ≈ [-1.745741459277817, 2.80427120993032, -0.6433328069840143]
    @test dbeta ≈ [0.8, 0.0, 1.8]
    @test dX[1, :] ≈ [-0.2514695048226982, 0.1351074538761989, -0.040061000612684985]
    @test dX[4, :] ≈ [0.2244527108511118, -0.1535117479685656, 0.09089478082556916]

    Yinf = batch_norm_inference(X201, GAMMA201, BETA201, RUNMEAN201, RUNVAR201; eps=EPS201)
    @test Yinf[1, :] ≈ [-1.4142100268524473, -1.3093972277663308, -1.4999993750011718]
end

@testset "Ch201 BatchNorm properties and edge cases" begin
    # With gamma=1, beta=0 the output is centered, unit-variance per feature.
    Y0, _ = batch_norm_forward(X201, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]; eps=1e-12)
    n = size(X201, 1)
    @test vec(sum(Y0; dims=1)) ./ n ≈ [0.0, 0.0, 0.0] atol = 1e-9
    @test vec(sum(abs2, Y0; dims=1)) ./ n ≈ [1.0, 1.0, 1.0] atol = 1e-6

    # gamma = sqrt(var + eps), beta = mean recovers the input exactly.
    eps = 1e-5
    _, c = batch_norm_forward(X201, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]; eps=eps)
    Yid, _ = batch_norm_forward(X201, sqrt.(c.var .+ eps), c.mean; eps=eps)
    @test Yid ≈ X201 atol = 1e-9

    # dbeta equals the column sum of dY.
    _, cache = batch_norm_forward(X201, GAMMA201, BETA201; eps=EPS201)
    _, _, dbeta = batch_norm_backward(DY201, cache)
    @test dbeta ≈ vec(sum(DY201; dims=1))

    @test_throws ArgumentError batch_norm_forward([1.0 2.0; NaN 3.0], [1.0, 1.0], [0.0, 0.0])
    @test_throws ArgumentError batch_norm_forward(X201, [1.0, 2.0], BETA201)
    @test_throws ArgumentError batch_norm_forward(X201, GAMMA201, BETA201; eps=0.0)
    @test_throws ArgumentError batch_norm_backward([1.0 2.0 3.0], cache)
    @test_throws ArgumentError batch_norm_inference(X201, GAMMA201, BETA201, RUNMEAN201, [-1.0, 1.0, 1.0])
end
