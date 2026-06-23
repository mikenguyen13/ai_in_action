using Test
using AIInAction.Ch202LayerNorm

# Shared fixtures: identical to the Python and Rust test suites.
const LN202_X = [2.0, 4.0, 6.0, 8.0]
const LN202_GAMMA = [1.5, 0.5, 1.0, 2.0]
const LN202_BETA = [0.1, -0.2, 0.0, 0.3]
const LN202_EPS = 1e-5

const LN202_EXPECTED_LN_PLAIN = [
    -1.3416394448610998,
    -0.4472131482870333,
    0.4472131482870333,
    1.3416394448610998,
]
const LN202_EXPECTED_LN_AFFINE = [
    -1.9124591672916496,
    -0.42360657414351666,
    0.4472131482870333,
    2.9832788897221993,
]
const LN202_EXPECTED_RMS_PLAIN = [
    0.36514831081206406,
    0.7302966216241281,
    1.0954449324361921,
    1.4605932432482562,
]
const LN202_EXPECTED_RMS_GAMMA = [
    0.5477224662180961,
    0.36514831081206406,
    1.0954449324361921,
    2.9211864864965125,
]

@testset "Ch202 LayerNorm/RMSNorm parity fixtures" begin
    @test layer_norm(LN202_X; eps=LN202_EPS) ≈ LN202_EXPECTED_LN_PLAIN
    @test layer_norm(LN202_X; gamma=LN202_GAMMA, beta=LN202_BETA, eps=LN202_EPS) ≈
          LN202_EXPECTED_LN_AFFINE
    @test rms_norm(LN202_X; eps=LN202_EPS) ≈ LN202_EXPECTED_RMS_PLAIN
    @test rms_norm(LN202_X; gamma=LN202_GAMMA, eps=LN202_EPS) ≈ LN202_EXPECTED_RMS_GAMMA
end

@testset "Ch202 properties and edge cases" begin
    # Zero mean, unit population variance for the plain layer norm.
    y = layer_norm(LN202_X; eps=0.0)
    n = length(y)
    @test sum(y) / n ≈ 0.0 atol = 1e-12
    @test sum(abs2, y) / n ≈ 1.0

    # Shift/scale invariance of LN.
    a, b = 3.0, 5.0
    base = layer_norm(LN202_X; eps=0.0)
    shifted = layer_norm([a * v + b for v in LN202_X]; eps=0.0)
    @test shifted ≈ base

    # Scale invariance of RMSNorm.
    scaled = rms_norm([4.0 * v for v in LN202_X]; eps=0.0)
    @test scaled ≈ rms_norm(LN202_X; eps=0.0)

    # Constant vector -> all zeros (gamma=1, beta=0) with eps regularization.
    yc = layer_norm([3.0, 3.0, 3.0]; eps=1e-5)
    @test all(isfinite, yc)
    @test yc ≈ [0.0, 0.0, 0.0]

    # Batched application normalizes rows independently.
    M = [2.0 4.0 6.0 8.0; 1.0 1.0 1.0 1.0]
    outln = apply_layer_norm(M; eps=LN202_EPS)
    @test size(outln) == (2, 4)
    @test outln[1, :] ≈ LN202_EXPECTED_LN_PLAIN
    @test outln[2, :] ≈ [0.0, 0.0, 0.0, 0.0]

    Mr = [2.0 4.0 6.0 8.0; 2.0 4.0 6.0 8.0]
    outrms = apply_rms_norm(Mr; eps=LN202_EPS)
    @test outrms[1, :] ≈ LN202_EXPECTED_RMS_PLAIN
    @test outrms[2, :] ≈ LN202_EXPECTED_RMS_PLAIN

    @test_throws ArgumentError layer_norm(Float64[])
    @test_throws ArgumentError rms_norm([1.0, NaN, 3.0])
    @test_throws ArgumentError layer_norm(LN202_X; gamma=[1.0, 2.0])
    @test_throws ArgumentError layer_norm(LN202_X; gamma=LN202_GAMMA, beta=[1.0, 2.0])
    @test_throws ArgumentError rms_norm(LN202_X; eps=-1e-6)
end
