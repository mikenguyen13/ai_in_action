using Test
using AIInAction.Ch207Densenet

# Shared fixtures: identical to the Python and Rust test suites. Parity hinges
# on the shared 64-bit LCG, so the same seed produces the same weights (and
# hence the same forward pass) in every language.
const CH207_EXPECTED_UNIFORMS = [
    0.07820865487829387,
    0.10169876029679303,
    0.60532332262523347,
    0.40121620369530075,
]
const CH207_X0 = [1.0, 0.6, -0.3, 0.9]
const CH207_EXPECTED_SIZES = [4, 7, 10, 13]
const CH207_EXPECTED_OUT = [
    1.0, 0.6, -0.3, 0.9,
    0.6279286728409486, 0.11130697455664373, 0.08343734603269459,
    0.0, 0.9451360437126258, 1.4047640097915077,
    0.0, 0.0, 0.8720018399583354,
]

@testset "Ch207 densenet channel/param arithmetic" begin
    @test dense_block_channel_sizes(4, 4, 3) == [4, 8, 12, 16]
    @test dense_block_channel_sizes(4, 3, 3) == [4, 7, 10, 13]

    @test dense_block_param_count(4, 4, 3, 4) == 2112
    @test dense_block_param_count(8, 12, 4, 4) == 25728
    @test dense_block_param_count(64, 32, 6, 4) == 331776

    @test transition_output_channels(64, 0.5) == 32
    @test transition_output_channels(128, 0.5) == 64

    @test plain_block_param_count(16, 16, 3) == 6912
    @test plain_block_param_count(64, 64, 6) == 221184
end

@testset "Ch207 densenet parameter efficiency" begin
    # Same total output width, dense connectivity uses far fewer parameters
    # than a plain stack of matched width -- the claim behind Section 4.2.
    dense = dense_block_param_count(16, 12, 8, 4)
    plain = plain_block_param_count(16, 16 + 12 * 8, 8)
    @test dense < plain
end

@testset "Ch207 densenet LCG and forward-pass parity fixtures" begin
    rng = Lcg(0)
    got = [next_uniform!(rng) for _ in 1:4]
    @test got ≈ CH207_EXPECTED_UNIFORMS atol = 1e-12

    out, sizes = dense_block_forward(CH207_X0, 3, 3, 2)
    @test sizes == CH207_EXPECTED_SIZES
    @test out ≈ CH207_EXPECTED_OUT atol = 1e-9

    # A second fixture pinning the smaller demo used in the docstring.
    out2, sizes2 = dense_block_forward([1.0, -1.0, 0.5, 0.5], 2, 2, 0)
    @test sizes2 == [4, 6, 8]
    @test length(out2) == 8
end

@testset "Ch207 densenet variant param totals" begin
    @test densenet_dense_param_total("121") == 6_860_800
    @test densenet_dense_param_total("169") == 12_316_672
    @test densenet_dense_param_total("201") == 17_854_464
    @test densenet_dense_param_total("264") == 30_240_768
end

@testset "Ch207 densenet edge cases" begin
    @test_throws ArgumentError dense_block_channel_sizes(-1, 4, 3)
    @test_throws ArgumentError dense_block_channel_sizes(4, 0, 3)
    @test_throws ArgumentError dense_block_channel_sizes(4, 4, 0)
    @test_throws ArgumentError transition_output_channels(64, 0.0)
    @test_throws ArgumentError transition_output_channels(64, 1.5)
    @test_throws ArgumentError densenet_dense_param_total("bogus")
    @test_throws ArgumentError dense_block_forward(Float64[], 2, 2, 0)
    @test_throws ArgumentError dense_block_forward([1.0, NaN], 2, 2, 0)
end
