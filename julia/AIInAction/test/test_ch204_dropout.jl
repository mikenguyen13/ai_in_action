using Test
using AIInAction.Ch204Dropout

# Shared fixtures: identical to the Python and Rust test suites. Parity hinges on
# the shared 64-bit LCG, so the same seed drops the same units in every language.
const CH204_EXPECTED_UNIFORMS = [
    0.5682303266439076,
    0.2254634289477513,
    0.41283831882951183,
    0.6303980498395979,
    0.6801478072421157,
    0.02622891069993838,
]
const CH204_EXPECTED_MASK_8 = [0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0]
const CH204_EXPECTED_H = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
const CH204_EXPECTED_OUT_8 = [0.0, 4.0, 6.0, 0.0, 0.0, 12.0, 14.0, 16.0]

@testset "Ch204 dropout parity fixtures" begin
    rng = Lcg(42)
    got = [next_uniform!(rng) for _ in 1:6]
    @test got ≈ CH204_EXPECTED_UNIFORMS atol = 1e-12

    @test bernoulli_mask(8, 0.5, 42) ≈ CH204_EXPECTED_MASK_8

    out, mask = inverted_dropout(CH204_EXPECTED_H, 0.5, 42)
    @test out ≈ CH204_EXPECTED_OUT_8
    @test mask ≈ CH204_EXPECTED_MASK_8
end

@testset "Ch204 dropout properties" begin
    # Generator basics.
    rng = Lcg(123)
    for _ in 1:1000
        u = next_uniform!(rng)
        @test 0.0 <= u < 1.0
    end
    @test [next_uniform!(Lcg(7)) for _ in 1:5] == [next_uniform!(Lcg(7)) for _ in 1:5]

    # Scaling.
    @test expected_scale(0.5) ≈ 2.0
    @test expected_scale(0.8) ≈ 1.25
    @test expected_scale(1.0) ≈ 1.0

    # p == 1 is the identity.
    @test bernoulli_mask(5, 1.0, 7) ≈ ones(5)
    out1, mask1 = inverted_dropout(CH204_EXPECTED_H, 1.0, 42)
    @test out1 ≈ CH204_EXPECTED_H
    @test mask1 ≈ ones(length(CH204_EXPECTED_H))

    # Entries are 0 or 1/p.
    p = 0.3
    inv = 1.0 / p
    for v in bernoulli_mask(50, p, 99)
        @test isapprox(v, 0.0; atol=1e-9) || isapprox(v, inv; atol=1e-9)
    end

    # Mean of a long mask approaches 1.
    m = bernoulli_mask(20000, 0.5, 2024)
    @test isapprox(sum(m) / length(m), 1.0; atol=0.05)

    # Dropped units are exactly zero.
    out, mask = inverted_dropout(CH204_EXPECTED_H, 0.5, 42)
    for i in eachindex(mask)
        if mask[i] == 0.0
            @test out[i] == 0.0
        end
    end

    # Expectation preserved when averaging over seeds.
    h = [3.0, -1.0, 4.0, 2.0]
    trials = 6000
    acc = zeros(length(h))
    for seed in 0:(trials - 1)
        o, _ = inverted_dropout(h, 0.5, seed)
        acc .+= o
    end
    @test acc ./ trials ≈ h atol = 0.15
end

@testset "Ch204 dropout edge cases" begin
    @test_throws ArgumentError Lcg(-1)
    @test_throws ArgumentError expected_scale(0.0)
    @test_throws ArgumentError expected_scale(1.5)
    @test_throws ArgumentError expected_scale(-0.2)
    @test_throws ArgumentError bernoulli_mask(0, 0.5, 1)
    @test_throws ArgumentError inverted_dropout(Float64[], 0.5, 1)
    @test_throws ArgumentError inverted_dropout([1.0, NaN], 0.5, 1)
    @test_throws ArgumentError inverted_dropout([1.0, 2.0], 0.0, 1)
end
