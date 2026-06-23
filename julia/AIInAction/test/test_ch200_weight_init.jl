using Test
using AIInAction.Ch200WeightInit

# Shared fixtures: identical to the Python and Rust test suites. Every const is
# prefixed CH200_ so it cannot collide with other test files.
const CH200_GAIN_RELU = 1.4142135623730951
const CH200_GAIN_TANH = 1.6666666666666667
const CH200_GAIN_LEAKY_02 = 1.3867504905630728

const CH200_XAVIER_46_STD = 0.4472135954999579
const CH200_XAVIER_46_BOUND = 0.7745966692414833
const CH200_HE_8_STD = 0.5
const CH200_HE_8_BOUND = 0.8660254037844386

# xavier_normal(fan_in=3, fan_out=2, seed=42) -> shape (2, 3)
const CH200_XAVIER_NORMAL_3_2_SEED42 = [
    0.262291800404047 -0.5640783697534434 1.0938907166332181
    0.34508066325448 -0.683313150259559 -1.1250423158375467
]

# he_uniform(fan_in=3, fan_out=2, seed=7) -> shape (2, 3)
const CH200_HE_UNIFORM_3_2_SEED7 = [
    -0.3116085279902403 -1.3667290947514303 1.1335223795602536
    0.23456229026376593 -0.13451463415109 -0.7087146789818501
]

# he_normal(fan_in=4, fan_out=3, seed=123, :fan_in) -> shape (3, 4)
const CH200_HE_NORMAL_4_3_SEED123 = [
    0.5830829313806632 -0.1503944856454256 -0.30548437167556053 -0.00772328473476067
    0.43836777367788293 0.45770047377992695 0.6982243250890897 -0.151260320705838
    -0.16215580034083707 -1.1099931389726774 0.23040596666525975 -0.4700322147558535
]

const CH200_TOL = 1e-9

@testset "Ch200 weight-init gain factors" begin
    @test calculate_gain("relu") ≈ CH200_GAIN_RELU
    @test calculate_gain("tanh") ≈ CH200_GAIN_TANH
    @test calculate_gain("linear") == 1.0
    @test calculate_gain("sigmoid") == 1.0
    @test calculate_gain("leaky_relu"; param=0.2) ≈ CH200_GAIN_LEAKY_02
    @test calculate_gain("leaky_relu"; param=1.0) ≈ 1.0
    @test_throws ArgumentError calculate_gain("gelu_bananas")
    @test_throws ArgumentError calculate_gain("leaky_relu"; param=-1.0)
end

@testset "Ch200 theoretical scales" begin
    s = xavier_scale(4, 6)
    @test s.std ≈ CH200_XAVIER_46_STD atol = CH200_TOL
    @test s.bound ≈ CH200_XAVIER_46_BOUND atol = CH200_TOL

    s2 = xavier_scale(7, 11; gain=1.3)
    @test s2.bound ≈ s2.std * sqrt(3.0) atol = CH200_TOL

    h = he_scale(8)
    @test h.std ≈ CH200_HE_8_STD atol = CH200_TOL
    @test h.bound ≈ CH200_HE_8_BOUND atol = CH200_TOL
    @test he_scale(2).std ≈ 1.0 atol = CH200_TOL
end

@testset "Ch200 seeded sampling parity fixtures" begin
    w1 = xavier_normal(3, 2; seed=42)
    @test size(w1) == (2, 3)
    @test w1 ≈ CH200_XAVIER_NORMAL_3_2_SEED42 atol = CH200_TOL

    w2 = he_uniform(3, 2; seed=7)
    @test size(w2) == (2, 3)
    @test w2 ≈ CH200_HE_UNIFORM_3_2_SEED7 atol = CH200_TOL

    w3 = he_normal(4, 3; seed=123, mode=:fan_in)
    @test size(w3) == (3, 4)
    @test w3 ≈ CH200_HE_NORMAL_4_3_SEED123 atol = CH200_TOL

    @test he_normal(5, 5; seed=99) == he_normal(5, 5; seed=99)
    @test !(he_normal(5, 5; seed=1) ≈ he_normal(5, 5; seed=2))

    bound = he_scale(16).bound
    wu = he_uniform(16, 16; seed=5)
    @test all(abs.(wu) .<= bound + CH200_TOL)
end

@testset "Ch200 validation and edge cases" begin
    @test_throws ArgumentError xavier_normal(0, 4)
    @test_throws ArgumentError xavier_uniform(4, -1)
    @test_throws ArgumentError he_normal(4, 4; mode=:fan_sideways)
    @test_throws ArgumentError xavier_normal(4, 4; gain=0.0)
    @test_throws ArgumentError he_normal(4, 4; gain=0.0)
    @test_throws ArgumentError he_scale(0)
end
