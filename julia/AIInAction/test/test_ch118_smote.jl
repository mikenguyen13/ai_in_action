using Test
using AIInAction.Ch118Smote

# Shared fixtures: identical to the Python and Rust test suites.
const MINORITY = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]]
const SEED = 42
const K = 2

# Synthetic points from smote(MINORITY, 4; k=2, seed=42).
const EXPECTED_SMOTE = [
    [0.17625009082257748, 0.0],
    [1.222554265987128, 0.777445734012872],
    [2.0256639048457146, 0.02566390484571457],
    [2.763079992495477, 1.0],
]

# LCG(42) next_float! sequence.
const EXPECTED_LCG = [0.252345174784, 0.088125045411, 0.577281198232, 0.222554265987]

@testset "Ch118 SMOTE parity fixtures" begin
    @test euclidean([0.0, 0.0], [3.0, 4.0]) ≈ 5.0

    rng = LCG(SEED)
    for expected in EXPECTED_LCG
        @test next_float!(rng) ≈ expected atol = 1e-9
    end

    # k_nearest uses 1-based indexing in Julia: Python [1,2] -> Julia [2,3].
    @test k_nearest(MINORITY, 1, K) == [2, 3]
    @test k_nearest(MINORITY, 2, K) == [1, 3]
    @test k_nearest(MINORITY, 3, K) == [2, 4]
    @test k_nearest(MINORITY, 4, K) == [3, 2]

    pts = smote(MINORITY, 4; k = K, seed = SEED)
    @test length(pts) == 4
    for (got, expected) in zip(pts, EXPECTED_SMOTE)
        @test got[1] ≈ expected[1] atol = 1e-9
        @test got[2] ≈ expected[2] atol = 1e-9
    end
end

@testset "Ch118 SMOTE edge cases" begin
    @test smote_sample([0.0, 0.0], [2.0, 4.0], 0.5) == [1.0, 2.0]
    @test smote_sample([1.0, 1.0], [3.0, 5.0], 0.0) == [1.0, 1.0]
    @test smote(MINORITY, 0; k = K, seed = SEED) == Vector{Vector{Float64}}()

    @test_throws ArgumentError euclidean([1.0, 2.0], [1.0])
    @test_throws ArgumentError LCG(-1)
    @test_throws ArgumentError k_nearest(MINORITY, 1, 4)
    @test_throws ArgumentError smote(MINORITY, -1; k = K, seed = SEED)
    @test_throws ArgumentError smote([[0.0, 0.0], [1.0, 1.0]], 3; k = 5, seed = SEED)
    @test_throws ArgumentError smote(Vector{Vector{Float64}}(), 3; k = 2, seed = SEED)
    @test_throws ArgumentError smote_sample([0.0], [1.0], 1.5)
end
