using Test
using AIInAction.Ch132ClusteringValidation

# Shared fixtures: identical to the Python and Rust test suites.
const X132 = [
    1.0 1.0
    1.5 2.0
    1.0 0.5
    8.0 8.0
    8.5 7.5
    7.5 8.5
]
const LABELS132 = [0, 0, 0, 1, 1, 1]
const LABELS_TRUE132 = [0, 0, 0, 1, 1, 1]
const LABELS_PRED132 = [0, 0, 1, 1, 2, 2]

@testset "Ch132 clustering validation parity fixtures" begin
    @test silhouette_score(X132, LABELS132) ≈ 0.8954900167230767
    @test davies_bouldin_index(X132, LABELS132) ≈ 0.11157205284841143
    @test calinski_harabasz_index(X132, LABELS132) ≈ 240.14285714285714
    @test adjusted_rand_index(LABELS_TRUE132, LABELS_PRED132) ≈ 0.24242424242424246
end

@testset "Ch132 additional fixtures" begin
    X1d = reshape([0.0, 0.1, 10.0, 10.1], 4, 1)
    @test silhouette_score(X1d, [0, 0, 1, 1]) ≈ 0.9899997499937498
    @test adjusted_rand_index(LABELS_TRUE132, LABELS_TRUE132) ≈ 1.0
    @test adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) ≈ 1.0
    @test adjusted_rand_index([0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]) <= 1e-9
    @test adjusted_rand_index([0, 0, 0], [5, 5, 5]) ≈ 1.0
end

@testset "Ch132 edge cases" begin
    Xline = reshape([0.0, 1.0, 2.0], 3, 1)
    @test_throws ArgumentError silhouette_score(Xline, [0, 0, 0])
    @test_throws ArgumentError davies_bouldin_index(Xline, [0, 0, 0])
    @test_throws ArgumentError calinski_harabasz_index(Xline, [0, 0, 0])
    @test_throws ArgumentError silhouette_score(Xline, [0, 1])
    @test_throws ArgumentError adjusted_rand_index([0, 0, 1], [0, 1])
    Xdup = reshape([0.0, 0.0, 0.0, 0.0], 4, 1)
    @test_throws ArgumentError davies_bouldin_index(Xdup, [0, 1, 0, 1])
    @test_throws ArgumentError calinski_harabasz_index(reshape([0.0, 5.0], 2, 1), [0, 1])
end
