using Test
using AIInAction.Ch146Lof

# Shared fixtures: identical to the Python and Rust test suites. Note Julia uses
# 1-based indices, so neighbor/anomaly index fixtures are the Python values + 1.
const X146 = [
    0.0 0.0
    0.0 1.0
    1.0 0.0
    1.0 1.0
    8.0 8.0
]
const K146 = 2

const LOF146_KDIST = [1.0, 1.0, 1.0, 1.0, 10.63014581273465]
const LOF146_LRD = [1.0, 1.0, 1.0, 1.0, 0.09742011681639788]
const LOF146_SCORES = [1.0, 1.0, 1.0, 1.0, 10.264820374673157]
# Python 0-based knn was [[1,2],[0,3],[0,3],[1,2],[3,1]]; +1 for Julia.
const LOF146_KNN = [[2, 3], [1, 4], [1, 4], [2, 3], [4, 2]]

@testset "Ch146 LOF parity fixtures" begin
    @test euclidean([0.0, 0.0], [3.0, 4.0]) ≈ 5.0

    nbrs = knn_distances(X146, K146)
    @test nbrs == LOF146_KNN

    @test k_distance(X146, K146) ≈ LOF146_KDIST
    @test lrd(X146, K146) ≈ LOF146_LRD
    @test lof_scores(X146, K146) ≈ LOF146_SCORES
end

@testset "Ch146 LOF properties and edge cases" begin
    scores = lof_scores(X146, K146)
    @test all(abs.(scores[1:4] .- 1.0) .< 1e-9)
    @test argmax(scores) == 5

    @test top_anomalies(X146, K146, 1) == [5]
    @test top_anomalies(X146, K146, 2)[1] == 5
    @test length(top_anomalies(X146, K146, 3)) == 3

    # Duplicate coincident points must not produce nan/inf scores.
    dup = [0.0 0.0; 0.0 0.0; 1.0 0.0; 0.0 1.0]
    sdup = lof_scores(dup, 2)
    @test length(sdup) == 4
    @test all(isfinite, sdup)

    # A uniform line has no outlier: all scores stay modest and finite.
    grid = reshape(Float64[0, 1, 2, 3, 4], 5, 1)
    sgrid = lof_scores(grid, 2)
    @test all(isfinite, sgrid)
    @test all(0.5 .<= sgrid .<= 1.5)

    @test_throws ArgumentError lof_scores(X146, 5)
    @test_throws ArgumentError lof_scores(X146, 0)
    @test_throws ArgumentError lof_scores([0.0 0.0; NaN 1.0; 1.0 1.0], 1)
    @test_throws ArgumentError euclidean([1.0, 2.0], [1.0])
    @test_throws ArgumentError top_anomalies(X146, K146, 0)
    @test_throws ArgumentError top_anomalies(X146, K146, 99)
end
