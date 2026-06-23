using Test
using AIInAction.Ch125AgglomerativeClustering

# Shared fixtures: identical to the Python and Rust test suites.
const PTS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [10.0, 10.0],
    [10.0, 11.0],
]

# (third-merge height, root height) per linkage.
const EXPECTED = Dict(
    "single" => (1.0, 13.45362404707371),
    "complete" => (1.414213562373095, 14.866068747318508),
    "average" => (1.2071067811865475, 14.045043082079953),
    "ward" => (1.2909944487358056, 21.73323108360405),
)

@testset "Agglomerative clustering parity fixtures" begin
    for lk in LINKAGES
        m = linkage_matrix(PTS, lk)
        @test length(m) == 4
        @test Int(m[1][1]) == 0 && Int(m[1][2]) == 1
        @test m[1][3] ≈ 1.0
        @test m[1][4] ≈ 2.0
        h_third, h_root = EXPECTED[lk]
        @test m[3][3] ≈ h_third
        @test m[4][3] ≈ h_root
        @test m[4][4] ≈ 5.0
        for row in m
            @test row[1] < row[2]
        end
        for t in 1:(length(m) - 1)
            @test m[t + 1][3] >= m[t][3] - 1e-9
        end
    end
end

@testset "fcluster cuts" begin
    for lk in LINKAGES
        m = linkage_matrix(PTS, lk)
        @test fcluster(m, 2) == [0, 0, 0, 1, 1]
        @test fcluster(m, 5) == [0, 1, 2, 3, 4]
        @test fcluster(m, 1) == [0, 0, 0, 0, 0]
    end
end

@testset "Cophenetic and collinear" begin
    m = linkage_matrix(PTS, "single")
    coph = cophenetic_distances(m)
    root = EXPECTED["single"][2]
    @test coph[1, 2] ≈ 1.0
    @test coph[4, 5] ≈ 1.0
    @test coph[1, 4] ≈ root
    @test coph[1, 1] == 0.0

    c = linkage_matrix([[0.0], [0.0], [5.0]], "single")
    @test Int(c[1][1]) == 0 && Int(c[1][2]) == 1
    @test c[1][3] ≈ 0.0
    @test c[2][3] ≈ 5.0
    @test c[2][4] ≈ 3.0
end

@testset "Edge cases" begin
    @test_throws ArgumentError linkage_matrix(PTS, "centroid")
    @test_throws ArgumentError linkage_matrix([[1.0, 2.0]], "single")
    @test_throws ArgumentError linkage_matrix(Vector{Vector{Float64}}(), "single")
    @test_throws ArgumentError linkage_matrix([[0.0, 0.0], [1.0]], "single")
    m = linkage_matrix(PTS, "ward")
    @test_throws ArgumentError fcluster(m, 0)
    @test_throws ArgumentError fcluster(m, 6)
end
