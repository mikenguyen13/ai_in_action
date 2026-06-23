using Test

# Allow standalone execution as well as inclusion from runtests.jl.
if !isdefined(Main, :Ch133ClusteringAtScale)
    include(joinpath(@__DIR__, "..", "src", "ch133_clustering_at_scale.jl"))
    using .Ch133ClusteringAtScale
end

# Shared fixtures: identical to the Python and Rust suites (1e-9 tolerance).
const POINTS = [
    [0.0, 0.0], [0.2, 0.1], [0.1, 0.2],
    [5.0, 5.0], [5.2, 4.9], [4.9, 5.1],
    [0.0, 5.0], [0.1, 4.8], [-0.1, 5.2],
]
const INIT = [[1.0, 1.0], [4.0, 4.0], [1.0, 4.0]]
const TOL = 1e-9

@testset "Ch133 RNG and geometry" begin
    @test next_below!(Lcg(7), 10) == 8
    @test squared_distance([0.0, 0.0], [3.0, 4.0]) == 25.0
    j, d = nearest_centroid([0.1, 0.1], INIT)
    @test j == 1                      # 1-based: first centroid
    @test isapprox(d, 1.62; atol=TOL)
    @test_throws ArgumentError squared_distance([1.0], [1.0, 2.0])
end

@testset "Ch133 clustering feature" begin
    cf = cf_from_points(POINTS)
    @test cf.n == 9
    @test isapprox(cf.ls[1], 15.4; atol=TOL)
    @test isapprox(cf.ls[2], 30.299999999999997; atol=TOL)
    @test isapprox(cf.ss, 226.27000000000004; atol=TOL)
    @test isapprox(centroid(cf), [1.7111111111111112, 3.3666666666666663]; atol=TOL)
    @test isapprox(radius(cf), 3.29829735349904; atol=TOL)

    left = cf_from_points(POINTS[1:3])
    right = cf_from_points(POINTS[4:end])
    merged = merge_cf(left, right)
    @test merged.n == cf.n
    @test isapprox(merged.ls, cf.ls; atol=TOL)
    @test isapprox(merged.ss, cf.ss; atol=TOL)

    @test radius(cf_from_points([[2.0, 3.0]])) == 0.0
    @test_throws ArgumentError centroid(ClusteringFeature(0, [0.0, 0.0], 0.0))
end

@testset "Ch133 mini-batch k-means" begin
    out = mini_batch_kmeans(POINTS, INIT; batch_size=4, n_iter=20, seed=42)
    expected = [
        [0.09999999999999999, 0.10769230769230771],
        [5.042307692307693, 5.000000000000001],
        [-0.0035714285714285696, 5.007142857142858],
    ]
    for (row, exp) in zip(out, expected)
        @test isapprox(row, exp; atol=TOL)
    end
    @test isapprox(inertia(POINTS, out), 0.20727712534718032; atol=TOL)
    @test mini_batch_kmeans(POINTS, INIT; batch_size=4, n_iter=0, seed=1) == INIT
    @test_throws ArgumentError mini_batch_kmeans(POINTS, INIT; batch_size=0, n_iter=1, seed=1)
end

@testset "Ch133 canopy clustering" begin
    out = canopy_clustering(POINTS; t1=2.0, t2=1.0, seed=7)
    @test out == [[3, 4, 5], [6, 7, 8], [0, 1, 2]]
    covered = Set{Int}()
    for c in out
        union!(covered, c)
    end
    @test covered == Set(0:(length(POINTS) - 1))
    @test_throws ArgumentError canopy_clustering(POINTS; t1=1.0, t2=2.0, seed=7)
end

@testset "Ch133 k-means|| seeding" begin
    seeds = kmeans_parallel_init(POINTS, 3; oversampling=2.0, n_rounds=3, seed=123)
    expected = [[0.2, 0.1], [5.2, 4.9], [-0.1, 5.2]]
    @test length(seeds) == 3
    for (row, exp) in zip(seeds, expected)
        @test isapprox(row, exp; atol=TOL)
    end
    @test_throws ArgumentError kmeans_parallel_init(POINTS, 99; oversampling=2.0, n_rounds=1, seed=1)
end
