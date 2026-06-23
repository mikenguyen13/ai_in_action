using Test

# Ch143Isomap is included directly so this test file is self-contained even before
# the module is wired into AIInAction.jl's index. Once wired in, this include is
# harmless (the module is redefined identically).
include(joinpath(@__DIR__, "..", "src", "ch143_isomap.jl"))
using .Ch143Isomap

# Shared fixtures: identical to the Python and Rust test suites. Six points tracing
# an open zig-zag (an intrinsically 1-D path) embedded in the plane.
const X143 = [
    0.0 0.0
    1.0 0.6
    2.2 0.0
    3.2 0.6
    4.6 0.0
    6.2 0.6
]

const ISOMAP143_EMB1 = [
    -2.9371839938074595,
    -2.0650406540284134,
    -0.7481485843361929,
    0.42401358188469096,
    1.9113800872248041,
    3.4149795630625714,
]

@testset "Ch143 Isomap parity fixtures" begin
    d = pairwise_distances(X143)
    @test d[1, 2] ≈ 1.16619037896906
    @test d[1, 1] == 0.0
    @test d ≈ d'

    r = fit_isomap(X143; n_components=1, n_neighbors=2)
    @test r.geodesic_distances[1, 6] ≈ 6.36619037896906
    @test r.geodesic_distances[2, 5] ≈ 4.030985786641715
    @test r.embedding[:, 1] ≈ ISOMAP143_EMB1
    @test r.eigenvalues ≈ [28.946415792110297]

    r2 = fit_isomap(X143; n_components=2, n_neighbors=2)
    @test r2.eigenvalues ≈ [28.946415792110297, 0.34784148875645043]
end

@testset "Ch143 Isomap properties and edge cases" begin
    r = fit_isomap(X143; n_components=1, n_neighbors=2)
    # Embedding unfolds the path monotonically and is centered.
    y = r.embedding[:, 1]
    @test all(y[i] < y[i + 1] for i in 1:length(y)-1)
    @test sum(y) ≈ 0.0 atol = 1e-9
    # Geodesic exceeds the chord across the full path.
    @test r.geodesic_distances[1, 6] > pairwise_distances(X143)[1, 6]

    # Deterministic sign convention.
    r2 = fit_isomap(X143; n_components=2, n_neighbors=2)
    for c in 1:2
        col = r2.embedding[:, c]
        k = argmax(abs.(col))
        @test col[k] > 0.0
    end

    # k-NN graph is symmetric with a zero diagonal and the expected sparsity.
    adj = knn_graph(X143, 2)
    @test adj == adj'
    @test all(adj[i, i] == 0.0 for i in 1:size(adj, 1))
    @test isfinite(adj[1, 2])
    @test isinf(adj[1, 5])

    # Floyd-Warshall on a simple chain 1 - 2 - 3.
    chain = [0.0 1.0 Inf; 1.0 0.0 1.0; Inf 1.0 0.0]
    sp = graph_shortest_paths(chain)
    @test sp[1, 3] ≈ 2.0
    @test sp[1, 2] ≈ 1.0

    # Classical MDS reproduces exact Euclidean geometry.
    pts = reshape([0.0, 1.0, 2.0, 4.0], 4, 1)
    Dp = pairwise_distances(pts)
    Y, _ = classical_mds(Dp, 1)
    @test pairwise_distances(Y) ≈ Dp atol = 1e-9

    @test_throws ArgumentError fit_isomap([1.0 2.0])
    @test_throws ArgumentError fit_isomap(X143; n_neighbors=10)
    @test_throws ArgumentError fit_isomap(X143; n_components=7, n_neighbors=2)

    far = [0.0 0.0; 0.1 0.0; 100.0 0.0; 100.1 0.0]
    @test_throws ArgumentError fit_isomap(far; n_components=1, n_neighbors=1)
end
