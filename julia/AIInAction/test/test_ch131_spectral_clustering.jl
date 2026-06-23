using Test
using AIInAction.Ch131SpectralClustering

# Local diagonal helper (avoid a LinearAlgebra dependency in tests).
diag_of(M) = [M[i, i] for i in 1:size(M, 1)]

# Shared fixtures: identical to the Python and Rust test suites.
const X_BLOBS = [
    0.0  0.0
    0.2  0.1
    0.1 -0.2
    5.0  5.0
    5.2  4.9
    4.9  5.1
]
const SIGMA = 1.0
const SC_TOL = 1e-9
aff_w01 = exp(-((0.2)^2 + (0.1)^2) / (2.0 * SIGMA * SIGMA))

const KM_POINTS = [
    0.0   0.0
    0.1   0.0
    10.0  10.0
    10.1  9.9
]

@testset "ch131 rbf_affinity" begin
    W = rbf_affinity(X_BLOBS, SIGMA)
    @test size(W) == (6, 6)
    @test isapprox(W, W'; atol=SC_TOL)
    @test all(abs.(diag_of(W)) .< SC_TOL)
    @test isapprox(W[1, 2], aff_w01; atol=SC_TOL)
    @test W[1, 4] < 1e-9
    @test_throws ArgumentError rbf_affinity(X_BLOBS, 0.0)
    @test_throws ArgumentError rbf_affinity(X_BLOBS, -1.0)
end

@testset "ch131 normalized_laplacian" begin
    W = rbf_affinity(X_BLOBS, SIGMA)
    L = normalized_laplacian(W)
    @test all(abs.(diag_of(L) .- 1.0) .< SC_TOL)
    @test isapprox(L, L'; atol=SC_TOL)
    vals, _ = jacobi_eigh(L)
    @test abs(vals[1]) < 1e-8
    @test all(vals .>= -1e-8)
    @test_throws ArgumentError normalized_laplacian([0.0 0.0; 0.0 0.0])
    @test_throws ArgumentError normalized_laplacian([0.0 1.0; 2.0 0.0])
end

@testset "ch131 jacobi_eigh" begin
    A = [2.0 1.0; 1.0 2.0]
    vals, vecs = jacobi_eigh(A)
    @test isapprox(vals[1], 1.0; atol=SC_TOL)
    @test isapprox(vals[2], 3.0; atol=SC_TOL)
    inv_sqrt2 = 1.0 / sqrt(2.0)
    @test isapprox(vecs[1, 1], inv_sqrt2; atol=SC_TOL)
    @test isapprox(vecs[2, 1], -inv_sqrt2; atol=SC_TOL)
    B = [4.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 2.0]
    vb, _ = jacobi_eigh(B)
    @test isapprox(vb, [1.0, 2.0, 4.0]; atol=SC_TOL)
    @test_throws ArgumentError jacobi_eigh([1.0 2.0; 3.0 4.0])
end

@testset "ch131 spectral_embedding" begin
    W = rbf_affinity(X_BLOBS, SIGMA)
    U = spectral_embedding(W, 2)
    @test size(U) == (6, 2)
    for i in 1:6
        @test isapprox(sqrt(sum(abs2, U[i, :])), 1.0; atol=SC_TOL)
    end
    @test_throws ArgumentError spectral_embedding(W, 0)
    @test_throws ArgumentError spectral_embedding(W, 7)
end

@testset "ch131 kmeans" begin
    labels, centers = kmeans(KM_POINTS, 2)
    @test labels == [0, 0, 1, 1]
    @test isapprox(centers[1, :], [0.05, 0.0]; atol=SC_TOL)
    @test isapprox(centers[2, :], [10.05, 9.95]; atol=SC_TOL)
    l1, _ = kmeans(KM_POINTS, 1)
    @test l1 == [0, 0, 0, 0]
    @test_throws ArgumentError kmeans(KM_POINTS, 5)
end

@testset "ch131 spectral_clustering" begin
    labels = spectral_clustering(X_BLOBS, 2, SIGMA)
    @test labels == [0, 0, 0, 1, 1, 1]
    @test labels[1] == labels[2] == labels[3]
    @test labels[4] == labels[5] == labels[6]
    @test labels[1] != labels[4]
end
