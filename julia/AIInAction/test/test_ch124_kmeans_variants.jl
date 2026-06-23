using Test
using AIInAction.Ch124KMeansVariants

# Shared fixtures: identical to the Python and Rust suites. Julia uses 1-based
# labels and 1-based medoid/cluster indices, so label vectors are the Python
# values plus one.

const X_BLOBS = [0.0 0.0; 1.0 0.5; 0.5 1.0; 8.0 8.0; 9.0 8.5; 8.5 9.0]
const C0 = [0.0 0.0; 9.0 9.0]

@testset "Ch124 K-Means variants parity fixtures" begin
    @testset "Lloyd step" begin
        labels, c = lloyd_step(X_BLOBS, C0)
        @test labels == [1, 1, 1, 2, 2, 2]
        @test c ≈ [0.5 0.5; 8.5 8.5]
        @test inertia(X_BLOBS, c) ≈ 2.0
    end

    @testset "Mini-batch update" begin
        batch = [1.0 1.0; 9.0 9.0; 2.0 0.0; 8.0 10.0]
        c, cnt = mini_batch_update([0.0 0.0; 10.0 10.0], [0.0, 0.0], batch)
        @test c ≈ [1.5 0.5; 8.5 9.5]
        @test cnt == [2.0, 2.0]
    end

    @testset "K-medians" begin
        @test kmedians_centroid([1.0 5.0; 2.0 100.0; 3.0 6.0]) ≈ [2.0, 6.0]
        @test kmedians_centroid(reshape([1.0, 2.0, 3.0, 100.0], 4, 1)) ≈ [2.0]
    end

    @testset "PAM assignment" begin
        D = [0.0 2.0 5.0 9.0; 2.0 0.0 4.0 7.0; 5.0 4.0 0.0 3.0; 9.0 7.0 3.0 0.0]
        labels, cost = pam_assign_cost(D, [1, 4])
        @test labels == [1, 1, 2, 2]
        @test cost ≈ 5.0
    end

    @testset "Kernel K-Means" begin
        XK = reshape([0.0, 0.2, 5.0, 5.2], 4, 1)
        K = rbf_kernel_matrix(XK, 0.5)
        @test K[1, 2] ≈ 0.9801986733067553
        @test K[1, 1] ≈ 1.0
        kd = kernel_assignment_distances(K, [1, 1, 2, 2], 2)
        @test kd[1, 1] ≈ 0.009900663346622318
        @test kd[1, 2] ≈ 1.990094266187928
    end

    @testset "Fuzzy c-means" begin
        XF = reshape([0.0, 1.0, 5.0, 6.0], 4, 1)
        CF = reshape([0.5, 5.5], 2, 1)
        U = fuzzy_memberships(XF, CF, 2.0)
        @test U[1, 1] ≈ 0.9918032786885246
        @test U[1, 2] ≈ 0.00819672131147541
        @test all(sum(U, dims=2) .≈ 1.0)
        C = fuzzy_centroids(XF, U, 2.0)
        @test C[1, 1] ≈ 0.4985105160463291
        @test C[2, 1] ≈ 5.501489483953671
    end

    @testset "Bisecting split" begin
        XB = [0.0 0.0; 0.5 0.3; 8.0 8.0; 8.4 7.6]
        labels, c, sse = bisecting_split(XB, [0.0 0.0; 8.0 8.0])
        @test labels == [1, 1, 2, 2]
        @test c ≈ [0.25 0.15; 8.2 7.8]
        @test sse ≈ 0.33
    end
end

@testset "Ch124 edge cases" begin
    CF = reshape([0.5, 5.5], 2, 1)
    U = fuzzy_memberships(reshape([0.5, 5.5], 2, 1), CF, 2.0)
    @test U ≈ [1.0 0.0; 0.0 1.0]

    XK = reshape([0.0, 0.2, 5.0, 5.2], 4, 1)
    K = rbf_kernel_matrix(XK, 0.5)
    kd = kernel_assignment_distances(K, [1, 1, 1, 1], 2)
    @test all(isinf, kd[:, 2])

    @test_throws ArgumentError rbf_kernel_matrix(XK, 0.0)
    @test_throws ArgumentError fuzzy_memberships(XK, reshape([0.0], 1, 1), 1.0)
    @test_throws ArgumentError lloyd_step([0.0 0.0], reshape([0.0], 1, 1))
    @test_throws ArgumentError bisecting_split([0.0 0.0], reshape([0.0], 1, 1))
end
