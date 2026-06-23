using Test
using AIInAction.Ch164ClusteringMetrics

# Shared fixtures: identical to the Python and Rust test suites. All top-level
# consts are prefixed CM164_ so they cannot collide with other test files.
const CM164_LT = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
const CM164_LP = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
const CM164_TOL = 1e-9

@testset "Ch164 clustering metrics parity fixtures" begin
    @test contingency_matrix(CM164_LT, CM164_LP) == [2 2 0 0; 0 1 2 0; 0 0 0 3]

    @test entropy(CM164_LT) ≈ 1.0888999753452238 atol = CM164_TOL
    @test entropy(CM164_LP) ≈ 1.366158847569202 atol = CM164_TOL

    @test mutual_information(CM164_LT, CM164_LP) ≈ 0.8979457248567799 atol = CM164_TOL
    @test mutual_information(CM164_LT, CM164_LP) ≈ mutual_information(CM164_LP, CM164_LT) atol = CM164_TOL

    @test normalized_mutual_information(CM164_LT, CM164_LP; average_method=:arithmetic) ≈ 0.7315064848758445 atol = CM164_TOL
    @test normalized_mutual_information(CM164_LT, CM164_LP; average_method=:geometric) ≈ 0.7362164101692431 atol = CM164_TOL
    @test normalized_mutual_information(CM164_LT, CM164_LP; average_method=:min) ≈ 0.8246356370538958 atol = CM164_TOL
    @test normalized_mutual_information(CM164_LT, CM164_LP; average_method=:max) ≈ 0.6572776851348504 atol = CM164_TOL

    @test homogeneity(CM164_LT, CM164_LP) ≈ 0.8246356370538958 atol = CM164_TOL
    @test completeness(CM164_LT, CM164_LP) ≈ 0.6572776851348504 atol = CM164_TOL

    @test v_measure(CM164_LT, CM164_LP) ≈ 0.7315064848758445 atol = CM164_TOL
    @test v_measure(CM164_LT, CM164_LP; beta=2.0) ≈ 0.7049682606092936 atol = CM164_TOL

    @test fowlkes_mallows_index(CM164_LT, CM164_LP) ≈ 0.6123724356957946 atol = CM164_TOL
end

@testset "Ch164 clustering metrics properties and edge cases" begin
    # V-measure is the harmonic mean of homogeneity and completeness.
    h = homogeneity(CM164_LT, CM164_LP)
    c = completeness(CM164_LT, CM164_LP)
    @test v_measure(CM164_LT, CM164_LP) ≈ 2h * c / (h + c) atol = CM164_TOL

    # Perfect agreement up to relabeling.
    a = [0, 0, 1, 1, 2, 2]
    b = [2, 2, 0, 0, 1, 1]
    @test mutual_information(a, b) ≈ entropy(a) atol = CM164_TOL
    @test normalized_mutual_information(a, b) ≈ 1.0 atol = CM164_TOL
    @test homogeneity(a, b) ≈ 1.0 atol = CM164_TOL
    @test completeness(a, b) ≈ 1.0 atol = CM164_TOL
    @test v_measure(a, b) ≈ 1.0 atol = CM164_TOL
    @test fowlkes_mallows_index(a, b) ≈ 1.0 atol = CM164_TOL

    # Independent partitions: zero mutual information.
    ai = [0, 0, 1, 1]
    bi = [0, 1, 0, 1]
    @test mutual_information(ai, bi) ≈ 0.0 atol = CM164_TOL
    @test normalized_mutual_information(ai, bi) ≈ 0.0 atol = CM164_TOL
    @test v_measure(ai, bi) ≈ 0.0 atol = CM164_TOL

    # Single-cluster-each degenerate case is perfect by convention.
    s = [0, 0, 0]
    @test entropy(s) == 0.0
    @test normalized_mutual_information(s, s) == 1.0
    @test homogeneity(s, s) == 1.0
    @test completeness(s, s) == 1.0

    # All singletons: no co-clustered pair, FM is zero.
    @test fowlkes_mallows_index([0, 1, 2, 3], [3, 2, 1, 0]) == 0.0

    @test_throws ArgumentError mutual_information([0, 0, 1], [0, 1])
    @test_throws ArgumentError entropy(Int[])
    @test_throws ArgumentError normalized_mutual_information(CM164_LT, CM164_LP; average_method=:harmonic)
    @test_throws ArgumentError v_measure(CM164_LT, CM164_LP; beta=-1.0)
end
