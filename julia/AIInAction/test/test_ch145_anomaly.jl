using Test
using AIInAction.Ch145Anomaly

# Shared fixtures: identical to the Python and Rust test suites.
const CH145_XZ = [10.0, 12.0, 11.0, 13.0, 9.0, 11.5, 40.0]

const CH145_XMAHAL = [
    2.0 1.0
    3.0 2.0
    4.0 2.5
    5.0 4.0
    6.0 5.0
    2.5 6.0
]
const CH145_PT = [10.0 1.0]

const CH145_XKDE = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]
const CH145_QKDE = [2.5, 10.0]

const CH145_XGRUBBS = [1.0, 2.0, 1.5, 1.8, 2.2, 5.0]

@testset "Ch145 anomaly parity fixtures" begin
    # z-score
    z = zscores(CH145_XZ)
    @test z[1] ≈ -0.4737231192137665
    @test z[7] ≈ 2.251807155714753
    @test zscore_flags(CH145_XZ; threshold=2.0) == [false, false, false, false, false, false, true]

    # Mahalanobis
    d2 = mahalanobis_sq(CH145_XMAHAL)
    @test d2[1] ≈ 2.051080282894225
    @test d2[6] ≈ 4.120035750369162
    @test mahalanobis_sq(CH145_XMAHAL, CH145_PT)[1] ≈ 27.01727286857853

    # KDE
    dens = gaussian_kde(CH145_XKDE, CH145_QKDE; bandwidth=1.0)
    @test dens[1] ≈ 0.22915412139731456
    @test dens[2] ≈ 0.056991755250522066
    sc = kde_scores(CH145_XKDE, CH145_QKDE; bandwidth=1.0)
    @test sc[2] ≈ 2.8648486663373274
    @test sc[2] > sc[1]

    # Grubbs (index is 1-based in Julia: the outlier 5.0 is at position 6)
    r = grubbs_test(CH145_XGRUBBS; alpha=0.05)
    @test r.statistic ≈ 1.9489336934427666
    @test r.critical_value ≈ 1.887145117783933
    @test r.index == 6
    @test r.is_outlier == true

    # Special functions
    @test chi2_ppf(0.95, 2) ≈ 5.991464547107979
    @test chi2_ppf(0.99, 3) ≈ 11.344866730144357
    @test student_t_ppf(0.975, 10) ≈ 2.228138851986274
end

@testset "Ch145 anomaly properties and edge cases" begin
    # z-scores are centered.
    @test sum(zscores(CH145_XZ)) ≈ 0.0 atol = 1e-12

    # Mean squared Mahalanobis over the n training rows equals d*(n-1)/n.
    d2 = mahalanobis_sq(CH145_XMAHAL)
    @test sum(d2) / 6 ≈ 2 * 5 / 6

    # Grubbs accepts when there is no outlier.
    @test grubbs_test([1.0, 2.0, 3.0, 4.0, 5.0]; alpha=0.05).is_outlier == false

    @test_throws ArgumentError zscores([5.0, 5.0, 5.0])
    @test_throws ArgumentError mahalanobis_sq([1.0 1.0; 2.0 2.0; 3.0 3.0])
    @test_throws ArgumentError gaussian_kde(CH145_XKDE, CH145_QKDE; bandwidth=-1.0)
    @test_throws ArgumentError grubbs_test([1.0, 2.0])
    @test_throws ArgumentError chi2_ppf(1.5, 2)
end
