using Test
using AIInAction.Ch169Bootstrap

# Shared fixtures: identical to the Python and Rust test suites.
const CH169_DATA = [4.0, 8.0, 15.0, 16.0, 23.0, 42.0, 1.0, 9.0]
const CH169_SEED = 12345
const CH169_N = 500

@testset "Ch169 bootstrap parity fixtures" begin
    rp = bootstrap_mean_ci(CH169_DATA; n_resamples=CH169_N, alpha=0.025,
        method="percentile", seed=CH169_SEED)
    @test rp.estimate ≈ 14.75
    @test rp.standard_error ≈ 4.401794815207815
    @test rp.replicates[1:5] ≈ [13.25, 24.125, 16.0, 17.125, 10.375]
    @test sum(rp.replicates) ≈ 7401.0
    @test rp.ci_low ≈ 7.434375
    @test rp.ci_high ≈ 24.506249999999994

    rb = bootstrap_mean_ci(CH169_DATA; n_resamples=CH169_N, alpha=0.025,
        method="bca", seed=CH169_SEED)
    @test rb.ci_low ≈ 8.55046123816017
    @test rb.ci_high ≈ 26.839660615922583
end

@testset "Ch169 special functions" begin
    @test norm_ppf(0.975) ≈ 1.959963986120195 atol = 1e-9
    @test norm_ppf(0.025) ≈ -norm_ppf(0.975) atol = 1e-12
    # norm_cdf is built on the ~1.2e-7-accurate erf shared by all three languages.
    @test norm_cdf(1.0) ≈ 0.8413447386043253 atol = 1e-12
    @test norm_ppf(0.95) ≈ 1.6448536269514722 atol = 1e-8
    for p in (0.01, 0.1, 0.3, 0.5, 0.8, 0.99)
        @test norm_cdf(norm_ppf(p)) ≈ p atol = 1e-7
    end
    @test quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.25) ≈ 2.0
    @test quantile([10.0, 20.0, 30.0], 0.0) == 10.0
    @test quantile([10.0, 20.0, 30.0], 1.0) == 30.0
end

@testset "Ch169 properties and edge cases" begin
    a = bootstrap_mean_ci(CH169_DATA; n_resamples=CH169_N, seed=CH169_SEED, method="percentile")
    b = bootstrap_mean_ci(CH169_DATA; n_resamples=CH169_N, seed=CH169_SEED, method="percentile")
    @test a.replicates == b.replicates

    rb = bootstrap_mean_ci(CH169_DATA; n_resamples=CH169_N, seed=CH169_SEED, method="bca")
    @test rb.ci_low < rb.estimate < rb.ci_high

    rc = bootstrap_mean_ci([5.0, 5.0, 5.0, 5.0]; n_resamples=100, seed=7, method="percentile")
    @test rc.estimate ≈ 5.0
    @test rc.standard_error ≈ 0.0
    @test rc.ci_low ≈ 5.0
    @test rc.ci_high ≈ 5.0

    @test_throws ArgumentError bootstrap_mean_ci([1.0])
    @test_throws ArgumentError bootstrap_mean_ci([1.0, 2.0, NaN])
    @test_throws ArgumentError bootstrap_mean_ci(CH169_DATA; method="studentized")
    @test_throws ArgumentError bootstrap_mean_ci(CH169_DATA; alpha=0.5)
    @test_throws ArgumentError bootstrap_mean_ci(CH169_DATA; n_resamples=0)
    @test_throws ArgumentError bootstrap_mean_ci(CH169_DATA; seed=-1)
    @test_throws ArgumentError norm_ppf(0.0)
    @test_throws ArgumentError norm_ppf(1.0)
end
