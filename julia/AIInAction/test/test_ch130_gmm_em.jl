using Test
using AIInAction.Ch130GmmEm

# Shared fixtures: identical to the Python and Rust test suites.
const X130 = reshape(Float64[0, 1, 2, 1, 9, 10, 11, 10], 8, 1)
init130() = GMMParams([0.5, 0.5], reshape(Float64[0.0, 8.0], 2, 1),
                      [reshape([1.0], 1, 1), reshape([1.0], 1, 1)])
const TOL130 = 1e-9

@testset "Ch130 GMM EM parity fixtures" begin
    @test gaussian_pdf([0.0], [0.0], reshape([1.0], 1, 1)) ≈ 1.0 / sqrt(2pi) atol = TOL130
    @test gaussian_pdf([0.0, 0.0], [0.0, 0.0], [1.0 0.0; 0.0 1.0]) ≈ 1.0 / (2pi) atol = TOL130

    g = e_step(X130, init130())
    @test size(g) == (8, 2)
    @test all(abs.(sum(g; dims=2) .- 1.0) .< TOL130)
    @test g[1, 1] ≈ 0.9999999999999873 atol = TOL130
    @test g[1, 2] ≈ 1.2664165549094016e-14 atol = TOL130

    @test log_likelihood(X130, init130()) ≈ -24.89668559750626 atol = TOL130

    u = m_step(X130, g; reg_covar=0.0)
    @test u.weights[1] ≈ 0.5 atol = 1e-6
    @test u.means[1, 1] ≈ 1.0 atol = 1e-6
    @test u.means[2, 1] ≈ 10.0 atol = 1e-6

    r = fit_gmm(X130, init130(); max_iter=200, tol=1e-10, reg_covar=0.0)
    @test r.converged
    @test all(r.history[2:end] .>= r.history[1:end-1] .- 1e-12)
    @test r.log_likelihood ≈ -14.124096987877165 atol = 1e-7
    means = sort(r.params.means[:, 1])
    @test means[1] ≈ 1.0 atol = 1e-6
    @test means[2] ≈ 10.0 atol = 1e-6
end

@testset "Ch130 edge cases" begin
    @test_throws ArgumentError gaussian_pdf([0.0], [0.0], reshape([0.0], 1, 1))
    bad = GMMParams([0.3, 0.3], reshape(Float64[0.0, 8.0], 2, 1),
                    [reshape([1.0], 1, 1), reshape([1.0], 1, 1)])
    @test_throws ArgumentError e_step(X130, bad)
    g = e_step(X130, init130())
    @test_throws ArgumentError m_step(X130, g; reg_covar=-1.0)
    @test_throws ArgumentError fit_gmm(X130, init130(); max_iter=0)
end
