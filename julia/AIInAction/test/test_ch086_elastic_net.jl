using Test
using AIInAction.Ch086ElasticNet

# Shared fixtures: identical to the Python and Rust test suites.
const X86 = [
    1.0 2.0
    2.0 1.0
    3.0 4.0
    4.0 3.0
    5.0 6.0
]
const Y86 = [2.0, 3.0, 5.0, 7.0, 8.0]

const EN_COEF = [1.1076803723827306, 0.22885958106994972]
const EN_INTERCEPT = 0.9446082234279691
const EN_PRED = [
    2.510007757950599,
    3.3888285492633803,
    5.18308766485596,
    6.061908456168741,
    7.856167571761321,
]
const LASSO_COEF = [1.1, 0.0]
const LASSO_INTERCEPT = 1.7
const RIDGE_COEF = [0.795939086294827, 0.4060913705581682]
const RIDGE_INTERCEPT = 1.312690355329381
const OLS_COEF = [1.6, 0.0]
const OLS_INTERCEPT = 0.2

@testset "Ch086 soft threshold" begin
    @test soft_threshold(3.0, 1.0) ≈ 2.0
    @test soft_threshold(-3.0, 1.0) ≈ -2.0
    @test soft_threshold(0.5, 1.0) == 0.0
    @test_throws ArgumentError soft_threshold(1.0, -0.5)
end

@testset "Ch086 Elastic Net parity fixtures" begin
    coef, b0 = elastic_net_fit(X86, Y86, 0.5; alpha = 0.5, max_iter = 10000, tol = 1e-12)
    @test coef ≈ EN_COEF atol = 1e-9
    @test b0 ≈ EN_INTERCEPT atol = 1e-9
    @test elastic_net_predict(X86, EN_COEF, EN_INTERCEPT) ≈ EN_PRED atol = 1e-9
end

@testset "Ch086 limiting cases" begin
    lc, lb = elastic_net_fit(X86, Y86, 1.0; alpha = 1.0, max_iter = 10000, tol = 1e-12)
    @test lc ≈ LASSO_COEF atol = 1e-9
    @test lb ≈ LASSO_INTERCEPT atol = 1e-9
    @test lc[2] == 0.0

    rc, rb = elastic_net_fit(X86, Y86, 1.0; alpha = 0.0, max_iter = 10000, tol = 1e-12)
    @test rc ≈ RIDGE_COEF atol = 1e-9
    @test rb ≈ RIDGE_INTERCEPT atol = 1e-9

    oc, ob = elastic_net_fit(X86, Y86, 0.0; alpha = 0.5, max_iter = 10000, tol = 1e-12)
    @test oc ≈ OLS_COEF atol = 1e-7
    @test ob ≈ OLS_INTERCEPT atol = 1e-7
end

@testset "Ch086 validation" begin
    @test_throws ArgumentError elastic_net_fit(reshape([1.0, 2.0], 2, 1), [1.0], 0.1)
    @test_throws ArgumentError elastic_net_fit(X86, Y86, 0.1; alpha = 1.5)
    @test_throws ArgumentError elastic_net_fit(X86, Y86, -1.0)
    @test_throws ArgumentError elastic_net_predict(X86, [1.0], 0.0)
end
