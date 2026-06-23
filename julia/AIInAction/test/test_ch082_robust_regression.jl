using Test
using AIInAction.Ch082RobustRegression

# Shared fixtures: identical to the Python and Rust test suites.
const X_HUBER = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]
const Y_HUBER = [1.0, 3.0, 5.0, 20.0, 9.0, 11.0]
const OLS_HUBER_COEF = [2.2380952380952381, 2.3714285714285714]

const X_WLS = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
const Y_WLS = [1.0, 3.0, 5.0, 7.0]
const W_WLS = [1.0, 2.0, 3.0, 4.0]

const X_GLS = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
const Y_GLS = [1.0, 2.0, 2.5]
const COV_GLS = [[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]]
const GLS_COEF = [1.05, 0.75]

const X_Q = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
const Y_Q = [1.0, 2.0, 3.0, 4.0, 50.0]

@testset "Ch082 primitives" begin
    @test solve_linear([[2.0, 0.0], [0.0, 4.0]], [2.0, 8.0]) ≈ [1.0, 2.0]
    @test predict([[1.0, 2.0], [1.0, 3.0]], [1.0, 1.0]) ≈ [3.0, 4.0]
    @test vandermonde([0.0, 2.0], 2) == [[1.0, 0.0, 0.0], [1.0, 2.0, 4.0]]
    @test_throws ArgumentError solve_linear([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])
end

@testset "Ch082 parity fixtures" begin
    @test fit_ols(X_HUBER, Y_HUBER) ≈ OLS_HUBER_COEF atol=1e-9

    res = fit_huber(X_HUBER, Y_HUBER; delta=1.345)
    @test res.coef ≈ [1.0, 2.0] atol=1e-7
    @test res.converged

    @test fit_wls(X_WLS, Y_WLS, W_WLS) ≈ [1.0, 2.0] atol=1e-9
    @test fit_gls(X_GLS, Y_GLS, COV_GLS) ≈ GLS_COEF atol=1e-9
    @test fit_quantile(X_Q, Y_Q; tau=0.5) ≈ [1.0, 1.0] atol=1e-5
end

@testset "Ch082 basis expansion" begin
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ys = [2.0 - x + 3.0 * x^2 for x in xs]
    Phi = vandermonde(xs, 2)
    @test fit_ols(Phi, ys) ≈ [2.0, -1.0, 3.0] atol=1e-9
end

@testset "Ch082 validation" begin
    @test_throws ArgumentError fit_ols([[1.0], [1.0]], [1.0])
    @test_throws ArgumentError fit_wls(X_WLS, Y_WLS, [1.0, -1.0, 1.0, 1.0])
    @test_throws ArgumentError fit_huber(X_WLS, Y_WLS; delta=0.0)
    @test_throws ArgumentError fit_quantile(X_WLS, Y_WLS; tau=1.5)
    @test_throws ArgumentError fit_gls(X_GLS, Y_GLS, [[1.0, 0.0], [0.0, 1.0]])
    @test_throws ArgumentError vandermonde([1.0, 2.0], -1)
end
