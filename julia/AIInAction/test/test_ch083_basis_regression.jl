using Test
using AIInAction.Ch083BasisRegression

# Shared fixtures: identical to the Python and Rust test suites. The polynomial
# data lie exactly on y = 1 + 2x - x^2, so degree-2 OLS recovers [1, 2, -1].
const X083 = [-2.0, -1.0, 0.0, 1.0, 2.0]
const Y083 = [-7.0, -2.0, 1.0, 2.0, 1.0]

const EXPECTED_OLS = [1.0, 2.0, -1.0]
const EXPECTED_RIDGE = [0.59090909090909061, 1.8181818181818181, -0.8545454545454545]
const EXPECTED_DOF_OLS = 3.0
const EXPECTED_DOF_RIDGE = 2.5363636363636362

@testset "Ch083 polynomial regression parity fixtures" begin
    phi = polynomial_design(X083, 2)
    @test size(phi) == (5, 3)
    @test phi[1, :] ≈ [1.0, -2.0, 4.0]

    beta = fit_ridge(phi, Y083, 0.0)
    @test beta ≈ EXPECTED_OLS atol = 1e-9
    @test predict(phi, beta) ≈ Y083 atol = 1e-9

    betar = fit_ridge(phi, Y083, 1.0)
    @test betar ≈ EXPECTED_RIDGE atol = 1e-9

    @test effective_dof(phi, 0.0) ≈ EXPECTED_DOF_OLS atol = 1e-9
    @test effective_dof(phi, 1.0) ≈ EXPECTED_DOF_RIDGE atol = 1e-9
end

@testset "Ch083 RBF basis" begin
    phi = rbf_design([-1.0, 0.0, 1.0], [-1.0, 1.0], 1.0; include_bias=true)
    @test size(phi) == (3, 3)
    @test phi[1, 3] ≈ 0.1353352832366127 atol = 1e-12
    @test phi[2, 2] ≈ 0.6065306597126334 atol = 1e-12
end

@testset "Ch083 estimator and edge cases" begin
    model = BasisRegression(degree=2, penalty=0.0)
    Ch083BasisRegression.fit!(model, X083, Y083)
    @test model.coef ≈ EXPECTED_OLS atol = 1e-9
    @test predict(model, X083) ≈ Y083 atol = 1e-9
    @test predict(model, [3.0])[1] ≈ (1.0 + 2.0 * 3.0 - 3.0^2) atol = 1e-9

    @test_throws ArgumentError polynomial_design(X083, -1)
    @test_throws ArgumentError polynomial_design(Float64[], 2)
    @test_throws ArgumentError fit_ridge(polynomial_design(X083, 2), [1.0, 2.0], 0.0)
    @test_throws ArgumentError fit_ridge(polynomial_design(X083, 2), Y083, -0.5)
    @test_throws ArgumentError rbf_design(X083, [-1.0, 1.0], 0.0)
    @test_throws ArgumentError predict(BasisRegression(degree=2), X083)
    @test_throws ArgumentError Ch083BasisRegression.fit!(BasisRegression(basis="spline"), X083, Y083)
end
