using Test
using AIInAction.Ch192Momentum

# Shared fixtures: identical to the Python and Rust test suites. Every top-level
# const is prefixed with MOM192 so it cannot collide with other test files.
const MOM192_STEP_THETA = [1.0, 2.0]
const MOM192_STEP_VEL = [0.5, -0.5]
const MOM192_STEP_GRAD = [0.2, 0.4]
const MOM192_EXP_STEP_THETA = [0.935, 2.005]
const MOM192_EXP_STEP_VEL = [0.65, -0.04999999999999999]

const MOM192_QUAD_H = [3.0 0.2; 0.2 1.0]
const MOM192_QUAD_B = [1.0, -2.0]
const MOM192_THETA0 = [0.0, 0.0]

const MOM192_EXP_5ITER_THETA = [1.2893499712, -3.1595351615999996]
const MOM192_EXP_5ITER_VEL = [1.884893344, 3.068733408]
const MOM192_EXP_5ITER_HIST = [
    3.1622776601683795,
    1.90275589606234,
    1.339506583783745,
    2.072914156257514,
    1.9343282392460306,
]

const MOM192_EXP_OPT_BETA = 0.6694214876033059

@testset "Ch192 momentum parity fixtures" begin
    th, v = momentum_step(MOM192_STEP_THETA, MOM192_STEP_VEL, MOM192_STEP_GRAD, 0.1, 0.9)
    @test th ≈ MOM192_EXP_STEP_THETA
    @test v ≈ MOM192_EXP_STEP_VEL

    g = quadratic_gradient(MOM192_QUAD_H, MOM192_QUAD_B)
    r = minimize(g, MOM192_THETA0; alpha=0.2, beta=0.9, max_iter=5, tol=0.0)
    @test r.n_iter == 5
    @test r.converged == false
    @test r.theta ≈ MOM192_EXP_5ITER_THETA
    @test r.velocity ≈ MOM192_EXP_5ITER_VEL
    @test r.history ≈ MOM192_EXP_5ITER_HIST

    @test optimal_beta(1.0, 100.0) ≈ MOM192_EXP_OPT_BETA
end

@testset "Ch192 momentum properties and edge cases" begin
    # beta = 0 is plain gradient descent on a single step.
    th0, v0 = momentum_step([1.0], [7.0], [2.0], 0.1, 0.0)
    @test v0 ≈ [2.0]
    @test th0 ≈ [0.8]

    g = quadratic_gradient(MOM192_QUAD_H, MOM192_QUAD_B)
    rc = minimize(g, MOM192_THETA0; alpha=0.2, beta=0.9, max_iter=5000, tol=1e-10)
    @test rc.converged == true
    @test rc.theta ≈ MOM192_QUAD_B atol = 1e-7
    @test rc.grad_norm <= 1e-10

    # beta = 0 recovers the gradient-descent minimum.
    gg = quadratic_gradient([2.0][:, :], [5.0])
    rg = minimize(gg, [0.0]; alpha=0.3, beta=0.0, max_iter=500, tol=1e-12)
    @test rg.converged == true
    @test rg.theta ≈ [5.0]

    # Momentum needs fewer steps than plain GD on an ill-conditioned quadratic.
    Hbad = [50.0 0.0; 0.0 1.0]
    gb = quadratic_gradient(Hbad, [0.0, 0.0])
    plain = minimize(gb, [1.0, 1.0]; alpha=0.02, beta=0.0, max_iter=10000, tol=1e-6)
    fast = minimize(gb, [1.0, 1.0]; alpha=0.02, beta=0.9, max_iter=10000, tol=1e-6)
    @test plain.converged && fast.converged
    @test fast.n_iter < plain.n_iter

    @test optimal_beta(2.0, 2.0) ≈ 0.0 atol = 1e-15

    @test_throws ArgumentError minimize(g, MOM192_THETA0; alpha=0.0, beta=0.5)
    @test_throws ArgumentError minimize(g, MOM192_THETA0; alpha=0.1, beta=1.0)
    @test_throws ArgumentError momentum_step([1.0], [0.0], [1.0], 0.1, -0.1)
    @test_throws ArgumentError momentum_step([1.0, 2.0], [0.0], [1.0, 1.0], 0.1, 0.5)
    @test_throws ArgumentError minimize(g, MOM192_THETA0; alpha=0.1, beta=0.5, max_iter=0)
    @test_throws ArgumentError quadratic_gradient([1.0 0.0; 0.0 1.0], [1.0])
    @test_throws ArgumentError optimal_beta(10.0, 1.0)
end
