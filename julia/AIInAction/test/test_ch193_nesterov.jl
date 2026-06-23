using Test
using AIInAction.Ch193Nesterov

# Shared fixtures: identical to the Python and Rust test suites.
# Quadratic f(x) = 0.5 x^T A x - b^T x, grad f(x) = A x - b, with
#   A = [[5, 1], [1, 3]],  b = [1, 2],  x* = [1/14, 9/14].
const CH193_A = [5.0 1.0; 1.0 3.0]
const CH193_B = [1.0, 2.0]
const CH193_ETA = 1.0 / 6.0
const CH193_BETA = 0.5
const CH193_XSTAR = [0.07142857142857142, 0.6428571428571429]

ch193_grad(x) = CH193_A * Vector{Float64}(x) .- CH193_B

const CH193_CONVEX5_X = [0.06917044734055175, 0.6483203299202533]
const CH193_CONVEX5_GN = 0.015285826582542654
const CH193_CONVEX5_HIST = [
    0.8498365855987974,
    0.4746668747398631,
    0.2123582666221123,
    0.05611771750996081,
    0.015285826582542654,
]

const CH193_MOM5_X = [0.061631944444444475, 0.666087962962963]
const CH193_MOM5_GN = 0.06519733559752104
const CH193_MOM5_HIST = [
    0.8498365855987974,
    0.3004626062886657,
    0.021960261528947072,
    0.07158169488919522,
    0.06519733559752104,
]

@testset "Ch193 Nesterov parity fixtures" begin
    rc = nesterov_convex(ch193_grad, [0.0, 0.0], CH193_ETA; max_iter=5, tol=0.0)
    @test rc.n_iter == 5
    @test rc.converged == false
    @test rc.x ≈ CH193_CONVEX5_X
    @test rc.grad_norm ≈ CH193_CONVEX5_GN
    @test rc.history ≈ CH193_CONVEX5_HIST

    rm = nesterov_momentum(ch193_grad, [0.0, 0.0], CH193_ETA, CH193_BETA; max_iter=5, tol=0.0)
    @test rm.n_iter == 5
    @test rm.converged == false
    @test rm.x ≈ CH193_MOM5_X
    @test rm.grad_norm ≈ CH193_MOM5_GN
    @test rm.history ≈ CH193_MOM5_HIST
end

@testset "Ch193 Nesterov convergence and edge cases" begin
    rc = nesterov_convex(ch193_grad, [0.0, 0.0], CH193_ETA; max_iter=10000, tol=1e-10)
    @test rc.converged
    @test rc.x ≈ CH193_XSTAR atol = 1e-7
    @test rc.grad_norm <= 1e-10

    rm = nesterov_momentum(ch193_grad, [0.0, 0.0], CH193_ETA, CH193_BETA; max_iter=10000, tol=1e-10)
    @test rm.converged
    @test rm.x ≈ CH193_XSTAR atol = 1e-7
    @test rm.grad_norm <= 1e-10

    # Starting at the optimum stops at iteration 1.
    ropt = nesterov_convex(ch193_grad, CH193_XSTAR, CH193_ETA; max_iter=100, tol=1e-9)
    @test ropt.converged
    @test ropt.n_iter == 1

    # beta = 0 reduces to plain gradient descent.
    r0 = nesterov_momentum(ch193_grad, [0.0, 0.0], CH193_ETA, 0.0; max_iter=1, tol=0.0)
    @test r0.x ≈ (-CH193_ETA .* ch193_grad([0.0, 0.0]))

    rh = nesterov_convex(ch193_grad, [0.0, 0.0], CH193_ETA; max_iter=7, tol=0.0)
    @test length(rh.history) == rh.n_iter == 7

    @test_throws ArgumentError nesterov_convex(ch193_grad, [0.0, 0.0], 0.0)
    @test_throws ArgumentError nesterov_momentum(ch193_grad, [0.0, 0.0], CH193_ETA, 1.0)
    @test_throws ArgumentError nesterov_momentum(ch193_grad, [0.0, 0.0], CH193_ETA, -0.1)
    @test_throws ArgumentError nesterov_convex(ch193_grad, [0.0, 0.0], CH193_ETA; max_iter=0)
    @test_throws ArgumentError nesterov_convex(ch193_grad, Float64[], CH193_ETA)
    @test_throws ArgumentError nesterov_convex(x -> [0.0], [0.0, 0.0], CH193_ETA)
end
