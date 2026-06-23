using Test
using AIInAction.Ch194Adagrad

# Shared fixtures: identical to the Python and Rust test suites.
const CH194_A = [1.0, 4.0, 0.25]
const CH194_B = [2.0, -1.0, 5.0]
const CH194_THETA0 = [0.0, 0.0, 0.0]
const CH194_ETA = 0.5
const CH194_EPS = 1e-8

ch194_grad(theta) = quadratic_grad(theta, CH194_A, CH194_B)

@testset "Ch194 AdaGrad parity fixtures" begin
    s = init_state(CH194_THETA0; learning_rate=CH194_ETA, epsilon=CH194_EPS)
    @test s.accumulator == [0.0, 0.0, 0.0]
    @test ch194_grad(s.theta) ≈ [-2.0, 4.0, -1.25]

    adagrad_step(s, ch194_grad(s.theta))
    @test s.theta ≈ [0.4999999975, -0.49999999875, 0.49999999600000006]
    @test s.accumulator ≈ [4.0, 16.0, 1.5625]
    @test effective_learning_rate(s) ≈ [0.24999999875, 0.1249999996875, 0.39999999680000003]

    s3 = init_state(CH194_THETA0; learning_rate=CH194_ETA, epsilon=CH194_EPS)
    for _ in 1:3
        adagrad_step(s3, ch194_grad(s3.theta))
    end
    @test s3.theta ≈ [1.0163655300219518, -0.843601262889007, 1.0977190862943662]
    @test s3.accumulator ≈ [7.690000015612001, 21.22229126752294, 3.9125960778289572]

    res = minimize(ch194_grad, CH194_THETA0; learning_rate=CH194_ETA,
        epsilon=CH194_EPS, max_iter=10000, tol=1e-9)
    @test res.converged
    @test res.theta ≈ [2.0, -1.0, 5.0] atol = 1e-6
    @test res.n_steps == 641
    @test res.grad_norm <= 1e-9
end

@testset "Ch194 AdaGrad properties and edge cases" begin
    # Constant unit gradient: G = t, so elr -> eta / (sqrt(t) + eps).
    sc = init_state([0.0]; learning_rate=1.0, epsilon=1e-8)
    for t in 1:100
        adagrad_step(sc, [1.0])
        @test effective_learning_rate(sc)[1] ≈ 1.0 / (sqrt(t) + 1e-8)
    end

    # Accumulator is monotone nondecreasing; effective rate nonincreasing.
    sm = init_state(CH194_THETA0; learning_rate=CH194_ETA, epsilon=CH194_EPS)
    prev_acc = copy(sm.accumulator)
    prev_elr = effective_learning_rate(sm)
    for _ in 1:20
        adagrad_step(sm, ch194_grad(sm.theta))
        @test all(sm.accumulator .>= prev_acc .- 1e-15)
        cur_elr = effective_learning_rate(sm)
        @test all(cur_elr .<= prev_elr .+ 1e-15)
        prev_acc = copy(sm.accumulator)
        prev_elr = cur_elr
    end

    @test quadratic_value([1.5, 0.0, 4.0], CH194_A, CH194_B) ≈
          0.5 * (1.0 * 0.25 + 4.0 * 1.0 + 0.25 * 1.0)
    @test quadratic_grad([1.5, 0.0, 4.0], CH194_A, CH194_B) ≈ [-0.5, 4.0, -0.25]

    rfull = minimize(ch194_grad, CH194_THETA0; learning_rate=CH194_ETA,
        epsilon=CH194_EPS, max_iter=5, tol=0.0)
    @test rfull.n_steps == 5
    @test !rfull.converged

    @test_throws ArgumentError init_state(CH194_THETA0; learning_rate=0.0)
    @test_throws ArgumentError init_state(CH194_THETA0; epsilon=-1e-8)
    @test_throws ArgumentError init_state(Float64[])
    @test_throws ArgumentError init_state([0.0, NaN])

    se = init_state(CH194_THETA0; learning_rate=CH194_ETA, epsilon=CH194_EPS)
    @test_throws ArgumentError adagrad_step(se, [1.0, 2.0])
    @test_throws ArgumentError adagrad_step(se, [1.0, Inf, 0.0])

    @test_throws ArgumentError minimize(ch194_grad, CH194_THETA0; max_iter=0)
    @test_throws ArgumentError minimize(ch194_grad, CH194_THETA0; tol=-1.0)
    @test_throws ArgumentError quadratic_grad([0.0, 0.0], [1.0, 0.0], [0.0, 0.0])
end
