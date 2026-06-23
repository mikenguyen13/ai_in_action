using Test
using AIInAction.Ch195Rmsprop

# Shared fixtures: identical to the Python and Rust test suites.
const CH195_PARAMS0 = [1.0, -2.0, 0.5]
const CH195_GRAD = [0.1, -0.3, 2.0]
const CH195_LR = 0.01
const CH195_BETA = 0.9
const CH195_EPS = 1e-8

const CH195_V1 = [0.0009999999999999998, 0.008999999999999998, 0.3999999999999999]
const CH195_P1 = [0.968377233398313, -1.9683772267316493, 0.4683772238983162]
const CH195_V2 = [0.0018999999999999998, 0.017099999999999997, 0.7599999999999998]
const CH195_P2 = [0.9454356652744136, -1.9454356550989789, 0.44543565077441793]
const CH195_MIN_PARAMS = [0.6027968620415073, 0.6027968532310017]
const CH195_MIN_V = [0.8446178332851308, 13.513885189461563]

@testset "Ch195 RMSProp parity fixtures" begin
    s = init_state(CH195_PARAMS0; lr=CH195_LR, beta=CH195_BETA, eps=CH195_EPS)
    @test s.v == [0.0, 0.0, 0.0]
    @test s.step_count == 0

    s1 = rmsprop_step(s, CH195_GRAD)
    @test s1.v ≈ CH195_V1
    @test s1.params ≈ CH195_P1
    @test s1.step_count == 1

    s2 = rmsprop_step(s1, CH195_GRAD)
    @test s2.v ≈ CH195_V2
    @test s2.params ≈ CH195_P2
    @test s2.step_count == 2

    cvec = [1.0, 4.0]
    sm = minimize(x -> cvec .* x, [2.0, 2.0], 10; lr=0.1, beta=CH195_BETA, eps=CH195_EPS)
    @test sm.params ≈ CH195_MIN_PARAMS
    @test sm.v ≈ CH195_MIN_V
    @test sm.step_count == 10
end

@testset "Ch195 RMSProp properties and edge cases" begin
    # beta=0: v = g^2, update is -lr*g/(|g|+eps).
    sb = init_state([5.0]; lr=0.1, beta=0.0, eps=1e-12)
    sb1 = rmsprop_step(sb, [2.0])
    @test sb1.params[1] ≈ 5.0 - 0.1 * 2.0 / (2.0 + 1e-12)

    # zero steps returns the initial params.
    cvec = [1.0, 4.0]
    s0 = minimize(x -> cvec .* x, [2.0, 2.0], 0; lr=0.1)
    @test s0.params == [2.0, 2.0]
    @test s0.step_count == 0

    # descent direction opposes the gradient sign on the first step.
    s = init_state(CH195_PARAMS0; lr=CH195_LR, beta=CH195_BETA, eps=CH195_EPS)
    s1 = rmsprop_step(s, CH195_GRAD)
    @test all(sign.(s1.params .- CH195_PARAMS0) .== .-sign.(CH195_GRAD))

    @test_throws ArgumentError rmsprop_step(s, [0.1, 0.2])
    @test_throws ArgumentError rmsprop_step(s, [0.1, Inf, 0.2])
    @test_throws ArgumentError init_state([1.0, NaN])
    @test_throws ArgumentError init_state(Float64[])
    @test_throws ArgumentError init_state(CH195_PARAMS0; lr=0.0)
    @test_throws ArgumentError init_state(CH195_PARAMS0; beta=1.0)
    @test_throws ArgumentError init_state(CH195_PARAMS0; eps=-1e-8)
    @test_throws ArgumentError minimize(x -> x, [1.0], -1)
end
