using Test
using AIInAction.Ch197Adamw

# Shared fixtures: identical to the Python and Rust test suites.
const ADAMW_STEP_PARAMS = [1.0, -2.0, 0.5]
const ADAMW_STEP_GRAD = [0.5, -1.0, 2.0]
const ADAMW_STEP_CONFIG = AdamWConfig(; lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8,
                                       weight_decay=0.01)

const ADAMW_EXPECTED_THETA = [0.899000002, -1.898000001, 0.3995000005]
const ADAMW_EXPECTED_M = [0.04999999999999999, -0.09999999999999998, 0.19999999999999996]
const ADAMW_EXPECTED_V = [0.0002500000000000002, 0.0010000000000000009, 0.0040000000000000036]
const ADAMW_QUAD_X200 = [1.999943097328645, -1.000007218001001]
const ADAMW_QUAD_WD50 = [1.7895443334390244, 0.604294026757625]

# Diagonal quadratic: grad(x) = A .* (x .- target).
const ADAMW_QUAD_A = [3.0, 1.0]
const ADAMW_QUAD_TARGET = [2.0, -1.0]
adamw_quad_grad(x) = ADAMW_QUAD_A .* (x .- ADAMW_QUAD_TARGET)

@testset "Ch197 AdamW parity fixtures" begin
    st = init_state(3)
    theta = adamw_step!(ADAMW_STEP_PARAMS, ADAMW_STEP_GRAD, st, ADAMW_STEP_CONFIG)
    @test theta ≈ ADAMW_EXPECTED_THETA
    @test st.m ≈ ADAMW_EXPECTED_M
    @test st.v ≈ ADAMW_EXPECTED_V
    @test st.t == 1

    x200 = minimize(adamw_quad_grad, [0.0, 0.0], AdamWConfig(; lr=0.1), 200)
    @test x200 ≈ ADAMW_QUAD_X200

    xwd = minimize(adamw_quad_grad, [5.0, 5.0],
                   AdamWConfig(; lr=0.1, weight_decay=0.01), 50)
    @test xwd ≈ ADAMW_QUAD_WD50
end

@testset "Ch197 AdamW properties and edge cases" begin
    x = minimize(adamw_quad_grad, [0.0, 0.0], AdamWConfig(; lr=0.1), 500)
    @test x ≈ ADAMW_QUAD_TARGET atol = 1e-4

    # With gradient exactly zero, only decoupled decay acts.
    stz = init_state(2)
    thz = adamw_step!([4.0, -6.0], [0.0, 0.0], stz,
                      AdamWConfig(; lr=0.1, weight_decay=0.2))
    @test thz ≈ [4.0 * (1 - 0.02), -6.0 * (1 - 0.02)]

    # t increments on each step.
    sti = init_state(2)
    p = [1.0, 1.0]
    for expected_t in 1:3
        p = adamw_step!(p, [0.1, 0.1], sti, AdamWConfig(; lr=0.1))
        @test sti.t == expected_t
    end

    @test_throws ArgumentError AdamWConfig(; lr=0.0)
    @test_throws ArgumentError AdamWConfig(; beta1=1.0)
    @test_throws ArgumentError AdamWConfig(; beta2=-0.1)
    @test_throws ArgumentError AdamWConfig(; eps=0.0)
    @test_throws ArgumentError AdamWConfig(; weight_decay=-0.01)
    @test_throws ArgumentError init_state(0)
    @test_throws ArgumentError minimize(adamw_quad_grad, [0.0, 0.0], AdamWConfig(), 0)

    stm = init_state(3)
    @test_throws ArgumentError adamw_step!([1.0, 2.0, 3.0], [0.1, 0.2], stm, AdamWConfig())
    stn = init_state(2)
    @test_throws ArgumentError adamw_step!([1.0, 2.0], [NaN, 0.1], stn, AdamWConfig())
end
