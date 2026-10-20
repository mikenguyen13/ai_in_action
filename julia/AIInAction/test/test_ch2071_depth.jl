using Test
using AIInAction.Ch2071Depth

# Shared fixtures: identical to the Python and Rust test suites. Every quantity
# is deterministic (closed forms, Simpson quadrature on fixed nodes, fixed step
# RK4), so the three implementations agree to floating point tolerance.
const CH2071_TOL = 1e-9

const CH2071_GE_Z2 = 0.9999999999999986
const CH2071_GE_Z4 = 2.9999999999999942
const CH2071_GE_COS = 0.6065306597126348
const CH2071_GE_COS_N40_LIM6 = 0.6065306577125951
const CH2071_GE2_SQ_N40_LIM6 = 1.999999841083864
const CH2071_GE2_COSPROD_N20_LIM5 = 0.08206418533372323

const CH2071_LM_RELU = 0.9999999999999994
const CH2071_LM_TANH = 0.6414417355967605
const CH2071_LM_LINEAR = 0.9399999999999971
const CH2071_LM_TANH_Q2 = 1.2999393641598669
const CH2071_QSTAR_TANH = 0.41803720053418575
const CH2071_QSTAR_RELU = 0.40000000000256286
const CH2071_QSTAR_TANH_CHAOS = 2.195493928033813
const CH2071_CM_RELU = 0.6089975654758696
const CH2071_CM_TANH = 0.54974715475882
const CH2071_CM_TANH_N40 = 0.37652618700891255
const CH2071_CM_LINEAR_N40 = -0.27000003475872764
const CH2071_CHI1_TANH = 0.9386362681983815
const CH2071_CHI1_RELU = 0.9986701923986636
const CH2071_CHI1_LINEAR = 0.8999999999999972
const CH2071_XI_09 = 9.491221581029905

const CH2071_RCM_INPUTS = [-1.0, -0.5, 0.0, 0.5, 0.9, 1.0]
const CH2071_RCM = [0.0, 0.1089977810442294, 0.3183098861837907, 0.6089977810442293,
                    0.9095383988446721, 1.0]
const CH2071_TRAJ_PLAIN = [0.2, 0.4246975638632654, 0.5598220427417753, 0.6495461163619815,
                           0.7129656727390657, 0.7598110813130384]
const CH2071_TRAJ_RES = [0.2, 0.24493951277265308, 0.2860269749375514, 0.3237085998221533,
                         0.35836528648096716, 0.39032419037801597]
const CH2071_TRAJ_NEG = [-0.9, 0.009538398844672072, 0.3230935658001504, 0.49662004436592994]
const CH2071_COLLAPSE_PLAIN = 0.44413219804902115
const CH2071_COLLAPSE_RES = 11.103304951225528
const CH2071_ODE = 0.3839566908351801
const CH2071_ODE_COARSE = 0.2991227756181538
const CH2071_RV1 = 1024.0
const CH2071_RV2 = 1.3524069147107642

const CH2071_DLT_DEPTHS = [1, 2, 3, 5, 10]
const CH2071_DLT = [0.0, 0.5493061443340553, 1.0179617754504038, 2.4934025172272927,
                    32.14439124998376]
const CH2071_DLT_OPTS = 2.9016436644053982
const CH2071_DLA_DEPTHS = [3, 5, 10]
const CH2071_DLA = [2.0, 2.6666666666666665, 32.0]
const CH2071_DLA_S = 2.777777777777778

const CH2071_PLD4 = [0.0625, 0.25, 0.375, 0.25, 0.0625]
const CH2071_EPD = [0.19753086419753096, 0.3950617283950618, 0.29629629629629634,
                    0.09876543209876541, 0.012345679012345677]
const CH2071_EPD_ZERO = [1.0, 0.0, 0.0, 0.0]
const CH2071_SAW_XS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
const CH2071_SAW = [0.0, 0.8, 0.3999999999999999, 0.0, 0.40000000000000036, 0.0]

@testset "Ch2071 depth activations" begin
    @test ACTIVATIONS == ("linear", "relu", "tanh")
    @test activation("linear", -1.5) == -1.5
    @test activation("relu", -1.5) == 0.0
    @test activation("relu", 2.0) == 2.0
    @test activation("relu", 0.0) == 0.0
    @test activation("tanh", 0.5) ≈ tanh(0.5) atol = 1e-15
    @test activation_derivative("linear", 3.0) == 1.0
    @test activation_derivative("relu", 0.0) == 0.0
    @test activation_derivative("relu", 0.1) == 1.0
    @test activation_derivative("tanh", 0.5) ≈ 1.0 - tanh(0.5)^2 atol = 1e-15
    @test_throws ArgumentError activation("sigmoid", 0.0)
    @test_throws ArgumentError activation_derivative("gelu", 0.0)
    @test_throws ArgumentError length_map(1.0, 1.0, 0.0, "softplus")
end

@testset "Ch2071 depth quadrature" begin
    @test gaussian_expectation(z -> z * z) ≈ CH2071_GE_Z2 atol = CH2071_TOL
    @test gaussian_expectation(z -> z^4) ≈ CH2071_GE_Z4 atol = CH2071_TOL
    @test gaussian_expectation(cos) ≈ CH2071_GE_COS atol = CH2071_TOL
    @test gaussian_expectation(cos; n=40, lim=6.0) ≈ CH2071_GE_COS_N40_LIM6 atol = CH2071_TOL
    @test CH2071_GE_COS ≈ exp(-0.5) atol = 1e-12
    @test gaussian_expectation_2d((a, b) -> (a + b)^2; n=40, lim=6.0) ≈ CH2071_GE2_SQ_N40_LIM6 atol = CH2071_TOL
    @test gaussian_expectation_2d((a, b) -> cos(a) * cos(2.0 * b); n=20, lim=5.0) ≈ CH2071_GE2_COSPROD_N20_LIM5 atol = CH2071_TOL
    @test_throws ArgumentError gaussian_expectation(cos; n=3)
    @test_throws ArgumentError gaussian_expectation(cos; n=0)
    @test_throws ArgumentError gaussian_expectation_2d((a, b) -> a * b; n=5)
end

@testset "Ch2071 depth length map and fixed point" begin
    @test length_map(1.0, 2.0, 0.0, "relu") ≈ CH2071_LM_RELU atol = CH2071_TOL
    @test length_map(1.0, 1.5, 0.05, "tanh") ≈ CH2071_LM_TANH atol = CH2071_TOL
    @test length_map(0.7, 1.2, 0.1, "linear") ≈ CH2071_LM_LINEAR atol = CH2071_TOL
    @test length_map(2.0, 2.5, 0.0, "tanh") ≈ CH2071_LM_TANH_Q2 atol = CH2071_TOL
    for q in (0.25, 1.0, 3.0)
        @test length_map(q, 2.0, 0.0, "relu") ≈ q rtol = 1e-12
    end
    @test_throws ArgumentError length_map(1.0, 0.0, 0.0, "relu")
    @test_throws ArgumentError length_map(1.0, 1.0, -0.1, "relu")
    @test_throws ArgumentError length_map(-1.0, 1.0, 0.0, "relu")

    @test length_fixed_point(1.5, 0.05, "tanh") ≈ CH2071_QSTAR_TANH atol = CH2071_TOL
    @test length_fixed_point(1.5, 0.1, "relu") ≈ CH2071_QSTAR_RELU atol = CH2071_TOL
    @test length_fixed_point(4.0, 0.05, "tanh"; q0=0.5) ≈ CH2071_QSTAR_TANH_CHAOS atol = CH2071_TOL
    @test CH2071_QSTAR_RELU ≈ 0.1 / (1.0 - 0.75) atol = 1e-9
    @test_throws ErrorException length_fixed_point(3.0, 0.0, "relu")
    @test_throws ErrorException length_fixed_point(1.5, 0.05, "tanh"; max_iter=2)
end

@testset "Ch2071 depth correlation map, chi1, depth scale" begin
    cm = correlation_map(0.5, 1.0, 2.0, 0.0, "relu")
    @test cm ≈ CH2071_CM_RELU atol = CH2071_TOL
    @test cm ≈ relu_correlation_map(0.5) atol = 1e-6
    @test correlation_map(0.5, CH2071_QSTAR_TANH, 1.5, 0.05, "tanh") ≈ CH2071_CM_TANH atol = CH2071_TOL
    @test correlation_map(0.3, CH2071_QSTAR_TANH, 1.5, 0.05, "tanh"; n=40) ≈ CH2071_CM_TANH_N40 atol = CH2071_TOL
    @test correlation_map(-0.4, 0.8, 1.3, 0.2, "linear"; n=40) ≈ CH2071_CM_LINEAR_N40 atol = CH2071_TOL
    @test_throws ArgumentError correlation_map(1.5, 1.0, 2.0, 0.0, "relu"; n=4)
    @test_throws ArgumentError correlation_map(0.5, 1.0, -2.0, 0.0, "relu"; n=4)
    @test_throws ArgumentError correlation_map(0.5, 0.0, 2.0, 0.0, "relu"; n=4)

    @test chi1(CH2071_QSTAR_TANH, 1.5, "tanh") ≈ CH2071_CHI1_TANH atol = CH2071_TOL
    @test chi1(1.0, 2.0, "relu") ≈ CH2071_CHI1_RELU atol = CH2071_TOL
    @test chi1(1.0, 0.9, "linear") ≈ CH2071_CHI1_LINEAR atol = CH2071_TOL
    @test_throws ArgumentError chi1(1.0, 0.0, "tanh")
    @test_throws ArgumentError chi1(-1.0, 1.0, "tanh")

    @test correlation_depth_scale(0.9) ≈ CH2071_XI_09 atol = CH2071_TOL
    @test correlation_depth_scale(1.0) == Inf
    @test_throws ArgumentError correlation_depth_scale(0.0)
end

@testset "Ch2071 depth ReLU kernel and residual networks" begin
    @test [relu_correlation_map(c) for c in CH2071_RCM_INPUTS] ≈ CH2071_RCM atol = CH2071_TOL
    @test_throws ArgumentError relu_correlation_map(1.01)
    @test_throws ArgumentError relu_correlation_map(-1.01)

    @test relu_correlation_trajectory(0.2, 5) ≈ CH2071_TRAJ_PLAIN atol = CH2071_TOL
    @test relu_correlation_trajectory(0.2, 5; alpha=0.5) ≈ CH2071_TRAJ_RES atol = CH2071_TOL
    @test relu_correlation_trajectory(-0.9, 3) ≈ CH2071_TRAJ_NEG atol = CH2071_TOL
    @test relu_correlation_trajectory(0.3, 0) == [0.3]
    @test_throws ArgumentError relu_correlation_trajectory(0.2, -1)
    @test_throws ArgumentError relu_correlation_trajectory(0.2, 3; alpha=-0.1)

    @test relu_collapse_asymptotic(10) ≈ CH2071_COLLAPSE_PLAIN atol = CH2071_TOL
    @test relu_collapse_asymptotic(10; alpha=0.5) ≈ CH2071_COLLAPSE_RES atol = CH2071_TOL
    @test_throws ArgumentError relu_collapse_asymptotic(0)
    @test_throws ArgumentError relu_collapse_asymptotic(10; alpha=0.0)

    @test relu_kernel_ode(0.2, 1.0) ≈ CH2071_ODE atol = CH2071_TOL
    @test relu_kernel_ode(-0.3, 2.0; steps=50) ≈ CH2071_ODE_COARSE atol = CH2071_TOL
    @test_throws ArgumentError relu_kernel_ode(0.2, 1.0; steps=0)
    @test_throws ArgumentError relu_kernel_ode(0.2, -1.0)

    # With alpha = 1 / sqrt(L) the residual trajectory approaches the ODE limit.
    target = relu_kernel_ode(0.2, 1.0)
    errs = [abs(relu_correlation_trajectory(0.2, L; alpha=1 / sqrt(L))[end] - target)
            for L in (10, 100, 1000)]
    @test errs[1] > errs[2] > errs[3]
    @test errs[3] < 2e-4

    @test residual_variance(1.0, 10, 1.0) ≈ CH2071_RV1 atol = CH2071_TOL
    @test residual_variance(0.5, 100, 0.1) ≈ CH2071_RV2 atol = CH2071_TOL
    @test_throws ArgumentError residual_variance(-1.0, 10, 1.0)
    @test_throws ArgumentError residual_variance(1.0, -1, 1.0)
    @test_throws ArgumentError residual_variance(1.0, 10, -1.0)
end

@testset "Ch2071 depth deep linear learning time" begin
    @test [deep_linear_learning_time(N, 0.5) for N in CH2071_DLT_DEPTHS] ≈ CH2071_DLT atol = CH2071_TOL
    @test deep_linear_learning_time(4, 0.3; s=2.0, frac=0.9, n=200) ≈ CH2071_DLT_OPTS atol = CH2071_TOL
    @test deep_linear_learning_time(1, 0.9) == 0.0
    @test [deep_linear_time_asymptotic(N, 0.5) for N in CH2071_DLA_DEPTHS] ≈ CH2071_DLA atol = CH2071_TOL
    @test deep_linear_time_asymptotic(4, 0.3; s=2.0) ≈ CH2071_DLA_S atol = CH2071_TOL
    gaps = [abs(deep_linear_learning_time(N, 0.5) / deep_linear_time_asymptotic(N, 0.5) - 1.0)
            for N in (5, 10, 20)]
    @test gaps[1] > gaps[2] > gaps[3]
    @test gaps[3] < 1e-4
    @test_throws ArgumentError deep_linear_learning_time(0, 0.5)
    @test_throws ArgumentError deep_linear_learning_time(3, 0.0)
    @test_throws ArgumentError deep_linear_learning_time(3, 0.5; s=0.0)
    @test_throws ArgumentError deep_linear_learning_time(3, 0.5; frac=1.0)
    @test_throws ArgumentError deep_linear_learning_time(3, 0.5; n=7)
    @test_throws ArgumentError deep_linear_time_asymptotic(2, 0.5)
    @test_throws ArgumentError deep_linear_time_asymptotic(3, -0.5)
end

@testset "Ch2071 depth residual paths" begin
    @test path_length_distribution(4) ≈ CH2071_PLD4 atol = CH2071_TOL
    @test path_length_distribution(0) == [1.0]
    @test sum(path_length_distribution(54)) ≈ 1.0 atol = 1e-12
    @test effective_path_distribution(4, 0.5) ≈ CH2071_EPD atol = CH2071_TOL
    @test effective_path_distribution(3, 0.0) ≈ CH2071_EPD_ZERO atol = CH2071_TOL
    @test sum(effective_path_distribution(54, 0.2)) ≈ 1.0 atol = 1e-12
    @test_throws ArgumentError path_length_distribution(-1)
    @test_throws ArgumentError effective_path_distribution(-1, 0.5)
    @test_throws ArgumentError effective_path_distribution(4, -0.5)
end

@testset "Ch2071 depth sawtooth" begin
    @test [sawtooth(x, 3) for x in CH2071_SAW_XS] ≈ CH2071_SAW atol = CH2071_TOL
    @test sawtooth(0.37, 0) == 0.37
    @test_throws ArgumentError sawtooth(0.5, -1)
    for L in 1:6
        grid = 4 * 2^L
        ys = [sawtooth(i / grid, L) for i in 0:grid]
        @test count_linear_pieces(ys) == 2^L
    end
    xs = [i / 64 for i in 0:64]
    @test count_linear_pieces([2.0 * x + 1.0 for x in xs]) == 1
    @test count_linear_pieces([abs(x - 0.33) for x in xs]) == 2
    @test_throws ArgumentError count_linear_pieces([1.0, 2.0])
end
