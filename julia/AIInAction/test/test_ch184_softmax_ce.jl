using Test
using AIInAction.Ch184SoftmaxCE

# Shared fixtures: identical to the Python and Rust test suites. Labels are
# 1-based here (Julia convention); they select the same classes as Python's 0-based
# [0, 1, 2], so all numeric results match.
const Z184 = [
    2.0 1.0 0.1
    0.5 2.5 0.3
    1.0 1.0 1.0
]
const Y184 = [1, 2, 3]

const Z184_SOFTMAX_ROW1 = [0.6590011388859679, 0.24243297070471392, 0.09856589040931818]
const Z184_LOG_SOFTMAX_ROW1 = [-0.41703001627783354, -1.4170300162778335, -2.3170300162778332]
const Z184_LOSS = 0.5785639426554937
const Z184_LOSS_SMOOTH = 0.6574528315443826
const Z184_GRAD = [
    -0.1136662870380107 0.08081099023490464 0.03285529680310606
    0.03620124343567079 -0.06584031473611691 0.029639071300446088
    0.1111111111111111 0.1111111111111111 -0.22222222222222224
]
const Z184_GRAD_SMOOTH = [
    -0.09144406481578848 0.06969987912379354 0.02174418569199495
    0.025090132324559682 -0.0436180925138947 0.018527960189334978
    0.09999999999999999 0.09999999999999999 -0.20000000000000004
]

@testset "Ch184 softmax cross-entropy parity fixtures" begin
    p = softmax(Z184)
    @test p[1, :] ≈ Z184_SOFTMAX_ROW1
    @test vec(sum(p; dims=2)) ≈ ones(3)
    @test all(p .> 0)

    ls = log_softmax(Z184)
    @test ls[1, :] ≈ Z184_LOG_SOFTMAX_ROW1
    @test ls ≈ log.(softmax(Z184))

    @test cross_entropy_loss(Z184, Y184) ≈ Z184_LOSS
    @test cross_entropy_loss(Z184, Y184; label_smoothing=0.1) ≈ Z184_LOSS_SMOOTH

    @test cross_entropy_grad(Z184, Y184) ≈ Z184_GRAD
    @test cross_entropy_grad(Z184, Y184; label_smoothing=0.1) ≈ Z184_GRAD_SMOOTH
end

@testset "Ch184 properties and edge cases" begin
    g = cross_entropy_grad(Z184, Y184)
    @test vec(sum(g; dims=2)) ≈ zeros(3) atol = 1e-12

    # Equal logits => uniform softmax => loss = log(K).
    @test cross_entropy_loss([0.0 0.0 0.0], [2]) ≈ log(3.0)

    # Finite-difference check of the gradient.
    h = 1e-6
    num = similar(Z184)
    for i in 1:size(Z184, 1), j in 1:size(Z184, 2)
        zp = copy(Z184); zp[i, j] += h
        zm = copy(Z184); zm[i, j] -= h
        num[i, j] = (cross_entropy_loss(zp, Y184) - cross_entropy_loss(zm, Y184)) / (2h)
    end
    @test g ≈ num atol = 1e-7

    # Stability under huge logits.
    zbig = [1000.0 0.0; 0.0 1000.0]
    pbig = softmax(zbig)
    @test all(isfinite, pbig)
    @test pbig[1, 1] ≈ 1.0
    @test cross_entropy_loss(zbig, [1, 2]) ≈ 0.0 atol = 1e-9

    @test_throws ArgumentError cross_entropy_loss(reshape([1.0, 2.0], 2, 1), [1, 1])
    @test_throws ArgumentError softmax([1.0 Inf])
    @test_throws ArgumentError cross_entropy_loss(Z184, [1, 2, 4])
    @test_throws ArgumentError cross_entropy_loss(Z184, [1, 2])
    @test_throws ArgumentError cross_entropy_loss(Z184, Y184; label_smoothing=1.0)
end
