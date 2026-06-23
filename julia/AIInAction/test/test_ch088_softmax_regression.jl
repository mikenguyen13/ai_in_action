using Test
using AIInAction.Ch088SoftmaxRegression

# --- Shared fixtures: identical to the Python and Rust test suites. ---
const SOFTMAX_Z = [1.0 2.0 3.0; 1.0 1.0 1.0]
const SOFTMAX_ROW0 = [0.09003057317038046, 0.24472847105479764, 0.6652409557748218]

const CE_PROBS = [0.7 0.2 0.1; 0.1 0.8 0.1]
const CE_Y = [0, 1]
const CE_EXPECTED = 0.2899092476264711

const TRAIN_X = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0; 2.0 2.0; 2.0 0.0]
const TRAIN_Y = [0, 1, 2, 1, 2, 1]
const TRAIN_LR = 0.5
const TRAIN_ITERS = 200
const TRAIN_PRED = [0, 1, 2, 1, 2, 1]
const TRAIN_PROBA_ROW0 = [0.8222160377229213, 0.16537455654418495, 0.012409405732893796]

const TOL = 1e-9

@testset "Softmax regression parity fixtures" begin
    p = softmax(SOFTMAX_Z)
    @test isapprox(p[1, :], SOFTMAX_ROW0; atol=TOL)
    @test isapprox(p[2, :], fill(1 / 3, 3); atol=TOL)
    @test isapprox(sum(p; dims=2), ones(2, 1); atol=TOL)

    @test isapprox(cross_entropy(CE_PROBS, CE_Y), CE_EXPECTED; atol=TOL)

    model = fit!(SoftmaxRegression(; learning_rate=TRAIN_LR, n_iter=TRAIN_ITERS), TRAIN_X, TRAIN_Y)
    @test predict(model, TRAIN_X) == TRAIN_PRED
    proba0 = predict_proba(model, reshape([0.0, 0.0], 1, 2))
    @test isapprox(vec(proba0), TRAIN_PROBA_ROW0; atol=TOL)
    @test isapprox(sum(proba0), 1.0; atol=TOL)
    @test model.n_classes == 3
    @test size(model.W) == (2, 3)
end

@testset "Softmax regression edge cases" begin
    @test_throws ArgumentError softmax(reshape([1.0, Inf], 1, 2))
    @test_throws ArgumentError cross_entropy([0.5 0.5], [0, 1])
    @test_throws ArgumentError cross_entropy(reshape([0.5, 0.5], 1, 2), [2])
    @test_throws ArgumentError SoftmaxRegression(; learning_rate=0.0)
    @test_throws ArgumentError SoftmaxRegression(; n_iter=0)
    @test_throws ArgumentError SoftmaxRegression(; l2=-1.0)
    @test_throws ArgumentError fit!(SoftmaxRegression(), [0.0 0.0; 1.0 1.0], [0])
    @test_throws ArgumentError predict(SoftmaxRegression(), reshape([0.0, 0.0], 1, 2))
    let m = fit!(SoftmaxRegression(; n_iter=10), TRAIN_X, TRAIN_Y)
        @test_throws ArgumentError predict(m, reshape([0.0, 0.0, 0.0], 1, 3))
    end
end
