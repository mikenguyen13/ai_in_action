using Test
using AIInAction.Ch089SoftmaxRegression

# Shared fixtures: identical to the Python and Rust test suites.
const SOFTMAX_Z = [1.0, 2.0, 3.0]
const SOFTMAX_EXPECTED = [0.09003057317038046, 0.24472847105479764, 0.6652409557748218]
const SOFTMAX_T2_EXPECTED = [0.1863237232258476, 0.3071958857184984, 0.506480391055654]
const LSE_EXPECTED = 0.6931471805599453
const CE_Z = [2.0, 1.0, 0.0]
const CE_EXPECTED = 0.4076059644443806

const FIT_X = [0.0 0.0; 1.0 0.0; 0.0 1.0]
const FIT_Y = [0, 1, 2]
const FIT_W = [-0.20927700627356688 -0.20927700627356688;
    0.41855401254713376 -0.20927700627356688;
    -0.20927700627356688 0.41855401254713376]
const FIT_B = [0.0258904318973107, -0.012945215948655314, -0.012945215948655314]
const FIT_LOSS = 0.8486364761645943
const FIT_PROBA = [0.3420186020672228 0.3289906989663886 0.3289906989663886;
    0.265668759951062 0.4787821251716869 0.25554911487725124;
    0.265668759951062 0.25554911487725124 0.4787821251716869]

const TOL = 1e-9

@testset "Softmax primitives parity" begin
    @test isapprox(softmax(SOFTMAX_Z), SOFTMAX_EXPECTED; atol=TOL)
    @test isapprox(sum(softmax(SOFTMAX_Z)), 1.0; atol=TOL)
    @test isapprox(softmax(SOFTMAX_Z; temperature=2.0), SOFTMAX_T2_EXPECTED; atol=TOL)
    @test isapprox(log_sum_exp([0.0, 0.0]), LSE_EXPECTED; atol=TOL)
    @test isapprox(cross_entropy_from_logits(CE_Z, 0), CE_EXPECTED; atol=TOL)
end

@testset "Softmax properties" begin
    @test isapprox(softmax(SOFTMAX_Z .+ 100.0), softmax(SOFTMAX_Z); atol=TOL)
    @test isapprox(softmax([1000.0, 1000.0, 1000.0]), fill(1 / 3, 3); atol=TOL)
end

@testset "Classifier parity fixtures" begin
    m = SoftmaxRegression(; learning_rate=1.0, n_iter=2, l2=0.0)
    fit!(m, FIT_X, FIT_Y)
    @test isapprox(m.W, FIT_W; atol=TOL)
    @test isapprox(m.b, FIT_B; atol=TOL)
    @test isapprox(loss(m, FIT_X, FIT_Y), FIT_LOSS; atol=TOL)
    @test isapprox(predict_proba(m, FIT_X), FIT_PROBA; atol=TOL)
end

@testset "Classifier converges" begin
    X = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0; 2.0 2.0; 2.0 0.0]
    y = [0, 1, 2, 2, 2, 1]
    m = SoftmaxRegression(; learning_rate=0.5, n_iter=300)
    fit!(m, X, y)
    @test predict(m, X) == y
end

@testset "Error handling" begin
    @test_throws ArgumentError softmax(Float64[])
    @test_throws ArgumentError softmax([1.0, 2.0]; temperature=0.0)
    @test_throws ArgumentError cross_entropy_from_logits([1.0, 2.0], 2)
    @test_throws ArgumentError SoftmaxRegression(; learning_rate=0.0)
    @test_throws ArgumentError SoftmaxRegression(; n_iter=0)
    m = SoftmaxRegression()
    @test_throws ArgumentError predict(m, FIT_X)  # not fitted
end
