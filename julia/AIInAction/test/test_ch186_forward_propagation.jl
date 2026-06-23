using Test
using AIInAction.Ch186ForwardPropagation

# Shared fixtures: identical to the Python and Rust test suites. The network is the
# worked chapter example: a 2-2-1 net with a ReLU hidden layer and sigmoid output.
const FP186_W1 = [0.5 -0.2; 0.1 0.4]
const FP186_B1 = [0.1, -0.3]
const FP186_W2 = [0.7 -0.6]
const FP186_B2 = [0.2]
# d0 = 2 features, m = 2 examples as columns.
const FP186_X = [1.0 0.0; 2.0 1.0]

fp186_net() = [
    make_layer(FP186_W1, FP186_B1, "relu"),
    make_layer(FP186_W2, FP186_B2, "sigmoid"),
]

@testset "Ch186 forward propagation parity fixtures" begin
    net = fp186_net()
    Z1, A1 = forward_layer(net[1], FP186_X)
    @test vec(Z1) ≈ [0.2, 0.6, -0.1, 0.1]      # column-major flatten of [0.2 -0.1; 0.6 0.1]
    @test vec(A1) ≈ [0.2, 0.6, 0.0, 0.1]

    Z2, _ = forward_layer(net[2], A1)
    @test vec(Z2) ≈ [-0.02, 0.14]

    Yhat = forward(net, FP186_X)
    @test size(Yhat) == (1, 2)
    @test vec(Yhat) ≈ [0.49500016666000024, 0.5349429451582145]

    single = forward(net, [1.0, 2.0])
    @test size(single) == (1, 1)
    @test single[1, 1] ≈ 0.49500016666000024
    # First column of the batch equals the single-example result.
    @test Yhat[1, 1] ≈ single[1, 1]
end

@testset "Ch186 activations and properties" begin
    @test relu([-2.0, -0.5, 0.0, 0.5, 2.0]) ≈ [0.0, 0.0, 0.0, 0.5, 2.0]

    s = sigmoid([-1000.0, 0.0, 1000.0])
    @test s ≈ [0.0, 0.5, 1.0]
    @test all(isfinite, s)

    tanh_net = [make_layer([1.0 2.0 -1.0], [0.5], "tanh")]
    @test forward(tanh_net, [0.5, -0.5, 1.0])[1, 1] ≈ -0.7615941559557649

    # Two identity layers compose to a single affine map.
    l1 = make_layer([1.0 2.0; 0.0 1.0], [1.0, -1.0], "identity")
    l2 = make_layer([2.0 0.0], [0.5], "identity")
    x = [3.0, 4.0]
    out = forward([l1, l2], x)
    expected = [2.0 0.0] * ([1.0 2.0; 0.0 1.0] * x .+ [1.0, -1.0]) .+ [0.5]
    @test vec(out) ≈ vec(expected)
end

@testset "Ch186 validation and edge cases" begin
    @test_throws ArgumentError make_layer([1.0 2.0], [0.0, 0.0], "relu")   # bias mismatch
    @test_throws ArgumentError make_layer([1.0], [0.0], "softmax")          # unknown activation
    @test_throws ArgumentError make_layer([1.0 NaN], [0.0], "relu")         # non-finite
    @test_throws ArgumentError apply_activation("gelu", [0.0])
    @test_throws ArgumentError forward(Ch186ForwardPropagation.Layer[], [1.0])
    bad = [make_layer([1.0 1.0], [0.0], "relu"), make_layer([1.0 1.0], [0.0], "relu")]
    @test_throws ArgumentError forward(bad, [1.0, 1.0])
    @test_throws ArgumentError forward(fp186_net(), [1.0, 2.0, 3.0])
    @test_throws ArgumentError forward(fp186_net(), [1.0, Inf])
end
