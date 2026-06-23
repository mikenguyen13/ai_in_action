using Test
using AIInAction.Ch187Backprop

# Shared fixtures: identical to the Python and Rust test suites.
const BP187_W1 = [0.10 0.20 -0.30; 0.40 -0.50 0.60]
const BP187_B1 = [0.10, -0.20]
const BP187_W2 = [0.70 -0.80; -0.10 0.30]
const BP187_B2 = [0.05, -0.05]
const BP187_X = [1.0, -2.0, 0.5]
const BP187_Y = [0.3, -0.7]

bp187_net() = make_mlp([BP187_W1, BP187_W2], [BP187_B1, BP187_B2])

@testset "Ch187 backprop parity fixtures" begin
    net = bp187_net()
    zs, acts = forward(net, BP187_X)
    @test zs[1] ≈ [-0.3500000000000001, 1.5]
    @test acts[2] ≈ [0.4133824210826699, 0.8175744761936437]
    @test acts[end] ≈ [-0.3146918861970461, 0.15393410074982605]
    @test acts[end] ≈ zs[end]  # linear output

    @test squared_error_loss(net, BP187_X, BP187_Y) ≈ 0.5535247816899481

    gW, gb = backprop(net, BP187_X, BP187_Y)
    @test vec(gW[1]') ≈ [
        -0.12505050629624692, 0.25010101259249384, -0.06252525314812346,
        0.11155166358278021, -0.22310332716556042, 0.055775831791390104,
    ]
    @test gb[1] ≈ [-0.12505050629624692, 0.11155166358278021]
    @test vec(gW[2]') ≈ [
        -0.25410282013600793, -0.5025563968780328,
        0.35300134601301564, 0.6981547251244291,
    ]
    @test gb[2] ≈ [-0.6146918861970461, 0.853934100749826]
end

@testset "Ch187 backprop properties and edge cases" begin
    net = bp187_net()

    # BP1 with a linear output: delta^L == a^L - y == grad_b at L.
    gW, gb = backprop(net, BP187_X, BP187_Y)
    _, acts = forward(net, BP187_X)
    @test gb[end] ≈ acts[end] .- BP187_Y

    # Analytic gradients match the central-difference estimate.
    ngW, ngb = numerical_gradient(net, BP187_X, BP187_Y)
    for l in 1:length(gW)
        @test gW[l] ≈ ngW[l] atol = 1e-7
        @test gb[l] ≈ ngb[l] atol = 1e-7
    end

    # Zero residual gives zero gradient.
    out = forward(net, BP187_X)[2][end]
    z_gW, z_gb = backprop(net, BP187_X, out)
    for g in z_gW
        @test all(abs.(g) .< 1e-12)
    end
    for g in z_gb
        @test all(abs.(g) .< 1e-12)
    end

    @test sigmoid(0.0) ≈ 0.5
    @test sigmoid_prime(0.0) ≈ 0.25
    @test sigmoid(1000.0) ≈ 1.0
    @test sigmoid(-1000.0) ≈ 0.0 atol = 1e-12

    @test_throws ArgumentError make_mlp([BP187_W1, BP187_W2], [BP187_B1])
    @test_throws ArgumentError make_mlp([[1.0 2.0], [1.0 2.0 3.0]], [[0.0], [0.0]])
    @test_throws ArgumentError forward(net, [1.0, 2.0])
    @test_throws ArgumentError backprop(net, BP187_X, [0.0, 0.0, 0.0])
    @test_throws ArgumentError numerical_gradient(net, BP187_X, BP187_Y; eps=0.0)
end
