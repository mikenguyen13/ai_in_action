using Test
using AIInAction.ChClickgraph

# Shared fixtures: identical to the Python (tests/test_ch_clickgraph.py) and Rust suites.
# Names are prefixed with CG_ so they cannot collide with other test files.
const CG_PAPER_SERIES = [9.0, 14.0, 17.0]
const CG_SCALED = [-1.875, 0.0, 1.125]
const CG_BURST = [-0.10833333333333331, 0.016666666666666663, 0.09166666666666667]

# 1-based node ids (Python: 0-based)
const CG_SESSIONS = [[1, 2, 3], [1, 2], [1, 4, 1, 2], [4, 4, 5]]
const CG_SESSION_COUNTS = [4.0, 3.0, 1.0, 3.0, 1.0]
const CG_SESSION_W = [
    0.0 3.0 0.0 1.0 0.0
    0.0 0.0 1.0 0.0 0.0
    0.0 0.0 0.0 0.0 0.0
    1.0 0.0 0.0 0.0 1.0
    0.0 0.0 0.0 0.0 0.0
]

const CG_EMA_123_07 = 2.61
const CG_LSTM_X = [1.0; 2.0; 0.5;;]
const CG_LSTM_H = [-0.007358975720281007, -0.3247841816214326]
const CG_ATT_H = [0.5, -0.2, 0.1, 0.8]
const CG_ATT_E = [0.2, 1.0, 0.5]
const CG_ATT_Z = [0.13473023562592665, -0.13723523527276552]
const CG_ATT_ALPHA = [0.3514830614166104, 0.28412005829708203, 0.3643968802863075]
const CG_TOTAL_QL = 1.2
const CG_WQL_09 = 0.16
const CG_GATED = [2.5, 4.424234314520019]
const CG_BURST_FC = [0.0, 7.6159415595576485]

@testset "ChClickgraph" begin
    @testset "scaling and burstiness" begin
        r = robust_median_scale(CG_PAPER_SERIES)
        @test r.median == 14.0
        @test r.mad ≈ 8 / 3
        @test r.scaled ≈ CG_SCALED
        c = robust_median_scale([3.0, 3.0, 3.0])
        @test c.mad == 0.0 && c.scaled == zeros(3)
        @test_throws ArgumentError robust_median_scale(Float64[])
        b = burstiness(CG_PAPER_SERIES)
        @test isapprox(b, CG_BURST; atol=1e-12)
        @test isapprox(sum(b), 0.0; atol=1e-12)
        @test burstiness([0, 0, 0]) == zeros(3)
        @test_throws ArgumentError burstiness([1.0, -1.0])
    end

    @testset "graph construction" begin
        counts, W = session_graph(CG_SESSIONS, 5)
        @test counts == CG_SESSION_COUNTS
        @test W == CG_SESSION_W
        @test_throws ArgumentError session_graph([[1, 9]], 5)
        @test upstream_neighbors(W, 2) == [1]
        @test upstream_neighbors(W, 1) == [4]
        @test upstream_neighbors(W, 5) == [4]
        @test upstream_neighbors(W, 4) == [1]
        B = hcat(burstiness([9, 14, 17]), burstiness([2, 4, 4]), burstiness([1, 1, 1]))'
        @test neighbor_burstiness(B, [1, 2]) ≈ (B[1, :] .+ B[2, :]) ./ 2
        @test neighbor_burstiness(B, Int[]) == zeros(3)
        @test average_graph([[1.0 0.0; 0.0 1.0], [3.0 0.0; 0.0 3.0]]) == [2.0 0.0; 0.0 2.0]
        @test normalize_edge_weights([0.0 4.0; 2.0 0.0], [2.0, 0.0]) == [0.0 0.0; 1.0 0.0]
    end

    @testset "trend, softmax, sigmoid" begin
        @test ema_last([1.0, 2.0, 3.0], 0.7) ≈ CG_EMA_123_07
        @test burst_forecast([0.0, 1.0], 10.0) ≈ CG_BURST_FC
        @test_throws ArgumentError ema_last([1.0], 0.0)
        @test softmax([1000.0, 1000.0]) == [0.5, 0.5]
        @test sigmoid(0.0) == 0.5
        @test isapprox(sigmoid(-800.0), 0.0; atol=1e-300)
    end

    @testset "embeddings" begin
        W, b = det_matrix(8, 3), vec(det_matrix(1, 8, 0.3))
        h = lstm_embed(CG_LSTM_X, W, b)
        @test isapprox(h, CG_LSTM_H; atol=1e-12)
        @test all(abs.(h) .< 1)
        @test_throws ArgumentError lstm_embed(CG_LSTM_X, W[:, 1:2], b)

        Wq, Wk, Wv = det_matrix(3, 4, 0.1), det_matrix(3, 4, 0.2), det_matrix(2, 4, 0.3)
        we, Hn = vec(det_matrix(1, 3, 0.4)), det_matrix(3, 4, 0.5)
        z, a = attention_neighbor_embedding(CG_ATT_H, Hn, CG_ATT_E, Wq, Wk, Wv, we)
        @test isapprox(a, CG_ATT_ALPHA; atol=1e-12)
        @test isapprox(z, CG_ATT_Z; atol=1e-12)
        @test sum(a) ≈ 1.0
        perm = [3, 1, 2]
        z2, a2 = attention_neighbor_embedding(CG_ATT_H, Hn[perm, :], CG_ATT_E[perm], Wq, Wk, Wv, we)
        @test z2 ≈ z
        @test a2 ≈ a[perm]
        z0, a0 = attention_neighbor_embedding(CG_ATT_H, zeros(0, 4), Float64[], Wq, Wk, Wv, we)
        @test z0 == zeros(2) && isempty(a0)
    end

    @testset "forecast assembly and quantile loss" begin
        @test gated_forecast([1.0, 2.0], [0.5, -0.5], [2.0, 4.0], [0.0, 1.0]) ≈ CG_GATED
        @test_throws ArgumentError gated_forecast([1.0], [1.0, 2.0], [1.0], [0.0])
        @test pinball_loss(10.0, 8.0, 0.75) == 1.5
        @test pinball_loss(8.0, 10.0, 0.75) ≈ 0.5
        preds = [1.0 2.0 3.0; 2.0 2.0 2.0]
        @test total_quantile_loss([1.5, 2.5], preds, [0.1, 0.5, 0.9]) ≈ CG_TOTAL_QL
        @test weighted_quantile_loss([10, 20, 0], [8, 25, 1], 0.9) ≈ CG_WQL_09
        @test_throws ArgumentError pinball_loss(1.0, 1.0, 1.0)
        @test_throws ArgumentError weighted_quantile_loss([0.0], [1.0], 0.5)
        @test interval_coverage([1, 2, 3, 4], [0, 0, 0, 0], [2, 2, 2, 2]) == 0.5
    end
end
