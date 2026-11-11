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
const CG_RNN_H = [0.7327923787582081, 0.20401955903303856]
const CG_GRU_H = [-0.18247873874066795, -0.6512644859666139]
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

        Wr, br = det_matrix(2, 3), vec(det_matrix(1, 2, 0.3))
        hr = rnn_embed(CG_LSTM_X, Wr, br)
        @test isapprox(hr, CG_RNN_H; atol=1e-12)
        @test all(abs.(hr) .< 1)
        @test isapprox(rnn_embed([0.7;;], Wr, br), tanh.(Wr[:, 1] .* 0.7 .+ br); atol=1e-15)
        @test_throws ArgumentError rnn_embed(CG_LSTM_X, Wr[:, 1:2], br)
        @test_throws ArgumentError rnn_embed(zeros(0, 1), Wr, br)

        Wg, bih, bhh = det_matrix(6, 3), vec(det_matrix(1, 6, 0.3)), vec(det_matrix(1, 6, 0.6))
        hg = gru_embed(CG_LSTM_X, Wg, bih, bhh)
        @test isapprox(hg, CG_GRU_H; atol=1e-12)
        @test all(abs.(hg) .< 1)
        # reset and update biases enter only as a sum; the candidate's do not
        mih, mhh = copy(bih), copy(bhh)
        mih[1:4] .+= bhh[1:4]; mhh[1:4] .= 0.0
        @test isapprox(gru_embed(CG_LSTM_X, Wg, mih, mhh), hg; atol=1e-14)
        mih, mhh = copy(bih), copy(bhh)
        mih[5:6] .+= bhh[5:6]; mhh[5:6] .= 0.0
        @test maximum(abs.(gru_embed(CG_LSTM_X, Wg, mih, mhh) .- hg)) > 1e-3
        @test_throws ArgumentError gru_embed(CG_LSTM_X, Wg[:, 1:2], bih, bhh)
        @test_throws ArgumentError gru_embed(CG_LSTM_X, Wg, bih, bhh[1:5])
        # a saturated update gate keeps the zero initial state
        shut = copy(bih); shut[3:4] .= 50.0
        @test maximum(abs.(gru_embed(CG_LSTM_X, Wg, shut, bhh))) < 1e-20
        # constant update gate, history-free candidate: h_T is the EMA of the candidates
        W0 = copy(Wg); W0[1:4, :] .= 0.0; W0[5:6, 2:end] .= 0.0
        zeta = 0.4
        b0 = copy(bih); b0[3:4] .= zeta
        h0 = gru_embed(CG_LSTM_X, W0, b0, zeros(6))
        alpha = 1 - sigmoid(zeta)
        for k in 1:2
            n = [tanh(W0[4 + k, 1] * x + b0[4 + k]) for x in CG_LSTM_X[:, 1]]
            @test isapprox(h0[k], ema_last(vcat(0.0, n), alpha); atol=1e-14)
        end

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
