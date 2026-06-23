using Test
using AIInAction.Ch162RankingMetrics

# Shared fixtures: identical to the Python and Rust test suites. Every const is
# prefixed RM162_ so it cannot collide with other test files.
const RM162_MRR_QUERIES = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
const RM162_AP_Q1 = [1, 0, 1, 0, 0, 1]
const RM162_AP_Q2 = [0, 1, 1, 0]
const RM162_NDCG_Q1 = [3, 2, 0, 1, 2]
const RM162_NDCG_Q2 = [0, 0, 2, 1]

@testset "Ch162 ranking metrics parity fixtures" begin
    @test reciprocal_rank(RM162_MRR_QUERIES[1]) ≈ 1.0
    @test reciprocal_rank(RM162_MRR_QUERIES[2]) ≈ 0.3333333333333333
    @test reciprocal_rank(RM162_MRR_QUERIES[3]) ≈ 0.5
    @test mean_reciprocal_rank(RM162_MRR_QUERIES) ≈ 0.6111111111111112

    @test average_precision(RM162_AP_Q1) ≈ 0.7222222222222222
    @test average_precision(RM162_AP_Q2) ≈ 0.5833333333333333
    @test mean_average_precision([RM162_AP_Q1, RM162_AP_Q2]) ≈ 0.6527777777777778

    @test precision_at_k(RM162_AP_Q1, 3) ≈ 0.6666666666666666

    @test dcg(RM162_NDCG_Q1) ≈ 10.484024240491392
    @test dcg([3, 2, 2, 1, 0]) ≈ 10.823465818787767
    @test ndcg(RM162_NDCG_Q1) ≈ 0.9686383655679718
    @test ndcg(RM162_NDCG_Q2) ≈ 0.531730627995306
    @test mean_ndcg([RM162_NDCG_Q1, RM162_NDCG_Q2]) ≈ 0.7501844967816389
end

@testset "Ch162 ranking metrics properties and edge cases" begin
    @test reciprocal_rank([0, 0, 1, 0]; k=2) == 0.0
    @test reciprocal_rank([0, 0, 1, 0]; k=3) ≈ 1 / 3
    @test reciprocal_rank([0, 0, 0]) == 0.0

    @test average_precision(RM162_AP_Q1; n_relevant=5) ≈ 0.43333333333333335
    @test average_precision(RM162_AP_Q1; k=3) ≈ 5 / 9
    @test average_precision([1, 1, 1, 0, 0]) ≈ 1.0
    @test average_precision([0, 0, 0]) == 0.0

    @test precision_at_k([1, 0, 1], 10) ≈ 0.2

    @test ndcg([3, 2, 2, 1, 0]) ≈ 1.0
    @test ndcg([0, 0, 0]) == 0.0
    @test dcg([3, 0, 0]; exponential=false) ≈ 3.0
    @test ndcg(RM162_NDCG_Q1; exponential=true) != ndcg(RM162_NDCG_Q1; exponential=false)

    @test_throws ArgumentError mean_reciprocal_rank([])
    @test_throws ArgumentError mean_average_precision([])
    @test_throws ArgumentError mean_ndcg([])
    @test_throws ArgumentError dcg([1, -1, 0])
    @test_throws ArgumentError reciprocal_rank([1, 0]; k=0)
    @test_throws ArgumentError precision_at_k([1, 0], 0)
    @test_throws ArgumentError average_precision([1, 1, 1]; n_relevant=2)
    @test_throws ArgumentError mean_average_precision([RM162_AP_Q1, RM162_AP_Q2]; n_relevant=[3])
end
