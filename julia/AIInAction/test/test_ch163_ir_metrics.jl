using Test
using AIInAction.Ch163IrMetrics

# Shared fixtures: identical to the Python and Rust test suites. Names are prefixed
# with IR163 so top-level consts cannot collide with other test files.
const IR163_RANKING = [1, 0, 1, 1, 0, 1]   # relevant at ranks 1, 3, 4, 6
const IR163_NUM_RELEVANT = 5               # one relevant doc never retrieved
const IR163_GRADES = [3, 2, 3, 0, 1, 2]
const IR163_IDEAL_GRADES = [3, 3, 2, 2, 1, 0]
const IR163_QUERY_SET = [
    [1, 0, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
]

@testset "Ch163 IR metrics parity fixtures" begin
    @test precision_at_k(IR163_RANKING, 1) ≈ 1.0
    @test precision_at_k(IR163_RANKING, 3) ≈ 0.6666666666666666
    @test precision_at_k(IR163_RANKING, 6) ≈ 0.6666666666666666
    @test precision_at_k(IR163_RANKING, 100) ≈ 0.6666666666666666   # clamped

    @test recall_at_k(IR163_RANKING, 3, IR163_NUM_RELEVANT) ≈ 0.4
    @test recall_at_k(IR163_RANKING, 6, IR163_NUM_RELEVANT) ≈ 0.8

    @test average_precision(IR163_RANKING; num_relevant=IR163_NUM_RELEVANT) ≈ 0.6166666666666666
    @test average_precision(IR163_RANKING) ≈ 0.7708333333333333
    @test average_precision([1, 1, 1]) ≈ 1.0

    @test dcg_at_k(IR163_GRADES, 3) ≈ 12.392789260714373
    @test ndcg_at_k(IR163_GRADES, 3) ≈ 0.9594535145926796
    @test ndcg_at_k(IR163_IDEAL_GRADES, 6) ≈ 1.0
    @test ndcg_at_k(IR163_GRADES, 6; ideal_grades=IR163_IDEAL_GRADES) ≈ 0.9488107485678985

    @test reciprocal_rank(IR163_RANKING) ≈ 1.0
    @test reciprocal_rank([0, 0, 1, 0]) ≈ 1.0 / 3.0
    @test reciprocal_rank([0, 0, 0]) == 0.0

    @test mean_average_precision(IR163_QUERY_SET) ≈ 0.46249999999999997
    @test mean_reciprocal_rank(IR163_QUERY_SET) ≈ 0.5555555555555556
end

@testset "Ch163 IR metrics edge cases" begin
    @test_throws ArgumentError precision_at_k([1, 2, 0], 3)
    @test_throws ArgumentError precision_at_k([1, 0, 1], 0)
    @test_throws ArgumentError recall_at_k([1, 1, 1], 3, 2)
    @test_throws ArgumentError recall_at_k([1, 0, 1], 3, 0)
    @test_throws ArgumentError dcg_at_k([1, -1, 2], 3)
    @test_throws ArgumentError ndcg_at_k([0, 0, 0], 3)
    @test_throws ArgumentError precision_at_k(Int[], 1)
    @test_throws ArgumentError dcg_at_k(Int[], 1)
    @test_throws ArgumentError mean_average_precision(Vector{Vector{Int}}())
    @test_throws ArgumentError mean_reciprocal_rank(Vector{Vector{Int}}())
    @test_throws ArgumentError mean_average_precision([[1, 0], [0, 1]]; num_relevant=[1])
end
