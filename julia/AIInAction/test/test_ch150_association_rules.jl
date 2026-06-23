using Test
using AIInAction.Ch150AssociationRules

# Shared fixtures: identical to the Python and Rust suites.
const CH150_TRANSACTIONS = [
    [1, 2, 5],
    [2, 4],
    [2, 3],
    [1, 2, 4],
    [1, 3],
    [2, 3],
    [1, 3],
    [1, 2, 3, 5],
    [1, 2, 3],
]
const CH150_MIN_SUPPORT = 2.0 / 9.0

const CH150_EXPECTED = Dict(
    [1] => 6 / 9,
    [2] => 7 / 9,
    [3] => 6 / 9,
    [4] => 2 / 9,
    [5] => 2 / 9,
    [1, 2] => 4 / 9,
    [1, 3] => 4 / 9,
    [1, 5] => 2 / 9,
    [2, 3] => 4 / 9,
    [2, 4] => 2 / 9,
    [2, 5] => 2 / 9,
    [1, 2, 3] => 2 / 9,
    [1, 2, 5] => 2 / 9,
)

@testset "Ch150 frequent itemsets" begin
    fis = apriori(CH150_TRANSACTIONS, CH150_MIN_SUPPORT)
    @test length(fis) == 13
    for (k, v) in CH150_EXPECTED
        @test fis[k] ≈ v
    end

    fg = fpgrowth(CH150_TRANSACTIONS, CH150_MIN_SUPPORT)
    @test sort(collect(keys(fg))) == sort(collect(keys(fis)))
    for (k, v) in CH150_EXPECTED
        @test fg[k] ≈ v
    end
end

@testset "Ch150 support helper" begin
    @test support(CH150_TRANSACTIONS, [1, 2]) ≈ 4 / 9
    @test support(CH150_TRANSACTIONS, [2]) ≈ 7 / 9
    @test support(CH150_TRANSACTIONS, [4, 5]) ≈ 0.0
end

@testset "Ch150 association rules" begin
    fis = apriori(CH150_TRANSACTIONS, CH150_MIN_SUPPORT)
    rules = association_rules(fis, 0.7)
    @test length(rules) == 6
    keyset = Set((r.antecedent, r.consequent) for r in rules)
    @test keyset == Set([
        ([1, 5], [2]),
        ([2, 5], [1]),
        ([4], [2]),
        ([5], [1]),
        ([5], [1, 2]),
        ([5], [2]),
    ])

    r = first(x for x in rules if x.antecedent == [5] && x.consequent == [1, 2])
    @test r.support ≈ 2 / 9
    @test r.confidence ≈ 1.0
    @test r.lift ≈ 2.25
    @test r.leverage ≈ 0.12345679012345678
    @test isinf(r.conviction)

    r2 = first(x for x in rules if x.antecedent == [2, 5] && x.consequent == [1])
    @test r2.lift ≈ 1.5
    @test r2.leverage ≈ 0.07407407407407407
end

@testset "Ch150 finite conviction and ordering" begin
    fis = apriori(CH150_TRANSACTIONS, CH150_MIN_SUPPORT)
    rules = association_rules(fis, 0.5)
    r = first(x for x in rules if x.antecedent == [1] && x.consequent == [2])
    @test r.confidence ≈ 2 / 3
    @test r.lift ≈ 0.8571428571428571
    @test r.leverage ≈ -0.07407407407407407
    @test r.conviction ≈ 0.6666666666666665

    all_rules = association_rules(fis, 0.0)
    confs = [x.confidence for x in all_rules]
    @test confs == sort(confs; rev = true)
end

@testset "Ch150 edge cases" begin
    @test_throws ArgumentError apriori(Vector{Vector{Int}}(), 0.5)
    @test_throws ArgumentError apriori(CH150_TRANSACTIONS, 0.0)
    @test_throws ArgumentError apriori(CH150_TRANSACTIONS, 1.5)
    fis = apriori(CH150_TRANSACTIONS, CH150_MIN_SUPPORT)
    @test_throws ArgumentError association_rules(fis, 1.5)
    @test_throws ArgumentError support(CH150_TRANSACTIONS, Int[])

    # Apriori and FP-Growth agree on dense data across thresholds.
    dense = [[1, 2, 3, 4], [1, 2, 3], [1, 2, 4], [1, 2], [2, 3, 4]]
    for ms in (0.2, 0.4, 0.6, 0.8)
        a = apriori(dense, ms)
        f = fpgrowth(dense, ms)
        @test sort(collect(keys(a))) == sort(collect(keys(f)))
        for k in keys(a)
            @test a[k] ≈ f[k]
        end
    end
end
