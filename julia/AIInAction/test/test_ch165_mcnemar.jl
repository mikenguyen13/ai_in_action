using Test
using AIInAction.Ch165Mcnemar

# Shared fixtures: identical to the Python and Rust test suites.
const MC165_CORRECT_A = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1]
const MC165_CORRECT_B = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
const MC165_TOL = 1e-9

# (b, c) => (chi2_corr, p_corr, chi2_raw, p_raw, exact_p)
const MC165_CASES = Dict(
    (12, 5) => (2.1176470588235294, 0.14561009539686698,
        2.8823529411764706, 0.08955507441364255, 0.1434631347656256),
    (30, 15) => (4.355555555555555, 0.03688842570704986,
        5.0, 0.025347318677468256, 0.03569780355519456),
    (3, 1) => (0.25, 0.6170750774519738,
        1.0, 0.31731050786291404, 0.6249999999999994),
)

@testset "Ch165 McNemar parity fixtures" begin
    t = contingency_table(MC165_CORRECT_A, MC165_CORRECT_B)
    @test (t.a, t.b, t.c, t.d) == (2, 6, 3, 1)
    @test n_total(t) == 12
    @test n_discordant(t) == 9

    for ((b, c), (chi2_corr, p_corr, chi2_raw, p_raw, exact_p)) in MC165_CASES
        rc = mcnemar_test(b, c; exact=false, correction=true)
        @test rc.method == "chi2"
        @test rc.statistic ≈ chi2_corr atol = MC165_TOL
        @test rc.p_value ≈ p_corr atol = MC165_TOL

        rr = mcnemar_test(b, c; exact=false, correction=false)
        @test rr.statistic ≈ chi2_raw atol = MC165_TOL
        @test rr.p_value ≈ p_raw atol = MC165_TOL

        re = mcnemar_test(b, c; exact=true)
        @test re.method == "exact"
        @test re.statistic ≈ Float64(min(b, c)) atol = MC165_TOL
        @test re.p_value ≈ exact_p atol = MC165_TOL
    end
end

@testset "Ch165 McNemar properties and edge cases" begin
    @test mcnemar_test(12, 5).method == "exact"
    @test mcnemar_test(30, 15).method == "chi2"
    @test mcnemar_test(3, 1).method == "exact"

    # Two-sided test is symmetric in b and c.
    r1 = mcnemar_test(30, 15; exact=false)
    r2 = mcnemar_test(15, 30; exact=false)
    @test r1.p_value ≈ r2.p_value atol = MC165_TOL
    @test r1.statistic ≈ r2.statistic atol = MC165_TOL
    e1 = mcnemar_test(12, 5; exact=true)
    e2 = mcnemar_test(5, 12; exact=true)
    @test e1.p_value ≈ e2.p_value atol = MC165_TOL

    # b == c is the strongest evidence for H_0.
    req = mcnemar_test(10, 10; exact=false)
    @test req.statistic ≈ 0.0 atol = MC165_TOL
    @test req.p_value ≈ 1.0 atol = MC165_TOL
    @test mcnemar_test(10, 10; exact=true).p_value ≈ 1.0 atol = MC165_TOL

    @test_throws ArgumentError mcnemar_test(0, 0)
    @test_throws ArgumentError mcnemar_test(-1, 5)
    @test_throws ArgumentError contingency_table([1, 0, 1], [1, 0])
    @test_throws ArgumentError contingency_table(Int[], Int[])
end
