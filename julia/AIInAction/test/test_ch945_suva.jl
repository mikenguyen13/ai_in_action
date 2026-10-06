using Test
using LinearAlgebra
using AIInAction.Ch945Suva

# Shared fixtures: identical to the Python (tests/test_ch945_suva.py) and Rust suites.
# Names are prefixed with SV945_ so they cannot collide with other test files.
const SV945_CDF_POINTS = [(-1.96, 0.02499789514822045), (0.0, 0.5), (0.5, 0.691462461274013),
    (2.5, 0.9937903346742238), (-8.0, 6.220960520142783e-16), (10.0, 1.0)]

const SV945_GRID_COUNT = 240
const SV945_GRID_FIRST = (0, 0, 0, 200)
const SV945_GRID_LAST = (600, 600, 600, 400)
const SV945_RECIP_COUNT = 120
const SV945_RECIP_FIRST = (0, 0, 200, 0)
const SV945_RECIP_LAST = (600, 600, 400, 600)
const SV945_RECIP_01 = [(0, 0, 1, 0), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1),
    (1, 1, 0, 1)]

const SV945_MOTIVES = [((200, 400, 600, 200), (1, 1, 1, -1)), ((0, 600, 600, 0), (1, 1, 0, 0)),
    ((400, 400, 200, 600), (-1, -1, 1, 0))]

const SV945_OLS_X = [[1.0, Float64(i)] for i in 0:7]
const SV945_OLS_Y = [1.0, 2.5, 2.0, 4.5, 4.0, 6.5, 5.5, 8.0]
const SV945_OLS_CLUSTERS = [0, 0, 1, 1, 2, 2, 3, 3]
const SV945_OLS_COEF = [1.041666666666666, 0.916666666666667]
const SV945_OLS_SE = [0.25027848001436165, 0.05787698374909847]

const SV945_DEMEAN_VALUES = [1.0, 3.0, 2.0, 6.0, 10.0]
const SV945_DEMEAN_GROUPS = ["a", "a", "b", "b", "b"]
const SV945_DEMEAN_OUT = [-1.0, 1.0, -4.0, 0.0, 4.0]

const SV945_DIFF_PROP_ARGS = (30, 50, 18, 50)
const SV945_DIFF_PROP_D = 0.24
const SV945_DIFF_PROP_SE = 0.09699484522385712

const SV945_KAPPA_A = [0, 0, 1, 1, 2, 2, 0, 1]
const SV945_KAPPA_B = [0, 1, 1, 1, 2, 0, 0, 1]
const SV945_KAPPA = 0.6097560975609756

const SV945_AUC_SCORES = [0.1, 0.4, 0.35, 0.8, 0.4, 0.2]
const SV945_AUC_LABELS = [0, 0, 1, 1, 1, 0]
const SV945_AUC = 0.8333333333333334

const SV945_D_X = [5.0, 6.0, 7.0, 8.0]
const SV945_D_Y = [3.0, 4.0, 5.0]
const SV945_COHENS_D = 2.1128856368212916

const SV945_TOST_PASS_ARGS = (0.02, 0.04, 0.1)
const SV945_TOST_PASS = 0.02275013194817923
const SV945_TOST_FAIL_ARGS = (0.12, 0.04, 0.1)
const SV945_TOST_FAIL = 0.691462461274013

const SV945_DEFF_5_02 = 1.8
const SV945_N_045_030 = 160
const SV945_N_045_030_DEFF = 288
const SV945_N_05_04_POWER90 = 515
const SV945_Z_90 = 1.2815515655446004

const SV945_DPO_01 = 2.1972245773362196
const SV945_DPO_025_BETA2 = 0.5493061443340549

const SV945_COLLAPSE_IN = ["u", "u", "s", "s", "f", "s"]
const SV945_COLLAPSE_OUT = ["u", "s", "f", "s"]

const SV945_TREE_PATHS = [(["understanding", "self_interest", "self_interest"], "B1"),
    (["understanding", "self_interest"], "B2"),
    (["understanding", "fairness"], "B2"),
    (["self_interest"], "B1")]
const SV945_TREE_ROOT_COUNT = 4
const SV945_TREE_UNDERSTANDING_COUNT = 3
const SV945_TREE_UNDERSTANDING_ACTIONS = Dict("B1" => 1, "B2" => 2)
const SV945_TREE_PATH_PROB = 0.25  # understanding -> self_interest -> B1 = 3/4 * 2/3 * 1/2

@testset "Ch945 normal CDF (Hart and West)" begin
    for (x, expected) in SV945_CDF_POINTS
        @test normal_cdf(x) ≈ expected rtol = 1e-12 atol = 1e-15
    end
    for i in -900:900
        x = i / 100
        @test normal_cdf(x) + normal_cdf(-x) ≈ 1.0 atol = 1e-14
    end
    @test normal_cdf(40.0) == 1.0
    @test normal_cdf(-40.0) == 0.0
    @test normal_cdf(-Z_975) ≈ 0.025 atol = 1e-12
    @test normal_cdf(Z_80) ≈ 0.8 atol = 1e-12
    @test normal_cdf(SV945_Z_90) ≈ 0.9 atol = 1e-12
    @test_throws ArgumentError normal_cdf(NaN)
    @test_throws ArgumentError normal_cdf(Inf)
end

@testset "Ch945 payoff grid" begin
    g = payoff_grid()
    @test g isa Vector{NTuple{4,Int}}
    @test length(g) == SV945_GRID_COUNT
    @test g[1] == SV945_GRID_FIRST
    @test g[end] == SV945_GRID_LAST
    r = payoff_grid(; game=:reciprocity)
    @test length(r) == SV945_RECIP_COUNT
    @test r[1] == SV945_RECIP_FIRST
    @test r[end] == SV945_RECIP_LAST
    @test payoff_grid((0, 1); game=:reciprocity) == SV945_RECIP_01
    @test payoff_grid((0, 1); game="reciprocity") == SV945_RECIP_01
    @test length(payoff_grid((0, 1))) == 12

    # The option better for A is never strictly better for B.
    for (a1, b1, a2, b2) in r
        if a1 > a2
            @test b1 <= b2
        else
            @test a1 < a2 && b1 >= b2
        end
    end

    @test_throws ArgumentError payoff_grid(; game=:ultimatum)
    @test_throws ArgumentError payoff_grid((1, 1))
    @test_throws ArgumentError payoff_grid((5,))
end

@testset "Ch945 motive features" begin
    for (payoffs, expected) in SV945_MOTIVES
        @test motive_features(payoffs...) == expected
        @test motive_features(payoffs...) isa NTuple{4,Int}
    end
    @test length(MOTIVES) == 4
    @test_throws ArgumentError motive_features(NaN, 0, 0, 0)
end

@testset "Ch945 OLS with CR1 cluster-robust SE" begin
    r = ols_cluster(SV945_OLS_X, SV945_OLS_Y, SV945_OLS_CLUSTERS)
    @test r isa OlsResult
    @test r.coef ≈ SV945_OLS_COEF rtol = 1e-10
    @test r.se ≈ SV945_OLS_SE rtol = 1e-10
    for i in 1:2
        @test r.coef[i] ≈ SV945_OLS_COEF[i] rtol = 1e-10
        @test r.se[i] ≈ SV945_OLS_SE[i] rtol = 1e-10
    end
    @test r.n == 8
    @test r.n_clusters == 4

    # Matrix input gives the same fit as a vector of rows.
    xm = hcat(ones(8), collect(0.0:7.0))
    rmat = ols_cluster(xm, SV945_OLS_Y, SV945_OLS_CLUSTERS)
    @test rmat.coef == r.coef
    @test rmat.se == r.se

    z = zscores(r)
    @test z ≈ r.coef ./ r.se
    p = p_values(r)
    @test p[1] ≈ 2.0 * normal_cdf(-abs(z[1]))
    @test all(0.0 .<= p .<= 1.0)
    iv = ci(r)
    @test length(iv) == 2
    @test iv[2][1] ≈ r.coef[2] - Z_975 * r.se[2]
    @test iv[2][2] ≈ r.coef[2] + Z_975 * r.se[2]

    # Recovers known coefficients on deterministic, nearly noiseless data.
    x1 = collect(1.0:30.0)
    x2 = [Float64(mod(i * i, 7)) for i in 1:30]
    yy = [0.5 + 2.0 * x1[i] - x2[i] + 0.01 * sin(i) for i in 1:30]
    rr = ols_cluster([[1.0, x1[i], x2[i]] for i in 1:30], yy, collect(1:30))
    @test rr.coef ≈ [0.5, 2.0, -1.0] atol = 0.03
    lo, hi = ci(rr)[2]
    @test lo < 2.0 < hi

    @test_throws ArgumentError ols_cluster(fill([1.0, 1.0], 5), [1, 2, 3, 4, 5], [0, 1, 0, 1, 0])
    @test_throws ArgumentError ols_cluster(fill([1.0], 3), [1.0, 2.0, 3.0], [0, 0, 0])
    @test_throws ArgumentError ols_cluster([[1.0]], [1.0], [0])
    @test_throws ArgumentError ols_cluster([[1.0], [1.0, 2.0]], [1.0, 2.0], [0, 1])
    @test_throws ArgumentError ols_cluster(SV945_OLS_X, SV945_OLS_Y, [0, 1])
end

@testset "Ch945 within transform" begin
    @test demean_by_group(SV945_DEMEAN_VALUES, SV945_DEMEAN_GROUPS) ≈ SV945_DEMEAN_OUT
    @test_throws ArgumentError demean_by_group([1.0, 2.0], ["a"])

    # Frisch-Waugh-Lovell: demeaned slopes equal slopes with group dummies.
    n = 20
    g = [(t - 1) ÷ 4 + 1 for t in 1:n]
    w = [mod(7 * t, 3) == 0 ? 1.0 : 0.0 for t in 1:n]
    y = [0.3 * w[t] + 0.5 * g[t] + 0.1 * sin(t) for t in 1:n]
    within = ols_cluster([[v] for v in demean_by_group(w, g)], demean_by_group(y, g), g)
    dummies = Float64[g[t] == j for t in 1:n, j in 1:5]
    full = hcat(w, dummies) \ y
    @test within.coef[1] ≈ full[1] rtol = 1e-10
end

@testset "Ch945 difference of proportions" begin
    d, se = diff_proportions(SV945_DIFF_PROP_ARGS...)
    @test d ≈ SV945_DIFF_PROP_D rtol = 1e-12
    @test se ≈ SV945_DIFF_PROP_SE rtol = 1e-12
    @test_throws ArgumentError diff_proportions(5, 4, 1, 4)
    @test_throws ArgumentError diff_proportions(1, 0, 1, 4)
    @test_throws ArgumentError diff_proportions(1.5, 4, 1, 4)
end

@testset "Ch945 measurement quality" begin
    @test cohen_kappa(SV945_KAPPA_A, SV945_KAPPA_B) ≈ SV945_KAPPA rtol = 1e-12
    @test cohen_kappa(["x", "y"], ["x", "y"]) == 1.0
    @test cohen_kappa(["x", "x"], ["x", "x"]) == 1.0
    @test cohen_kappa([:a, :b, :b], [:a, :b, :b]) == 1.0
    @test_throws ArgumentError cohen_kappa(Int[], Int[])
    @test_throws ArgumentError cohen_kappa([1, 2], [1])

    @test auc_score(SV945_AUC_SCORES, SV945_AUC_LABELS) ≈ SV945_AUC rtol = 1e-12
    @test auc_score(SV945_AUC_SCORES, Bool.(SV945_AUC_LABELS)) ≈ SV945_AUC rtol = 1e-12
    @test auc_score([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 1.0
    @test auc_score([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    @test_throws ArgumentError auc_score([0.1, 0.2], [1, 1])
    @test_throws ArgumentError auc_score([0.1, 0.2], [0, 2])
    @test_throws ArgumentError auc_score([0.1, 0.2], [0])
end

@testset "Ch945 audit decisions" begin
    @test cohens_d(SV945_D_X, SV945_D_Y) ≈ SV945_COHENS_D rtol = 1e-12
    @test_throws ArgumentError cohens_d([1.0], [1.0, 2.0])
    @test_throws ArgumentError cohens_d([1.0, 1.0], [1.0, 1.0])

    @test tost_p_value(SV945_TOST_PASS_ARGS...) ≈ SV945_TOST_PASS rtol = 1e-12
    @test tost_p_value(SV945_TOST_FAIL_ARGS...) ≈ SV945_TOST_FAIL rtol = 1e-12
    @test tost_p_value(SV945_TOST_PASS_ARGS...) < 0.05
    @test tost_p_value(SV945_TOST_FAIL_ARGS...) > 0.05
    @test_throws ArgumentError tost_p_value(0.0, 0.0, 0.1)
    @test_throws ArgumentError tost_p_value(0.0, 0.04, 0.0)
    @test_throws ArgumentError tost_p_value(NaN, 0.04, 0.1)
end

@testset "Ch945 power planning" begin
    @test design_effect(5, 0.2) ≈ SV945_DEFF_5_02
    @test samples_per_arm(0.45, 0.30) == SV945_N_045_030
    @test samples_per_arm(0.45, 0.30) isa Int
    @test samples_per_arm(0.45, 0.30; deff=SV945_DEFF_5_02) == SV945_N_045_030_DEFF
    @test samples_per_arm(0.5, 0.4; z_beta=SV945_Z_90) == SV945_N_05_04_POWER90
    @test_throws ArgumentError samples_per_arm(0.3, 0.3)
    @test_throws ArgumentError samples_per_arm(0.45, 0.30; deff=0.5)
    @test_throws ArgumentError samples_per_arm(1.5, 0.30)
    @test_throws ArgumentError design_effect(0.5, 0.2)
    @test_throws ArgumentError design_effect(5, 1.2)
end

@testset "Ch945 smoothed DPO margin" begin
    @test dpo_smoothed_margin(0.1) ≈ SV945_DPO_01 rtol = 1e-14
    @test dpo_smoothed_margin(0.25, 2.0) ≈ SV945_DPO_025_BETA2 rtol = 1e-14
    @test_throws ArgumentError dpo_smoothed_margin(0.0)
    @test_throws ArgumentError dpo_smoothed_margin(0.5)
    @test_throws ArgumentError dpo_smoothed_margin(0.1, 0.0)
end

@testset "Ch945 reasoning trees" begin
    @test collapse_repeats(SV945_COLLAPSE_IN) == SV945_COLLAPSE_OUT
    @test collapse_repeats(String[]) == String[]

    t = tree_from_paths(SV945_TREE_PATHS)
    @test t isa ReasoningTree
    @test t.root.count == SV945_TREE_ROOT_COUNT
    node = tree_node(t, ["understanding"])
    @test node isa TreeNode
    @test node.count == SV945_TREE_UNDERSTANDING_COUNT
    @test node.actions == SV945_TREE_UNDERSTANDING_ACTIONS
    @test path_probability(t, ["understanding", "self_interest", "B1"]) ≈ SV945_TREE_PATH_PROB
    @test path_probability(t, ["understanding", "altruism"]) == 0.0
    @test_throws KeyError tree_node(t, ["understanding", "altruism"])

    # Probabilities of complete paths sum to one.
    complete = unique([(collapse_repeats(c), a) for (c, a) in SV945_TREE_PATHS])
    @test length(complete) == 4
    total = sum(path_probability(t, vcat(c, [a])) for (c, a) in complete)
    @test total ≈ 1.0

    # add! builds the same tree incrementally.
    t2 = ReasoningTree()
    @test_throws ArgumentError path_probability(t2, ["understanding"])
    for (c, a) in SV945_TREE_PATHS
        add!(t2, c, a)
    end
    @test t2.root.count == SV945_TREE_ROOT_COUNT
    @test tree_node(t2, ["understanding"]).actions == SV945_TREE_UNDERSTANDING_ACTIONS
end
