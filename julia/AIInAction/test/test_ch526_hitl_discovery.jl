using Test
using Random
using AIInAction.Ch526HitlDiscovery

# Shared fixtures: identical to the Python (tests/test_ch526_hitl_discovery.py) and Rust
# suites. Names are prefixed with HD526_ so they cannot collide with other test files.
# Molecule indices are 1-based here, so every Python index i appears as i + 1.
const HD526_H_HALF = 0.6931471805599453  # binary_entropy(0.5)
const HD526_H_09 = 0.3250829733914482  # binary_entropy(0.9)

# S=3 draws, N=2 molecules, K=2 properties; Python PROBS[s][n][k] is PROBS[s, n, k] here.
const HD526_PROBS_NESTED = [
    [[0.9, 0.2], [0.5, 0.5]],
    [[0.1, 0.2], [0.5, 0.5]],
    [[0.5, 0.2], [0.9, 0.1]],
]
const HD526_PROBS = [HD526_PROBS_NESTED[s][n][k] for s in 1:3, n in 1:2, k in 1:2]
const HD526_MU = [0.5 0.2; 0.6333333333333333 0.3666666666666667]
const HD526_ALEATORIC = [0.44777104244761395 0.5004024235381879;
                         0.5704591115037796 0.5704591115037796]
const HD526_EPISTEMIC = [0.24537613811233133 0.0;
                         0.08669864999356092 0.08669864999356092]
const HD526_UCB_BETA2 = [1.9727922061357857, 2.363003220017392]  # sigma_d = sqrt(mu (1 - mu)), beta = 2

const HD526_TOPK_SCORES = [0.3, 0.9, 0.9, 0.1, 0.5]
const HD526_TOPK_3 = [1, 2, 4] .+ 1
const HD526_TOPK_3_EXCLUDING_1 = [2, 4, 0] .+ 1

const HD526_MERGE_UNC = [5, 2, 7] .+ 1
const HD526_MERGE_SRCH = [2, 9, 5] .+ 1
const HD526_MERGED = [2, 9, 5, 7] .+ 1

const HD526_TARGETS = Bool[1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
const HD526_KNOWN = Bool[1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
const HD526_SEARCH_LIST = [0, 1, 3, 4, 6, 7] .+ 1
const HD526_UNC_LIST = [2, 5, 8, 9, 3] .+ 1
const HD526_EXPERT_PICK_5 = [0, 3, 6, 2, 5] .+ 1  # explore_share 0.5, budget 5

const HD526_HIT_30_OF_2500 = 0.012
const HD526_RECALL_30_OF_148 = 0.20270270270270271

@testset "Ch526 binary entropy" begin
    @test binary_entropy(0.5) ≈ HD526_H_HALF
    @test binary_entropy(0.9) ≈ HD526_H_09
    @test binary_entropy(0.0) ≈ 0.0 atol = 1e-9
    @test binary_entropy(0.1) ≈ binary_entropy(0.9)
    @test binary_entropy([0.5, 0.9]) ≈ [HD526_H_HALF, HD526_H_09]
end

@testset "Ch526 uncertainty decomposition" begin
    mu, al, ep = decompose_uncertainty(HD526_PROBS)
    @test size(mu) == (2, 2)
    @test mu ≈ HD526_MU atol = 1e-12
    @test al ≈ HD526_ALEATORIC atol = 1e-9
    @test ep ≈ HD526_EPISTEMIC atol = 1e-9

    # S x N input means a single property
    mu1, _, _ = decompose_uncertainty(HD526_PROBS[:, :, 1])
    @test size(mu1) == (2, 1)
    @test vec(mu1) ≈ HD526_MU[:, 1] atol = 1e-12

    rng = Xoshiro(0)
    ps = 0.01 .+ 0.98 .* rand(rng, 8, 30, 3)
    mu, al, ep = decompose_uncertainty(ps)
    @test al .+ ep ≈ binary_entropy.(mu) atol = 1e-9
    @test all(ep .>= 0)
    same = repeat(ps[1:1, :, :], 8, 1, 1)
    @test maximum(decompose_uncertainty(same)[3]) < 1e-9
end

@testset "Ch526 ucb scores" begin
    got = ucb_scores(HD526_MU, sqrt.(HD526_MU .* (1 .- HD526_MU)), 2.0)
    @test got ≈ HD526_UCB_BETA2 atol = 1e-12
    # beta = 0 reduces to the summed mean
    @test ucb_scores(HD526_MU, sqrt.(HD526_MU .* (1 .- HD526_MU)), 0.0) ≈ vec(sum(HD526_MU; dims=2))
    @test_throws ArgumentError ucb_scores(HD526_MU, zeros(3, 2), 1.0)
end

@testset "Ch526 top_k and merge" begin
    @test top_k(HD526_TOPK_SCORES, 3) == HD526_TOPK_3
    mask = falses(5)
    mask[2] = true  # Python index 1
    @test top_k(HD526_TOPK_SCORES, 3, mask) == HD526_TOPK_3_EXCLUDING_1
    @test length(top_k(HD526_TOPK_SCORES, 10, mask)) == 4

    @test merge_shortlists(HD526_MERGE_UNC, HD526_MERGE_SRCH) == HD526_MERGED
end

@testset "Ch526 expert" begin
    e = Expert(1.0, 0.5)
    e.known_mask = copy(HD526_KNOWN)
    e.known_target = copy(HD526_TARGETS)
    @test expert_select(e, HD526_SEARCH_LIST, HD526_UNC_LIST, 5) == HD526_EXPERT_PICK_5

    # Everything is a known non-target: the budget is still spent from the dropped ones.
    e2 = Expert(1.0, 0.0)
    e2.known_mask = trues(10)
    e2.known_target = falses(10)
    pick = expert_select(e2, [1, 2, 3], [4, 5, 6], 4)
    @test length(pick) == 4 && length(unique(pick)) == 4

    @test_throws ArgumentError expert_select(Expert(0.5), [1], [2], 1)

    # reveal! is deterministic given the RNG and respects the knowledge rate
    e3 = Expert(0.5)
    reveal!(e3, trues(2000), Xoshiro(1))
    @test 0.45 < count(e3.known_mask) / 2000 < 0.55
    @test e3.known_target == trues(2000)
    e4 = Expert(0.5)
    reveal!(e4, trues(2000), Xoshiro(1))
    @test e3.known_mask == e4.known_mask
end

@testset "Ch526 metrics" begin
    @test hit_rate(30, 50, 50) ≈ HD526_HIT_30_OF_2500
    @test recall_rate(30, 148) ≈ HD526_RECALL_30_OF_148
    @test is_target([1 1 1; 1 0 1]) == [true, false]
end

# Julia port of the Python _synthetic generator: the target is the intersection of two
# half-spaces. Own RNG, so the pool is statistically (not bit-for-bit) the same.
function hd526_synthetic(n=1500, d=8, seed=0)
    rng = Xoshiro(seed)
    x = randn(rng, n, d)
    y = hcat(Int.(x[:, 1] .+ 0.3 .* x[:, 3] .> 1.2), Int.(x[:, 2] .- 0.3 .* x[:, 4] .> 1.0))
    return x, y
end

function hd526_fp(seed=0)
    return (xl, yl) -> begin
        ens = BootstrapLogisticEnsemble(; n_members=6, epochs=80, seed=seed)
        fit!(ens, xl, yl)
        return xq -> predict_samples(ens, xq)
    end
end

@testset "Ch526 bootstrap ensemble learns" begin
    x, y = hd526_synthetic()
    ens = BootstrapLogisticEnsemble(; n_members=8, epochs=150, seed=1)
    @test_throws ArgumentError predict_samples(ens, x)
    @test fit!(ens, x[1:600, :], y[1:600, :]) === ens
    ps = predict_samples(ens, x[601:end, :])
    @test size(ps) == (8, 900, 2)
    mu = dropdims(sum(ps; dims=1); dims=1) ./ 8
    acc = sum((mu .> 0.5) .== (y[601:end, :] .== 1)) / length(mu)
    @test acc > 0.85
end

@testset "Ch526 simulation shapes and budget" begin
    x, y = hd526_synthetic()
    r = simulate("hitl", x, y, hd526_fp(); budget=20, rounds=6, q=20, h=20,
                 knowledge=0.5, seed=3)
    @test r isa SimResult
    @test length(r.hits_by_round) == 6
    @test issorted(r.hits_by_round)
    @test r.hits_by_round[end] <= r.n_targets
    @test hit_rate(r) ≈ r.hits_by_round[end] / (20 * 6)
    @test length(recall_by_round(r)) == 6
    @test_throws ArgumentError simulate("bogus", x, y, hd526_fp(); rounds=1)
end

@testset "Ch526 every strategy runs" begin
    x, y = hd526_synthetic()
    for s in STRATEGIES
        r = simulate(s, x, y, hd526_fp(); budget=10, rounds=4, q=10, h=10,
                     warmup_rounds=2, human_view=60, seed=2)
        @test length(r.hits_by_round) == 4
        @test issorted(r.hits_by_round)
        @test r.hits_by_round[end] <= r.n_targets
    end
    @test length(STRATEGIES) == 9
end

@testset "Ch526 hitl beats random on the synthetic pool" begin
    x, y = hd526_synthetic()
    rnd = sum(simulate("random", x, y, hd526_fp(s); budget=20, rounds=10, seed=s).hits_by_round[end]
              for s in 0:3) / 4
    hit = sum(simulate("hitl", x, y, hd526_fp(s); budget=20, rounds=10, q=20, h=20,
                       knowledge=0.75, seed=s).hits_by_round[end] for s in 0:3) / 4
    @test hit > 2 * rnd
end

@testset "Ch526 more expert knowledge does not hurt" begin
    x, y = hd526_synthetic()
    avg(k) = sum(simulate("hitl", x, y, hd526_fp(s); budget=20, rounds=10, q=20, h=20,
                          knowledge=k, seed=s).hits_by_round[end] for s in 0:4) / 5
    @test avg(1.0) >= avg(0.0)
end
