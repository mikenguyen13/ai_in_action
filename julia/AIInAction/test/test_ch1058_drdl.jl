using Test
using AIInAction.Ch1058Drdl

# Shared fixtures: identical to the Python (tests/test_ch1058_drdl.py) and Rust suites.
# Names are prefixed with D1058_ so they cannot collide with other test files.
# Rows are companies, columns are horizons. Times are 1-based horizon indices in all
# three languages; comparable pairs are 1-based here (Python: 0-based); code indices
# stay 0-based to match the Python/Rust fixtures.
const D1058_OUTPUTS = [-1.0 0.5 -0.3;
                       0.2 -2.0 0.1;
                       -3.0 -1.0 1.0;
                       0.0 0.0 0.0]
const D1058_CUM_PROBS = [0.3077993724446536 0.8668928334755382 0.9365446603138965;
                         0.7051836792708419 0.7425006392477038 0.9147279853227372;
                         0.048568007099546597 0.3414181774380723 0.9565414790857123;
                         0.6321205588285577 0.8646647167633873 0.950212931632136]
const D1058_TIMES = [1, 2, 3, 3]
const D1058_EVENTS = [1, 1, 1, 0]
const D1058_PAIRS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]  # Python: (0, 1), ...
const D1058_FSL_G1 = 0.0696463795589742
const D1058_FSL_G05 = 0.1259078886595625
const D1058_FSL_G2 = 0.023133174339533064
const D1058_BCE = 2.376713158199545
const D1058_CINDEX = 0.5
const D1058_AUC = [1 / 3, 0.75, 1 / 3]
const D1058_BRIER = [0.34459024562633167, 0.23705872071716394, 0.22902278877569854]

const D1058_EMB = [0.1 0.9; 1.0 0.0; 0.6 0.6; -0.9 0.1]
const D1058_CODEBOOK = [1.0 0.0; 0.0 1.0; -1.0 0.0]
const D1058_CODES = [1, 0, 0, 2]  # 0-based code indices
const D1058_MASK_CODES = [2, 0, 2, 1, 0, 3]
const D1058_MASK = [1, 1, 0, 1, 0, 1]
const D1058_ORTH_CODEBOOK = [1.0 0.0; 1.0 1.0; 0.0 2.0]
const D1058_ORTH = 1.0
const D1058_SIGMA_03 = ((0.5, 0.2672140914204164), (1.0, 0.14888503362331806),
                        (2.0, 0.04496964958360028))
const D1058_PROJ_MAIN = [1.0, 0.0, 1.0]
const D1058_PROJ_AUX = [-1.0, 2.0, 0.0]
const D1058_PROJ_OUT = [-0.5, 2.0, 0.5]

# A perfectly ranked risk matrix for times [1, 2, 3] and events [1, 1, 0].
const D1058_RANKED = [0.9 0.95 0.99; 0.1 0.8 0.9; 0.05 0.1 0.2]

# Elementwise closeness with matching shapes.
function _d1058_close(a, b; atol::Real=1e-12)
    size(a) == size(b) || return false
    for (x, y) in zip(a, b)
        abs(x - y) <= atol || return false
    end
    return true
end

function _d1058_dot(x::AbstractVector, y::AbstractVector)
    s = 0.0
    for k in 1:length(x)
        s += x[k] * y[k]
    end
    return s
end

# Deterministic 64-bit LCG (Knuth MMIX constants) returning uniforms in [-1, 1), so the
# property test needs no Random dependency and is identical on every Julia version.
function _d1058_lcg_uniforms(count::Integer, seed::UInt64)
    state = seed
    out = Vector{Float64}(undef, count)
    for k in 1:count
        state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f
        out[k] = 2.0 * (Float64(state >> 11) / 9007199254740992.0) - 1.0
    end
    return out
end

@testset "Ch1058 risk accumulation fixture and monotonicity" begin
    f = risk_accumulation(D1058_OUTPUTS)
    @test f isa Matrix{Float64}
    @test _d1058_close(f, D1058_CUM_PROBS)
    @test all(f[i, u + 1] > f[i, u] for i in 1:size(f, 1), u in 1:(size(f, 2) - 1))
    @test all(x -> 0 < x < 1, f)
    v = risk_accumulation(D1058_OUTPUTS[1, :])
    @test v isa Vector{Float64}
    @test _d1058_close(v, D1058_CUM_PROBS[1, :])
end

@testset "Ch1058 risk accumulation is a cloglog hazard model" begin
    # F(t_u) = 1 - prod_{j<=u} (1 - h_j) with h_j = 1 - exp(-exp(o_j)).
    o = D1058_OUTPUTS
    expected = similar(o)
    for i in 1:size(o, 1)
        surv = 1.0
        for u in 1:size(o, 2)
            h = 1.0 - exp(-exp(o[i, u]))
            surv *= 1.0 - h
            expected[i, u] = 1.0 - surv
        end
    end
    @test _d1058_close(risk_accumulation(o), expected)
end

@testset "Ch1058 risk accumulation validation" begin
    @test_throws ArgumentError risk_accumulation(Float64[])
    @test_throws ArgumentError risk_accumulation([1.0, NaN])
    @test_throws ArgumentError risk_accumulation(zeros(2, 0))
end

@testset "Ch1058 nearest codes and first-occurrence mask" begin
    @test nearest_codes(D1058_EMB, D1058_CODEBOOK) == D1058_CODES
    @test first_occurrence_mask(D1058_MASK_CODES) == D1058_MASK
    @test first_occurrence_mask(Int[]) == Int[]
    @test_throws ArgumentError nearest_codes([1.0 2.0 3.0], D1058_CODEBOOK)
    @test_throws ArgumentError first_occurrence_mask([0, -1])
end

@testset "Ch1058 nearest codes tie goes to lowest index" begin
    @test nearest_codes([0.0 0.0], [1.0 0.0; -1.0 0.0]) == [0]
end

@testset "Ch1058 orthogonal loss" begin
    @test orthogonal_loss(D1058_ORTH_CODEBOOK) ≈ D1058_ORTH atol = 1e-12
    identity4 = [i == j ? 1.0 : 0.0 for i in 1:4, j in 1:4]
    @test orthogonal_loss(identity4) ≈ 0.0 atol = 1e-15
    @test orthogonal_loss([2.0 0.0; 5.0 0.0]) ≈ 1.0
    @test_throws ArgumentError orthogonal_loss([0.0 0.0; 1.0 0.0])
end

@testset "Ch1058 focal sigma" begin
    for (g, v) in D1058_SIGMA_03
        @test focal_sigma(0.3, g) ≈ v atol = 1e-12
    end
    @test focal_sigma(0.0, 0.7) == 0.0
    @test focal_sigma(50.0, 1.0) ≈ 1.0
    @test_throws ArgumentError focal_sigma(-0.1, 1.0)
    @test_throws ArgumentError focal_sigma(0.1, 0.0)
end

@testset "Ch1058 comparable pairs fixture" begin
    @test comparable_pairs(D1058_TIMES, D1058_EVENTS) == D1058_PAIRS
end

@testset "Ch1058 focal survival loss fixture" begin
    @test focal_survival_loss(D1058_CUM_PROBS, D1058_TIMES, D1058_EVENTS; gamma=1.0) ≈
          D1058_FSL_G1 atol = 1e-12
    @test focal_survival_loss(D1058_CUM_PROBS, D1058_TIMES, D1058_EVENTS) ≈
          D1058_FSL_G1 atol = 1e-12
    @test focal_survival_loss(D1058_CUM_PROBS, D1058_TIMES, D1058_EVENTS; gamma=0.5) ≈
          D1058_FSL_G05 atol = 1e-12
    @test focal_survival_loss(D1058_CUM_PROBS, D1058_TIMES, D1058_EVENTS; gamma=2.0) ≈
          D1058_FSL_G2 atol = 1e-12
end

@testset "Ch1058 focal survival loss is zero when perfectly ranked" begin
    @test focal_survival_loss(D1058_RANKED, [1, 2, 3], [1, 1, 0]) == 0.0
    @test focal_survival_loss(D1058_RANKED, [3, 3, 3], [0, 0, 0]) == 0.0
end

@testset "Ch1058 multiperiod BCE fixture" begin
    @test multiperiod_bce(D1058_CUM_PROBS, D1058_TIMES, D1058_EVENTS) ≈ D1058_BCE atol = 1e-12
    y = cumulative_labels(D1058_TIMES, D1058_EVENTS, 3)
    @test y isa Matrix{Float64}
    @test y == [1 1 1; 0 1 1; 0 0 1; 0 0 0]
end

@testset "Ch1058 survival target validation" begin
    @test_throws ArgumentError multiperiod_bce(D1058_CUM_PROBS, [0, 1, 2, 3], D1058_EVENTS)
    @test_throws ArgumentError multiperiod_bce(D1058_CUM_PROBS, [1, 2, 3, 4], D1058_EVENTS)
    @test_throws ArgumentError multiperiod_bce(D1058_CUM_PROBS, D1058_TIMES, [1, 2, 0, 0])
    @test_throws ArgumentError cumulative_labels(D1058_TIMES, D1058_EVENTS, 0)
    @test_throws ArgumentError td_auc(D1058_CUM_PROBS, D1058_TIMES, D1058_EVENTS, 0)
    @test_throws ArgumentError td_auc(D1058_CUM_PROBS, D1058_TIMES, D1058_EVENTS, 4)
end

@testset "Ch1058 project conflicting gradients" begin
    out = project_conflicting(D1058_PROJ_MAIN, D1058_PROJ_AUX)
    @test _d1058_close(out, D1058_PROJ_OUT)
    @test abs(_d1058_dot(out, D1058_PROJ_MAIN)) <= 1e-12
    @test _d1058_close(project_conflicting([1.0, 0.0], [0.5, 1.0]), [0.5, 1.0])
    @test _d1058_close(project_conflicting([0.0, 0.0], [0.5, 1.0]), [0.5, 1.0])
    @test_throws ArgumentError project_conflicting([1.0, 0.0], [1.0, 0.0, 0.0])
end

@testset "Ch1058 combined step never conflicts with the main loss" begin
    u = _d1058_lcg_uniforms(200 * 12, UInt64(0))
    ok = true
    conflicts = 0
    for k in 1:200
        base = 12 * (k - 1)
        a = u[(base + 1):(base + 6)]
        b = u[(base + 7):(base + 12)]
        _d1058_dot(a, b) < 0 && (conflicts += 1)
        total = a .+ project_conflicting(a, b)
        ok &= _d1058_dot(total, a) >= _d1058_dot(a, a) - 1e-9
    end
    @test ok
    @test conflicts > 0  # the projection branch was actually exercised
end

@testset "Ch1058 metrics fixture" begin
    @test td_concordance(D1058_CUM_PROBS, D1058_TIMES, D1058_EVENTS) ≈ D1058_CINDEX atol = 1e-12
    for h in 1:3
        @test td_auc(D1058_CUM_PROBS, D1058_TIMES, D1058_EVENTS, h) ≈ D1058_AUC[h] atol = 1e-12
    end
    @test _d1058_close(brier_scores(D1058_CUM_PROBS, D1058_TIMES, D1058_EVENTS), D1058_BRIER)
end

@testset "Ch1058 concordance perfect and reversed" begin
    @test td_concordance(D1058_RANKED, [1, 2, 3], [1, 1, 0]) == 1.0
    @test td_concordance(D1058_RANKED[end:-1:1, :], [1, 2, 3], [1, 1, 0]) == 0.0
    @test_throws ArgumentError td_concordance(D1058_RANKED, [3, 3, 3], [0, 0, 0])
end
