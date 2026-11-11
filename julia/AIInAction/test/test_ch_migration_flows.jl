using Test
using LinearAlgebra: eigvals
using AIInAction.ChMigrationFlows

# Shared fixtures: identical to the Python (tests/test_ch_migration_flows.py) and Rust
# suites. Names are prefixed with MF_ so they cannot collide with other test files.
const MF_YJ_X = [-5.0, -0.3, 0.0, 2.0, 10.0]
const MF_YJ_07 = [-3.57877726629599, -0.28800168756136063, 0.0, 1.6538132571065618, 6.225223813548735]
const MF_YJ_00 = [-1.791759469228055, -0.262364264467491, 0.0, 1.0986122886681098, 2.3978952727983707]
const MF_YJ_M05 = [-1.183503419072274, -0.24588396138594154, 0.0, 0.8452994616207485, 1.3969773108444727]
const MF_YJ_DERIV_07 = [0.5841906810678655, 0.9243086009945788, 1.0, 0.7192230933248643, 0.4870596972258284]
const MF_YJ_SAMPLE = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 40.0, 100.0, -1.0, -3.0, -7.0]
const MF_YJ_LOGLIK_05 = -33.338467549439706
const MF_YJ_FIT = -0.01
const MF_ELASTICITY = 0.8307692307692308

const MF_STOCK = [100.0 5.0 2.0; 3.0 80.0 4.0; 1.0 6.0 60.0]
const MF_BIRTHS = [1.5, 1.0, 0.8]
const MF_MORT = [0.01, 0.02, 0.015]
const MF_STEP = [100.2 4.9 2.27; 2.37 79.4 4.54; 0.0900000000000003 5.88 60.800000000000004]
const MF_OD = [0.0 2.4000000000000004 3.6000000000000005; 1.8000000000000003 0.0 5.4; 2.4000000000000004 4.800000000000001 0.0]
const MF_NET = [-1.7999999999999998, 0.0, 1.799999999999999]

const MF_IPF_TABLE = [10.0 20.0 30.0; 15.0 25.0 5.0; 8.0 2.0 40.0]
const MF_IPF_ROWS = [60.0, 50.0, 50.0]
const MF_IPF_COLS = [45.0, 55.0, 60.0]
const MF_IPF = [13.584306505781386 24.27835009115688 22.137343404288874; 18.723560813157828 27.886171700008326 3.3902674980586913; 12.692132681060787 2.835478208834802 34.47238909765244]

const MF_BAL_S1 = [50.0 5.0 2.0; 3.0 40.0 4.0; 1.0 6.0 30.0]
const MF_BAL_S2 = [48.0 7.0 3.0; 4.0 42.0 3.0; 2.0 5.0 33.0]
const MF_BAL_BIRTHS = [2.0, 1.5, 1.0]
const MF_BAL1 = [49.68615791114227 5.279992903186189 2.1138491735349096; 2.877097428817112 40.78378334376581 4.079119232039964; 0.9767446600406219 6.226223753048012 31.19703159442512]
const MF_BAL2 = [48.32400928274534 6.712813076986998 2.8831776522759487; 4.118486084752095 41.19277564057946 2.9487382738222503; 2.0175046325025683 4.80441128243354 31.7780840739018]
const MF_REL1 = [0.006276841777154658 0.055998580637237796 0.05692458676745482; 0.04096752372762932 0.019594583594145298 0.019779808009990907; 0.02325533995937812 0.03770395884133526 0.039901053147503994]
const MF_REL2 = [0.006750193390527848 0.04102670328757174 0.03894078257468377; 0.029621521188023747 0.019219627605250894 0.017087242059249903; 0.008752316251284142 0.03911774351329207 0.03702775533630914]

const MF_WEIGHT_INPUT = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
const MF_WEIGHTS = [2.0, 1.5238095238095237, 0.7619047619047619, 0.5, 0.5, 0.5]
const MF_WL2 = 1.21875
const MF_SURVIVAL = [1.0, 0.99, 0.9702, 0.9410939999999999]
const MF_CALIBRATE = 1.22456821045488
const MF_STOCK_NEXT = [110.0 3.0 9.0; 2.0 90.0 1.0; 4.0 6.0 55.0]
const MF_BASELINE = [0.0 0.0 1.4; 0.0 0.0 0.0; 0.6 0.0 0.0]
const MF_GRAVITY = -6.6899999999999995
const MF_LOG_RMSE = 0.18451760994274952
const MF_PEARSON = 0.9986517555689657
const MF_ENSEMBLE = [1.0 2.0; 2.0 4.0; 4.0 5.0; 5.0 9.0]
const MF_ENSEMBLE_MEAN = [3.0, 5.0]
const MF_ENSEMBLE_STD = [1.8257418583505538, 2.943920288775949]

const MF_GATED_Z = [0.5, -0.2, 0.0, 1.5]
const MF_GATED_AC = [0.3, -1.0, 2.0, 0.0]
const MF_GATED_AG = [0.0, 2.0, -1.0, 3.0]
const MF_GATED_NEXT = [0.39565630622579545, -0.266943664380472, 0.7047606324503499, 1.42886119023365]
const MF_GATED_DZ = [0.5, 0.8807970779778823, 0.2689414213699951, 0.9525741268224334]
const MF_GATED_DAC = [0.4575684809133146, 0.05006216869470697, 0.051649891596191815, 0.047425873177566635]
const MF_GATED_DAG = [0.052171846887102274, 0.058963783975451815, -0.18953932621681382, 0.067764989596368]
const MF_CARRY_BIRTH = [0.012, 0.01]
const MF_CARRY_MORT = [0.01, 0.008]
const MF_CARRY = [1.002 0.0 0.012 0.0; 0.0 0.992 0.0 0.0; 0.0 0.0 0.99 0.0; 0.0 0.01 0.0 1.002]

function mf_flows()
    t = zeros(3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3
        j != k && (t[i, j, k] = 0.1 * i * (j + 1) * k)  # Python: 0.1 (i+1)(j+2)(k+1), 0-based
    end
    return t
end

@testset "ChMigrationFlows Yeo-Johnson" begin
    @test yj_sym(MF_YJ_X, 0.7) ≈ MF_YJ_07 atol = 1e-12
    @test yj_sym(MF_YJ_X, 0.0) ≈ MF_YJ_00 atol = 1e-12
    @test yj_sym(MF_YJ_X, -0.5) ≈ MF_YJ_M05 atol = 1e-12
    @test yj_sym(MF_YJ_X, 1.0) ≈ MF_YJ_X atol = 1e-12
    @test [yj_sym_deriv(x, 0.7) for x in MF_YJ_X] ≈ MF_YJ_DERIV_07 atol = 1e-12
    for lam in (0.7, 0.0, 1.0, 1.6)
        @test yj_sym_inv(yj_sym(MF_YJ_X, lam), lam) ≈ MF_YJ_X atol = 1e-10
        h = 1e-6
        num = [(yj_sym(x + h, lam) - yj_sym(x - h, lam)) / (2h) for x in MF_YJ_X]
        @test num ≈ [yj_sym_deriv(x, lam) for x in MF_YJ_X] atol = 1e-6
    end
    @test_throws ArgumentError yj_sym_inv(5.0, -0.5)
    @test_throws ArgumentError yj_sym(NaN, 0.5)
    @test_throws ArgumentError yj_profile_loglik([1.0], 0.5)
    @test yj_profile_loglik(MF_YJ_SAMPLE, 0.5) ≈ MF_YJ_LOGLIK_05 atol = 1e-10
    @test fit_yj_lambda(MF_YJ_SAMPLE) ≈ MF_YJ_FIT atol = 1e-12
    @test elasticity_chain(0.9, 12.0, 0.0) ≈ MF_ELASTICITY atol = 1e-12
    @test elasticity_chain(1.0, 1e6, 0.0) ≈ 1e6 / (1e6 + 1.0) atol = 1e-12
end

@testset "ChMigrationFlows accounting" begin
    t = mf_flows()
    nxt = stock_step(MF_STOCK, MF_BIRTHS, MF_MORT, t)
    @test nxt ≈ MF_STEP atol = 1e-12
    expected_change = sum(MF_BIRTHS) - sum(MF_MORT' .* MF_STOCK)
    @test sum(nxt) - sum(MF_STOCK) ≈ expected_change atol = 1e-12
    f = od_flows(t)
    @test f ≈ MF_OD atol = 1e-12
    @test net_migration(f) ≈ MF_NET atol = 1e-12
    @test abs(sum(net_migration(f))) < 1e-12
    @test_throws ArgumentError stock_step(MF_STOCK, MF_BIRTHS, [0.1, 0.2, 1.5], t)
    bad = copy(t)
    bad[1, 1, 2] = -1.0
    @test_throws ArgumentError stock_step(MF_STOCK, MF_BIRTHS, MF_MORT, bad)
    @test_throws ArgumentError stock_step(MF_STOCK, MF_BIRTHS[1:2], MF_MORT, t)
end

@testset "ChMigrationFlows ipf and balancing" begin
    fit, iters = ipf(MF_IPF_TABLE, MF_IPF_ROWS, MF_IPF_COLS)
    @test fit ≈ MF_IPF atol = 1e-8
    @test vec(sum(fit, dims=2)) ≈ MF_IPF_ROWS atol = 1e-6
    @test vec(sum(fit, dims=1)) ≈ MF_IPF_COLS atol = 1e-6
    @test 1 < iters < 100
    @test_throws ArgumentError ipf(MF_IPF_TABLE, MF_IPF_ROWS, [45.0, 55.0, 70.0])
    @test_throws ArgumentError ipf([0.0 0.0; 1.0 1.0], [1.0, 1.0], [1.0, 1.0])
    deaths = 0.02 .* MF_BAL_S2
    b1, b2, r1, r2 = demographic_balance(MF_BAL_S1, MF_BAL_S2, MF_BAL_BIRTHS, deaths)
    @test b1 ≈ MF_BAL1 atol = 1e-8
    @test b2 ≈ MF_BAL2 atol = 1e-8
    @test r1 ≈ MF_REL1 atol = 1e-8
    @test r2 ≈ MF_REL2 atol = 1e-8
end

@testset "ChMigrationFlows loss and calibration" begin
    w = uncertainty_weights(MF_WEIGHT_INPUT)
    @test w ≈ MF_WEIGHTS atol = 1e-12
    @test minimum(w) >= 0.5 && maximum(w) <= 2.0
    @test issorted(w, rev=true)
    @test weighted_l2([1.0, 2.0, 3.0, 4.0], [1.5, 1.5, 3.5, 2.0], [1.0, 0.5, 2.0, 1.0]) ≈ MF_WL2 atol = 1e-12
    @test_throws ArgumentError weighted_l2([1.0], [1.0, 2.0], [1.0])
    @test_throws ArgumentError uncertainty_weights([-0.1, 0.2])
    s = survival_fraction([0.01, 0.02, 0.03])
    @test s ≈ MF_SURVIVAL atol = 1e-12
    @test calibrate_offset([10.0, 12.0, 14.0, 15.0], [9.0, 10.5, 13.0, 13.2], [1.0, 1.0, 2.0, 0.5], s) ≈ MF_CALIBRATE atol = 1e-12
    @test calibrate_offset([-1.0, -1.0, -1.0], [2.0, 1.0, 3.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]) ≈ -1.0 atol = 1e-12
    pred = [5.0, 6.0, 7.0, 8.0]
    @test calibrate_offset(pred .+ 2.5 .* s, pred, ones(4), s) ≈ 2.5 atol = 1e-12
    @test_throws ArgumentError calibrate_offset([1.0], [1.0, 2.0], [1.0], [1.0])
end

@testset "ChMigrationFlows baseline and metrics" begin
    @test stock_difference_baseline(MF_STOCK, MF_STOCK_NEXT, 5.0) ≈ MF_BASELINE atol = 1e-12
    @test_throws ArgumentError stock_difference_baseline(MF_STOCK, MF_STOCK_NEXT, 0.0)
    @test gravity_log_rate([-8.7, 0.8, -0.7, 0.8, 0.5], 1.0, 2.0, -0.3, 1.0, 0.4) ≈ MF_GRAVITY atol = 1e-12
    @test log_rmse([1.0, 2.0, 3.0, 4.0], [1.1, 1.8, 3.6, 3.0]) ≈ MF_LOG_RMSE atol = 1e-12
    @test pearson([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1]) ≈ MF_PEARSON atol = 1e-12
    @test isnan(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    @test coverage([1.0, 2.0, 3.0, 4.0], [1.1, 2.5, 2.9, 3.0], fill(0.2, 4)) == 0.5
    m, sd = ensemble_mean_std(MF_ENSEMBLE)
    @test m ≈ MF_ENSEMBLE_MEAN atol = 1e-12
    @test sd ≈ MF_ENSEMBLE_STD atol = 1e-12
    @test_throws ArgumentError ensemble_mean_std([1.0 2.0])
end

@testset "ChMigrationFlows recurrence" begin
    z, ac, ag = MF_GATED_Z, MF_GATED_AC, MF_GATED_AG
    @test gated_latent_update(z, ac, ag) ≈ MF_GATED_NEXT atol = 1e-12
    g, dc, dg = gated_latent_partials(z, ac, ag)
    @test g ≈ MF_GATED_DZ atol = 1e-12
    @test dc ≈ MF_GATED_DAC atol = 1e-12
    @test dg ≈ MF_GATED_DAG atol = 1e-12
    h = 1e-6
    f = gated_latent_update
    @test (f(z .+ h, ac, ag) .- f(z .- h, ac, ag)) ./ (2h) ≈ g atol = 1e-8
    @test (f(z, ac .+ h, ag) .- f(z, ac .- h, ag)) ./ (2h) ≈ dc atol = 1e-8
    @test (f(z, ac, ag .+ h) .- f(z, ac, ag .- h)) ./ (2h) ≈ dg atol = 1e-8
    @test all(abs.(f([0.99, -0.99, 0.0], [50.0, -50.0, 3.0], [-3.0, 0.0, 4.0])) .< 1.0)
    @test f(z, ac, fill(40.0, 4)) ≈ z atol = 1e-12
    @test f(z, ac, fill(-40.0, 4)) ≈ tanh.(ac) atol = 1e-12
    @test_throws ArgumentError f([0.1, 0.2], [0.1], [0.1, 0.2])
    @test_throws ArgumentError f([0.1], [NaN], [0.0])

    a = stock_carry_jacobian(MF_CARRY_BIRTH, MF_CARRY_MORT)
    @test a ≈ MF_CARRY atol = 1e-12
    @test sort(real.(eigvals(a))) ≈ sort(vcat(1 .- MF_CARRY_MORT, 1 .- MF_CARRY_MORT .+ MF_CARRY_BIRTH)) atol = 1e-12
    t = mf_flows()
    br = [0.012, 0.01, 0.015]
    step(s) = stock_step(s, br .* vec(sum(s, dims=1)), MF_MORT, t)
    a3 = stock_carry_jacobian(br, MF_MORT)
    rowmajor(m) = vec(permutedims(m))
    @test rowmajor(step(MF_STOCK) .- step(MF_STOCK_NEXT)) ≈ a3 * rowmajor(MF_STOCK .- MF_STOCK_NEXT) atol = 1e-12
    @test_throws ArgumentError stock_carry_jacobian([0.01, 0.02], [0.01])
    @test_throws ArgumentError stock_carry_jacobian([0.01], [1.5])
end
