using Test
using AIInAction.Ch139Nmf

# Shared fixtures: identical to the Python and Rust test suites.
const CH139_V_BLOCK = [
    1.0 1.0 0.0 0.0
    1.0 1.0 0.0 0.0
    0.0 0.0 2.0 2.0
    0.0 0.0 2.0 2.0
]

const CH139_V_DENSE = [
    1.0 2.0 3.0
    4.0 5.0 6.0
    7.0 8.0 9.0
]

const CH139_TOL = 1e-9

@testset "Ch139 NMF parity fixtures" begin
    r = fit_nmf(CH139_V_BLOCK, 2; max_iter=300, seed=0)
    expected_W = [
        0.8504210044717101 0.0
        0.8504210044717101 0.0
        0.0 1.3943383444007382
        0.0 1.3943383444007382
    ]
    expected_H = [
        1.175888171504758 1.175888171504758 0.0 0.0
        0.0 0.0 1.4343720862275404 1.4343720862275404
    ]
    @test r.W ≈ expected_W atol = CH139_TOL
    @test r.H ≈ expected_H atol = CH139_TOL
    @test r.error ≈ 0.0 atol = 1e-6

    rd = fit_nmf(CH139_V_DENSE, 2; max_iter=100, seed=7)
    expected_Wd = [
        0.7214042811455682 0.2319152540718632
        0.3793167032779777 0.9034766028860397
        0.0933041476566348 1.5614929744526524
    ]
    expected_Hd = [
        1.8692478223292252e-3 1.1082573543977547 2.3545687125460262
        4.4657474657315506 5.0631720186370552 5.6307847244507254
    ]
    @test rd.W ≈ expected_Wd atol = CH139_TOL
    @test rd.H ≈ expected_Hd atol = CH139_TOL
    @test rd.error ≈ 0.06848010125776947 atol = CH139_TOL
end

@testset "Ch139 NMF properties and edge cases" begin
    model = fit_nmf(CH139_V_BLOCK, 2; max_iter=300, seed=0)
    @test all(x -> x >= 0.0, model.W)
    @test all(x -> x >= 0.0, model.H)
    @test reconstruct(model) ≈ CH139_V_BLOCK atol = 1e-6

    Hn = transform(model, CH139_V_BLOCK; max_iter=300)
    expected_Hn = [
        1.1758881714856226 1.1758881714856226 0.0 0.0
        0.0 0.0 1.4343720862268226 1.4343720862268226
    ]
    @test Hn ≈ expected_Hn atol = CH139_TOL
    @test reconstruction_error(CH139_V_BLOCK, model.W, Hn) ≈ 0.0 atol = 1e-6

    @test reconstruction_error([1.0 0.0; 0.0 1.0], reshape([1.0, 0.0], 2, 1), [1.0 0.0]) ≈ 1.0 atol = CH139_TOL

    @test_throws ArgumentError fit_nmf([1.0 -1.0; 2.0 3.0], 1)
    @test_throws ArgumentError fit_nmf([1.0 NaN; 2.0 3.0], 1)
    @test_throws ArgumentError fit_nmf(CH139_V_BLOCK, 0)
    @test_throws ArgumentError fit_nmf(CH139_V_DENSE, 5)
    @test_throws ArgumentError fit_nmf(CH139_V_BLOCK, 2; max_iter=0)
    @test_throws ArgumentError fit_nmf(CH139_V_BLOCK, 2; seed=-1)
    @test_throws ArgumentError transform(model, [1.0 2.0; 3.0 4.0])
end
