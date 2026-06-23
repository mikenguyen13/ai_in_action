using Test
using AIInAction.Ch161Calibration

# Shared fixtures: identical to the Python and Rust test suites. Names are prefixed
# with CAL161_ so they cannot collide with other test files.
const CAL161_CONF = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3]
const CAL161_CORRECT = [1, 1, 0, 1, 1, 0, 1, 0, 0, 1]

@testset "Ch161 calibration parity fixtures" begin
    @test expected_calibration_error(CAL161_CONF, CAL161_CORRECT; n_bins=5) ≈ 0.195
    @test maximum_calibration_error(CAL161_CONF, CAL161_CORRECT; n_bins=5) ≈ 0.7
    @test expected_calibration_error(CAL161_CONF, CAL161_CORRECT; n_bins=10) ≈ 0.285
    @test brier_score(CAL161_CONF, CAL161_CORRECT) ≈ 0.23274999999999996

    rc = reliability_curve(CAL161_CONF, CAL161_CORRECT; n_bins=5)
    @test length(rc.bins) == 5
    @test rc.n_samples == 10
    occ = occupied(rc)
    @test [b.count for b in occ] == [1, 3, 2, 4]
    @test occ[1].accuracy ≈ 1.0
    @test occ[1].confidence ≈ 0.3
    @test gap(occ[1]) ≈ 0.7
    @test occ[4].count == 4
    @test occ[4].accuracy ≈ 0.75
    @test occ[4].confidence ≈ 0.875
end

@testset "Ch161 calibration properties and edge cases" begin
    rc = reliability_curve(CAL161_CONF, CAL161_CORRECT; n_bins=5)
    @test rc.bins[1].count == 0
    @test rc.bins[1].accuracy == 0.0
    @test rc.bins[1].confidence == 0.0

    cal161_pc = [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75]
    cal161_pcorr = [1, 0, 0, 0, 1, 1, 1, 0]
    @test expected_calibration_error(cal161_pc, cal161_pcorr; n_bins=2) ≈ 0.0 atol = 1e-12
    @test maximum_calibration_error(cal161_pc, cal161_pcorr; n_bins=2) ≈ 0.0 atol = 1e-12

    rc1 = reliability_curve([1.0, 1.0], [1, 0]; n_bins=4)
    @test rc1.bins[end].count == 2
    @test rc1.bins[end].accuracy ≈ 0.5
    @test rc1.bins[end].confidence ≈ 1.0

    @test_throws ArgumentError expected_calibration_error([0.5, 0.5], [1]; n_bins=5)
    @test_throws ArgumentError expected_calibration_error(Float64[], Int[]; n_bins=5)
    @test_throws ArgumentError expected_calibration_error(CAL161_CONF, CAL161_CORRECT; n_bins=0)
    @test_throws ArgumentError reliability_curve([1.5], [1]; n_bins=5)
    @test_throws ArgumentError reliability_curve([0.5], [2]; n_bins=5)
end
