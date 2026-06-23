using Test
using AIInAction.Ch138Fastica

# Shared fixtures: identical to the Python and Rust test suites.
# Observed mixtures X = S * A^T (samples in rows, signals in columns).
const X138 = [
    1.0 -2.0
    0.0 5.0
    4.0 2.0
    -2.0 -6.0
    -3.0 1.0
    4.0 7.0
    -3.0 -4.0
    1.0 3.0
]

const X138_MEAN = [0.25, 0.75]
const X138_UNMIXING = [
    0.7846180154937991 0.6199794914048147
    -0.6199794914048147 0.784618015493799
]
const X138_COMPONENTS = [
    0.3534839303856012 0.0016237326826006237
    0.2994403038579343 -0.2922019487902003
]
const X138_SOURCE_ROW0 = [0.2606476829120492, 1.0281355870665014]

@testset "Ch138 FastICA parity fixtures" begin
    r = fit_ica(X138; n_components=2, max_iter=200)
    @test r.mean ≈ X138_MEAN
    @test r.unmixing ≈ X138_UNMIXING
    @test r.components ≈ X138_COMPONENTS

    S = transform(r, X138)
    @test S[1, :] ≈ X138_SOURCE_ROW0
end

@testset "Ch138 FastICA properties and edge cases" begin
    r = fit_ica(X138; n_components=2, max_iter=200)
    # Unmixing rotation is orthogonal.
    @test r.unmixing * r.unmixing' ≈ [1.0 0.0; 0.0 1.0] atol = 1e-9
    # Mixing inverts the unmixing operator.
    @test r.components * r.mixing ≈ [1.0 0.0; 0.0 1.0] atol = 1e-9

    # Recovered sources are uncorrelated.
    S = transform(r, X138)
    n = size(X138, 1)
    Smu = vec(sum(S; dims=1) ./ n)
    Sc = S .- Smu'
    cov12 = sum(Sc[:, 1] .* Sc[:, 2]) / (n - 1)
    @test cov12 ≈ 0.0 atol = 1e-9

    # Reconstruction from sources recovers the centered data.
    Xc = X138 .- X138_MEAN'
    @test S * r.mixing' ≈ Xc atol = 1e-9

    # Default n_components is d.
    rdef = fit_ica(X138)
    @test size(rdef.components, 1) == 2

    # Fixed iteration count.
    r37 = fit_ica(X138; n_components=2, max_iter=37)
    @test r37.n_iter == 37

    @test_throws ArgumentError fit_ica([1.0 2.0])
    @test_throws ArgumentError fit_ica(X138; n_components=5)
    @test_throws ArgumentError fit_ica(X138; n_components=2, max_iter=0)
    @test_throws ArgumentError transform(r, [1.0 2.0 3.0])
end
