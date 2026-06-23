using Test
using AIInAction.Ch136Pca

# Shared fixtures: identical to the Python and Rust test suites.
const X136 = [
    2.5 2.4
    0.5 0.7
    2.2 2.9
    1.9 2.2
    3.1 3.0
    2.3 2.7
    2.0 1.6
    1.0 1.1
    1.5 1.6
    1.1 0.9
]

@testset "Ch136 PCA parity fixtures" begin
    r = fit_pca(X136; n_components=2)
    @test r.mean ≈ [1.81, 1.91]
    @test r.components[1, :] ≈ [0.6778733985280119, 0.735178655544408]
    @test r.components[2, :] ≈ [0.735178655544408, -0.6778733985280119]
    @test r.explained_variance ≈ [1.2840277121727839, 0.0490833989383273]
    @test r.explained_variance_ratio ≈ [0.963181314348646, 0.0368186856513541]

    scores = transform(r, X136)
    @test scores[1, :] ≈ [0.8279701862010882, 0.1751153070469155]

    r1 = fit_pca(X136; n_components=1)
    @test reconstruction_error(r1, X136) ≈ 0.04417505904449458
end

@testset "Ch136 PCA properties and edge cases" begin
    r = fit_pca(X136; n_components=2)
    @test reconstruction_error(r, X136) ≈ 0.0 atol = 1e-12
    @test r.components * r.components' ≈ [1.0 0.0; 0.0 1.0] atol = 1e-12
    @test sum(r.explained_variance_ratio) ≈ 1.0

    recovered = inverse_transform(r, transform(r, X136))
    @test recovered ≈ X136 atol = 1e-12

    rw = fit_pca(X136; n_components=2, whiten=true)
    sw = transform(rw, X136)
    n = size(X136, 1)
    v = vec(sum(abs2, sw .- sum(sw; dims=1) ./ n; dims=1) ./ (n - 1))
    @test v ≈ [1.0, 1.0] atol = 1e-9

    rs = fit_pca(X136; n_components=2, scale=true)
    # The second component is a tied-magnitude eigenvector whose overall sign is
    # numerically arbitrary, so compare it up to sign.
    @test rs.components[1, :] ≈ [0.7071067811865476, 0.7071067811865475]
    @test abs.(rs.components[2, :]) ≈ [0.7071067811865475, 0.7071067811865476]
    @test rs.explained_variance_ratio ≈ [0.9629646363461227, 0.0370353636538773]

    @test_throws ArgumentError fit_pca([1.0 2.0])
    @test_throws ArgumentError fit_pca(X136; n_components=5)
    @test_throws ArgumentError fit_pca([1.0 5.0; 2.0 5.0; 3.0 5.0]; n_components=2, scale=true)
end
