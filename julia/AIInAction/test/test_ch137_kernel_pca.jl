using Test
using AIInAction.Ch137KernelPca

# Shared fixtures: identical to the Python and Rust test suites.
const KPCA137_X = [
    0.0 0.0
    1.0 0.0
    0.0 1.0
    1.0 1.0
    2.0 1.0
    1.0 2.0
]
const KPCA137_Z = [0.5 0.5]
const KPCA137_RBF = ("rbf", Dict("gamma" => 0.5))
const KPCA137_LINEAR = ("linear", Dict{String,Float64}())

const KPCA137_RBF_EIGENVALUES = [1.2737124650823262, 0.8646647167633879]
const KPCA137_RBF_EVR = [0.42052544762205024, 0.28547535415413783]
const KPCA137_RBF_ALPHA0 = [
    0.5236557939308785,
    0.24722334224292025,
    0.24722334224291986,
    -0.16977678408044233,
    -0.4241628471681383,
    -0.42416284716813824,
]
const KPCA137_RBF_TRAIN_PROJ0 = [
    0.666986912142342,
    0.31489145267412133,
    0.3148914526741211,
    -0.21624680616484987,
    -0.5402615056628672,
    -0.5402615056628672,
]
const KPCA137_RBF_ABS_TRAIN_PROJ1 = [
    0.0,
    0.46493674751609687,
    0.4649367475160969,
    0.0,
    0.4649367475160967,
    0.46493674751609687,
]
const KPCA137_RBF_PROJ_Z0 = 0.3931535807027422

const KPCA137_LINEAR_EIGENVALUES = [3.666666666666668, 1.9999999999999991]
const KPCA137_LINEAR_EVR = [0.6470588235294118, 0.352941176470588]
const KPCA137_LINEAR_TRAIN_PROJ0 = [
    1.1785113019775793,
    0.47140452079103184,
    0.4714045207910317,
    -0.23570226039551584,
    -0.9428090415820635,
    -0.9428090415820635,
]

const KPCA137_TOL = 1e-9

@testset "Ch137 Kernel PCA parity fixtures (RBF)" begin
    m = fit_kernel_pca(KPCA137_X; n_components=2, kernel=KPCA137_RBF)
    @test m.eigenvalues ≈ KPCA137_RBF_EIGENVALUES atol = KPCA137_TOL
    @test m.explained_variance_ratio ≈ KPCA137_RBF_EVR atol = KPCA137_TOL
    @test m.alphas[:, 1] ≈ KPCA137_RBF_ALPHA0 atol = KPCA137_TOL

    proj = transform(m, KPCA137_X)
    @test proj[:, 1] ≈ KPCA137_RBF_TRAIN_PROJ0 atol = KPCA137_TOL
    # Second component sign is numerically arbitrary; compare up to sign.
    @test abs.(proj[:, 2]) ≈ KPCA137_RBF_ABS_TRAIN_PROJ1 atol = KPCA137_TOL

    pz = transform(m, KPCA137_Z)
    @test pz[1, 1] ≈ KPCA137_RBF_PROJ_Z0 atol = KPCA137_TOL

    # beta_i^k = mu_k * alpha_i^k for the training set.
    @test proj ≈ m.alphas .* m.eigenvalues' atol = KPCA137_TOL
end

@testset "Ch137 Kernel PCA linear kernel and properties" begin
    m = fit_kernel_pca(KPCA137_X; n_components=2, kernel=KPCA137_LINEAR)
    @test m.eigenvalues ≈ KPCA137_LINEAR_EIGENVALUES atol = KPCA137_TOL
    @test m.explained_variance_ratio ≈ KPCA137_LINEAR_EVR atol = KPCA137_TOL
    proj = transform(m, KPCA137_X)
    @test proj[:, 1] ≈ KPCA137_LINEAR_TRAIN_PROJ0 atol = KPCA137_TOL

    # alpha normalization: (alpha^k)^T alpha^k == 1 / mu_k.
    mr = fit_kernel_pca(KPCA137_X; n_components=2, kernel=KPCA137_RBF)
    for k in 1:2
        @test mr.alphas[:, k]' * mr.alphas[:, k] ≈ 1 / mr.eigenvalues[k] atol = KPCA137_TOL
    end

    # Default n_components respects kernel rank.
    @test size(fit_kernel_pca(KPCA137_X; kernel=KPCA137_LINEAR).alphas, 2) == 2
    @test size(fit_kernel_pca(KPCA137_X; kernel=KPCA137_RBF).alphas, 2) == 5

    # Linear Gram and RBF diagonal.
    K = kernel_matrix(KPCA137_X, KPCA137_X, KPCA137_LINEAR)
    @test K ≈ KPCA137_X * KPCA137_X' atol = KPCA137_TOL
    Kr = kernel_matrix(KPCA137_X, KPCA137_X, KPCA137_RBF)
    @test [Kr[i, i] for i in 1:size(KPCA137_X, 1)] ≈ ones(size(KPCA137_X, 1)) atol = KPCA137_TOL
end

@testset "Ch137 Kernel PCA edge cases" begin
    @test_throws ArgumentError fit_kernel_pca([1.0 2.0]; kernel=KPCA137_LINEAR)
    @test_throws ArgumentError fit_kernel_pca(KPCA137_X; n_components=6, kernel=KPCA137_LINEAR)
    @test_throws ArgumentError fit_kernel_pca(KPCA137_X; n_components=2,
        kernel=("sigmoid", Dict{String,Float64}()))
    @test_throws ArgumentError kernel_matrix(KPCA137_X, KPCA137_X,
        ("rbf", Dict("gamma" => -1.0)))
    m = fit_kernel_pca(KPCA137_X; n_components=2, kernel=KPCA137_RBF)
    @test_throws ArgumentError transform(m, [1.0 2.0 3.0])
end
