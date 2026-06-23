"""Tests for aiinaction.ch137_kernel_pca, the shared cross-language source of truth.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers (1e-9 tolerance), which keeps the three
implementations at parity. Components with tied-magnitude eigenvector entries have a
numerically arbitrary overall sign, so those are compared up to sign (absolute
value), exactly as the Rust and Julia suites do.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aiinaction.ch137_kernel_pca import (
    KernelPCAResult,
    fit_kernel_pca,
    kernel_matrix,
    transform,
)

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
X137 = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 1.0],
    [1.0, 2.0],
]
Z137 = [[0.5, 0.5]]
RBF = ("rbf", {"gamma": 0.5})
LINEAR = ("linear", {})

# RBF (gamma=0.5), n_components=2.
RBF_EIGENVALUES = [1.2737124650823262, 0.8646647167633879]
RBF_EVR = [0.42052544762205024, 0.28547535415413783]
RBF_ALPHA0 = [
    0.5236557939308785,
    0.24722334224292025,
    0.24722334224291986,
    -0.16977678408044233,
    -0.4241628471681383,
    -0.42416284716813824,
]
RBF_TRAIN_PROJ0 = [
    0.666986912142342,
    0.31489145267412133,
    0.3148914526741211,
    -0.21624680616484987,
    -0.5402615056628672,
    -0.5402615056628672,
]
RBF_ABS_TRAIN_PROJ1 = [
    0.0,
    0.46493674751609687,
    0.4649367475160969,
    0.0,
    0.4649367475160967,
    0.46493674751609687,
]
RBF_PROJ_Z0 = 0.3931535807027422

# Linear kernel, n_components=2.
LINEAR_EIGENVALUES = [3.666666666666668, 1.9999999999999991]
LINEAR_EVR = [0.6470588235294118, 0.352941176470588]
LINEAR_TRAIN_PROJ0 = [
    1.1785113019775793,
    0.47140452079103184,
    0.4714045207910317,
    -0.23570226039551584,
    -0.9428090415820635,
    -0.9428090415820635,
]

TOL = 1e-9


def _fit_rbf() -> KernelPCAResult:
    return fit_kernel_pca(X137, n_components=2, kernel=RBF)


def test_rbf_eigenvalues_match_fixture():
    m = _fit_rbf()
    assert m.eigenvalues == pytest.approx(RBF_EIGENVALUES, abs=TOL)


def test_rbf_explained_variance_ratio_matches_fixture():
    m = _fit_rbf()
    assert m.explained_variance_ratio == pytest.approx(RBF_EVR, abs=TOL)


def test_rbf_first_alpha_matches_fixture():
    m = _fit_rbf()
    assert list(m.alphas[:, 0]) == pytest.approx(RBF_ALPHA0, abs=TOL)


def test_rbf_train_projection_first_component_matches_fixture():
    m = _fit_rbf()
    proj = transform(m, X137)
    assert list(proj[:, 0]) == pytest.approx(RBF_TRAIN_PROJ0, abs=TOL)


def test_rbf_train_projection_second_component_up_to_sign():
    m = _fit_rbf()
    proj = transform(m, X137)
    assert list(np.abs(proj[:, 1])) == pytest.approx(RBF_ABS_TRAIN_PROJ1, abs=TOL)


def test_rbf_out_of_sample_projection_matches_fixture():
    m = _fit_rbf()
    pz = transform(m, Z137)
    assert pz[0, 0] == pytest.approx(RBF_PROJ_Z0, abs=TOL)


def test_train_projection_shortcut_equals_mu_times_alpha():
    # beta_i^k = mu_k * alpha_i^k for the training set.
    m = _fit_rbf()
    proj = transform(m, X137)
    expected = m.alphas * m.eigenvalues
    assert proj == pytest.approx(expected, abs=TOL)


def test_linear_kernel_eigenvalues_match_fixture():
    m = fit_kernel_pca(X137, n_components=2, kernel=LINEAR)
    assert m.eigenvalues == pytest.approx(LINEAR_EIGENVALUES, abs=TOL)


def test_linear_kernel_projection_matches_fixture():
    m = fit_kernel_pca(X137, n_components=2, kernel=LINEAR)
    proj = transform(m, X137)
    assert list(proj[:, 0]) == pytest.approx(LINEAR_TRAIN_PROJ0, abs=TOL)
    assert m.explained_variance_ratio == pytest.approx(LINEAR_EVR, abs=TOL)


def test_explained_variance_ratio_sums_to_at_most_one():
    m = fit_kernel_pca(X137, n_components=5, kernel=RBF)
    assert sum(m.explained_variance_ratio) <= 1.0 + TOL


def test_default_n_components_for_full_rank_kernel():
    # The RBF kernel is full rank here, so the default retains n - 1 components.
    m = fit_kernel_pca(X137, kernel=RBF)
    assert m.n_components == len(X137) - 1


def test_default_n_components_respects_kernel_rank():
    # The linear kernel on 2-D data has rank 2, so only 2 positive components exist.
    m = fit_kernel_pca(X137, kernel=LINEAR)
    assert m.n_components == 2


def test_kernel_matrix_linear_is_gram():
    K = kernel_matrix(X137, X137, LINEAR)
    expected = np.asarray(X137) @ np.asarray(X137).T
    assert K == pytest.approx(expected, abs=TOL)


def test_kernel_matrix_rbf_diagonal_is_one():
    K = kernel_matrix(X137, X137, RBF)
    assert list(np.diag(K)) == pytest.approx([1.0] * len(X137), abs=TOL)


def test_centered_kernel_alpha_normalization():
    # (alpha^k)^T alpha^k == 1 / mu_k.
    m = _fit_rbf()
    for k in range(m.n_components):
        norm_sq = float(m.alphas[:, k] @ m.alphas[:, k])
        assert norm_sq == pytest.approx(1.0 / m.eigenvalues[k], abs=TOL)


def test_too_few_samples_errors():
    with pytest.raises(ValueError):
        fit_kernel_pca([[1.0, 2.0]], kernel=LINEAR)


def test_bad_n_components_errors():
    with pytest.raises(ValueError):
        fit_kernel_pca(X137, n_components=6, kernel=LINEAR)


def test_unknown_kernel_errors():
    with pytest.raises(ValueError):
        fit_kernel_pca(X137, n_components=2, kernel=("sigmoid", {}))


def test_non_finite_input_errors():
    bad = [[0.0, 0.0], [1.0, math.nan], [0.0, 1.0]]
    with pytest.raises(ValueError):
        fit_kernel_pca(bad, n_components=1, kernel=LINEAR)


def test_transform_feature_mismatch_errors():
    m = _fit_rbf()
    with pytest.raises(ValueError):
        transform(m, [[1.0, 2.0, 3.0]])


def test_rbf_negative_gamma_errors():
    with pytest.raises(ValueError):
        kernel_matrix(X137, X137, ("rbf", {"gamma": -1.0}))
