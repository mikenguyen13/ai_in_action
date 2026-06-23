"""Tests for aiinaction.ch136_pca, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. The data matrix is the classic 2-D PCA tutorial set (Smith, 2002).
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch136_pca

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
X = [
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9],
]

EXPECTED = {
    "mean": [1.81, 1.91],
    "components": [
        [0.6778733985280119, 0.735178655544408],
        [0.735178655544408, -0.6778733985280119],
    ],
    "explained_variance": [1.2840277121727839, 0.0490833989383273],
    "explained_variance_ratio": [0.963181314348646, 0.0368186856513541],
    "score_row0": [0.8279701862010882, 0.1751153070469155],
    "recon_error_1": 0.04417505904449458,
}


def test_mean_matches_fixture():
    r = ch136_pca.fit_pca(X, n_components=2)
    assert r.mean == pytest.approx(EXPECTED["mean"])


def test_components_match_fixture():
    r = ch136_pca.fit_pca(X, n_components=2)
    assert r.components.ravel().tolist() == pytest.approx(
        np.asarray(EXPECTED["components"]).ravel().tolist()
    )


def test_explained_variance_matches_fixture():
    r = ch136_pca.fit_pca(X, n_components=2)
    assert r.explained_variance.tolist() == pytest.approx(EXPECTED["explained_variance"])


def test_explained_variance_ratio_matches_fixture():
    r = ch136_pca.fit_pca(X, n_components=2)
    assert r.explained_variance_ratio.tolist() == pytest.approx(
        EXPECTED["explained_variance_ratio"]
    )


def test_transform_first_row_matches_fixture():
    r = ch136_pca.fit_pca(X, n_components=2)
    scores = ch136_pca.transform(r, X)
    assert scores[0].tolist() == pytest.approx(EXPECTED["score_row0"])


def test_reconstruction_error_one_component():
    r = ch136_pca.fit_pca(X, n_components=1)
    err = ch136_pca.reconstruction_error(r, X)
    assert err == pytest.approx(EXPECTED["recon_error_1"])


def test_full_rank_reconstruction_is_exact():
    r = ch136_pca.fit_pca(X, n_components=2)
    err = ch136_pca.reconstruction_error(r, X)
    assert err == pytest.approx(0.0, abs=1e-12)


def test_components_are_orthonormal():
    r = ch136_pca.fit_pca(X, n_components=2)
    gram = r.components @ r.components.T
    assert gram.ravel().tolist() == pytest.approx(np.eye(2).ravel().tolist(), abs=1e-12)


def test_explained_variance_ratio_sums_to_one_full_rank():
    r = ch136_pca.fit_pca(X, n_components=2)
    assert float(r.explained_variance_ratio.sum()) == pytest.approx(1.0)


def test_inverse_transform_round_trip():
    r = ch136_pca.fit_pca(X, n_components=2)
    scores = ch136_pca.transform(r, X)
    recovered = ch136_pca.inverse_transform(r, scores)
    assert recovered.ravel().tolist() == pytest.approx(
        np.asarray(X, dtype=float).ravel().tolist(), abs=1e-12
    )


def test_whiten_gives_unit_variance_scores():
    r = ch136_pca.fit_pca(X, n_components=2, whiten=True)
    scores = ch136_pca.transform(r, X)
    var = scores.var(axis=0, ddof=1)
    assert var.tolist() == pytest.approx([1.0, 1.0], abs=1e-9)


def test_scaled_pca_components_match():
    r = ch136_pca.fit_pca(X, n_components=2, scale=True)
    # Correlation PCA of this 2-feature set yields the +/-45-degree axes. The first
    # component is well-determined; the second is a tied-magnitude eigenvector whose
    # overall sign is numerically arbitrary, so compare it up to sign (abs values).
    assert r.components[0].tolist() == pytest.approx(
        [0.7071067811865476, 0.7071067811865475]
    )
    assert np.abs(r.components[1]).tolist() == pytest.approx(
        [0.7071067811865475, 0.7071067811865476]
    )
    assert r.explained_variance_ratio.tolist() == pytest.approx(
        [0.9629646363461227, 0.0370353636538773]
    )


def test_sign_convention_is_deterministic():
    # Largest-magnitude entry of each loading row must be positive.
    r = ch136_pca.fit_pca(X, n_components=2)
    for row in r.components:
        k = int(np.argmax(np.abs(row)))
        assert row[k] > 0.0


def test_default_n_components_is_min_n_d():
    r = ch136_pca.fit_pca(X)
    assert r.n_components == 2


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least 2 samples"):
        ch136_pca.fit_pca([[1.0, 2.0]])


def test_non_finite_raises():
    with pytest.raises(ValueError, match="non-finite"):
        ch136_pca.fit_pca([[1.0, 2.0], [float("nan"), 3.0]])


def test_bad_n_components_raises():
    with pytest.raises(ValueError, match="n_components must be in"):
        ch136_pca.fit_pca(X, n_components=5)


def test_transform_feature_mismatch_raises():
    r = ch136_pca.fit_pca(X, n_components=2)
    with pytest.raises(ValueError, match="features but model was fit on"):
        ch136_pca.transform(r, [[1.0, 2.0, 3.0]])


def test_scale_zero_variance_feature_raises():
    bad = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
    with pytest.raises(ValueError, match="zero variance"):
        ch136_pca.fit_pca(bad, n_components=2, scale=True)
