"""Tests for aiinaction.ch138_fastica, the shared cross-language fixtures.

These fixtures are the single source of truth: the Julia and Rust suites assert
against the same numbers, which is what keeps the three implementations at parity.

The fixture is two synthetic sources mixed by a fixed 2x2 matrix
``A = [[2, 1], [1, 3]]`` over 8 samples, giving the observed matrix ``X`` below.
FastICA is run deterministically (identity init, 200 fixed iterations, Jacobi
symmetric orthogonalization), so the recovered operator is reproducible.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction.ch138_fastica import fit_ica, transform

# Observed mixtures X = S @ A^T (samples in rows, signals in columns).
X138 = [
    [1.0, -2.0],
    [0.0, 5.0],
    [4.0, 2.0],
    [-2.0, -6.0],
    [-3.0, 1.0],
    [4.0, 7.0],
    [-3.0, -4.0],
    [1.0, 3.0],
]

EXPECTED_MEAN = [0.25, 0.75]
EXPECTED_UNMIXING = [
    [0.7846180154937991, 0.6199794914048147],
    [-0.6199794914048147, 0.784618015493799],
]
EXPECTED_COMPONENTS = [
    [0.3534839303856012, 0.0016237326826006237],
    [0.2994403038579343, -0.2922019487902003],
]
EXPECTED_SOURCE_ROW0 = [0.2606476829120492, 1.0281355870665014]


def test_mean_matches_fixture():
    r = fit_ica(X138, n_components=2, max_iter=200)
    assert r.mean == pytest.approx(EXPECTED_MEAN)


def test_unmixing_matches_fixture():
    r = fit_ica(X138, n_components=2, max_iter=200)
    assert np.asarray(r.unmixing).ravel().tolist() == pytest.approx(
        np.asarray(EXPECTED_UNMIXING).ravel().tolist()
    )


def test_components_match_fixture():
    r = fit_ica(X138, n_components=2, max_iter=200)
    assert np.asarray(r.components).ravel().tolist() == pytest.approx(
        np.asarray(EXPECTED_COMPONENTS).ravel().tolist()
    )


def test_transform_first_row_matches_fixture():
    r = fit_ica(X138, n_components=2, max_iter=200)
    S = transform(r, X138)
    assert S[0].tolist() == pytest.approx(EXPECTED_SOURCE_ROW0)


def test_unmixing_is_orthogonal():
    r = fit_ica(X138, n_components=2, max_iter=200)
    W = np.asarray(r.unmixing)
    assert (W @ W.T).ravel().tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0], abs=1e-9)


def test_recovered_sources_are_uncorrelated():
    r = fit_ica(X138, n_components=2, max_iter=200)
    S = transform(r, X138)
    # Off-diagonal of the source covariance should vanish (independence implies
    # uncorrelatedness).
    c = np.cov(S, rowvar=False)
    assert c[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_mixing_inverts_components():
    r = fit_ica(X138, n_components=2, max_iter=200)
    # components @ mixing = I for the square full-rank case.
    prod = np.asarray(r.components) @ np.asarray(r.mixing)
    assert prod.ravel().tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0], abs=1e-9)


def test_reconstruction_from_sources():
    r = fit_ica(X138, n_components=2, max_iter=200)
    S = transform(r, X138)
    Xc = np.asarray(X138) - np.asarray(r.mean)
    recon = S @ np.asarray(r.mixing).T
    assert recon.ravel().tolist() == pytest.approx(Xc.ravel().tolist(), abs=1e-9)


def test_default_n_components_is_d():
    r = fit_ica(X138)
    assert r.n_components == 2
    assert r.n_features == 2


def test_n_iter_equals_max_iter():
    r = fit_ica(X138, n_components=2, max_iter=37)
    assert r.n_iter == 37


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least 2 samples"):
        fit_ica([[1.0, 2.0]])


def test_bad_n_components_raises():
    with pytest.raises(ValueError, match="n_components must be in"):
        fit_ica(X138, n_components=5)


def test_bad_max_iter_raises():
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        fit_ica(X138, n_components=2, max_iter=0)


def test_non_finite_raises():
    bad = [[1.0, 2.0], [float("nan"), 3.0]]
    with pytest.raises(ValueError, match="non-finite"):
        fit_ica(bad)


def test_transform_feature_mismatch_raises():
    r = fit_ica(X138, n_components=2)
    with pytest.raises(ValueError, match="features but model was fit on"):
        transform(r, [[1.0, 2.0, 3.0]])
