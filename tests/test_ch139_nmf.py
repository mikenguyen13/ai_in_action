"""Tests for aiinaction.ch139_nmf, including the shared cross-language fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. All three use the identical deterministic LCG initializer, fill order, and
fixed iteration counts, so the iterates agree bit-for-bit up to 1e-9.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction.ch139_nmf import (
    fit_nmf,
    reconstruct,
    reconstruction_error,
    transform,
)
from aiinaction.ch139_nmf import _seeded_uniform

# Shared fixture: a clean block-diagonal matrix that NMF recovers near-exactly.
V_BLOCK = [
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 2.0],
    [0.0, 0.0, 2.0, 2.0],
]

# Shared fixture: a dense, low-rank-deficient matrix (rank-2 approximated).
V_DENSE = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
]

TOL = 1e-9


def test_seeded_uniform_matches_fixture():
    # The LCG initializer is the shared anchor for all three languages.
    out = _seeded_uniform(2, 2, 0)
    expected = np.array(
        [
            [0.2360679730223333, 0.2785669087249397],
            [0.8195337600029228, 0.6678668978466031],
        ]
    )
    assert out == pytest.approx(expected, abs=TOL)


def test_block_factors_match_fixture():
    r = fit_nmf(V_BLOCK, 2, max_iter=300, seed=0)
    expected_W = np.array(
        [
            [0.8504210044717101, 0.0],
            [0.8504210044717101, 0.0],
            [0.0, 1.3943383444007382],
            [0.0, 1.3943383444007382],
        ]
    )
    expected_H = np.array(
        [
            [1.175888171504758, 1.175888171504758, 0.0, 0.0],
            [0.0, 0.0, 1.4343720862275404, 1.4343720862275404],
        ]
    )
    assert r.W == pytest.approx(expected_W, abs=TOL)
    assert r.H == pytest.approx(expected_H, abs=TOL)
    assert r.n_components == 2
    assert r.n_features == 4


def test_block_reconstruction_is_near_exact():
    r = fit_nmf(V_BLOCK, 2, max_iter=300, seed=0)
    assert r.error == pytest.approx(0.0, abs=1e-6)
    recon = reconstruct(r)
    assert recon == pytest.approx(np.asarray(V_BLOCK), abs=1e-6)


def test_dense_factors_match_fixture():
    r = fit_nmf(V_DENSE, 2, max_iter=100, seed=7)
    expected_W = np.array(
        [
            [0.7214042811455682, 0.2319152540718632],
            [0.3793167032779777, 0.9034766028860397],
            [0.0933041476566348, 1.5614929744526524],
        ]
    )
    expected_H = np.array(
        [
            [1.8692478223292252e-03, 1.1082573543977547e00, 2.3545687125460262e00],
            [4.4657474657315506e00, 5.0631720186370552e00, 5.6307847244507254e00],
        ]
    )
    assert r.W == pytest.approx(expected_W, abs=TOL)
    assert r.H == pytest.approx(expected_H, abs=TOL)
    assert r.error == pytest.approx(0.06848010125776947, abs=TOL)


def test_factors_are_non_negative():
    r = fit_nmf(V_DENSE, 2, max_iter=100, seed=7)
    assert np.all(r.W >= 0.0)
    assert np.all(r.H >= 0.0)


def test_transform_matches_fixture():
    model = fit_nmf(V_BLOCK, 2, max_iter=300, seed=0)
    H_new = transform(model, V_BLOCK, max_iter=300)
    expected = np.array(
        [
            [1.1758881714856226, 1.1758881714856226, 0.0, 0.0],
            [0.0, 0.0, 1.4343720862268226, 1.4343720862268226],
        ]
    )
    assert H_new == pytest.approx(expected, abs=TOL)
    assert reconstruction_error(V_BLOCK, model.W, H_new) == pytest.approx(0.0, abs=1e-6)


def test_reconstruction_error_helper():
    err = reconstruction_error(
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0], [0.0]],
        [[1.0, 0.0]],
    )
    # Residual is [[0,0],[0,1]] -> Frobenius norm 1.
    assert err == pytest.approx(1.0, abs=TOL)


def test_negative_input_raises():
    with pytest.raises(ValueError, match="non-negative"):
        fit_nmf([[1.0, -1.0], [2.0, 3.0]], 1)


def test_non_finite_input_raises():
    with pytest.raises(ValueError, match="non-finite"):
        fit_nmf([[1.0, float("nan")], [2.0, 3.0]], 1)


def test_bad_n_components_raises():
    with pytest.raises(ValueError, match="n_components"):
        fit_nmf(V_BLOCK, 0)
    with pytest.raises(ValueError, match="n_components"):
        fit_nmf(V_DENSE, 5)


def test_bad_max_iter_raises():
    with pytest.raises(ValueError, match="max_iter"):
        fit_nmf(V_BLOCK, 2, max_iter=0)


def test_negative_seed_raises():
    with pytest.raises(ValueError, match="seed"):
        fit_nmf(V_BLOCK, 2, seed=-1)


def test_transform_feature_mismatch_raises():
    model = fit_nmf(V_BLOCK, 2, max_iter=50, seed=0)
    with pytest.raises(ValueError, match="features"):
        transform(model, [[1.0, 2.0], [3.0, 4.0]])


def test_reconstruction_error_shape_mismatch_raises():
    with pytest.raises(ValueError, match="rows"):
        reconstruction_error([[1.0, 2.0]], [[1.0], [2.0]], [[1.0, 2.0]])
