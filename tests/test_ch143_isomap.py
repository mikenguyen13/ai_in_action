"""Tests for aiinaction.ch143_isomap, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. The data are six points tracing an open zig-zag (an intrinsically 1-D path)
embedded in the plane, so Isomap with one component should unfold it onto a line.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aiinaction import ch143_isomap

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
X = [
    [0.0, 0.0],
    [1.0, 0.6],
    [2.2, 0.0],
    [3.2, 0.6],
    [4.6, 0.0],
    [6.2, 0.6],
]

# d(x_0, x_1) = sqrt(1.0^2 + 0.6^2) = sqrt(1.36).
STEP01 = math.sqrt(1.36)  # 1.16619037896906

EXPECTED = {
    "pdist_0_1": 1.16619037896906,
    "geodesic_0_5": 6.36619037896906,
    "geodesic_1_4": 4.030985786641715,
    "embedding1": [
        -2.9371839938074595,
        -2.0650406540284134,
        -0.7481485843361929,
        0.42401358188469096,
        1.9113800872248041,
        3.4149795630625714,
    ],
    "eigenvalue1": 28.946415792110297,
    "eigenvalues2": [28.946415792110297, 0.34784148875645043],
}

TOL = 1e-9


def test_pairwise_distance_fixture():
    d = ch143_isomap.pairwise_distances(X)
    assert d[0, 1] == pytest.approx(EXPECTED["pdist_0_1"], abs=TOL)
    assert d[0, 0] == 0.0
    # Symmetric.
    assert d.ravel().tolist() == pytest.approx(d.T.ravel().tolist())


def test_knn_graph_is_symmetric_and_sparse():
    adj = ch143_isomap.knn_graph(X, n_neighbors=2)
    assert np.array_equal(adj, adj.T) or np.allclose(adj, adj.T, equal_nan=True)
    # Diagonal is zero.
    assert all(adj[i, i] == 0.0 for i in range(adj.shape[0]))
    # Endpoint 0 reaches neighbors 1 and 2 only (finite); 3,4,5 are inf.
    assert math.isfinite(adj[0, 1])
    assert math.isfinite(adj[0, 2])
    assert math.isinf(adj[0, 4])


def test_geodesic_distances_fixture():
    r = ch143_isomap.fit_isomap(X, n_components=1, n_neighbors=2)
    geo = r.geodesic_distances
    assert geo[0, 5] == pytest.approx(EXPECTED["geodesic_0_5"], abs=TOL)
    assert geo[1, 4] == pytest.approx(EXPECTED["geodesic_1_4"], abs=TOL)
    # Geodesic from 0 to 5 is the 5-hop path, strictly longer than the chord.
    chord = ch143_isomap.pairwise_distances(X)[0, 5]
    assert geo[0, 5] > chord


def test_embedding1_matches_fixture():
    r = ch143_isomap.fit_isomap(X, n_components=1, n_neighbors=2)
    assert r.embedding.ravel().tolist() == pytest.approx(EXPECTED["embedding1"], abs=TOL)


def test_eigenvalue1_matches_fixture():
    r = ch143_isomap.fit_isomap(X, n_components=1, n_neighbors=2)
    assert r.eigenvalues.tolist() == pytest.approx([EXPECTED["eigenvalue1"]], abs=TOL)


def test_eigenvalues_two_components_match_fixture():
    r = ch143_isomap.fit_isomap(X, n_components=2, n_neighbors=2)
    assert r.eigenvalues.tolist() == pytest.approx(EXPECTED["eigenvalues2"], abs=TOL)


def test_embedding_preserves_order():
    # The unfolded 1-D coordinate should be monotone along the path.
    r = ch143_isomap.fit_isomap(X, n_components=1, n_neighbors=2)
    y = r.embedding.ravel()
    assert all(y[i] < y[i + 1] for i in range(len(y) - 1))


def test_embedding_is_centered():
    r = ch143_isomap.fit_isomap(X, n_components=1, n_neighbors=2)
    assert float(r.embedding.sum()) == pytest.approx(0.0, abs=1e-9)


def test_sign_convention_is_deterministic():
    r = ch143_isomap.fit_isomap(X, n_components=2, n_neighbors=2)
    for j in range(r.embedding.shape[1]):
        col = r.embedding[:, j]
        k = int(np.argmax(np.abs(col)))
        assert col[k] > 0.0


def test_result_shapes_and_properties():
    r = ch143_isomap.fit_isomap(X, n_components=2, n_neighbors=3)
    assert r.embedding.shape == (6, 2)
    assert r.n_samples == 6
    assert r.n_components == 2
    assert r.n_neighbors == 3
    assert r.geodesic_distances.shape == (6, 6)


def test_graph_shortest_paths_simple_chain():
    inf = float("inf")
    # Chain 0 - 1 - 2 with unit edges.
    adj = [
        [0.0, 1.0, inf],
        [1.0, 0.0, 1.0],
        [inf, 1.0, 0.0],
    ]
    d = ch143_isomap.graph_shortest_paths(adj)
    assert d[0][2] == pytest.approx(2.0)
    assert d[0][1] == pytest.approx(1.0)


def test_classical_mds_recovers_euclidean_geometry():
    # If distances are exactly Euclidean, MDS reproduces the configuration
    # (up to rotation/reflection); here the recovered 1-D coords reproduce the
    # original spacing.
    pts = [[0.0], [1.0], [2.0], [4.0]]
    D = ch143_isomap.pairwise_distances(pts)
    Y, vals = ch143_isomap.classical_mds(D, 1)
    # Reconstructed pairwise distances must match the originals.
    rec = ch143_isomap.pairwise_distances(Y.tolist())
    assert rec.ravel().tolist() == pytest.approx(np.asarray(D).ravel().tolist(), abs=1e-9)


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least 2 samples"):
        ch143_isomap.fit_isomap([[1.0, 2.0]])


def test_non_finite_raises():
    with pytest.raises(ValueError, match="non-finite"):
        ch143_isomap.fit_isomap([[1.0, 2.0], [float("nan"), 3.0], [3.0, 4.0]])


def test_bad_n_neighbors_raises():
    with pytest.raises(ValueError, match="n_neighbors must be in"):
        ch143_isomap.fit_isomap(X, n_neighbors=10)


def test_bad_n_components_raises():
    with pytest.raises(ValueError, match="n_components must be in"):
        ch143_isomap.fit_isomap(X, n_components=7, n_neighbors=2)


def test_disconnected_graph_raises():
    # Two well-separated clusters of two points, k=1 keeps them disconnected.
    far = [
        [0.0, 0.0],
        [0.1, 0.0],
        [100.0, 0.0],
        [100.1, 0.0],
    ]
    with pytest.raises(ValueError, match="disconnected"):
        ch143_isomap.fit_isomap(far, n_components=1, n_neighbors=1)
