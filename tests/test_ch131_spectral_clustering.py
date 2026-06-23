"""Tests for aiinaction.ch131_spectral_clustering and the shared fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the identical numbers and labels, which is what keeps the three
language ports at parity (1e-9 tolerance).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aiinaction import ch131_spectral_clustering as sc

# ---------------------------------------------------------------------------
# Shared fixtures (identical in julia/AIInAction/test and rust/aiinaction/src).
# ---------------------------------------------------------------------------

# Two well-separated blobs in 2-D: indices 0,1,2 vs 3,4,5.
X_BLOBS = [
    [0.0, 0.0],
    [0.2, 0.1],
    [0.1, -0.2],
    [5.0, 5.0],
    [5.2, 4.9],
    [4.9, 5.1],
]
SIGMA = 1.0

# Expected cluster labels (the first point seeds cluster 0).
EXPECTED_LABELS = [0, 0, 0, 1, 1, 1]

# RBF affinity reference entries.
AFF_W01 = math.exp(-((0.2) ** 2 + (0.1) ** 2) / (2.0 * SIGMA * SIGMA))  # ~0.9753099120283326

# Symmetric eigenproblem fixture A = [[2,1],[1,2]] -> eigenvalues {1, 3}.
A_SYM = [[2.0, 1.0], [1.0, 2.0]]
A_EIGVALS = [1.0, 3.0]
INV_SQRT2 = 1.0 / math.sqrt(2.0)

# k-means fixture.
KM_POINTS = [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 9.9]]
KM_LABELS = [0, 0, 1, 1]
KM_CENTERS = [[0.05, 0.0], [10.05, 9.95]]

TOL = 1e-9


# ---------------------------------------------------------------------------
# rbf_affinity
# ---------------------------------------------------------------------------

def test_rbf_affinity_symmetric_zero_diagonal():
    W = sc.rbf_affinity(X_BLOBS, SIGMA)
    assert W.shape == (6, 6)
    assert np.allclose(W, W.T, atol=TOL)
    assert np.allclose(np.diag(W), 0.0, atol=TOL)


def test_rbf_affinity_reference_entry():
    W = sc.rbf_affinity(X_BLOBS, SIGMA)
    assert W[0, 1] == pytest.approx(AFF_W01, abs=TOL)
    # Cross-cluster similarity is essentially zero.
    assert W[0, 3] < 1e-9


def test_rbf_affinity_rejects_bad_sigma():
    with pytest.raises(ValueError, match="sigma"):
        sc.rbf_affinity(X_BLOBS, 0.0)
    with pytest.raises(ValueError, match="sigma"):
        sc.rbf_affinity(X_BLOBS, -1.0)


def test_rbf_affinity_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        sc.rbf_affinity([1.0, 2.0, 3.0], SIGMA)


# ---------------------------------------------------------------------------
# normalized_laplacian
# ---------------------------------------------------------------------------

def test_normalized_laplacian_unit_diagonal_and_symmetric():
    W = sc.rbf_affinity(X_BLOBS, SIGMA)
    L = sc.normalized_laplacian(W)
    assert np.allclose(np.diag(L), 1.0, atol=TOL)
    assert np.allclose(L, L.T, atol=TOL)


def test_normalized_laplacian_psd_smallest_eigenvalue_zero():
    W = sc.rbf_affinity(X_BLOBS, SIGMA)
    L = sc.normalized_laplacian(W)
    vals, _ = sc.jacobi_eigh(L)
    assert vals[0] == pytest.approx(0.0, abs=1e-8)
    assert all(v >= -1e-8 for v in vals)


def test_normalized_laplacian_rejects_isolated_vertex():
    W = [[0.0, 0.0], [0.0, 0.0]]
    with pytest.raises(ValueError, match="positive degree"):
        sc.normalized_laplacian(W)


def test_normalized_laplacian_rejects_asymmetric():
    with pytest.raises(ValueError, match="symmetric"):
        sc.normalized_laplacian([[0.0, 1.0], [2.0, 0.0]])


# ---------------------------------------------------------------------------
# jacobi_eigh
# ---------------------------------------------------------------------------

def test_jacobi_eigh_values_ascending():
    vals, vecs = sc.jacobi_eigh(A_SYM)
    assert vals[0] == pytest.approx(A_EIGVALS[0], abs=TOL)
    assert vals[1] == pytest.approx(A_EIGVALS[1], abs=TOL)


def test_jacobi_eigh_orthonormal_sign_fixed_vectors():
    vals, vecs = sc.jacobi_eigh(A_SYM)
    # Columns orthonormal.
    gram = vecs.T @ vecs
    assert np.allclose(gram, np.eye(2), atol=TOL)
    # Eigenvector for lambda=1 is (1, -1)/sqrt2 (sign-fixed: first entry +).
    assert vecs[0, 0] == pytest.approx(INV_SQRT2, abs=TOL)
    assert vecs[1, 0] == pytest.approx(-INV_SQRT2, abs=TOL)
    # Reconstruction A v = lambda v.
    Av = np.asarray(A_SYM) @ vecs[:, 0]
    assert np.allclose(Av, vals[0] * vecs[:, 0], atol=TOL)


def test_jacobi_eigh_diagonal_matrix():
    vals, _ = sc.jacobi_eigh([[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    assert [round(v, 9) for v in vals] == [1.0, 2.0, 4.0]


def test_jacobi_eigh_rejects_asymmetric():
    with pytest.raises(ValueError, match="symmetric"):
        sc.jacobi_eigh([[1.0, 2.0], [3.0, 4.0]])


# ---------------------------------------------------------------------------
# spectral_embedding
# ---------------------------------------------------------------------------

def test_spectral_embedding_shape_and_unit_rows():
    W = sc.rbf_affinity(X_BLOBS, SIGMA)
    U = sc.spectral_embedding(W, 2)
    assert U.shape == (6, 2)
    row_norms = np.sqrt((U * U).sum(axis=1))
    assert np.allclose(row_norms, 1.0, atol=TOL)


def test_spectral_embedding_rejects_bad_k():
    W = sc.rbf_affinity(X_BLOBS, SIGMA)
    with pytest.raises(ValueError, match="k must be"):
        sc.spectral_embedding(W, 0)
    with pytest.raises(ValueError, match="k must be"):
        sc.spectral_embedding(W, 7)


# ---------------------------------------------------------------------------
# kmeans
# ---------------------------------------------------------------------------

def test_kmeans_labels_and_centers():
    labels, centers = sc.kmeans(KM_POINTS, 2)
    assert labels == KM_LABELS
    assert np.allclose(np.asarray(centers), np.asarray(KM_CENTERS), atol=TOL)


def test_kmeans_single_cluster():
    labels, centers = sc.kmeans(KM_POINTS, 1)
    assert labels == [0, 0, 0, 0]
    assert np.allclose(centers[0], np.asarray(KM_POINTS).mean(axis=0), atol=TOL)


def test_kmeans_rejects_bad_k():
    with pytest.raises(ValueError, match="k must be"):
        sc.kmeans(KM_POINTS, 5)


# ---------------------------------------------------------------------------
# spectral_clustering end-to-end
# ---------------------------------------------------------------------------

def test_spectral_clustering_separates_blobs():
    labels = sc.spectral_clustering(X_BLOBS, 2, SIGMA)
    assert labels == EXPECTED_LABELS


def test_spectral_clustering_label_consistency():
    # The two blobs must land in two distinct clusters of size 3 each.
    labels = sc.spectral_clustering(X_BLOBS, 2, SIGMA)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
