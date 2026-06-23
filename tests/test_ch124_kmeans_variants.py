"""Tests for aiinaction.ch124_kmeans_variants and the shared cross-language fixtures.

The numeric fixtures in this file are the single source of truth. The Julia suite
(``julia/AIInAction/test/test_ch124_kmeans_variants.jl``) and the Rust inline tests
(``rust/aiinaction/src/ch124_kmeans_variants.rs``) assert against these same numbers
to a 1e-9 tolerance, which is what keeps the three implementations at parity.
"""
from __future__ import annotations

import math

import pytest

from aiinaction import ch124_kmeans_variants as km


def _flat(rows):
    return [v for row in rows for v in row]


def approx2d(actual, expected):
    """Compare two lists-of-lists elementwise (pytest.approx rejects nesting)."""
    assert [len(r) for r in actual] == [len(r) for r in expected]
    assert _flat(actual) == pytest.approx(_flat(expected))

# --- Shared fixtures (mirrored verbatim in Julia and Rust) -------------------

# Two well-separated blobs in 2-D.
X_BLOBS = [[0.0, 0.0], [1.0, 0.5], [0.5, 1.0], [8.0, 8.0], [9.0, 8.5], [8.5, 9.0]]
C0 = [[0.0, 0.0], [9.0, 9.0]]
LLOYD_LABELS = [0, 0, 0, 1, 1, 1]
LLOYD_CENTROIDS = [[0.5, 0.5], [8.5, 8.5]]
LLOYD_INERTIA = 2.0

# Mini-batch update.
MB_C0 = [[0.0, 0.0], [10.0, 10.0]]
MB_COUNTS0 = [0.0, 0.0]
MB_BATCH = [[1.0, 1.0], [9.0, 9.0], [2.0, 0.0], [8.0, 10.0]]
MB_CENTROIDS = [[1.5, 0.5], [8.5, 9.5]]
MB_COUNTS = [2.0, 2.0]

# K-medians coordinatewise (lower) median.
KMED_MEMBERS = [[1.0, 5.0], [2.0, 100.0], [3.0, 6.0]]
KMED_CENTROID = [2.0, 6.0]
KMED_EVEN = [[1.0], [2.0], [3.0], [100.0]]
KMED_EVEN_CENTROID = [2.0]

# PAM assignment on a precomputed dissimilarity matrix.
PAM_D = [[0.0, 2.0, 5.0, 9.0], [2.0, 0.0, 4.0, 7.0], [5.0, 4.0, 0.0, 3.0], [9.0, 7.0, 3.0, 0.0]]
PAM_MEDOIDS = [0, 3]
PAM_LABELS = [0, 0, 1, 1]
PAM_COST = 5.0

# Kernel K-Means assignment distances.
XK = [[0.0], [0.2], [5.0], [5.2]]
GAMMA = 0.5
K01 = 0.9801986733067553
KERNEL_LABELS = [0, 0, 1, 1]
KD_00 = 0.009900663346622318
KD_01 = 1.990094266187928

# Fuzzy c-means.
XF = [[0.0], [1.0], [5.0], [6.0]]
CF = [[0.5], [5.5]]
M = 2.0
U_00 = 0.9918032786885246
U_01 = 0.00819672131147541
FC_0 = 0.4985105160463291
FC_1 = 5.501489483953671

# Bisecting (2-means) split.
XB = [[0.0, 0.0], [0.5, 0.3], [8.0, 8.0], [8.4, 7.6]]
BIS_INIT = [[0.0, 0.0], [8.0, 8.0]]
BIS_LABELS = [0, 0, 1, 1]
BIS_CENTROIDS = [[0.25, 0.15], [8.2, 7.8]]
BIS_SSE = 0.33


# --- Parity tests ------------------------------------------------------------

def test_lloyd_step_fixture():
    labels, centroids = km.lloyd_step(X_BLOBS, C0)
    assert labels == LLOYD_LABELS
    approx2d(centroids, LLOYD_CENTROIDS)
    assert km.inertia(X_BLOBS, centroids) == pytest.approx(LLOYD_INERTIA)


def test_mini_batch_update_fixture():
    c, cnt = km.mini_batch_update(MB_C0, MB_COUNTS0, MB_BATCH)
    approx2d(c, MB_CENTROIDS)
    assert cnt == pytest.approx(MB_COUNTS)


def test_kmedians_centroid_fixture():
    assert km.kmedians_centroid(KMED_MEMBERS) == pytest.approx(KMED_CENTROID)
    assert km.kmedians_centroid(KMED_EVEN) == pytest.approx(KMED_EVEN_CENTROID)


def test_pam_assign_cost_fixture():
    labels, cost = km.pam_assign_cost(PAM_D, PAM_MEDOIDS)
    assert labels == PAM_LABELS
    assert cost == pytest.approx(PAM_COST)


def test_kernel_assignment_fixture():
    K = km.rbf_kernel_matrix(XK, GAMMA)
    assert K[0][1] == pytest.approx(K01)
    assert K[0][0] == pytest.approx(1.0)
    kd = km.kernel_assignment_distances(K, KERNEL_LABELS, 2)
    assert kd[0][0] == pytest.approx(KD_00)
    assert kd[0][1] == pytest.approx(KD_01)
    # Each point's own cluster is the nearer one.
    assert all(min(range(2), key=lambda j, i=i: kd[i][j]) == KERNEL_LABELS[i] for i in range(4))


def test_fuzzy_fixture():
    U = km.fuzzy_memberships(XF, CF, M)
    assert U[0][0] == pytest.approx(U_00)
    assert U[0][1] == pytest.approx(U_01)
    # Memberships form a partition of unity.
    for row in U:
        assert sum(row) == pytest.approx(1.0)
    C = km.fuzzy_centroids(XF, U, M)
    assert C[0][0] == pytest.approx(FC_0)
    assert C[1][0] == pytest.approx(FC_1)


def test_bisecting_split_fixture():
    labels, centroids, sse = km.bisecting_split(XB, BIS_INIT)
    assert labels == BIS_LABELS
    approx2d(centroids, BIS_CENTROIDS)
    assert sse == pytest.approx(BIS_SSE)


# --- Edge cases --------------------------------------------------------------

def test_fuzzy_membership_on_centroid():
    # A point exactly on a centroid gets full membership there.
    U = km.fuzzy_memberships([[0.5], [5.5]], CF, M)
    assert U[0] == pytest.approx([1.0, 0.0])
    assert U[1] == pytest.approx([0.0, 1.0])


def test_kernel_empty_cluster_is_inf():
    K = km.rbf_kernel_matrix(XK, GAMMA)
    kd = km.kernel_assignment_distances(K, [0, 0, 0, 0], 2)
    assert all(math.isinf(kd[i][1]) for i in range(4))


def test_lloyd_empty_centroid_unchanged():
    # Centroid 1 captures nothing and is left as-is.
    labels, c = km.lloyd_step([[0.0], [0.1]], [[0.0], [100.0]])
    assert labels == [0, 0]
    assert c[1] == [100.0]


def test_inertia_zero_when_centroids_are_points():
    assert km.inertia([[1.0, 2.0]], [[1.0, 2.0]]) == 0.0


@pytest.mark.parametrize(
    "bad",
    [[], [[]], [1.0, 2.0]],
)
def test_matrix_validation_raises(bad):
    with pytest.raises(ValueError):
        km.inertia(bad, [[0.0]])


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="dimension"):
        km.lloyd_step([[0.0, 0.0]], [[0.0]])


def test_gamma_must_be_positive():
    with pytest.raises(ValueError, match="gamma"):
        km.rbf_kernel_matrix(XK, 0.0)


def test_fuzziness_exponent_validation():
    with pytest.raises(ValueError, match="m must be greater than 1"):
        km.fuzzy_memberships(XF, CF, 1.0)


def test_pam_medoid_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        km.pam_assign_cost(PAM_D, [0, 99])


def test_bisecting_requires_two_centroids():
    with pytest.raises(ValueError, match="exactly 2"):
        km.bisecting_split(XB, [[0.0, 0.0]])
