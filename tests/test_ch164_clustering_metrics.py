"""Tests for aiinaction.ch164_clustering_metrics (shared cross-language fixtures).

The numbers in EXPECTED are the single source of truth: the Julia and Rust test
suites assert against the identical labelings and values, which is what keeps the
three implementations at parity (1e-9 tolerance).

Fixture rationale: ``LT`` (3 true classes) and ``LP`` (4 predicted clusters) are
deliberately asymmetric so that H(true) != H(pred); this makes the four NMI
averaging methods, homogeneity, and completeness all take distinct values and
exercises the non-degenerate branches.
"""
from __future__ import annotations

import math

import pytest

from aiinaction import ch164_clustering_metrics as cm

# Shared fixtures mirrored in julia/AIInAction/test and rust/aiinaction/src.
LT = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
LP = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]

EXPECTED = {
    "h_true": 1.0888999753452238,
    "h_pred": 1.366158847569202,
    "mutual_information": 0.8979457248567799,
    "nmi_arithmetic": 0.7315064848758445,
    "nmi_geometric": 0.7362164101692431,
    "nmi_min": 0.8246356370538958,
    "nmi_max": 0.6572776851348504,
    "homogeneity": 0.8246356370538958,
    "completeness": 0.6572776851348504,
    "v_measure": 0.7315064848758445,
    "v_measure_beta2": 0.7049682606092936,
    "fowlkes_mallows": 0.6123724356957946,
}

TOL = 1e-9


def test_contingency_matrix():
    assert cm.contingency_matrix(LT, LP).tolist() == [
        [2, 2, 0, 0],
        [0, 1, 2, 0],
        [0, 0, 0, 3],
    ]


def test_entropy_matches_fixture():
    assert cm.entropy(LT) == pytest.approx(EXPECTED["h_true"], abs=TOL)
    assert cm.entropy(LP) == pytest.approx(EXPECTED["h_pred"], abs=TOL)


def test_mutual_information_matches_fixture():
    assert cm.mutual_information(LT, LP) == pytest.approx(
        EXPECTED["mutual_information"], abs=TOL
    )


def test_mutual_information_symmetric():
    assert cm.mutual_information(LT, LP) == pytest.approx(
        cm.mutual_information(LP, LT), abs=TOL
    )


@pytest.mark.parametrize(
    "method,key",
    [
        ("arithmetic", "nmi_arithmetic"),
        ("geometric", "nmi_geometric"),
        ("min", "nmi_min"),
        ("max", "nmi_max"),
    ],
)
def test_nmi_methods_match_fixture(method, key):
    assert cm.normalized_mutual_information(
        LT, LP, average_method=method
    ) == pytest.approx(EXPECTED[key], abs=TOL)


def test_homogeneity_completeness_match_fixture():
    assert cm.homogeneity(LT, LP) == pytest.approx(EXPECTED["homogeneity"], abs=TOL)
    assert cm.completeness(LT, LP) == pytest.approx(EXPECTED["completeness"], abs=TOL)


def test_v_measure_matches_fixture():
    assert cm.v_measure(LT, LP) == pytest.approx(EXPECTED["v_measure"], abs=TOL)
    assert cm.v_measure(LT, LP, beta=2.0) == pytest.approx(
        EXPECTED["v_measure_beta2"], abs=TOL
    )


def test_v_measure_is_harmonic_mean_of_h_and_c():
    h = cm.homogeneity(LT, LP)
    c = cm.completeness(LT, LP)
    expected = 2.0 * h * c / (h + c)
    assert cm.v_measure(LT, LP) == pytest.approx(expected, abs=TOL)


def test_fowlkes_mallows_matches_fixture():
    assert cm.fowlkes_mallows_index(LT, LP) == pytest.approx(
        EXPECTED["fowlkes_mallows"], abs=TOL
    )


def test_perfect_agreement_up_to_relabeling():
    a = [0, 0, 1, 1, 2, 2]
    b = [2, 2, 0, 0, 1, 1]  # same partition, different names
    assert cm.mutual_information(a, b) == pytest.approx(cm.entropy(a), abs=TOL)
    assert cm.normalized_mutual_information(a, b) == pytest.approx(1.0, abs=TOL)
    assert cm.homogeneity(a, b) == pytest.approx(1.0, abs=TOL)
    assert cm.completeness(a, b) == pytest.approx(1.0, abs=TOL)
    assert cm.v_measure(a, b) == pytest.approx(1.0, abs=TOL)
    assert cm.fowlkes_mallows_index(a, b) == pytest.approx(1.0, abs=TOL)


def test_independent_partitions_have_zero_mi():
    # Each predicted cluster splits each true class evenly -> independence.
    a = [0, 0, 1, 1]
    b = [0, 1, 0, 1]
    assert cm.mutual_information(a, b) == pytest.approx(0.0, abs=TOL)
    assert cm.normalized_mutual_information(a, b) == pytest.approx(0.0, abs=TOL)
    assert cm.v_measure(a, b) == pytest.approx(0.0, abs=TOL)


def test_single_cluster_each_is_degenerate_perfect_nmi():
    a = [0, 0, 0]
    b = [0, 0, 0]
    assert cm.entropy(a) == 0.0
    assert cm.normalized_mutual_information(a, b) == 1.0
    # Both components are 1.0 by convention -> homogeneity, completeness, v all 1.
    assert cm.homogeneity(a, b) == 1.0
    assert cm.completeness(a, b) == 1.0


def test_all_singletons_fm_is_zero():
    a = [0, 1, 2, 3]
    b = [3, 2, 1, 0]
    # No pair is co-clustered in either partition.
    assert cm.fowlkes_mallows_index(a, b) == 0.0


def test_reexported_silhouette_and_ari():
    # Re-exported from ch132; smoke-check the chapter's full API is importable.
    assert cm.silhouette_score(
        [[0.0], [0.1], [10.0], [10.1]], [0, 0, 1, 1]
    ) == pytest.approx(0.9899997499937498, abs=TOL)
    assert cm.adjusted_rand_index(LT, LP) == pytest.approx(
        0.4915254237288135, abs=TOL
    )


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        cm.mutual_information([0, 0, 1], [0, 1])


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        cm.entropy([])


def test_bad_average_method_raises():
    with pytest.raises(ValueError, match="average_method"):
        cm.normalized_mutual_information(LT, LP, average_method="harmonic")


def test_negative_beta_raises():
    with pytest.raises(ValueError, match="beta"):
        cm.v_measure(LT, LP, beta=-1.0)
