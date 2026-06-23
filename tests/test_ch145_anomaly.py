"""Tests for aiinaction.ch145_anomaly, including the shared cross-language fixtures.

The numeric fixtures below are the single source of truth: the Julia and Rust
test suites assert against the same numbers, which is what keeps the three
implementations at parity (1e-9 tolerance).
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction.ch145_anomaly import (
    chi2_ppf,
    gaussian_kde,
    grubbs_test,
    kde_scores,
    mahalanobis_sq,
    student_t_ppf,
    zscore_flags,
    zscores,
)

# --- Shared fixtures (mirrored in Julia and Rust suites) -------------------
X_Z = [10.0, 12.0, 11.0, 13.0, 9.0, 11.5, 40.0]

X_MAHAL = [
    [2.0, 1.0],
    [3.0, 2.0],
    [4.0, 2.5],
    [5.0, 4.0],
    [6.0, 5.0],
    [2.5, 6.0],
]
PT_MAHAL = [[10.0, 1.0]]

X_KDE = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]
Q_KDE = [2.5, 10.0]

X_GRUBBS = [1.0, 2.0, 1.5, 1.8, 2.2, 5.0]

EXPECTED = {
    "z_last": 2.251807155714753,
    "z_first": -0.4737231192137665,
    "mahal_row0": 2.051080282894225,
    "mahal_row5": 4.120035750369162,
    "mahal_pt": 27.01727286857853,
    "kde_q0": 0.22915412139731456,
    "kde_q1": 0.056991755250522066,
    "kde_score_q1": 2.8648486663373274,
    "grubbs_G": 1.9489336934427666,
    "grubbs_crit": 1.887145117783933,
    "chi2_95_2": 5.991464547107979,
    "chi2_99_3": 11.344866730144357,
    "t_975_10": 2.228138851986274,
}


# --- z-score ---------------------------------------------------------------
def test_zscores_match_fixture():
    z = zscores(X_Z)
    assert z[0] == pytest.approx(EXPECTED["z_first"])
    assert z[-1] == pytest.approx(EXPECTED["z_last"])


def test_zscores_zero_mean():
    assert float(np.mean(zscores(X_Z))) == pytest.approx(0.0, abs=1e-12)


def test_zscore_flags():
    flags = zscore_flags(X_Z, threshold=2.0)
    assert flags.tolist() == [False, False, False, False, False, False, True]


def test_zscores_zero_std_raises():
    with pytest.raises(ValueError, match="undefined"):
        zscores([5.0, 5.0, 5.0])


def test_zscore_flags_bad_threshold():
    with pytest.raises(ValueError, match="positive"):
        zscore_flags(X_Z, threshold=0.0)


# --- Mahalanobis -----------------------------------------------------------
def test_mahalanobis_match_fixture():
    d2 = mahalanobis_sq(X_MAHAL)
    assert d2[0] == pytest.approx(EXPECTED["mahal_row0"])
    assert d2[5] == pytest.approx(EXPECTED["mahal_row5"])


def test_mahalanobis_external_point():
    d2 = mahalanobis_sq(X_MAHAL, PT_MAHAL)
    assert d2[0] == pytest.approx(EXPECTED["mahal_pt"])


def test_mahalanobis_mean_identity():
    # The mean squared Mahalanobis distance over the n training rows is exactly
    # d * (n - 1) / n for a ddof=1 covariance (here 2 * 5 / 6).
    d2 = mahalanobis_sq(X_MAHAL)
    n, d = 6, 2
    assert float(np.mean(d2)) == pytest.approx(d * (n - 1) / n)


def test_mahalanobis_singular_raises():
    # Second feature is a perfect copy of the first -> singular covariance.
    X = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    with pytest.raises(ValueError, match="singular"):
        mahalanobis_sq(X)


def test_mahalanobis_feature_mismatch_raises():
    with pytest.raises(ValueError, match="features"):
        mahalanobis_sq(X_MAHAL, [[1.0, 2.0, 3.0]])


# --- KDE -------------------------------------------------------------------
def test_kde_match_fixture():
    dens = gaussian_kde(X_KDE, Q_KDE, bandwidth=1.0)
    assert dens[0] == pytest.approx(EXPECTED["kde_q0"])
    assert dens[1] == pytest.approx(EXPECTED["kde_q1"])


def test_kde_scores_match_fixture():
    scores = kde_scores(X_KDE, Q_KDE, bandwidth=1.0)
    assert scores[1] == pytest.approx(EXPECTED["kde_score_q1"])
    # Outlier (10.0) is rarer than the cluster point (2.5).
    assert scores[1] > scores[0]


def test_kde_bad_bandwidth_raises():
    with pytest.raises(ValueError, match="positive"):
        gaussian_kde(X_KDE, Q_KDE, bandwidth=-1.0)


# --- Grubbs ----------------------------------------------------------------
def test_grubbs_match_fixture():
    r = grubbs_test(X_GRUBBS, alpha=0.05)
    assert r.statistic == pytest.approx(EXPECTED["grubbs_G"])
    assert r.critical_value == pytest.approx(EXPECTED["grubbs_crit"])
    assert r.index == 5
    assert r.is_outlier is True


def test_grubbs_no_outlier():
    r = grubbs_test([1.0, 2.0, 3.0, 4.0, 5.0], alpha=0.05)
    assert r.is_outlier is False


def test_grubbs_too_few_raises():
    with pytest.raises(ValueError, match="at least 3"):
        grubbs_test([1.0, 2.0])


def test_grubbs_zero_std_raises():
    with pytest.raises(ValueError, match="undefined"):
        grubbs_test([4.0, 4.0, 4.0])


# --- Special functions -----------------------------------------------------
def test_chi2_ppf_match_fixture():
    assert chi2_ppf(0.95, 2) == pytest.approx(EXPECTED["chi2_95_2"])
    assert chi2_ppf(0.99, 3) == pytest.approx(EXPECTED["chi2_99_3"])


def test_student_t_ppf_match_fixture():
    assert student_t_ppf(0.975, 10) == pytest.approx(EXPECTED["t_975_10"])


def test_chi2_ppf_bad_args():
    with pytest.raises(ValueError):
        chi2_ppf(1.5, 2)
    with pytest.raises(ValueError):
        chi2_ppf(0.5, 0)
