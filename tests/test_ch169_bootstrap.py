"""Tests for aiinaction.ch169_bootstrap, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. Because the bootstrap uses the module's fully specified 64-bit LCG, the exact
resample indices (and therefore every replicate, the standard error, and both
interval endpoints) are reproducible across all three languages for a given seed.
"""
from __future__ import annotations

import math

import pytest

from aiinaction import ch169_bootstrap as boot

# Shared fixture sample, mirrored in julia/AIInAction/test and rust/aiinaction/src.
DATA = [4.0, 8.0, 15.0, 16.0, 23.0, 42.0, 1.0, 9.0]
SEED = 12345
N_RESAMPLES = 500

EXPECTED = {
    "estimate": 14.75,
    "standard_error": 4.401794815207815,
    "perc_low": 7.434375,
    "perc_high": 24.506249999999994,
    "bca_low": 8.55046123816017,
    "bca_high": 26.839660615922583,
    "replicates_head": [13.25, 24.125, 16.0, 17.125, 10.375],
    "replicates_sum": 7401.0,
    # Special-function fixtures.
    "norm_ppf_975": 1.959963986120195,
    "norm_cdf_1": 0.8413447386043253,
    "quantile_25": 2.0,
}


def _percentile_result():
    return boot.bootstrap_mean_ci(
        DATA, n_resamples=N_RESAMPLES, alpha=0.025, method="percentile", seed=SEED
    )


def _bca_result():
    return boot.bootstrap_mean_ci(
        DATA, n_resamples=N_RESAMPLES, alpha=0.025, method="bca", seed=SEED
    )


def test_estimate_matches_fixture():
    assert _percentile_result().estimate == pytest.approx(EXPECTED["estimate"])


def test_standard_error_matches_fixture():
    assert _percentile_result().standard_error == pytest.approx(EXPECTED["standard_error"])


def test_replicate_head_matches_fixture():
    r = _percentile_result()
    assert r.replicates[:5] == pytest.approx(EXPECTED["replicates_head"])


def test_replicate_sum_matches_fixture():
    r = _percentile_result()
    assert sum(r.replicates) == pytest.approx(EXPECTED["replicates_sum"])


def test_percentile_interval_matches_fixture():
    r = _percentile_result()
    assert r.ci_low == pytest.approx(EXPECTED["perc_low"])
    assert r.ci_high == pytest.approx(EXPECTED["perc_high"])


def test_bca_interval_matches_fixture():
    r = _bca_result()
    assert r.ci_low == pytest.approx(EXPECTED["bca_low"])
    assert r.ci_high == pytest.approx(EXPECTED["bca_high"])


def test_norm_ppf_matches_fixture():
    assert boot.norm_ppf(0.975) == pytest.approx(EXPECTED["norm_ppf_975"], abs=1e-9)


def test_norm_ppf_is_antisymmetric():
    assert boot.norm_ppf(0.025) == pytest.approx(-boot.norm_ppf(0.975), abs=1e-12)


def test_norm_cdf_matches_fixture():
    assert boot.norm_cdf(1.0) == pytest.approx(EXPECTED["norm_cdf_1"])


def test_norm_cdf_ppf_round_trip():
    # norm_cdf uses a ~1.2e-7-accurate erf (chosen for cross-language parity), so the
    # round trip is checked to that accuracy rather than to machine precision.
    for p in (0.01, 0.1, 0.3, 0.5, 0.8, 0.99):
        assert boot.norm_cdf(boot.norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_quantile_linear_rule():
    assert boot.quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.25) == pytest.approx(EXPECTED["quantile_25"])


def test_quantile_endpoints():
    s = [10.0, 20.0, 30.0]
    assert boot.quantile(s, 0.0) == 10.0
    assert boot.quantile(s, 1.0) == 30.0


def test_interval_contains_estimate():
    r = _bca_result()
    assert r.ci_low < r.estimate < r.ci_high


def test_determinism_same_seed():
    a = _percentile_result()
    b = _percentile_result()
    assert a.replicates == b.replicates


def test_different_seed_changes_replicates():
    a = boot.bootstrap_mean_ci(DATA, n_resamples=200, seed=1)
    b = boot.bootstrap_mean_ci(DATA, n_resamples=200, seed=2)
    assert a.replicates != b.replicates


def test_constant_data_gives_degenerate_interval():
    r = boot.bootstrap_mean_ci([5.0, 5.0, 5.0, 5.0], n_resamples=100, seed=7)
    assert r.estimate == pytest.approx(5.0)
    assert r.standard_error == pytest.approx(0.0)
    assert r.ci_low == pytest.approx(5.0)
    assert r.ci_high == pytest.approx(5.0)


def test_too_few_observations_raises():
    with pytest.raises(ValueError, match="at least 2 observations"):
        boot.bootstrap_mean_ci([1.0])


def test_non_finite_raises():
    with pytest.raises(ValueError, match="non-finite"):
        boot.bootstrap_mean_ci([1.0, 2.0, float("nan")])


def test_bad_method_raises():
    with pytest.raises(ValueError, match="method must be"):
        boot.bootstrap_mean_ci(DATA, method="studentized")


def test_bad_alpha_raises():
    with pytest.raises(ValueError, match="alpha must be"):
        boot.bootstrap_mean_ci(DATA, alpha=0.5)


def test_bad_n_resamples_raises():
    with pytest.raises(ValueError, match="n_resamples must be"):
        boot.bootstrap_mean_ci(DATA, n_resamples=0)


def test_negative_seed_raises():
    with pytest.raises(ValueError, match="seed must be non-negative"):
        boot.bootstrap_mean_ci(DATA, seed=-1)


def test_norm_ppf_out_of_range_raises():
    with pytest.raises(ValueError, match="open interval"):
        boot.norm_ppf(0.0)
    with pytest.raises(ValueError, match="open interval"):
        boot.norm_ppf(1.0)


def test_norm_ppf_accuracy_against_known_quantiles():
    # 90% two-sided -> 1.6448536..., a standard tabulated value.
    assert boot.norm_ppf(0.95) == pytest.approx(1.6448536269514722, abs=1e-8)


def test_math_isfinite_replicates():
    r = _percentile_result()
    assert all(math.isfinite(v) for v in r.replicates)
