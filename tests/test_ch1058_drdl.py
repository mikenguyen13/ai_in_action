"""Tests for aiinaction.ch1058_drdl, including shared cross-language fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which keeps the three implementations at parity.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch1058_drdl as d

# ---------------------------------------------------------------------------
# Shared fixtures (mirrored in julia/AIInAction/test and rust/aiinaction/src)
# ---------------------------------------------------------------------------
OUTPUTS = [[-1.0, 0.5, -0.3], [0.2, -2.0, 0.1], [-3.0, -1.0, 1.0], [0.0, 0.0, 0.0]]
CUM_PROBS = [
    [0.3077993724446536, 0.8668928334755382, 0.9365446603138965],
    [0.7051836792708419, 0.7425006392477038, 0.9147279853227372],
    [0.048568007099546597, 0.3414181774380723, 0.9565414790857123],
    [0.6321205588285577, 0.8646647167633873, 0.950212931632136],
]
TIMES = [1, 2, 3, 3]
EVENTS = [1, 1, 1, 0]
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
FSL_G1 = 0.0696463795589742
FSL_G05 = 0.1259078886595625
FSL_G2 = 0.023133174339533064
BCE = 2.376713158199545
CINDEX = 0.5
AUC = (1 / 3, 0.75, 1 / 3)
BRIER = (0.34459024562633167, 0.23705872071716394, 0.22902278877569854)

EMB = [[0.1, 0.9], [1.0, 0.0], [0.6, 0.6], [-0.9, 0.1]]
CODEBOOK = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
CODES = [1, 0, 0, 2]
MASK_CODES = [2, 0, 2, 1, 0, 3]
MASK = [1, 1, 0, 1, 0, 1]
ORTH_CODEBOOK = [[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]]
ORTH = 1.0
SIGMA_03 = {0.5: 0.2672140914204164, 1.0: 0.14888503362331806, 2.0: 0.04496964958360028}
PROJ_IN = ([1.0, 0.0, 1.0], [-1.0, 2.0, 0.0])
PROJ_OUT = [-0.5, 2.0, 0.5]


def test_risk_accumulation_fixture_and_monotone():
    f = d.risk_accumulation(OUTPUTS)
    np.testing.assert_allclose(f, CUM_PROBS, atol=1e-12)
    assert np.all(np.diff(f, axis=1) > 0)
    assert np.all((f > 0) & (f < 1))
    np.testing.assert_allclose(d.risk_accumulation(OUTPUTS[0]), CUM_PROBS[0], atol=1e-12)


def test_risk_accumulation_is_cloglog_hazard():
    # F(t_u) = 1 - prod_j (1 - h_j) with h_j = 1 - exp(-exp(o_j)).
    o = np.array(OUTPUTS)
    h = 1.0 - np.exp(-np.exp(o))
    f = 1.0 - np.cumprod(1.0 - h, axis=1)
    np.testing.assert_allclose(d.risk_accumulation(o), f, atol=1e-12)


def test_risk_accumulation_validation():
    with pytest.raises(ValueError):
        d.risk_accumulation([])
    with pytest.raises(ValueError):
        d.risk_accumulation([1.0, np.nan])


def test_nearest_codes_and_mask():
    assert d.nearest_codes(EMB, CODEBOOK).tolist() == CODES
    assert d.first_occurrence_mask(MASK_CODES).tolist() == MASK
    assert d.first_occurrence_mask([]).tolist() == []
    with pytest.raises(ValueError):
        d.nearest_codes([[1.0, 2.0, 3.0]], CODEBOOK)
    with pytest.raises(ValueError):
        d.first_occurrence_mask([0, -1])


def test_nearest_codes_tie_goes_to_lowest_index():
    assert d.nearest_codes([[0.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]]).tolist() == [0]


def test_orthogonal_loss():
    assert d.orthogonal_loss(ORTH_CODEBOOK) == pytest.approx(ORTH, abs=1e-12)
    assert d.orthogonal_loss(np.eye(4)) == pytest.approx(0.0, abs=1e-15)
    assert d.orthogonal_loss([[2.0, 0.0], [5.0, 0.0]]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        d.orthogonal_loss([[0.0, 0.0], [1.0, 0.0]])


def test_focal_sigma():
    for g, v in SIGMA_03.items():
        assert d.focal_sigma(0.3, g) == pytest.approx(v, abs=1e-12)
    assert d.focal_sigma(0.0, 0.7) == 0.0
    assert d.focal_sigma(50.0, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        d.focal_sigma(-0.1, 1.0)
    with pytest.raises(ValueError):
        d.focal_sigma(0.1, 0.0)


def test_comparable_pairs_fixture():
    assert d.comparable_pairs(TIMES, EVENTS) == PAIRS


def test_focal_survival_loss_fixture():
    assert d.focal_survival_loss(CUM_PROBS, TIMES, EVENTS, 1.0) == pytest.approx(FSL_G1, abs=1e-12)
    assert d.focal_survival_loss(CUM_PROBS, TIMES, EVENTS, 0.5) == pytest.approx(FSL_G05, abs=1e-12)
    assert d.focal_survival_loss(CUM_PROBS, TIMES, EVENTS, 2.0) == pytest.approx(FSL_G2, abs=1e-12)


def test_focal_survival_loss_zero_when_perfectly_ranked():
    f = [[0.9, 0.95, 0.99], [0.1, 0.8, 0.9], [0.05, 0.1, 0.2]]
    assert d.focal_survival_loss(f, [1, 2, 3], [1, 1, 0]) == 0.0
    assert d.focal_survival_loss(f, [3, 3, 3], [0, 0, 0]) == 0.0


def test_multiperiod_bce_fixture():
    assert d.multiperiod_bce(CUM_PROBS, TIMES, EVENTS) == pytest.approx(BCE, abs=1e-12)
    y = d.cumulative_labels(TIMES, EVENTS, 3)
    assert y.tolist() == [[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]]


def test_survival_target_validation():
    with pytest.raises(ValueError):
        d.multiperiod_bce(CUM_PROBS, [0, 1, 2, 3], EVENTS)
    with pytest.raises(ValueError):
        d.multiperiod_bce(CUM_PROBS, [1, 2, 3, 4], EVENTS)
    with pytest.raises(ValueError):
        d.multiperiod_bce(CUM_PROBS, TIMES, [1, 2, 0, 0])


def test_project_conflicting():
    out = d.project_conflicting(*PROJ_IN)
    np.testing.assert_allclose(out, PROJ_OUT, atol=1e-12)
    assert float(np.dot(out, PROJ_IN[0])) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(d.project_conflicting([1.0, 0.0], [0.5, 1.0]), [0.5, 1.0])
    np.testing.assert_allclose(d.project_conflicting([0.0, 0.0], [0.5, 1.0]), [0.5, 1.0])


def test_combined_step_never_conflicts_with_main_loss():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.normal(size=6), rng.normal(size=6)
        total = a + d.project_conflicting(a, b)
        assert float(total @ a) >= float(a @ a) - 1e-9


def test_metrics_fixture():
    assert d.td_concordance(CUM_PROBS, TIMES, EVENTS) == pytest.approx(CINDEX, abs=1e-12)
    for h, v in enumerate(AUC, start=1):
        assert d.td_auc(CUM_PROBS, TIMES, EVENTS, h) == pytest.approx(v, abs=1e-12)
    np.testing.assert_allclose(d.brier_scores(CUM_PROBS, TIMES, EVENTS), BRIER, atol=1e-12)


def test_concordance_perfect_and_reversed():
    f = np.array([[0.9, 0.95, 0.99], [0.1, 0.8, 0.9], [0.05, 0.1, 0.2]])
    assert d.td_concordance(f, [1, 2, 3], [1, 1, 0]) == 1.0
    assert d.td_concordance(f[::-1], [1, 2, 3], [1, 1, 0]) == 0.0
    with pytest.raises(ValueError):
        d.td_concordance(f, [3, 3, 3], [0, 0, 0])
