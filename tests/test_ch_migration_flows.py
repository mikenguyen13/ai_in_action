"""Tests for aiinaction.ch_migration_flows, including shared cross-language fixtures.

The fixtures are the single source of truth: the Julia and Rust suites assert the same
numbers. The last block checks the synthetic world and (if torch is installed) a
two-epoch smoke run of the recurrent model.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aiinaction import ch_migration_flows as m

# ---------------------------------------------------------------------------
# Shared fixtures (mirrored in julia/AIInAction/test and rust/aiinaction/src)
# ---------------------------------------------------------------------------
YJ_X = [-5.0, -0.3, 0.0, 2.0, 10.0]
YJ_07 = [-3.57877726629599, -0.28800168756136063, 0.0, 1.6538132571065618, 6.225223813548735]
YJ_00 = [-1.791759469228055, -0.262364264467491, 0.0, 1.0986122886681098, 2.3978952727983707]
YJ_M05 = [-1.183503419072274, -0.24588396138594154, 0.0, 0.8452994616207485, 1.3969773108444727]
YJ_DERIV_07 = [0.5841906810678655, 0.9243086009945788, 1.0, 0.7192230933248643, 0.4870596972258284]
YJ_SAMPLE = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 40.0, 100.0, -1.0, -3.0, -7.0]
YJ_LOGLIK_05 = -33.338467549439706
YJ_FIT = -0.01
ELASTICITY = 0.8307692307692308  # du/dphi 0.9, x 12, lambda 0

STOCK = [[100.0, 5.0, 2.0], [3.0, 80.0, 4.0], [1.0, 6.0, 60.0]]
BIRTHS = [1.5, 1.0, 0.8]
MORT = [0.01, 0.02, 0.015]
STEP_EXPECTED = [[100.2, 4.9, 2.27], [2.37, 79.4, 4.54], [0.0900000000000003, 5.88, 60.800000000000004]]
OD_EXPECTED = [[0.0, 2.4000000000000004, 3.6000000000000005], [1.8000000000000003, 0.0, 5.4], [2.4000000000000004, 4.800000000000001, 0.0]]
NET_EXPECTED = [-1.7999999999999998, 0.0, 1.799999999999999]

IPF_TABLE = [[10.0, 20.0, 30.0], [15.0, 25.0, 5.0], [8.0, 2.0, 40.0]]
IPF_ROWS = [60.0, 50.0, 50.0]
IPF_COLS = [45.0, 55.0, 60.0]
IPF_EXPECTED = [[13.584306505781386, 24.27835009115688, 22.137343404288874], [18.723560813157828, 27.886171700008326, 3.3902674980586913], [12.692132681060787, 2.835478208834802, 34.47238909765244]]

BAL_S1 = [[50.0, 5.0, 2.0], [3.0, 40.0, 4.0], [1.0, 6.0, 30.0]]
BAL_S2 = [[48.0, 7.0, 3.0], [4.0, 42.0, 3.0], [2.0, 5.0, 33.0]]
BAL_BIRTHS = [2.0, 1.5, 1.0]  # deaths = 0.02 * S2
BAL1 = [[49.68615791114227, 5.279992903186189, 2.1138491735349096], [2.877097428817112, 40.78378334376581, 4.079119232039964], [0.9767446600406219, 6.226223753048012, 31.19703159442512]]
BAL2 = [[48.32400928274534, 6.712813076986998, 2.8831776522759487], [4.118486084752095, 41.19277564057946, 2.9487382738222503], [2.0175046325025683, 4.80441128243354, 31.7780840739018]]
REL1 = [[0.006276841777154658, 0.055998580637237796, 0.05692458676745482], [0.04096752372762932, 0.019594583594145298, 0.019779808009990907], [0.02325533995937812, 0.03770395884133526, 0.039901053147503994]]
REL2 = [[0.006750193390527848, 0.04102670328757174, 0.03894078257468377], [0.029621521188023747, 0.019219627605250894, 0.017087242059249903], [0.008752316251284142, 0.03911774351329207, 0.03702775533630914]]

WEIGHT_INPUT = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
WEIGHTS = [2.0, 1.5238095238095237, 0.7619047619047619, 0.5, 0.5, 0.5]
WL2 = 1.21875
SURVIVAL = [1.0, 0.99, 0.9702, 0.9410939999999999]  # mortality [0.01, 0.02, 0.03]
CALIBRATE = 1.22456821045488  # obs [10,12,14,15], pred [9,10.5,13,13.2], w [1,1,2,0.5]
CALIBRATE_CONSTRAINED = -1.0  # obs -1s, pred [2,1,3], w [1,0,0], s ones
STOCK_NEXT = [[110.0, 3.0, 9.0], [2.0, 90.0, 1.0], [4.0, 6.0, 55.0]]
BASELINE = [[0.0, 0.0, 1.4], [0.0, 0.0, 0.0], [0.6, 0.0, 0.0]]  # interval 5
GRAVITY = -6.6899999999999995  # beta (-8.7, .8, -.7, .8, .5), args (1, 2, -.3, 1, .4)
LOG_RMSE = 0.18451760994274952
PEARSON = 0.9986517555689657
COVERAGE = 0.5
ENSEMBLE_SAMPLES = [[1.0, 2.0], [2.0, 4.0], [4.0, 5.0], [5.0, 9.0]]
ENSEMBLE_MEAN = [3.0, 5.0]
ENSEMBLE_STD = [1.8257418583505538, 2.943920288775949]


def close(a, b, tol=1e-12):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=tol, atol=tol)


def test_yj_fixtures_and_identity():
    close(m.yj_sym(YJ_X, 0.7), YJ_07)
    close(m.yj_sym(YJ_X, 0.0), YJ_00)
    close(m.yj_sym(YJ_X, -0.5), YJ_M05)
    close(m.yj_sym(YJ_X, 1.0), YJ_X)  # lambda = 1 is the identity
    close(m.yj_sym_deriv(YJ_X, 0.7), YJ_DERIV_07)


@pytest.mark.parametrize("lam", [0.7, 0.0, 1.0, 1.6])
def test_yj_roundtrip_and_derivative(lam):
    x = np.array(YJ_X)
    close(m.yj_sym_inv(m.yj_sym(x, lam), lam), x, 1e-10)
    h = 1e-6
    num = (m.yj_sym(x + h, lam) - m.yj_sym(x - h, lam)) / (2 * h)
    close(num, m.yj_sym_deriv(x, lam), 1e-6)


def test_yj_errors_and_fit():
    with pytest.raises(ValueError):
        m.yj_sym_inv(5.0, -0.5)  # outside the range for lambda < 0
    with pytest.raises(ValueError):
        m.yj_sym([1.0, float("nan")], 0.5)
    with pytest.raises(ValueError):
        m.yj_profile_loglik([1.0], 0.5)
    close(m.yj_profile_loglik(YJ_SAMPLE, 0.5), YJ_LOGLIK_05)
    assert m.fit_yj_lambda(YJ_SAMPLE) == pytest.approx(YJ_FIT)
    # a right-skewed sample asks for a strongly compressing transform
    skew = np.random.default_rng(0).lognormal(0.0, 1.2, size=400)
    assert m.fit_yj_lambda(skew) < 0.5


def test_elasticity_chain():
    assert m.elasticity_chain(0.9, 12.0, 0.0) == pytest.approx(ELASTICITY)
    # with lambda 0, phi = log(x + 1), so a unit slope in phi is an elasticity x/(x+1)
    x = 1e6
    assert m.elasticity_chain(1.0, x, 0.0) == pytest.approx(x / (x + 1.0))


def test_stock_step_matches_fixture_and_conserves():
    t = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if j != k:
                    t[i, j, k] = 0.1 * (i + 1) * (j + 2) * (k + 1)
    nxt = m.stock_step(STOCK, BIRTHS, MORT, t)
    close(nxt, STEP_EXPECTED)
    # migration only moves people: total change is births minus deaths
    s = np.array(STOCK)
    expected_change = sum(BIRTHS) - float(np.sum(np.array(MORT)[None, :] * s))
    assert nxt.sum() - s.sum() == pytest.approx(expected_change)
    f = m.od_flows(t)
    close(f, OD_EXPECTED)
    close(m.net_migration(f), NET_EXPECTED)
    assert m.net_migration(f).sum() == pytest.approx(0.0, abs=1e-12)


def test_stock_step_validation():
    t = np.zeros((3, 3, 3))
    with pytest.raises(ValueError):
        m.stock_step(STOCK, BIRTHS, [0.1, 0.2, 1.5], t)
    bad = t.copy()
    bad[0, 0, 1] = -1.0
    with pytest.raises(ValueError):
        m.stock_step(STOCK, BIRTHS, MORT, bad)
    with pytest.raises(ValueError):
        m.stock_step(STOCK, BIRTHS[:2], MORT, t)


def test_ipf_fixture_and_errors():
    fit, iters = m.ipf(IPF_TABLE, IPF_ROWS, IPF_COLS)
    close(fit, IPF_EXPECTED, 1e-8)
    close(fit.sum(axis=1), IPF_ROWS, 1e-6)
    close(fit.sum(axis=0), IPF_COLS, 1e-6)
    assert 1 < iters < 100
    with pytest.raises(ValueError):
        m.ipf(IPF_TABLE, IPF_ROWS, [45.0, 55.0, 70.0])  # totals disagree
    with pytest.raises(ValueError):
        m.ipf([[0.0, 0.0], [1.0, 1.0]], [1.0, 1.0], [1.0, 1.0])  # empty row, positive target


def test_demographic_balance_fixture():
    deaths = 0.02 * np.array(BAL_S2)
    b1, b2, r1, r2 = m.demographic_balance(BAL_S1, BAL_S2, BAL_BIRTHS, deaths)
    close(b1, BAL1, 1e-8)
    close(b2, BAL2, 1e-8)
    close(r1, REL1, 1e-8)
    close(r2, REL2, 1e-8)
    # after adding births and deaths both tables share the same marginals
    a1 = b1 + np.diag(BAL_BIRTHS)
    a2 = b2 + deaths
    close(a1.sum(axis=0), a2.sum(axis=0), 1e-8)
    close(a1.sum(axis=1), a2.sum(axis=1), 1e-8)


def test_weights_and_loss():
    close(m.uncertainty_weights(WEIGHT_INPUT), WEIGHTS)
    w = m.uncertainty_weights(WEIGHT_INPUT)
    assert w.min() >= 0.5 and w.max() <= 2.0
    assert np.all(np.diff(w) <= 0)  # smaller uncertainty, larger weight
    assert m.weighted_l2([1, 2, 3, 4], [1.5, 1.5, 3.5, 2], [1, 0.5, 2, 1]) == pytest.approx(WL2)
    with pytest.raises(ValueError):
        m.weighted_l2([1.0], [1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        m.uncertainty_weights([-0.1, 0.2])


def test_calibration():
    s = m.survival_fraction([0.01, 0.02, 0.03])
    close(s, SURVIVAL)
    b = m.calibrate_offset([10, 12, 14, 15], [9, 10.5, 13, 13.2], [1, 1, 2, 0.5], s)
    assert b == pytest.approx(CALIBRATE)
    assert m.calibrate_offset([-1, -1, -1], [2, 1, 3], [1, 0, 0], [1, 1, 1]) == pytest.approx(
        CALIBRATE_CONSTRAINED
    )
    # exact recovery of a pure offset
    pred = np.array([5.0, 6.0, 7.0, 8.0])
    obs = pred + 2.5 * s
    assert m.calibrate_offset(obs, pred, np.ones(4), s) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        m.calibrate_offset([1.0], [1.0, 2.0], [1.0], [1.0])


def test_baseline_gravity_metrics():
    close(m.stock_difference_baseline(STOCK, STOCK_NEXT, 5.0), BASELINE)
    with pytest.raises(ValueError):
        m.stock_difference_baseline(STOCK, STOCK_NEXT, 0.0)
    assert m.gravity_log_rate([-8.7, 0.8, -0.7, 0.8, 0.5], 1.0, 2.0, -0.3, 1.0, 0.4) == pytest.approx(GRAVITY)
    assert m.log_rmse([1, 2, 3, 4], [1.1, 1.8, 3.6, 3.0]) == pytest.approx(LOG_RMSE)
    assert m.pearson([1, 2, 3, 4, 5], [2.1, 3.9, 6.2, 7.8, 10.1]) == pytest.approx(PEARSON)
    assert math.isnan(m.pearson([1, 1, 1], [1, 2, 3]))
    assert m.coverage([1, 2, 3, 4], [1.1, 2.5, 2.9, 3], [0.2] * 4) == COVERAGE
    mean, std = m.ensemble_mean_std(ENSEMBLE_SAMPLES)
    close(mean, ENSEMBLE_MEAN)
    close(std, ENSEMBLE_STD)
    with pytest.raises(ValueError):
        m.ensemble_mean_std([[1.0, 2.0]])


# ---------------------------------------------------------------------------
# Python-only: the synthetic world, the observations, and the recurrent model
# ---------------------------------------------------------------------------


def test_world_obeys_accounting_and_is_deterministic():
    w = m.make_world(seed=3)
    w2 = m.make_world(seed=3)
    np.testing.assert_array_equal(w.flows, w2.flows)
    assert np.all(w.flows >= 0) and np.all(w.stocks >= 0)
    idx = np.arange(w.n)
    assert np.all(w.flows[:, :, idx, idx] == 0)  # no self-flows
    for t in range(w.years):
        births = w.birth_rate * w.stocks[t].sum(axis=0)
        nxt = m.stock_step(w.stocks[t], births, w.mortality, w.flows[t])
        np.testing.assert_allclose(nxt, w.stocks[t + 1], rtol=1e-10)
    truth = m.simulate_truth(w)
    np.testing.assert_allclose(truth["net"].sum(axis=1), 0.0, atol=1e-8)


def test_conflict_shock_has_memory():
    w = m.make_world(seed=0)
    out6 = w.flows.sum(axis=1)[:, 6, :].sum(axis=1)
    # outflow jumps when the conflict starts (year 6) and decays slowly afterwards
    assert out6[7] > 5 * out6[3]
    assert out6[12] > out6[5] and out6[12] < out6[8]


def test_observations_are_partial_and_inconsistent():
    w = m.make_world(seed=0)
    obs = m.observe(w, seed=1)
    assert obs.stocks.shape == (5, w.n, w.n) and obs.stocks_bal.shape == (5, w.n, w.n)
    assert len(obs.net_countries) == 3 and obs.flows.shape == (w.years, 10)
    off = ~np.eye(w.n, dtype=bool)
    rel = np.abs(obs.stocks[2][off] / w.stocks[10][off] - 1.0)
    assert 0.02 < rel.mean() < 0.2  # raw tables are noisy copies of the truth
    assert np.all(obs.stock_rel_unc >= 0)
    assert obs.stocks_bal.min() > -1e-6


def test_train_ensemble_smoke():
    pytest.importorskip("torch")
    w = m.make_world(seed=0)
    obs = m.observe(w, seed=1)
    cfg = m.TrainConfig(epochs=3, members=2, pushforward_draws=2, latent=4, hidden=8, layers=2)
    res = m.train_ensemble(w, obs, cfg)
    assert res["samples"].shape == (4, w.years, w.n, w.n, w.n)
    assert np.all(np.isfinite(res["samples"])) and np.all(res["samples"] >= 0)
    rep = m.recovery_report(w, res["samples"], obs)
    assert set(rep) == {"observed corridors", "unobserved corridors"}
    el = m.feature_elasticity(w, res, years=(2,))
    assert set(el) == {"origin", "destination"}
