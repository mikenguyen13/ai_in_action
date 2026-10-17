"""Tests for aiinaction.ch526_hitl_discovery, including shared cross-language fixtures.

The fixtures are the single source of truth: the Julia and Rust test suites assert
against the same numbers, which keeps the three implementations at parity.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiinaction import ch526_hitl_discovery as h

# ---------------------------------------------------------------------------
# Shared fixtures (mirrored in julia/AIInAction/test and rust/aiinaction/src)
# ---------------------------------------------------------------------------
H_HALF = 0.6931471805599453  # binary_entropy(0.5)
H_09 = 0.3250829733914482  # binary_entropy(0.9)

# S=3 posterior draws, N=2 molecules, K=2 properties; PROBS[s][n][k]
PROBS = [
    [[0.9, 0.2], [0.5, 0.5]],
    [[0.1, 0.2], [0.5, 0.5]],
    [[0.5, 0.2], [0.9, 0.1]],
]
MU = [[0.5, 0.2], [0.6333333333333333, 0.3666666666666667]]
ALEATORIC = [[0.44777104244761395, 0.5004024235381879], [0.5704591115037796, 0.5704591115037796]]
EPISTEMIC = [[0.24537613811233133, 0.0], [0.08669864999356092, 0.08669864999356092]]
UCB_BETA2 = [1.9727922061357857, 2.363003220017392]  # sigma_d = sqrt(mu (1 - mu)), beta = 2

TOPK_SCORES = [0.3, 0.9, 0.9, 0.1, 0.5]
TOPK_3 = [1, 2, 4]
TOPK_3_EXCLUDING_1 = [2, 4, 0]

MERGE_UNC, MERGE_SRCH, MERGED = [5, 2, 7], [2, 9, 5], [2, 9, 5, 7]

TARGETS = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
KNOWN = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
SEARCH_LIST, UNC_LIST = [0, 1, 3, 4, 6, 7], [2, 5, 8, 9, 3]
EXPERT_PICK_5 = [0, 3, 6, 2, 5]  # explore_share 0.5, budget 5

HIT_30_OF_2500 = 0.012
RECALL_30_OF_148 = 0.20270270270270271


def test_binary_entropy():
    assert h.binary_entropy(0.5) == pytest.approx(H_HALF)
    assert h.binary_entropy(0.9) == pytest.approx(H_09)
    assert h.binary_entropy(0.0) == pytest.approx(0.0, abs=1e-9)
    assert h.binary_entropy(0.1) == pytest.approx(h.binary_entropy(0.9))


def test_decompose_uncertainty_fixture():
    mu, al, ep = h.decompose_uncertainty(np.array(PROBS))
    np.testing.assert_allclose(mu, MU, atol=1e-12)
    np.testing.assert_allclose(al, ALEATORIC, atol=1e-9)
    np.testing.assert_allclose(ep, EPISTEMIC, atol=1e-9)


def test_decomposition_identity():
    rng = np.random.default_rng(0)
    ps = rng.uniform(0.01, 0.99, size=(8, 30, 3))
    mu, al, ep = h.decompose_uncertainty(ps)
    np.testing.assert_allclose(al + ep, h.binary_entropy(mu), atol=1e-9)
    assert (ep >= 0).all()
    # identical draws carry no model uncertainty
    same = np.repeat(ps[:1], 8, axis=0)
    assert h.decompose_uncertainty(same)[2].max() < 1e-9


def test_ucb_scores_fixture():
    mu = np.array(MU)
    got = h.ucb_scores(mu, np.sqrt(mu * (1 - mu)), 2.0)
    np.testing.assert_allclose(got, UCB_BETA2, atol=1e-12)
    # beta = 0 reduces to the summed mean
    np.testing.assert_allclose(h.ucb_scores(mu, np.sqrt(mu * (1 - mu)), 0.0), mu.sum(1))


def test_top_k_ties_and_exclusion():
    s = np.array(TOPK_SCORES)
    assert h.top_k(s, 3).tolist() == TOPK_3
    mask = np.zeros(5, dtype=bool)
    mask[1] = True
    assert h.top_k(s, 3, mask).tolist() == TOPK_3_EXCLUDING_1
    assert len(h.top_k(s, 10, mask)) == 4


def test_merge_shortlists():
    assert h.merge_shortlists(MERGE_UNC, MERGE_SRCH).tolist() == MERGED


def test_expert_select_fixture():
    e = h.Expert(1.0, 0.5)
    e.known_mask = np.array(KNOWN, dtype=bool)
    e.known_target = np.array(TARGETS, dtype=bool)
    assert h.expert_select(e, SEARCH_LIST, UNC_LIST, 5).tolist() == EXPERT_PICK_5


def test_expert_overrides_known_negatives_and_spends_budget():
    e = h.Expert(1.0, 0.0)
    e.known_mask = np.ones(10, dtype=bool)
    e.known_target = np.zeros(10, dtype=bool)  # everything is a known non-target
    pick = h.expert_select(e, [0, 1, 2], [3, 4, 5], 4)
    assert len(pick) == 4 and len(set(pick.tolist())) == 4  # budget spent from the dropped ones


def test_metrics():
    assert h.hit_rate(30, 50, 50) == pytest.approx(HIT_30_OF_2500)
    assert h.recall_rate(30, 148) == pytest.approx(RECALL_30_OF_148)
    assert h.is_target(np.array([[1, 1, 1], [1, 0, 1]])).tolist() == [True, False]


def _synthetic(n=1500, d=8, seed=0):
    """Two-property pool where the target is the intersection of two half-spaces."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = np.stack([x[:, 0] + 0.3 * x[:, 2] > 1.2, x[:, 1] - 0.3 * x[:, 3] > 1.0], axis=1).astype(int)
    return x, y


def _fp(seed=0):
    def fit_predict(xl, yl):
        return h.BootstrapLogisticEnsemble(n_members=6, epochs=80, seed=seed).fit(xl, yl).predict_samples
    return fit_predict


def test_bootstrap_ensemble_learns():
    x, y = _synthetic()
    ens = h.BootstrapLogisticEnsemble(n_members=8, epochs=150, seed=1).fit(x[:600], y[:600])
    ps = ens.predict_samples(x[600:])
    assert ps.shape == (8, 900, 2)
    mu = ps.mean(0)
    acc = ((mu > 0.5).astype(int) == y[600:]).mean()
    assert acc > 0.85


def test_simulation_shapes_and_budget():
    x, y = _synthetic()
    r = h.simulate("hitl", x, y, _fp(), budget=20, rounds=6, q=20, h=20, knowledge=0.5, seed=3)
    assert len(r.hits_by_round) == 6
    assert all(b >= a for a, b in zip(r.hits_by_round, r.hits_by_round[1:]))
    assert r.hits_by_round[-1] <= r.n_targets
    assert r.hit_rate == pytest.approx(r.hits_by_round[-1] / (20 * 6))


def test_hitl_beats_random_on_synthetic_pool():
    x, y = _synthetic()
    rnd = np.mean([h.simulate("random", x, y, _fp(s), budget=20, rounds=10, seed=s).hits_by_round[-1]
                   for s in range(4)])
    hit = np.mean([h.simulate("hitl", x, y, _fp(s), budget=20, rounds=10, q=20, h=20,
                              knowledge=0.75, seed=s).hits_by_round[-1] for s in range(4)])
    assert hit > 2 * rnd


def test_more_expert_knowledge_does_not_hurt():
    x, y = _synthetic()

    def avg(k):
        return np.mean([h.simulate("hitl", x, y, _fp(s), budget=20, rounds=10, q=20, h=20,
                                   knowledge=k, seed=s).hits_by_round[-1] for s in range(5)])

    assert avg(1.0) >= avg(0.0)
