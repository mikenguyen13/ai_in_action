"""Tests for aiinaction.ch_clickgraph, including shared cross-language fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites assert
against the same numbers, which keeps the three implementations at parity.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aiinaction import ch_clickgraph as g

# ---------------------------------------------------------------------------
# Shared fixtures (mirrored in julia/AIInAction/test and rust/aiinaction/src)
# ---------------------------------------------------------------------------
PAPER_SERIES = [9.0, 14.0, 17.0]  # purchases of SKU A in the paper's Figure 3
SCALE_MEDIAN = 14.0
SCALE_MAD = 8.0 / 3.0
SCALED = [-1.875, 0.0, 1.125]
BURST = [-0.10833333333333331, 0.016666666666666663, 0.09166666666666667]

SESSIONS = [[0, 1, 2], [0, 1], [0, 3, 0, 1], [3, 3, 4]]  # 0-based node ids, 5 nodes
SESSION_COUNTS = [4.0, 3.0, 1.0, 3.0, 1.0]
SESSION_W = [
    [0.0, 3.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
]

EMA_123_07 = 2.61

LSTM_X = [[1.0], [2.0], [0.5]]
LSTM_H = [-0.007358975720281007, -0.3247841816214326]

ATT_H = [0.5, -0.2, 0.1, 0.8]
ATT_E = [0.2, 1.0, 0.5]
ATT_Z = [0.13473023562592665, -0.13723523527276552]
ATT_ALPHA = [0.3514830614166104, 0.28412005829708203, 0.3643968802863075]

PINBALL_075_10_8 = 1.5
TOTAL_QL = 1.2  # y=[1.5, 2.5], preds=[[1,2,3],[2,2,2]], q=[0.1,0.5,0.9]
WQL_09 = 0.16  # y=[10,20,0], yhat=[8,25,1]
GATED = [2.5, 4.424234314520019]  # base=[1,2], burst=[0.5,-0.5], neigh=[2,4], gate=[0,1]
BURST_FC = [0.0, 7.6159415595576485]  # tanh([0,1]) * 10


def lstm_params():
    return g.det_matrix(8, 3), g.det_matrix(1, 8, 0.3)[0]


def att_params():
    return (
        g.det_matrix(3, 4, 0.1),
        g.det_matrix(3, 4, 0.2),
        g.det_matrix(2, 4, 0.3),
        g.det_matrix(1, 3, 0.4)[0],
        g.det_matrix(3, 4, 0.5),
    )


def test_robust_median_scale_paper_example():
    r = g.robust_median_scale(PAPER_SERIES)
    assert r.median == SCALE_MEDIAN
    assert r.mad == pytest.approx(SCALE_MAD)
    assert r.scaled == pytest.approx(SCALED)


def test_robust_median_scale_constant_and_errors():
    r = g.robust_median_scale([3.0, 3.0, 3.0])
    assert r.mad == 0.0 and list(r.scaled) == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        g.robust_median_scale([])


def test_burstiness_paper_example_and_properties():
    b = g.burstiness(PAPER_SERIES)
    assert b == pytest.approx(BURST, abs=1e-12)
    assert b.sum() == pytest.approx(0.0, abs=1e-12)  # shares sum to one
    assert list(g.burstiness([0, 0, 0])) == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        g.burstiness([1.0, -1.0])


def test_session_graph_fixture():
    counts, W = g.session_graph(SESSIONS, 5)
    assert counts.tolist() == SESSION_COUNTS
    assert W.tolist() == SESSION_W
    with pytest.raises(ValueError):
        g.session_graph([[0, 9]], 5)


def test_upstream_and_neighbor_burstiness():
    _, W = g.session_graph(SESSIONS, 5)
    assert g.upstream_neighbors(W, 1) == [0]
    assert g.upstream_neighbors(W, 0) == [3]
    assert g.upstream_neighbors(W, 4) == [3]
    assert g.upstream_neighbors(W, 3) == [0]
    B = np.array([g.burstiness(r) for r in ([9, 14, 17], [2, 4, 4], [1, 1, 1])])
    nb = g.neighbor_burstiness(B, [0, 1])
    assert nb == pytest.approx((B[0] + B[1]) / 2)
    assert list(g.neighbor_burstiness(B, [])) == [0.0, 0.0, 0.0]


def test_average_and_normalize_graph():
    A = g.average_graph([np.eye(2), 3 * np.eye(2)])
    assert A.tolist() == [[2.0, 0.0], [0.0, 2.0]]
    N = g.normalize_edge_weights(np.array([[0.0, 4.0], [2.0, 0.0]]), [2.0, 0.0])
    assert N.tolist() == [[0.0, 0.0], [1.0, 0.0]]


def test_ema_and_burst_forecast():
    assert g.ema_last([1.0, 2.0, 3.0], 0.7) == pytest.approx(EMA_123_07)
    assert g.burst_forecast([0.0, 1.0], 10.0) == pytest.approx(BURST_FC)
    with pytest.raises(ValueError):
        g.ema_last([1.0], 0.0)


def test_softmax_and_sigmoid():
    assert g.softmax([1000.0, 1000.0]).tolist() == [0.5, 0.5]  # stable for large logits
    assert g.sigmoid(0.0) == 0.5
    assert g.sigmoid(-800.0) == pytest.approx(0.0, abs=1e-300)


def test_lstm_embed_fixture_and_bounds():
    W, b = lstm_params()
    h = g.lstm_embed(LSTM_X, W, b)
    assert h == pytest.approx(LSTM_H, abs=1e-12)
    assert np.all(np.abs(h) < 1.0)
    with pytest.raises(ValueError):
        g.lstm_embed(LSTM_X, W[:, :2], b)


def test_attention_fixture_and_properties():
    Wq, Wk, Wv, we, Hn = att_params()
    z, a = g.attention_neighbor_embedding(ATT_H, Hn, ATT_E, Wq, Wk, Wv, we)
    assert a == pytest.approx(ATT_ALPHA, abs=1e-12)
    assert z == pytest.approx(ATT_Z, abs=1e-12)
    assert a.sum() == pytest.approx(1.0)
    assert z == pytest.approx(a @ (Hn @ Wv.T))  # convex combination of values
    # permutation equivariance of the neighbor set
    perm = [2, 0, 1]
    z2, a2 = g.attention_neighbor_embedding(ATT_H, Hn[perm], np.array(ATT_E)[perm], Wq, Wk, Wv, we)
    assert z2 == pytest.approx(z)
    assert a2 == pytest.approx(a[perm])
    z0, a0 = g.attention_neighbor_embedding(ATT_H, np.zeros((0, 4)), [], Wq, Wk, Wv, we)
    assert z0.tolist() == [0.0, 0.0] and a0.size == 0


def test_gated_forecast_fixture():
    out = g.gated_forecast([1.0, 2.0], [0.5, -0.5], [2.0, 4.0], [0.0, 1.0])
    assert out == pytest.approx(GATED)
    with pytest.raises(ValueError):
        g.gated_forecast([1.0], [1.0, 2.0], [1.0], [0.0])


def test_pinball_and_totals():
    assert g.pinball_loss(10.0, 8.0, 0.75) == PINBALL_075_10_8
    assert g.pinball_loss(8.0, 10.0, 0.75) == pytest.approx(0.25 * 2.0)
    preds = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    assert g.total_quantile_loss([1.5, 2.5], preds, [0.1, 0.5, 0.9]) == pytest.approx(TOTAL_QL)
    assert g.weighted_quantile_loss([10, 20, 0], [8, 25, 1], 0.9) == pytest.approx(WQL_09)
    with pytest.raises(ValueError):
        g.pinball_loss(1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        g.weighted_quantile_loss([0.0], [1.0], 0.5)


def test_pinball_minimizer_is_the_quantile():
    rng = np.random.default_rng(3)
    y = rng.gamma(2.0, 3.0, 4000)
    grid = np.linspace(0, 30, 3001)
    for q in (0.1, 0.5, 0.9):
        risk = [np.mean([g.pinball_loss(v, c, q) for v in y[:400]]) for c in grid[::30]]
        best = grid[::30][int(np.argmin(risk))]
        assert abs(best - np.quantile(y[:400], q)) < 0.8


def test_interval_coverage():
    assert g.interval_coverage([1, 2, 3, 4], [0, 0, 0, 0], [2, 2, 2, 2]) == 0.5


def test_simulator_shapes_and_signal():
    cs = g.simulate_clickstream(n_products=24, n_days=40, n_categories=4, seed=1)
    assert cs.counts.shape == (40, 72) and cs.W.shape == (40, 72, 72)
    assert cs.promo.sum() > 0
    # promotions raise views of promoted products
    v = cs.counts[:, 0::3]
    assert v[cs.promo > 0].mean() > 1.5 * v[cs.promo == 0].mean()
    # determinism
    cs2 = g.simulate_clickstream(n_products=24, n_days=40, n_categories=4, seed=1)
    assert np.array_equal(cs.counts, cs2.counts)
    # edges only go where defined: view->cart and cart->purchase within a product
    assert np.all(cs.W[:, 0::3, 1::3].diagonal(axis1=1, axis2=2) >= 0)


def test_pinball_matches_numpy_vectorized():
    y = np.array([0.0, 3.0, 7.0])
    yhat = np.array([1.0, 3.0, 5.0])
    q = 0.3
    vec = np.maximum(q * (y - yhat), (q - 1) * (y - yhat))
    assert sum(g.pinball_loss(a, b, q) for a, b in zip(y, yhat)) == pytest.approx(vec.sum())
    assert math.isclose(g.weighted_quantile_loss(y, yhat, q), 2 * vec.sum() / np.abs(y).sum())
