"""Clickstream graphs: node embeddings, burstiness, graph attention, quantile loss.

Reference implementations for the chapter "Graph Embeddings for Clickstream Sales
Forecasting". The chapter uses Liu, Chen, Zhang, Mai, and Zhao (2026), ForecastClickGraph,
as a worked example of embeddings. This module implements the framework-free numeric
pieces so that they can be tested and ported:

* **Graph construction.** ``session_graph`` turns sessions (ordered lists of
  product-stage node ids) into node counts and a directed, transition-weighted
  adjacency matrix. ``average_graph``, ``normalize_edge_weights`` and
  ``upstream_neighbors`` build the stabilized graph used to pick neighbors.
* **Scaling.** ``robust_median_scale`` centers a series on its median and divides by
  the mean absolute deviation about the median (the form used in the paper's worked
  example).
* **Burstiness.** ``burstiness`` is the Lappas et al. (2009) score of a single
  interval, ``x_t / sum(x) - 1/T``. ``neighbor_burstiness`` averages it over the
  upstream neighbors of a node. ``ema_last`` and ``burst_forecast`` convert a network
  output in ``[-1, 1]`` (via ``tanh``) into a percentage deviation from the trend.
* **Embeddings.** ``lstm_embed`` is a plain LSTM forward pass that maps a time series
  to a fixed-size vector (a *time series embedding*). ``attention_neighbor_embedding``
  is a one-head graph transformer layer with an edge term: the softmax weights are a
  learned stand-in for conversion rates.
* **Forecast assembly and loss.** ``gated_forecast`` adds a sigmoid-gated neighbor
  forecast to the base and burst forecasts. ``pinball_loss``,
  ``total_quantile_loss`` and ``weighted_quantile_loss`` implement the quantile
  (pinball) loss and its scale-free summary; ``interval_coverage`` checks calibration.

Every function mirrors the Julia module ``AIInAction.ChClickgraph`` and the Rust module
``aiinaction::ch_clickgraph`` one-to-one, and the three test suites share fixtures.
``simulate_clickstream`` (a seeded generator of promotion-driven daily graphs used by the
chapter experiment) is Python only.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "RobustScaled",
    "robust_median_scale",
    "burstiness",
    "session_graph",
    "average_graph",
    "normalize_edge_weights",
    "upstream_neighbors",
    "neighbor_burstiness",
    "ema_last",
    "burst_forecast",
    "softmax",
    "sigmoid",
    "det_matrix",
    "lstm_embed",
    "attention_neighbor_embedding",
    "gated_forecast",
    "pinball_loss",
    "total_quantile_loss",
    "weighted_quantile_loss",
    "interval_coverage",
    "Clickstream",
    "simulate_clickstream",
]


# ---------------------------------------------------------------------------
# Scaling and burstiness
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RobustScaled:
    """Result of :func:`robust_median_scale`."""

    median: float
    mad: float
    scaled: np.ndarray


def robust_median_scale(x: Sequence[float]) -> RobustScaled:
    """Center on the median and divide by the mean absolute deviation about the median.

    If the deviation is zero (a constant series) the divisor is 1, so the output is the
    centered series rather than a division by zero.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("x must be a non-empty 1-D sequence")
    med = float(np.median(a))
    mad = float(np.mean(np.abs(a - med)))
    div = mad if mad > 0.0 else 1.0
    return RobustScaled(med, mad, (a - med) / div)


def burstiness(x: Sequence[float]) -> np.ndarray:
    """Lappas et al. burstiness of each single interval: ``x_t / sum(x) - 1 / T``.

    The score is positive when an interval holds more than its uniform share ``1/T`` of
    the window total. An all-zero window has no bursts and returns zeros.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("x must be a non-empty 1-D sequence")
    if np.any(a < 0):
        raise ValueError("counts must be non-negative")
    total = float(a.sum())
    if total <= 0.0:
        return np.zeros_like(a)
    return a / total - 1.0 / a.size


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def session_graph(
    sessions: Sequence[Sequence[int]], n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build node counts and a directed transition matrix from sessions.

    Each session is an ordered list of node ids (a node is a product-stage pair such as
    "SKU A view"). ``counts[i]`` is the number of occurrences of node ``i`` and
    ``W[i, j]`` the number of times a session moved directly from ``i`` to ``j``.
    Consecutive repeats of the same node are counted as occurrences but not as
    transitions.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    counts = np.zeros(n_nodes)
    W = np.zeros((n_nodes, n_nodes))
    for s in sessions:
        prev = -1
        for node in s:
            if not 0 <= node < n_nodes:
                raise ValueError(f"node id {node} outside [0, {n_nodes})")
            counts[node] += 1.0
            if prev >= 0 and prev != node:
                W[prev, node] += 1.0
            prev = node
    return counts, W


def average_graph(graphs: Sequence[np.ndarray]) -> np.ndarray:
    """Elementwise mean of daily adjacency matrices over a context window."""
    if len(graphs) == 0:
        raise ValueError("need at least one graph")
    stack = np.stack([np.asarray(g, dtype=float) for g in graphs])
    return stack.mean(axis=0)


def normalize_edge_weights(W: np.ndarray, mean_counts: Sequence[float]) -> np.ndarray:
    """Divide each edge ``i -> j`` by the average quantity of its target node ``j``.

    Targets with a non-positive average quantity leave their incoming weights at zero.
    """
    W = np.asarray(W, dtype=float)
    m = np.asarray(mean_counts, dtype=float)
    if W.shape[0] != W.shape[1] or m.size != W.shape[1]:
        raise ValueError("W must be square and mean_counts must match its size")
    out = np.zeros_like(W)
    ok = m > 0
    out[:, ok] = W[:, ok] / m[ok]
    return out


def upstream_neighbors(W: np.ndarray, j: int) -> list[int]:
    """Nodes ``i`` with a positive edge ``i -> j`` (excluding ``j`` itself), ascending."""
    W = np.asarray(W, dtype=float)
    return [i for i in range(W.shape[0]) if i != j and W[i, j] > 0.0]


def neighbor_burstiness(B: np.ndarray, upstream: Sequence[int]) -> np.ndarray:
    """Average the burstiness series (rows of ``B``) over the upstream neighbors.

    A node with no upstream neighbors gets a zero series.
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2:
        raise ValueError("B must be 2-D (nodes x time)")
    if len(upstream) == 0:
        return np.zeros(B.shape[1])
    return B[list(upstream)].mean(axis=0)


def ema_last(x: Sequence[float], alpha: float) -> float:
    """Final value of the exponential moving average ``s_t = a x_t + (1 - a) s_(t-1)``."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        raise ValueError("x must be non-empty")
    s = float(a[0])
    for v in a[1:]:
        s = alpha * float(v) + (1.0 - alpha) * s
    return s


def burst_forecast(raw: Sequence[float], trend: float) -> np.ndarray:
    """Turn raw network outputs into burst forecasts ``tanh(raw) * trend``.

    ``tanh`` bounds the relative deviation to ``[-1, 1]``, and the trend (an EMA of the
    node's own recent counts) converts it to count units.
    """
    return np.tanh(np.asarray(raw, dtype=float)) * float(trend)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def softmax(v: Sequence[float]) -> np.ndarray:
    """Numerically stable softmax of a 1-D vector."""
    a = np.asarray(v, dtype=float)
    if a.size == 0:
        raise ValueError("v must be non-empty")
    e = np.exp(a - a.max())
    return e / e.sum()


def sigmoid(z: Sequence[float] | float) -> np.ndarray:
    """Logistic function, stable for large ``|z|``."""
    a = np.asarray(z, dtype=float)
    ez = np.exp(-np.abs(a))
    return np.where(a >= 0, 1.0 / (1.0 + ez), ez / (1.0 + ez))


def det_matrix(rows: int, cols: int, salt: float = 0.0) -> np.ndarray:
    """Deterministic test matrix ``0.4 sin(0.7 (i cols + j + 1) + salt)`` (0-based i, j).

    The same formula is used in all three languages so cross-language fixtures need no
    shared random numbers.
    """
    i = np.arange(rows)[:, None]
    j = np.arange(cols)[None, :]
    return 0.4 * np.sin(0.7 * (i * cols + j + 1) + salt)


def lstm_embed(x_seq: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Final hidden state of a single-layer LSTM: an embedding of the whole series.

    ``x_seq`` has shape ``(T, d)``; ``W`` has shape ``(4H, d + H)`` acting on the
    concatenation ``[x_t; h_(t-1)]`` with gate blocks ordered input, forget, cell
    candidate, output; ``b`` has length ``4H``. Initial states are zero.
    """
    x_seq = np.asarray(x_seq, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    if x_seq.ndim != 2 or x_seq.shape[0] == 0:
        raise ValueError("x_seq must be a non-empty (T, d) array")
    d = x_seq.shape[1]
    H = W.shape[0] // 4
    if W.shape != (4 * H, d + H) or b.shape != (4 * H,):
        raise ValueError("W must be (4H, d+H) and b must be (4H,)")
    h = np.zeros(H)
    c = np.zeros(H)
    for x in x_seq:
        z = W @ np.concatenate([x, h]) + b
        i = sigmoid(z[:H])
        f = sigmoid(z[H : 2 * H])
        g = np.tanh(z[2 * H : 3 * H])
        o = sigmoid(z[3 * H :])
        c = f * c + i * g
        h = o * np.tanh(c)
    return h


def attention_neighbor_embedding(
    h_i: Sequence[float],
    H_nb: np.ndarray,
    e_nb: Sequence[float],
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    we: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """One-head graph transformer layer with scalar edge weights.

    With ``q = Wq h_i``, ``k_j = Wk h_j + we * e_ij`` and ``v_j = Wv h_j``, the attention
    weights are ``alpha_j = softmax_j(q . k_j / sqrt(d_k))`` and the neighbor embedding
    is ``z = sum_j alpha_j v_j``. Returns ``(z, alpha)``; a node with no neighbors gets
    ``z = 0`` and an empty ``alpha``.
    """
    h_i = np.asarray(h_i, dtype=float)
    H_nb = np.asarray(H_nb, dtype=float)
    Wq, Wk, Wv = (np.asarray(m, dtype=float) for m in (Wq, Wk, Wv))
    we = np.asarray(we, dtype=float)
    dk = Wq.shape[0]
    if Wk.shape[0] != dk or we.shape != (dk,):
        raise ValueError("Wq, Wk and we must share the key dimension")
    if H_nb.size == 0:
        return np.zeros(Wv.shape[0]), np.zeros(0)
    e = np.asarray(e_nb, dtype=float)
    if H_nb.ndim != 2 or e.shape != (H_nb.shape[0],):
        raise ValueError("H_nb must be (n_neighbors, dim) and e_nb must have n_neighbors entries")
    q = Wq @ h_i
    K = H_nb @ Wk.T + e[:, None] * we[None, :]
    V = H_nb @ Wv.T
    alpha = softmax(K @ q / math.sqrt(dk))
    return alpha @ V, alpha


# ---------------------------------------------------------------------------
# Forecast assembly and quantile loss
# ---------------------------------------------------------------------------
def gated_forecast(
    base: Sequence[float],
    burst: Sequence[float],
    neigh: Sequence[float],
    gate_logits: Sequence[float],
) -> np.ndarray:
    """Final forecast ``base + burst + sigmoid(gate_logits) * neigh`` (elementwise)."""
    base, burst, neigh, g = (np.asarray(v, dtype=float) for v in (base, burst, neigh, gate_logits))
    if not (base.shape == burst.shape == neigh.shape == g.shape):
        raise ValueError("all inputs must have the same shape")
    return base + burst + sigmoid(g) * neigh


def pinball_loss(y: float, yhat: float, q: float) -> float:
    """Quantile (pinball) loss ``q (y - yhat)_+ + (1 - q) (yhat - y)_+``."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    return q * max(0.0, y - yhat) + (1.0 - q) * max(0.0, yhat - y)


def total_quantile_loss(
    y: Sequence[float], preds: np.ndarray, quantiles: Sequence[float]
) -> float:
    """Sum of pinball losses over horizon steps and quantiles.

    ``y`` has length ``H`` and ``preds`` has shape ``(H, Q)`` with one column per
    quantile level.
    """
    y = np.asarray(y, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if preds.shape != (y.size, len(quantiles)):
        raise ValueError("preds must have shape (len(y), len(quantiles))")
    return float(
        sum(pinball_loss(y[h], preds[h, k], q) for h in range(y.size) for k, q in enumerate(quantiles))
    )


def weighted_quantile_loss(y: Sequence[float], yhat: Sequence[float], q: float) -> float:
    """Scale-free summary ``2 sum pinball_q / sum |y|`` (the wQL of probabilistic forecasting)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape or y.size == 0:
        raise ValueError("y and yhat must be non-empty and the same shape")
    denom = float(np.abs(y).sum())
    if denom == 0.0:
        raise ValueError("sum |y| is zero; wQL is undefined")
    total = sum(pinball_loss(float(a), float(b), q) for a, b in zip(y, yhat))
    return 2.0 * total / denom


def interval_coverage(y: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> float:
    """Fraction of observations inside ``[lo, hi]`` (inclusive)."""
    y, lo, hi = (np.asarray(v, dtype=float) for v in (y, lo, hi))
    if not (y.shape == lo.shape == hi.shape) or y.size == 0:
        raise ValueError("y, lo and hi must be non-empty and the same shape")
    return float(np.mean((y >= lo) & (y <= hi)))


# ---------------------------------------------------------------------------
# Simulator (Python only)
# ---------------------------------------------------------------------------
@dataclass
class Clickstream:
    """Simulated daily product-stage graphs.

    Node ``3 p + s`` is product ``p`` at stage ``s`` (0 view, 1 cart, 2 purchase).
    ``counts`` has shape ``(D, N)`` and ``W`` shape ``(D, N, N)`` (edge ``i -> j``).
    ``promo`` has shape ``(D, P)`` and marks days on which the product's category is
    in promotion; ``burst_day`` marks days on which a product's purchases are inflated
    by a promotion that started on an earlier day.
    """

    counts: np.ndarray
    W: np.ndarray
    promo: np.ndarray
    burst_day: np.ndarray
    category: np.ndarray
    n_products: int
    n_days: int


def simulate_clickstream(
    n_products: int = 120,
    n_days: int = 70,
    n_categories: int = 8,
    seed: int = 0,
    promo_gain: tuple[float, float] = (0.6, 2.4),
) -> Clickstream:
    """Generate promotion-driven clickstream counts and transition-weighted graphs.

    Each product has a base attention rate with weekly seasonality. A promotion
    multiplies the attention of every product in a category by ``1 + g_p`` for three
    days, where ``g_p`` is a product-specific gain. Views follow attention, carts are a
    binomial thinning of views, and purchases follow attention with a lag of one to
    three days, so a burst in views and carts precedes the burst in purchases. Half of
    the products are sparse (low base rate), which makes their own series noisy.
    Intra-product edges are view to cart, cart to purchase, and view to purchase;
    inter-product edges are view to view (a user moves on to a related product) and
    related-view to purchase (a user viewed a related product, then bought this one).

    This is a count-level generator, not a session-level one: edge weights are drawn
    directly rather than counted from simulated sessions (see ``session_graph``).
    """
    rng = np.random.default_rng(seed)
    P, D, S = n_products, n_days, 3
    N = P * S
    category = np.arange(P) % n_categories
    sparse = rng.random(P) < 0.5
    base = np.where(sparse, rng.lognormal(np.log(2.5), 0.4, P), rng.lognormal(np.log(14.0), 0.4, P))
    gain = rng.uniform(promo_gain[0], promo_gain[1], P)
    kappa = rng.uniform(0.2, 0.35, P)  # view -> cart
    beta = rng.uniform(0.25, 0.45, P)  # attention -> purchase
    # promotion calendar: each category gets promotions of 3 days with random gaps
    promo_cat = np.zeros((D, n_categories))
    for c in range(n_categories):
        t = int(rng.integers(3, 9))
        while t < D - 2:
            promo_cat[t : t + 3, c] = 1.0
            t += 3 + int(rng.integers(9, 18))
    promo = promo_cat[:, category]  # (D, P)
    season = 1.0 + 0.2 * np.sin(2 * np.pi * np.arange(D) / 7.0)
    u = base[None, :] * season[:, None] * (1.0 + gain[None, :] * promo)  # attention (D, P)
    views = rng.poisson(u).astype(float)
    carts = rng.binomial(views.astype(int), kappa[None, :]).astype(float)
    idx = np.arange(D)
    lag = (
        0.4 * u[np.maximum(idx - 1, 0)]
        + 0.4 * u[np.maximum(idx - 2, 0)]
        + 0.2 * u[np.maximum(idx - 3, 0)]
    )
    purchases = rng.poisson(beta[None, :] * lag).astype(float)
    burst_day = np.zeros((D, P))
    for t in range(1, D):
        burst_day[t] = (promo[max(0, t - 3) : t].max(axis=0) > 0).astype(float)
    counts = np.zeros((D, N))
    counts[:, 0::3] = views
    counts[:, 1::3] = carts
    counts[:, 2::3] = purchases
    # co-category affinity: each product has four fixed "related" products in its category
    related = np.zeros((P, 4), dtype=int)
    for p in range(P):
        peers = np.flatnonzero((category == category[p]) & (np.arange(P) != p))
        related[p] = rng.choice(peers, 4, replace=False)
    W = np.zeros((D, N, N))
    for t in range(D):
        for p in range(P):
            v, c, b = 3 * p, 3 * p + 1, 3 * p + 2
            W[t, v, c] = carts[t, p]
            W[t, c, b] = purchases[t, p]
            W[t, v, b] = rng.binomial(int(purchases[t, p]), 0.25)  # direct view to purchase
            out = rng.binomial(int(views[t, p]), 0.3)  # leave for a related product
            share = rng.multinomial(out, np.full(4, 0.25))
            also = rng.multinomial(rng.binomial(int(purchases[t, p]), 0.4), np.full(4, 0.25))
            for k, q in enumerate(related[p]):
                W[t, v, 3 * q] += share[k]  # view p -> view q
                W[t, 3 * q, b] += also[k]  # viewed related q, then bought p
    return Clickstream(counts, W, promo, burst_day, category, P, D)
    return Clickstream(counts, W, promo, burst_day, category, P, D)
