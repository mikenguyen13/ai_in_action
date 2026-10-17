"""Learning a latent migration flow tensor from partial, inconsistent aggregates.

Reference implementations for the chapter on Gaskin and Abel (2026), "Deep learning
four decades of human migration". The paper embeds a mechanistic stock-flow
accounting identity inside a recurrent neural network and trains the network so the
implied stocks, net migration and total flows match partial, noisy observations.
This module contains the pieces of that pipeline:

* **Stock-flow accounting.** ``stock_step`` advances the stock table one year with
  births, deaths and the flow tensor ``T[i, j, k]`` (born in ``i``, moving from ``j``
  to ``k``); ``od_flows`` and ``net_migration`` aggregate the tensor to the quantities
  that are actually observed.
* **Scaling.** The symmetrised Yeo-Johnson transform ``yj_sym`` with its inverse,
  derivative (for elasticities through the chain rule), and a profile likelihood.
* **Demographic balancing.** ``ipf`` (iterative proportional fitting) and
  ``demographic_balance`` reconcile two successive stock tables and yield a relative
  uncertainty per cell, which ``uncertainty_weights`` turns into loss weights.
* **Loss and calibration.** ``weighted_l2`` and ``calibrate_offset`` (the closed-form
  initial-stock offset, with the non-negativity constraint).
* **A synthetic world** (``make_world``) with gravity-style ground truth, a memory
  effect, and noisy partial observations, plus a **torch recurrent latent-state
  model** (``train_ensemble``) and recovery metrics. The torch section is Python only.

The numeric functions mirror the Julia module ``AIInAction.ChMigrationFlows`` and the
Rust module ``aiinaction::ch_migration_flows``; the three test suites share fixtures.
Tensors are stored as ``T[i, j, k]`` (Julia: ``T[i, j, k]``, 1-based; Rust: flat
row-major ``(i * n + j) * n + k``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "yj_sym",
    "yj_sym_inv",
    "yj_sym_deriv",
    "yj_profile_loglik",
    "fit_yj_lambda",
    "elasticity_chain",
    "stock_step",
    "od_flows",
    "net_migration",
    "ipf",
    "demographic_balance",
    "uncertainty_weights",
    "weighted_l2",
    "survival_fraction",
    "calibrate_offset",
    "stock_difference_baseline",
    "gravity_log_rate",
    "log_rmse",
    "pearson",
    "coverage",
    "ensemble_mean_std",
    "SyntheticWorld",
    "make_world",
    "simulate_truth",
    "observe",
    "Observations",
    "TrainConfig",
    "train_ensemble",
    "recovery_report",
    "feature_elasticity",
]

# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _finite_array(name: str, x, ndim: int | None = None) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _check_lambda(lam: float) -> float:
    lam = float(lam)
    if not math.isfinite(lam):
        raise ValueError(f"lambda must be finite, got {lam}")
    return lam


# --------------------------------------------------------------------------
# Symmetrised Yeo-Johnson transform (Gaskin and Abel eq. 8)
# --------------------------------------------------------------------------


def yj_sym(x, lam: float) -> np.ndarray:
    """phi_lambda(x) = sgn(x) ((|x|+1)^lambda - 1)/lambda, or sgn(x) log(|x|+1) at 0.

    lambda = 1 is the identity; lambda < 1 compresses large magnitudes; the sign
    symmetry treats negative values (net migration) like positive ones.
    """
    lam = _check_lambda(lam)
    a = _finite_array("x", x)
    mag = np.abs(a)
    if abs(lam) < 1e-12:
        body = np.log1p(mag)
    else:
        body = np.expm1(lam * np.log1p(mag)) / lam
    return np.sign(a) * body


def yj_sym_inv(y, lam: float) -> np.ndarray:
    """Inverse of ``yj_sym``. Raises if ``y`` is outside the range of ``yj_sym``."""
    lam = _check_lambda(lam)
    a = _finite_array("y", y)
    mag = np.abs(a)
    if abs(lam) < 1e-12:
        body = np.expm1(mag)
    else:
        inner = 1.0 + lam * mag
        if np.any(inner <= 0.0):
            raise ValueError("y is outside the range of the transform for this lambda")
        body = np.expm1(np.log(inner) / lam)
    return np.sign(a) * body


def yj_sym_deriv(x, lam: float) -> np.ndarray:
    """d phi / d x = (|x|+1)^(lambda-1), which is positive and even in x."""
    lam = _check_lambda(lam)
    a = _finite_array("x", x)
    return np.exp((lam - 1.0) * np.log1p(np.abs(a)))


def yj_profile_loglik(x, lam: float) -> float:
    """Profile Gaussian log-likelihood of the transformed sample (up to a constant).

    ``-n/2 log var(phi(x)) + (lambda - 1) sum log(|x|+1)``: the second term is the log
    Jacobian of the transform. Maximising over lambda is the paper's maximum
    likelihood choice of the scaling parameter.
    """
    a = _finite_array("x", x, ndim=1)
    if a.size < 2:
        raise ValueError("need at least two observations")
    phi = yj_sym(a, lam)
    var = float(np.mean((phi - phi.mean()) ** 2))
    if var <= 0.0:
        raise ValueError("transformed sample has zero variance")
    jac = (float(lam) - 1.0) * float(np.sum(np.log1p(np.abs(a))))
    return -0.5 * a.size * math.log(var) + jac


def fit_yj_lambda(x, grid=None) -> float:
    """Grid-search maximum likelihood lambda (default grid -1 to 2 in steps of 0.01)."""
    if grid is None:
        grid = np.arange(-100, 201) / 100.0
    best, best_ll = None, -math.inf
    for lam in grid:
        try:
            ll = yj_profile_loglik(x, float(lam))
        except ValueError:
            continue
        if ll > best_ll:
            best, best_ll = float(lam), ll
    if best is None:
        raise ValueError("no admissible lambda on the grid")
    return best


def elasticity_chain(du_dphi, x, lam: float) -> np.ndarray:
    """Elasticity of the output u = log T with respect to the raw covariate x.

    ``eps = (du/dphi) * phi'(x) * |x|``: the network differentiates with respect to
    the *transformed* input, and the chain rule converts that to a percentage
    response of the flow to a percentage change of the raw covariate.
    """
    d = _finite_array("du_dphi", du_dphi)
    xa = _finite_array("x", x)
    return d * yj_sym_deriv(xa, lam) * np.abs(xa)


# --------------------------------------------------------------------------
# Stock-flow accounting (Gaskin and Abel eqs. 2, 4, 5)
# --------------------------------------------------------------------------


def stock_step(stock, births, mortality, flows) -> np.ndarray:
    """One annual step of the accounting identity.

    ``S_ij(t+1) = S_ij + delta_ij B_j - mu_j S_ij + sum_k (T_ikj - T_ijk)``

    ``stock[i, j]`` is the number born in ``i`` living in ``j``; ``births[j]`` the
    births in ``j``; ``mortality[j]`` the annual death probability in ``j``;
    ``flows[i, j, k]`` the number born in ``i`` moving from ``j`` to ``k``.
    """
    s = _finite_array("stock", stock, ndim=2)
    n = s.shape[0]
    if s.shape != (n, n):
        raise ValueError("stock must be square")
    b = _finite_array("births", births, ndim=1)
    mu = _finite_array("mortality", mortality, ndim=1)
    t = _finite_array("flows", flows, ndim=3)
    if b.shape != (n,) or mu.shape != (n,) or t.shape != (n, n, n):
        raise ValueError("births, mortality, flows must match the stock dimension")
    if np.any(b < 0) or np.any(mu < 0) or np.any(mu > 1):
        raise ValueError("births must be non-negative and mortality in [0, 1]")
    if np.any(t < 0):
        raise ValueError("flows must be non-negative")
    arrivals = t.sum(axis=1)  # sum over origin j: people born in i arriving in k
    departures = t.sum(axis=2)  # sum over destination k: people born in i leaving j
    return s + np.diag(b) - mu[None, :] * s + arrivals - departures


def od_flows(flows) -> np.ndarray:
    """Total origin-destination flow ``F[j, k] = sum_i T[i, j, k]`` (eq. 4)."""
    t = _finite_array("flows", flows, ndim=3)
    return t.sum(axis=0)


def net_migration(f) -> np.ndarray:
    """``mu_j = sum_k F[k, j] - sum_k F[j, k]`` (eq. 5); sums to zero over countries."""
    a = _finite_array("F", f, ndim=2)
    if a.shape[0] != a.shape[1]:
        raise ValueError("F must be square")
    return a.sum(axis=0) - a.sum(axis=1)


def stock_difference_baseline(s_prev, s_next, interval: float) -> np.ndarray:
    """Classical stock differencing: ``max(S(t2) - S(t1), 0) / interval`` off the diagonal.

    This is the family of methods the paper compares against: it assumes people only
    move between their birth country and a destination, ignores births and deaths,
    and cannot resolve years inside the interval.
    """
    if interval <= 0:
        raise ValueError("interval must be positive")
    a = _finite_array("s_prev", s_prev, ndim=2)
    b = _finite_array("s_next", s_next, ndim=2)
    if a.shape != b.shape:
        raise ValueError("stock tables must have the same shape")
    out = np.maximum(b - a, 0.0) / float(interval)
    np.fill_diagonal(out, 0.0)
    return out


# --------------------------------------------------------------------------
# Iterative proportional fitting and demographic balancing
# --------------------------------------------------------------------------


def ipf(table, row_targets, col_targets, tol: float = 1e-10, max_iter: int = 10_000):
    """Iterative proportional fitting (Deming and Stephan, 1940).

    Alternately rescales rows and columns of a non-negative table until its marginals
    match the targets. Returns ``(fitted, iterations)``. The target totals must
    agree; a row or column with a positive target but no mass cannot be fitted.
    """
    x = _finite_array("table", table, ndim=2).copy()
    r = _finite_array("row_targets", row_targets, ndim=1)
    c = _finite_array("col_targets", col_targets, ndim=1)
    if x.shape != (r.size, c.size):
        raise ValueError("target lengths must match the table shape")
    if np.any(x < 0) or np.any(r < 0) or np.any(c < 0):
        raise ValueError("table and targets must be non-negative")
    if abs(r.sum() - c.sum()) > 1e-9 * max(1.0, r.sum()):
        raise ValueError("row and column targets must have the same total")
    if np.any((x.sum(axis=1) == 0) & (r > 0)) or np.any((x.sum(axis=0) == 0) & (c > 0)):
        raise ValueError("a target is positive but the table row or column is empty")
    scale = max(1.0, r.sum())
    for it in range(1, max_iter + 1):
        rs = x.sum(axis=1)
        x *= np.where(rs > 0, r / np.where(rs > 0, rs, 1.0), 0.0)[:, None]
        cs = x.sum(axis=0)
        x *= np.where(cs > 0, c / np.where(cs > 0, cs, 1.0), 0.0)[None, :]
        err = max(np.abs(x.sum(axis=1) - r).max(), np.abs(x.sum(axis=0) - c).max())
        if err <= tol * scale:
            return x, it
    return x, max_iter


def demographic_balance(s1, s2, births, deaths):
    """Balance two successive stock tables against births and deaths.

    Following the scheme the paper cites: add the interval's births (diagonal) to the
    first table and the deaths to the second, force both to the average of their
    marginals with ``ipf``, then subtract births and deaths again. Returns the two
    balanced tables and the relative adjustment ``|balanced - original| / original``
    of each cell, which serves as the cell's relative uncertainty.
    """
    a = _finite_array("s1", s1, ndim=2)
    b = _finite_array("s2", s2, ndim=2)
    d = _finite_array("deaths", deaths, ndim=2)
    g = _finite_array("births", births, ndim=1)
    n = a.shape[0]
    if a.shape != (n, n) or b.shape != (n, n) or d.shape != (n, n) or g.shape != (n,):
        raise ValueError("shape mismatch")
    a1 = a + np.diag(g)
    a2 = b + d
    rows = 0.5 * (a1.sum(axis=1) + a2.sum(axis=1))
    cols = 0.5 * (a1.sum(axis=0) + a2.sum(axis=0))
    f1, _ = ipf(a1, rows, cols)
    f2, _ = ipf(a2, rows, cols)
    bal1 = f1 - np.diag(g)
    bal2 = f2 - d
    eps = 1e-12
    rel1 = np.abs(bal1 - a) / np.maximum(np.abs(a), eps)
    rel2 = np.abs(bal2 - b) / np.maximum(np.abs(b), eps)
    return bal1, bal2, rel1, rel2


# --------------------------------------------------------------------------
# Loss weights, loss, initial-stock calibration
# --------------------------------------------------------------------------


def uncertainty_weights(rel_unc, lo: float = 0.5, hi: float = 2.0, floor: float = 1e-6):
    """Weights from relative uncertainty: inverse, mean-normalised, clamped to [lo, hi].

    Smaller uncertainty gives a larger weight. One pass (normalise to mean 1, then
    clamp) is used, so the mean of the clamped weights can differ slightly from 1.
    """
    u = _finite_array("rel_unc", rel_unc)
    if np.any(u < 0):
        raise ValueError("relative uncertainty must be non-negative")
    if not (0 < lo <= hi):
        raise ValueError("need 0 < lo <= hi")
    raw = 1.0 / np.maximum(u, floor)
    w = raw / raw.mean()
    return np.clip(w, lo, hi)


def weighted_l2(pred, target, weights) -> float:
    """``mean_k w_k (pred_k - target_k)^2``, the structure of eqs. 7 and 9."""
    p = _finite_array("pred", pred)
    t = _finite_array("target", target)
    w = _finite_array("weights", weights)
    if p.shape != t.shape or p.shape != w.shape:
        raise ValueError("pred, target, weights must have the same shape")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    return float(np.mean(w * (p - t) ** 2))


def survival_fraction(mortality) -> np.ndarray:
    """``s(t0) = 1`` and ``s(t) = prod_{t0 <= tau < t} (1 - mu(tau))``; length ``len + 1``."""
    mu = _finite_array("mortality", mortality, ndim=1)
    if np.any(mu < 0) or np.any(mu > 1):
        raise ValueError("mortality must lie in [0, 1]")
    return np.concatenate(([1.0], np.cumprod(1.0 - mu)))


def calibrate_offset(observed, predicted, weights, survival) -> float:
    """Closed-form initial-stock offset (eq. 11) with the non-negativity constraint.

    ``b = sum_t s w (S - S_hat) / sum_t w s^2`` minimises the weighted squared error
    of ``S_hat(t) + s(t) b``. The result is raised to the smallest ``b`` that keeps
    every shifted stock non-negative.
    """
    o = _finite_array("observed", observed, ndim=1)
    p = _finite_array("predicted", predicted, ndim=1)
    w = _finite_array("weights", weights, ndim=1)
    s = _finite_array("survival", survival, ndim=1)
    if not (o.shape == p.shape == w.shape == s.shape):
        raise ValueError("all series must have the same length")
    if np.any(w < 0) or np.any(s <= 0):
        raise ValueError("weights must be non-negative and survival positive")
    den = float(np.sum(w * s * s))
    if den <= 0.0:
        raise ValueError("weights have no mass")
    b = float(np.sum(s * w * (o - p)) / den)
    lower = float(np.max(-p / s))
    return max(b, lower)


# --------------------------------------------------------------------------
# Ground-truth gravity rate and recovery metrics
# --------------------------------------------------------------------------


def gravity_log_rate(beta, log_gdp_o, log_gdp_d, log_dist, lang, log_pop_d) -> np.ndarray:
    """``b0 + bg (log gdp_d - log gdp_o) + bd log dist + bl lang + bp log pop_d``.

    ``beta = (b0, bg, bd, bl, bp)``. This is the per-capita departure rate (log) used
    to generate the synthetic ground truth.
    """
    b = _finite_array("beta", beta, ndim=1)
    if b.size != 5:
        raise ValueError("beta must have five entries")
    go = _finite_array("log_gdp_o", log_gdp_o)
    gd = _finite_array("log_gdp_d", log_gdp_d)
    dist = _finite_array("log_dist", log_dist)
    lg = _finite_array("lang", lang)
    pd_ = _finite_array("log_pop_d", log_pop_d)
    return b[0] + b[1] * (gd - go) + b[2] * dist + b[3] * lg + b[4] * pd_


def log_rmse(truth, estimate, floor: float = 1e-9) -> float:
    """Root mean squared error on the natural-log scale (multiplicative accuracy)."""
    t = _finite_array("truth", truth)
    e = _finite_array("estimate", estimate)
    if t.shape != e.shape:
        raise ValueError("shape mismatch")
    d = np.log(np.maximum(e, floor)) - np.log(np.maximum(t, floor))
    return float(np.sqrt(np.mean(d * d)))


def pearson(x, y) -> float:
    """Pearson correlation of two samples (nan if either has zero variance)."""
    a = _finite_array("x", x).ravel()
    b = _finite_array("y", y).ravel()
    if a.shape != b.shape or a.size < 2:
        raise ValueError("need two samples of equal length >= 2")
    ac, bc = a - a.mean(), b - b.mean()
    den = math.sqrt(float(ac @ ac) * float(bc @ bc))
    return float(ac @ bc) / den if den > 0 else math.nan


def coverage(truth, mean, std, z: float = 1.96) -> float:
    """Share of truths inside ``mean +/- z std``."""
    t = _finite_array("truth", truth)
    m = _finite_array("mean", mean)
    s = _finite_array("std", std)
    if not (t.shape == m.shape == s.shape):
        raise ValueError("shape mismatch")
    return float(np.mean(np.abs(t - m) <= z * s))


def ensemble_mean_std(samples) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample standard deviation (ddof 1) over the first axis."""
    a = _finite_array("samples", samples)
    if a.ndim < 1 or a.shape[0] < 2:
        raise ValueError("need at least two samples")
    return a.mean(axis=0), a.std(axis=0, ddof=1)


# ==========================================================================
# Synthetic world, observations, and the torch recurrent model (Python only)
# ==========================================================================


@dataclass
class SyntheticWorld:
    """A small world with gravity-style ground-truth migration flows.

    Units are thousands of people. ``flows[t, i, j, k]`` is the number born in ``i``
    moving from ``j`` to ``k`` during year ``t``; ``stocks[t]`` is the stock table at
    the start of year ``t`` (so ``stocks`` has ``years + 1`` entries).
    """

    n: int
    years: int
    pop_series: np.ndarray  # (years + 1, n) total residents, from the true stocks
    gdp: np.ndarray  # (years + 1, n) GDP per capita, thousand USD
    conflict: np.ndarray  # (years, n) conflict intensity covariate
    log_dist: np.ndarray  # (n, n)
    lang: np.ndarray  # (n, n)
    birth_rate: np.ndarray  # (n,)
    mortality: np.ndarray  # (n,)
    beta: np.ndarray  # (5,) gravity coefficients
    flows: np.ndarray  # (years, n, n, n)
    stocks: np.ndarray  # (years + 1, n, n)
    meta: dict = field(default_factory=dict)


BETA_TRUE = np.array([-8.7, 0.8, -0.7, 0.8, 0.5])
RETURN_BONUS = 3.0  # people return to their birth country this much more often
SHOCK_GAIN = 4.0  # conflict raises outflow by this factor times the memory trace
SHOCK_DECAY = 0.8  # the trace decays 20 percent per year


def make_world(seed: int = 0, n: int = 8, years: int = 20) -> SyntheticWorld:
    """Generate covariates and ground-truth flows. Deterministic for a given seed."""
    if n < 3 or years < 5:
        raise ValueError("need n >= 3 and years >= 5")
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1, size=(n, 2))
    dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)) + 0.15
    log_dist = np.log(dist / dist.mean())
    cluster = rng.integers(0, 3, size=n)
    lang = (cluster[:, None] == cluster[None, :]).astype(float)
    np.fill_diagonal(lang, 1.0)
    pop0 = np.exp(rng.uniform(np.log(4_000), np.log(60_000), size=n))
    gdp0 = np.exp(rng.uniform(np.log(2.0), np.log(40.0), size=n))
    growth = rng.uniform(0.0, 0.04, size=n)
    gdp = gdp0[None, :] * np.exp(growth[None, :] * np.arange(years + 1)[:, None])
    mortality = rng.uniform(0.008, 0.011, size=n)
    birth_rate = mortality * rng.uniform(1.0, 1.4, size=n)
    conflict = np.zeros((years, n))
    conflict[6:9, 6 % n] = 1.0  # observed-net-migration country
    conflict[12:14, 2 % n] = 1.0  # a country with no annual migration data
    trace = np.zeros((years, n))
    acc = np.zeros(n)
    for t in range(years):
        acc = SHOCK_DECAY * acc + conflict[t]
        trace[t] = acc

    def rates(t: int, shock: bool) -> np.ndarray:
        """Per-capita departure rate m[j, k] at year t."""
        lg = np.log(gdp[t])
        lp = np.log(pop_ref / pop_ref.mean())
        m = np.exp(
            gravity_log_rate(
                BETA_TRUE, lg[:, None], lg[None, :], log_dist, lang, lp[None, :]
            )
        )
        if shock:
            m = m * (1.0 + SHOCK_GAIN * trace[t])[:, None]
        np.fill_diagonal(m, 0.0)
        return m

    # burn-in: reach an approximate demographic and migration steady state
    stock = np.diag(pop0 * 0.97) + 0.03 * pop0[None, :] / (n - 1) * (1 - np.eye(n))
    for _ in range(80):
        pop_ref = stock.sum(0)
        m = rates(0, shock=False)
        flows_b = stock[:, :, None] * m[None, :, :] * (1.0 + RETURN_BONUS * np.eye(n))[:, None, :]
        stock = stock_step(stock, birth_rate * stock.sum(0), mortality, flows_b)
    stock = stock * (pop0 / stock.sum(0))[None, :]  # restore the intended population sizes
    stocks = [stock]
    flows = []
    pops = [stock.sum(0)]
    for t in range(years):
        pop_ref = stocks[-1].sum(0)
        m = rates(t, shock=True)
        ft = stocks[-1][:, :, None] * m[None, :, :] * (1.0 + RETURN_BONUS * np.eye(n))[:, None, :]
        flows.append(ft)
        nxt = stock_step(stocks[-1], birth_rate * stocks[-1].sum(0), mortality, ft)
        stocks.append(nxt)
        pops.append(nxt.sum(0))
    return SyntheticWorld(
        n=n,
        years=years,
        pop_series=np.array(pops),
        gdp=gdp,
        conflict=conflict,
        log_dist=log_dist,
        lang=lang,
        birth_rate=birth_rate,
        mortality=mortality,
        beta=BETA_TRUE.copy(),
        flows=np.array(flows),
        stocks=np.array(stocks),
        meta={"shock_trace": trace, "seed": seed},
    )


def simulate_truth(world: SyntheticWorld) -> dict:
    """Aggregates of the ground-truth tensor: F[t, j, k] and net migration[t, j]."""
    f = world.flows.sum(axis=1)
    net = np.array([net_migration(f[t]) for t in range(world.years)])
    return {"F": f, "net": net}


@dataclass
class Observations:
    """What the analyst sees: five-yearly migrant stocks, a few countries' annual net
    migration, a few corridors' annual flows, all noisy and mutually inconsistent."""

    stock_years: np.ndarray  # (q + 1,) years with a stock table
    stocks: np.ndarray  # (q + 1, n, n) raw noisy tables (diagonal from population)
    stocks_bal: np.ndarray  # balanced tables
    stock_rel_unc: np.ndarray  # (q, n, n) relative uncertainty of the interval difference
    net_countries: np.ndarray  # (m,) indices
    net: np.ndarray  # (years, m) observed annual net migration
    flow_corridors: np.ndarray  # (c, 2) (origin, destination)
    flows: np.ndarray  # (years, c) observed annual flows
    flow_rel_unc: np.ndarray  # (c,) relative standard error
    meta: dict = field(default_factory=dict)


def observe(
    world: SyntheticWorld,
    seed: int = 1,
    stock_noise: float = 0.05,
    net_countries=(0, 3, 6),
    net_bias: float = 0.85,
    corridors=((0, 3), (3, 0), (3, 6), (6, 0), (1, 0), (4, 3), (2, 5), (5, 2), (7, 1), (6, 4)),
    corridor_noise=(0.10, 0.15, 0.20, 0.30, 0.40, 0.20, 0.15, 0.25, 0.30, 0.20),
) -> Observations:
    """Draw inconsistent partial observations from the ground truth.

    * Stocks, every fifth year, off-diagonal only: log-normal cell noise plus a
      per-table census coverage factor for each country of residence (so the tables
      disagree with each other and with the accounting identity). The diagonal is
      filled in from total population, which is treated as reliable.
    * Net migration for three countries, annual; the second country's registry counts
      only ``net_bias`` of true movements (a definitional inconsistency).
    * Flows for ten corridors, annual, with corridor-specific log-normal noise whose
      scale is reported as the relative standard error.
    """
    rng = np.random.default_rng(seed)
    n, years = world.n, world.years
    stock_years = np.arange(0, years + 1, 5)
    off = ~np.eye(n, dtype=bool)
    tables = []
    for ty in stock_years:
        true = world.stocks[ty]
        cover = np.exp(rng.normal(0.0, 0.04, size=n))[None, :]
        noisy = true * np.exp(rng.normal(0.0, stock_noise, size=(n, n))) * cover
        noisy = np.where(off, noisy, 0.0)
        pop = world.pop_series[ty]
        np.fill_diagonal(noisy, pop - noisy.sum(axis=0) + np.diag(noisy))
        tables.append(noisy)
    tables = np.array(tables)
    bal = []
    rel = []
    for q in range(len(stock_years) - 1):
        t1, t2 = stock_years[q], stock_years[q + 1]
        span = np.arange(t1, t2)
        births = (world.birth_rate[None, :] * world.pop_series[span]).sum(axis=0)
        avg = 0.5 * (tables[q] + tables[q + 1])
        deaths = world.mortality[None, :] * avg * (t2 - t1)
        b1, b2, r1, r2 = demographic_balance(tables[q], tables[q + 1], births, deaths)
        if q == 0:
            bal.append(b1)
        bal.append(b2)
        rel.append(np.maximum(r1, r2))
    bal = np.array(bal)
    net_countries = np.array(net_countries, dtype=int)
    truth = simulate_truth(world)
    scale = np.abs(truth["net"][:, net_countries]).mean(axis=0) + 1.0
    bias = np.ones(len(net_countries))
    bias[1] = net_bias
    net = truth["net"][:, net_countries] * bias[None, :] + rng.normal(
        0.0, 0.10, size=(years, len(net_countries))
    ) * scale[None, :]
    cor = np.array(corridors, dtype=int)
    sig = np.array(corridor_noise, dtype=float)
    flows = np.stack(
        [truth["F"][:, o, d] for o, d in cor], axis=1
    ) * np.exp(rng.normal(0.0, 1.0, size=(years, len(cor))) * sig[None, :])
    return Observations(
        stock_years=stock_years,
        stocks=tables,
        stocks_bal=bal,
        stock_rel_unc=np.array(rel),
        net_countries=net_countries,
        net=net,
        flow_corridors=cor,
        flows=flows,
        flow_rel_unc=sig,
        meta={"net_bias": net_bias},
    )


@dataclass
class TrainConfig:
    """Hyper-parameters of the toy recurrent model (the paper uses 100, 7 x 60, 15)."""

    latent: int = 8
    hidden: int = 32
    layers: int = 3
    celu_alpha: float = 8.0
    epochs: int = 300
    lr: float = 5e-3
    weight_decay: float = 0.0
    members: int = 5
    pushforward_draws: int = 10
    lambda_target: float = 0.7  # the paper's target scaling parameter
    lambda_input: float = 0.0  # log-like scaling of stocks and GDP inputs
    use_memory: bool = True  # False zeroes the latent state every year (ablation)
    seed: int = 0
    threads: int = 4


def _import_torch():
    import torch

    return torch


def _edge_index(n: int):
    idx = [(i, j, k) for i in range(n) for j in range(n) for k in range(n) if j != k]
    a = np.array(idx)
    return a[:, 0], a[:, 1], a[:, 2]


def _build_inputs(world: SyntheticWorld, lam_in: float):
    """Static and per-year edge features (before the stock features)."""
    n = world.n
    I, J, K = _edge_index(n)
    years = world.years
    rows = []
    for t in range(years):
        pop, gdp = world.pop_series[t], world.gdp[t]
        cols = [
            yj_sym(pop[J] / 1000.0, lam_in),
            yj_sym(pop[K] / 1000.0, lam_in),
            yj_sym(gdp[J], lam_in),
            yj_sym(gdp[K], lam_in),
            world.log_dist[J, K],
            world.lang[J, K],
            world.conflict[t, J],
            world.conflict[t, K],
            (I == J).astype(float),
            (I == K).astype(float),
        ]
        rows.append(np.stack(cols, axis=1))
    return (I, J, K), np.array(rows)  # (years, E, 10)


def _make_net(torch, n_in: int, cfg: TrainConfig):
    layers = []
    width = n_in
    for _ in range(cfg.layers):
        layers += [torch.nn.Linear(width, cfg.hidden), torch.nn.Tanh()]
        width = cfg.hidden
    layers += [torch.nn.Linear(width, 1 + cfg.latent), torch.nn.CELU(alpha=cfg.celu_alpha)]
    return torch.nn.Sequential(*layers)


def _rollout(torch, net, static, stock0, world, cfg, norm, steps, edge, trace=None):
    """Run the recurrent model for ``steps`` years, feeding stocks and latent state back."""
    I, J, K = edge
    n = world.n
    E = len(I)
    mu = torch.tensor(world.mortality, dtype=torch.float32)
    z = torch.zeros(E, cfg.latent)
    stock = stock0
    flows, stocks = [], [stock]
    births_rate = torch.tensor(world.birth_rate, dtype=torch.float32)
    It, Jt, Kt = (torch.as_tensor(a) for a in (I, J, K))
    for t in range(steps):
        s_ij = torch.log1p(stock[It, Jt].clamp(min=0.0))
        s_ik = torch.log1p(stock[It, Kt].clamp(min=0.0))
        feats = torch.cat(
            [
                static[t],
                ((s_ij - norm["s_mu"]) / norm["s_sd"])[:, None],
                ((s_ik - norm["s_mu"]) / norm["s_sd"])[:, None],
            ],
            dim=1,
        )
        if trace is not None:
            trace.append((feats.detach().clone(), z.detach().clone()))
        out = net(torch.cat([feats, z], dim=1))
        logT = out[:, 0]
        z = out[:, 1:] if cfg.use_memory else torch.zeros_like(z)
        T = torch.zeros(n, n, n).index_put((It, Jt, Kt), torch.exp(logT))
        births = births_rate * stock.sum(0)
        stock = (
            stock
            + torch.diag(births)
            - mu[None, :] * stock
            + T.sum(1)
            - T.sum(2)
        )
        flows.append(T)
        stocks.append(stock)
    return torch.stack(flows), torch.stack(stocks)


def _phi_t(torch, x, lam):
    mag = torch.abs(x)
    if abs(lam) < 1e-12:
        body = torch.log1p(mag)
    else:
        body = torch.expm1(lam * torch.log1p(mag)) / lam
    return torch.sign(x) * body


def train_ensemble(world: SyntheticWorld, obs: Observations, cfg: TrainConfig | None = None):
    """Train an ensemble of recurrent latent-state models; return flow samples.

    Returns a dict with ``samples`` (draws, years, n, n, n) of the flow tensor
    (ensemble members times initial-stock pushforward draws), the per-member final
    losses, the trained networks, and the normalisation constants (so elasticities
    can be computed afterwards).
    """
    torch = _import_torch()
    cfg = cfg or TrainConfig()
    torch.set_num_threads(cfg.threads)
    n, years = world.n, world.years
    edge, base = _build_inputs(world, cfg.lambda_input)
    I, J, K = edge
    base_flat = base.reshape(-1, base.shape[-1])
    mu_f, sd_f = base_flat.mean(0), base_flat.std(0) + 1e-9
    binary = [5, 6, 7, 8, 9]  # keep indicators and conflict on their raw scale
    mu_f[binary], sd_f[binary] = 0.0, 1.0
    static = torch.tensor((base - mu_f) / sd_f, dtype=torch.float32)
    s_obs = np.log1p(np.maximum(obs.stocks_bal[:, I, J], 0.0))
    norm = {"s_mu": float(s_obs.mean()), "s_sd": float(s_obs.std() + 1e-9)}
    n_in = static.shape[-1] + 2 + cfg.latent

    # --- targets -------------------------------------------------------------
    off = ~np.eye(n, dtype=bool)
    q = len(obs.stock_years) - 1
    d_obs = obs.stocks_bal[1:] - obs.stocks_bal[:-1]  # (q, n, n)
    lam_t = cfg.lambda_target
    phi_d = yj_sym(d_obs[:, off], lam_t)
    m_d, s_d = phi_d.mean(), phi_d.std() + 1e-9
    w_stock = uncertainty_weights(obs.stock_rel_unc[:, off].ravel()).reshape(q, -1)
    phi_n = yj_sym(obs.net / 1.0, lam_t)
    m_n, s_n = phi_n.mean(), phi_n.std() + 1e-9
    phi_f = yj_sym(obs.flows, lam_t)
    m_f, s_f = phi_f.mean(), phi_f.std() + 1e-9
    w_flow = np.broadcast_to(uncertainty_weights(obs.flow_rel_unc), obs.flows.shape).copy()
    T_ = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    tgt = {
        "d": T_((phi_d - m_d) / s_d),
        "wd": T_(w_stock),
        "n": T_((phi_n - m_n) / s_n),
        "f": T_((phi_f - m_f) / s_f),
        "wf": T_(w_flow),
    }
    offt = torch.as_tensor(off)
    net_idx = torch.as_tensor(obs.net_countries)
    cor_o = torch.as_tensor(obs.flow_corridors[:, 0])
    cor_d = torch.as_tensor(obs.flow_corridors[:, 1])
    stock0_np = np.maximum(obs.stocks_bal[0], 0.0)

    def loss_fn(flows, stocks):
        # stock differences over five-year intervals (off-diagonal cells)
        parts = []
        for qi in range(q):
            a, b = obs.stock_years[qi], obs.stock_years[qi + 1]
            diff = (stocks[b] - stocks[a])[offt]
            parts.append((_phi_t(torch, diff, lam_t) - m_d) / s_d)
        pred_d = torch.stack(parts)
        j_stock = (tgt["wd"] * (pred_d - tgt["d"]) ** 2).mean()
        F = flows.sum(1)  # (years, n, n)
        net_hat = F.sum(1) - F.sum(2)  # inflow minus outflow, (years, n)
        pred_n = (_phi_t(torch, net_hat[:, net_idx], lam_t) - m_n) / s_n
        j_net = ((pred_n - tgt["n"]) ** 2).mean()
        pred_f = (_phi_t(torch, F[:, cor_o, cor_d], lam_t) - m_f) / s_f
        j_flow = (tgt["wf"] * (pred_f - tgt["f"]) ** 2).mean()
        return j_stock + j_net + j_flow, (j_stock.item(), j_net.item(), j_flow.item())

    nets, losses, histories = [], [], []
    for m in range(cfg.members):
        torch.manual_seed(cfg.seed * 1000 + m)
        net = _make_net(torch, n_in, cfg)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        s0 = torch.tensor(stock0_np, dtype=torch.float32)
        hist = []
        for ep in range(cfg.epochs):
            opt.zero_grad()
            flows, stocks = _rollout(torch, net, static, s0, world, cfg, norm, years, edge)
            loss, parts = loss_fn(flows, stocks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if ep % 25 == 0 or ep == cfg.epochs - 1:
                hist.append((ep, float(loss.item()), *parts))
        nets.append(net)
        losses.append(hist[-1][1])
        histories.append(hist)

    # --- pushforward of the initial-stock uncertainty --------------------------
    rng = np.random.default_rng(cfg.seed + 77)
    rel0 = np.where(off, np.maximum(obs.stock_rel_unc[0], 0.02), 0.0)
    samples = []
    with torch.no_grad():
        for net in nets:
            for _ in range(cfg.pushforward_draws):
                s0 = stock0_np * np.exp(rng.normal(0.0, 1.0, size=(n, n)) * rel0)
                np.fill_diagonal(s0, world.pop_series[0] - (s0.sum(0) - np.diag(s0)))
                flows, _ = _rollout(
                    torch, net, static, torch.tensor(s0, dtype=torch.float32), world, cfg, norm, years, edge
                )
                samples.append(flows.numpy().astype(float))
    return {
        "samples": np.array(samples),
        "final_loss": losses,
        "history": histories,
        "nets": nets,
        "norm": norm,
        "static": static,
        "edge": edge,
        "cfg": cfg,
        "feature_stats": (mu_f, sd_f),
        "stock0": stock0_np,
    }


def recovery_report(world: SyntheticWorld, samples: np.ndarray, obs: Observations) -> dict:
    """Compare the ensemble with the ground truth on observed and unobserved corridors."""
    n = world.n
    truth_F = world.flows.sum(axis=1)  # (years, n, n)
    est = samples.sum(axis=2)  # (draws, years, n, n)
    mean, std = ensemble_mean_std(est)
    off = ~np.eye(n, dtype=bool)
    seen = np.zeros((n, n), dtype=bool)
    for o, d in obs.flow_corridors:
        seen[o, d] = True
    unseen = off & ~seen
    res = {}
    for name, mask in (("observed corridors", seen), ("unobserved corridors", unseen)):
        t, m, s = truth_F[:, mask], mean[:, mask], std[:, mask]
        res[name] = {
            "log_rmse": log_rmse(t, m),
            "pearson_log": pearson(np.log(t), np.log(np.maximum(m, 1e-9))),
            "coverage_95": coverage(t, m, s),
            "median_rel_err": float(np.median(np.abs(m - t) / t)),
        }
    return res


def feature_elasticity(world: SyntheticWorld, result: dict, years=(2, 6, 10, 14, 18)) -> dict:
    """Mean elasticity of ``log T`` with respect to GDP per capita of origin and destination.

    Differentiates each trained network with respect to its (transformed, standardised)
    GDP inputs and converts to a raw-covariate elasticity with ``elasticity_chain``.
    The gravity ground truth has elasticity ``-beta_g`` for the origin and ``+beta_g``
    for the destination, so the estimates can be checked.
    """
    torch = _import_torch()
    cfg: TrainConfig = result["cfg"]
    edge, static, norm = result["edge"], result["static"], result["norm"]
    mu_f, sd_f = result["feature_stats"]
    I, J, K = edge
    out = {"origin": [], "destination": []}
    stock0 = torch.tensor(np.maximum(result["stock0"], 0.0), dtype=torch.float32)
    for net in result["nets"]:
        trace: list = []
        with torch.no_grad():
            _rollout(torch, net, static, stock0, world, cfg, norm, world.years, edge, trace=trace)
        for t in years:
            feats, z = trace[t]
            feats = feats.clone().requires_grad_(True)
            u = net(torch.cat([feats, z], dim=1))[:, 0]
            u.sum().backward()
            g = feats.grad.numpy()
            for name, col, idx in (("origin", 2, J), ("destination", 3, K)):
                du_dphi = g[:, col] / sd_f[col]
                x = world.gdp[t][idx]
                out[name].append(float(np.mean(elasticity_chain(du_dphi, x, cfg.lambda_input))))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}
