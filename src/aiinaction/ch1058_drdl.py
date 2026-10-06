"""Numeric core of DRDL for dynamic financial risk prediction (Chapter 1058).

Reference implementations of the deterministic building blocks of the discrete and
regularized deep learning (DRDL) method of Wang, Che, Jiang, and Zhao (2026),
"Leveraging Multiview Data Through Discrete and Regularized Deep Learning for
Dynamic Financial Risk Prediction", Information Systems Research 37(3), together
with the discrete-time survival metrics used to evaluate it.

Notation. There are ``m`` prediction horizons ``t_1 < ... < t_m``. Each company
``i`` has an observed horizon index ``A_i`` in ``{1, ..., m}`` and an event flag
``delta_i``: ``delta_i = 1`` means the first risk event happened in horizon
``A_i``; ``delta_i = 0`` means no event was seen through ``A_i`` (censored). The
model outputs a cumulative risk matrix ``F`` with ``F[i, u] = P(T_i <= t_{u+1})``
(0-based column ``u``).

Contents:

* ``risk_accumulation``: DRDL's risk accumulation function (paper Eq. 8),
  ``p_u = 1 - exp(-sum_{j<=u} exp(o_j))``. It is the cumulative incidence of a
  discrete-time hazard model with a complementary log-log link, so it is
  monotone in ``u`` by construction.
* ``nearest_codes`` and ``first_occurrence_mask``: the discrete multiview
  recoder's nearest-codebook assignment (Eq. 2) and the mask-based view fusion
  decoder's redundancy mask (Eq. 3).
* ``orthogonal_loss``: the codebook orthogonality regularizer (Eq. 7).
* ``focal_sigma`` and ``focal_survival_loss``: the focal survival loss
  (Eqs. 9-10), which penalizes only wrongly ranked comparable pairs.
* ``multiperiod_bce``: the per-horizon binary cross-entropy on cumulative labels.
* ``project_conflicting``: adaptive gradient balancing (Eq. 11).
* ``td_concordance``, ``td_auc``, ``brier_scores``: Antolini time-dependent
  C-index, cumulative/dynamic AUC at a horizon, and per-horizon Brier scores.

The functions mirror the Julia module ``AIInAction.Ch1058Drdl`` and the Rust
module ``aiinaction::ch1058_drdl``; the three test suites share fixtures.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "risk_accumulation",
    "nearest_codes",
    "first_occurrence_mask",
    "orthogonal_loss",
    "focal_sigma",
    "comparable_pairs",
    "focal_survival_loss",
    "cumulative_labels",
    "multiperiod_bce",
    "project_conflicting",
    "td_concordance",
    "td_auc",
    "brier_scores",
]


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _as_matrix(name: str, x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _survival_targets(times, events, n: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times)
    e = np.asarray(events)
    if t.shape != (n,) or e.shape != (n,):
        raise ValueError(f"times and events must have shape ({n},)")
    if not np.all(np.equal(np.mod(t, 1), 0)):
        raise ValueError("times must be integer horizon indices")
    t = t.astype(np.int64)
    if t.min(initial=1) < 1 or t.max(initial=1) > m:
        raise ValueError(f"times must lie in 1..{m}")
    if not np.all((e == 0) | (e == 1)):
        raise ValueError("events must be 0 or 1")
    return t, e.astype(np.int64)


# --------------------------------------------------------------------------
# Risk accumulation function (Eq. 8)
# --------------------------------------------------------------------------


def risk_accumulation(outputs) -> np.ndarray:
    """Map unconstrained outputs to monotone cumulative risk probabilities.

    ``p_u = 1 - exp(-sum_{j=1}^{u} exp(o_j))``. Works on a 1-D vector of ``m``
    outputs or row-wise on an ``n x m`` matrix.
    """
    o = np.asarray(outputs, dtype=np.float64)
    if o.ndim not in (1, 2) or o.shape[-1] == 0:
        raise ValueError("outputs must be a non-empty 1-D or 2-D array")
    if not np.all(np.isfinite(o)):
        raise ValueError("outputs contain non-finite values")
    cum = np.cumsum(np.exp(o), axis=-1)
    return -np.expm1(-cum)


# --------------------------------------------------------------------------
# Discrete multiview recoder and mask-based fusion (Eqs. 2-4)
# --------------------------------------------------------------------------


def nearest_codes(embeddings, codebook) -> np.ndarray:
    """Index of the nearest codebook row (L2) for every embedding row.

    Ties go to the lowest index, matching ``argmin``.
    """
    e = _as_matrix("embeddings", embeddings)
    c = _as_matrix("codebook", codebook)
    if e.shape[1] != c.shape[1]:
        raise ValueError("embeddings and codebook must have the same dimension")
    if c.shape[0] == 0:
        raise ValueError("codebook must be non-empty")
    d2 = ((e[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).astype(np.int64)


def first_occurrence_mask(codes: Sequence[int]) -> np.ndarray:
    """``m_i = 0`` if code ``z_i`` already appeared among ``z_1..z_{i-1}``, else 1."""
    seen: set[int] = set()
    out = np.zeros(len(codes), dtype=np.int64)
    for i, z in enumerate(codes):
        z = int(z)
        if z < 0:
            raise ValueError("codes must be non-negative")
        if z not in seen:
            out[i] = 1
            seen.add(z)
    return out


def orthogonal_loss(codebook) -> float:
    """Frobenius norm of the strict upper triangle of the normalized codebook Gram matrix.

    For unit-normalized codes ``c_k``, this is ``sqrt(sum_{k<l} (c_k . c_l)^2)``,
    zero exactly when the codes are mutually orthogonal.
    """
    c = _as_matrix("codebook", codebook)
    norms = np.linalg.norm(c, axis=1)
    if np.any(norms == 0):
        raise ValueError("codebook rows must be non-zero")
    u = c / norms[:, None]
    g = u @ u.T
    return float(np.sqrt((np.triu(g, k=1) ** 2).sum()))


# --------------------------------------------------------------------------
# Focal survival loss (Eqs. 9-10)
# --------------------------------------------------------------------------


def focal_sigma(x, gamma: float):
    """``sigma(x) = 2 / (1 + exp(-x^gamma)) - 1`` for ``x >= 0`` (0 at 0, below 1)."""
    if not (math.isfinite(gamma) and gamma > 0):
        raise ValueError("gamma must be a positive finite number")
    a = np.asarray(x, dtype=np.float64)
    if np.any(a < 0) or not np.all(np.isfinite(a)):
        raise ValueError("x must be finite and non-negative")
    v = 2.0 / (1.0 + np.exp(-(a**gamma))) - 1.0
    return float(v) if v.ndim == 0 else v


def comparable_pairs(times, events) -> list[tuple[int, int]]:
    """Ordered pairs ``(i, j)`` where company ``i`` is known to fail before ``j``.

    ``delta_i = 1`` and either ``A_i < A_j``, or ``A_i == A_j`` with ``delta_j = 0``
    (``j`` survived the horizon in which ``i`` failed).
    """
    t = np.asarray(times, dtype=np.int64)
    e = np.asarray(events, dtype=np.int64)
    pairs = []
    for i in range(len(t)):
        if e[i] != 1:
            continue
        for j in range(len(t)):
            if i == j:
                continue
            if t[i] < t[j] or (t[i] == t[j] and e[j] == 0):
                pairs.append((i, j))
    return pairs


def focal_survival_loss(cum_probs, times, events, gamma: float = 1.0) -> float:
    """Mean focal penalty over comparable pairs (Eq. 9).

    For a comparable pair ``(i, j)`` both companies are compared at the horizon
    ``A_i`` where ``i`` failed. The pair is wrongly ranked when
    ``F[j, A_i] > F[i, A_i]``; its penalty is ``sigma(max(0, F[j, A_i] - F[i, A_i]))``,
    so correctly ranked pairs contribute exactly zero. Returns 0 if no pair is
    comparable.
    """
    f = _as_matrix("cum_probs", cum_probs)
    t, e = _survival_targets(times, events, f.shape[0], f.shape[1])
    pairs = comparable_pairs(t, e)
    if not pairs:
        return 0.0
    total = 0.0
    for i, j in pairs:
        u = t[i] - 1
        total += focal_sigma(max(0.0, f[j, u] - f[i, u]), gamma)
    return total / len(pairs)


# --------------------------------------------------------------------------
# Multiperiod BCE
# --------------------------------------------------------------------------


def cumulative_labels(times, events, m: int) -> np.ndarray:
    """``y[i, u] = 1`` if company ``i`` had its first event at or before horizon ``u+1``."""
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer")
    n = len(np.asarray(times))
    t, e = _survival_targets(times, events, n, int(m))
    u = np.arange(1, m + 1)[None, :]
    return ((e[:, None] == 1) & (t[:, None] <= u)).astype(np.float64)


def multiperiod_bce(cum_probs, times, events, eps: float = 1e-12) -> float:
    """Mean over companies of the summed per-horizon binary cross-entropy.

    Assumes censoring only at the end of the window (every company is observed
    through ``t_m`` unless it fails earlier), which is the setting of the paper's
    yearly, quarterly, and monthly windows.
    """
    f = _as_matrix("cum_probs", cum_probs)
    if np.any(f < 0) or np.any(f > 1):
        raise ValueError("cum_probs must lie in [0, 1]")
    y = cumulative_labels(times, events, f.shape[1])
    p = np.clip(f, eps, 1.0 - eps)
    bce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(bce.sum(axis=1).mean())


# --------------------------------------------------------------------------
# Adaptive gradient balancing (Eq. 11)
# --------------------------------------------------------------------------


def project_conflicting(g_main, g_aux) -> np.ndarray:
    """Remove the component of ``g_aux`` that conflicts with ``g_main``.

    If ``cos(g_main, g_aux) < 0`` return ``g_aux - (g_aux . g_main / ||g_main||^2) g_main``
    (orthogonal to ``g_main``); otherwise return ``g_aux`` unchanged.
    """
    a = np.asarray(g_main, dtype=np.float64).ravel()
    b = np.asarray(g_aux, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("gradients must have the same number of elements")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("gradients must be finite")
    nn = float(a @ a)
    dot = float(a @ b)
    if nn == 0.0 or dot >= 0.0:
        return b.copy()
    return b - (dot / nn) * a


# --------------------------------------------------------------------------
# Evaluation metrics
# --------------------------------------------------------------------------


def td_concordance(cum_probs, times, events) -> float:
    """Antolini's time-dependent concordance index.

    Over comparable pairs ``(i, j)``, the fraction with ``F[i, A_i] > F[j, A_i]``,
    counting ties as one half.
    """
    f = _as_matrix("cum_probs", cum_probs)
    t, e = _survival_targets(times, events, f.shape[0], f.shape[1])
    pairs = comparable_pairs(t, e)
    if not pairs:
        raise ValueError("no comparable pairs")
    score = 0.0
    for i, j in pairs:
        u = t[i] - 1
        if f[i, u] > f[j, u]:
            score += 1.0
        elif f[i, u] == f[j, u]:
            score += 0.5
    return score / len(pairs)


def td_auc(cum_probs, times, events, horizon: int) -> float:
    """Cumulative/dynamic AUC at horizon ``horizon`` (1-based).

    Cases had their event at or before ``horizon``; controls are known to be
    event-free through ``horizon``. Ties count one half.
    """
    f = _as_matrix("cum_probs", cum_probs)
    t, e = _survival_targets(times, events, f.shape[0], f.shape[1])
    if not (1 <= horizon <= f.shape[1]):
        raise ValueError("horizon out of range")
    col = f[:, horizon - 1]
    cases = col[(e == 1) & (t <= horizon)]
    controls = col[(t > horizon) | ((t == horizon) & (e == 0))]
    if len(cases) == 0 or len(controls) == 0:
        raise ValueError("need at least one case and one control")
    gt = (cases[:, None] > controls[None, :]).sum()
    eq = (cases[:, None] == controls[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(cases) * len(controls)))


def brier_scores(cum_probs, times, events) -> np.ndarray:
    """Per-horizon Brier score ``mean_i (y[i, u] - F[i, u])^2`` (end-of-window censoring)."""
    f = _as_matrix("cum_probs", cum_probs)
    y = cumulative_labels(times, events, f.shape[1])
    return ((y - f) ** 2).mean(axis=0)
