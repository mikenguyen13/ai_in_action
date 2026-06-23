"""Learning rate schedules from scratch.

Small, dependency-free reference implementations of the dominant learning-rate
schedule families used in deep-learning optimization: cosine annealing, linear
warmup, warmup followed by cosine decay, and the one-cycle policy (with its
antiphase momentum schedule). The public API mirrors the Julia
(`AIInAction.Ch198LrSchedules`) and Rust (`aiinaction::ch198_lr_schedules`)
implementations one-to-one; the cross-language parity tests assert that all three
agree to within floating-point tolerance on shared fixtures.

All schedules are pure functions of the integer step ``t`` and a small set of
hyperparameters. Steps are zero-indexed: ``t = 0`` is the first update and
``t = T`` is the final point of a horizon of ``T`` steps (so a schedule is
evaluated at ``T + 1`` points ``0, 1, ..., T`` if you sweep the whole run).

References
----------
Loshchilov and Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts",
ICLR 2017. Smith and Topin, "Super-Convergence", 2018.
"""
from __future__ import annotations

import math

__all__ = [
    "cosine_annealing",
    "linear_warmup",
    "warmup_cosine",
    "one_cycle",
    "one_cycle_momentum",
    "schedule_curve",
]


def _check_horizon(total_steps: int, name: str = "total_steps") -> int:
    if not isinstance(total_steps, (int,)) or isinstance(total_steps, bool):
        raise ValueError(f"{name} must be an integer, got {type(total_steps).__name__}")
    if total_steps < 1:
        raise ValueError(f"{name} must be >= 1, got {total_steps}")
    return total_steps


def _check_step(t: int) -> int:
    if not isinstance(t, (int,)) or isinstance(t, bool):
        raise ValueError(f"t must be an integer, got {type(t).__name__}")
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    return t


def _check_bounds(eta_max: float, eta_min: float) -> tuple[float, float]:
    eta_max = float(eta_max)
    eta_min = float(eta_min)
    if eta_max < eta_min:
        raise ValueError(f"eta_max ({eta_max}) must be >= eta_min ({eta_min})")
    if eta_min < 0.0:
        raise ValueError(f"eta_min must be >= 0, got {eta_min}")
    return eta_max, eta_min


def cosine_annealing(
    t: int,
    total_steps: int,
    eta_max: float = 0.1,
    eta_min: float = 0.0,
) -> float:
    r"""Cosine annealing learning rate at step ``t`` over a horizon of ``total_steps``.

    Follows half a cosine wave from ``eta_max`` at ``t = 0`` down to ``eta_min`` at
    ``t = total_steps``:

    .. math::
        \eta_t = \eta_{\min}
            + \tfrac12 (\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right).

    Steps beyond ``total_steps`` are clamped to ``eta_min`` (the schedule has
    bottomed out). ``t`` must be non-negative.

    Examples
    --------
    >>> round(cosine_annealing(0, 10, 0.1, 0.0), 6)
    0.1
    >>> round(cosine_annealing(5, 10, 0.1, 0.0), 6)
    0.05
    >>> round(cosine_annealing(10, 10, 0.1, 0.0), 6)
    0.0
    """
    t = _check_step(t)
    total_steps = _check_horizon(total_steps)
    eta_max, eta_min = _check_bounds(eta_max, eta_min)
    if t >= total_steps:
        return eta_min
    cos = math.cos(math.pi * t / total_steps)
    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + cos)


def linear_warmup(
    t: int,
    warmup_steps: int,
    eta_target: float,
    eta_start: float = 0.0,
) -> float:
    r"""Linear warmup from ``eta_start`` to ``eta_target`` over ``warmup_steps``.

    .. math::
        \eta_t = \eta_{\text{start}}
            + (\eta_{\text{target}} - \eta_{\text{start}})\,\frac{t}{T_w},
        \qquad 0 \le t \le T_w.

    At ``t = 0`` returns ``eta_start``; at ``t = warmup_steps`` and beyond returns
    ``eta_target``.

    Examples
    --------
    >>> round(linear_warmup(0, 4, 0.1), 6)
    0.0
    >>> round(linear_warmup(2, 4, 0.1), 6)
    0.05
    >>> round(linear_warmup(4, 4, 0.1), 6)
    0.1
    """
    t = _check_step(t)
    warmup_steps = _check_horizon(warmup_steps, "warmup_steps")
    eta_target = float(eta_target)
    eta_start = float(eta_start)
    if eta_target < 0.0 or eta_start < 0.0:
        raise ValueError("learning rates must be non-negative")
    if t >= warmup_steps:
        return eta_target
    frac = t / warmup_steps
    return eta_start + (eta_target - eta_start) * frac


def warmup_cosine(
    t: int,
    warmup_steps: int,
    total_steps: int,
    eta_max: float,
    eta_min: float = 0.0,
    eta_start: float = 0.0,
) -> float:
    r"""Linear warmup followed by cosine annealing: the standard modern recipe.

    For ``t < warmup_steps`` the rate ramps linearly from ``eta_start`` to
    ``eta_max``. For ``warmup_steps <= t <= total_steps`` the rate follows a cosine
    descent from ``eta_max`` to ``eta_min`` over the remaining
    ``total_steps - warmup_steps`` steps:

    .. math::
        \eta_t = \eta_{\min}
            + \tfrac12 (\eta_{\max} - \eta_{\min})
              \left(1 + \cos\frac{\pi (t - T_w)}{T - T_w}\right).

    ``warmup_steps`` must satisfy ``0 <= warmup_steps < total_steps``. Steps beyond
    ``total_steps`` clamp to ``eta_min``.

    Examples
    --------
    >>> round(warmup_cosine(0, 2, 10, 0.1), 6)
    0.0
    >>> round(warmup_cosine(2, 2, 10, 0.1), 6)
    0.1
    >>> round(warmup_cosine(10, 2, 10, 0.1), 6)
    0.0
    """
    t = _check_step(t)
    total_steps = _check_horizon(total_steps)
    eta_max, eta_min = _check_bounds(eta_max, eta_min)
    eta_start = float(eta_start)
    if eta_start < 0.0:
        raise ValueError(f"eta_start must be >= 0, got {eta_start}")
    if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool):
        raise ValueError(f"warmup_steps must be an integer, got {type(warmup_steps).__name__}")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps must satisfy 0 <= warmup_steps < total_steps={total_steps}, "
            f"got {warmup_steps}"
        )
    if t < warmup_steps:
        return eta_start + (eta_max - eta_start) * (t / warmup_steps)
    if t >= total_steps:
        return eta_min
    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    cos = math.cos(math.pi * progress)
    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + cos)


def one_cycle(
    t: int,
    total_steps: int,
    eta_max: float,
    eta_min: float | None = None,
    pct_start: float = 0.3,
) -> float:
    r"""One-cycle learning rate (Smith and Topin) with cosine ramps.

    A single cycle spanning the whole run. The rate rises from ``eta_min`` to
    ``eta_max`` over the first ``pct_start`` fraction of the horizon, then descends
    back to ``eta_min`` over the remainder, both legs shaped as half-cosines:

    .. math::
        \eta_t = \begin{cases}
          \eta_{\min} + \tfrac12 (\eta_{\max} - \eta_{\min})
            \left(1 - \cos\frac{\pi t}{t_1}\right), & 0 \le t \le t_1, \\[4pt]
          \eta_{\min} + \tfrac12 (\eta_{\max} - \eta_{\min})
            \left(1 + \cos\frac{\pi (t - t_1)}{T - t_1}\right), & t_1 < t \le T,
        \end{cases}

    where ``t1 = pct_start * total_steps``. ``eta_min`` defaults to
    ``eta_max / 25`` (the common fastai initial division factor). ``pct_start`` must
    lie in ``(0, 1)``.

    Examples
    --------
    >>> round(one_cycle(0, 10, 1.0, 0.0, pct_start=0.3), 6)
    0.0
    >>> round(one_cycle(3, 10, 1.0, 0.0, pct_start=0.3), 6)
    1.0
    >>> round(one_cycle(10, 10, 1.0, 0.0, pct_start=0.3), 6)
    0.0
    """
    t = _check_step(t)
    total_steps = _check_horizon(total_steps)
    eta_max = float(eta_max)
    if eta_min is None:
        eta_min = eta_max / 25.0
    eta_max, eta_min = _check_bounds(eta_max, eta_min)
    pct_start = float(pct_start)
    if not (0.0 < pct_start < 1.0):
        raise ValueError(f"pct_start must be in (0, 1), got {pct_start}")

    t1 = pct_start * total_steps
    if t >= total_steps:
        return eta_min
    if t <= t1:
        # Cosine ramp up: 0.5 * (1 - cos) goes from 0 to 1 as t goes 0 -> t1.
        cos = math.cos(math.pi * t / t1)
        return eta_min + 0.5 * (eta_max - eta_min) * (1.0 - cos)
    progress = (t - t1) / (total_steps - t1)
    cos = math.cos(math.pi * progress)
    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + cos)


def one_cycle_momentum(
    t: int,
    total_steps: int,
    mom_max: float = 0.95,
    mom_min: float = 0.85,
    pct_start: float = 0.3,
) -> float:
    r"""Antiphase momentum schedule for the one-cycle policy.

    Momentum moves opposite to the learning rate: it falls from ``mom_max`` to
    ``mom_min`` while the learning rate climbs, then rises back to ``mom_max`` while
    the learning rate descends. Both legs are cosine-shaped.

    ``mom_max`` and ``mom_min`` must lie in ``[0, 1)`` with ``mom_max >= mom_min``.

    Examples
    --------
    >>> round(one_cycle_momentum(0, 10, 0.95, 0.85, pct_start=0.3), 6)
    0.95
    >>> round(one_cycle_momentum(3, 10, 0.95, 0.85, pct_start=0.3), 6)
    0.85
    >>> round(one_cycle_momentum(10, 10, 0.95, 0.85, pct_start=0.3), 6)
    0.95
    """
    t = _check_step(t)
    total_steps = _check_horizon(total_steps)
    mom_max = float(mom_max)
    mom_min = float(mom_min)
    if not (0.0 <= mom_min <= mom_max < 1.0):
        raise ValueError(
            f"momenta must satisfy 0 <= mom_min <= mom_max < 1, got "
            f"mom_min={mom_min}, mom_max={mom_max}"
        )
    pct_start = float(pct_start)
    if not (0.0 < pct_start < 1.0):
        raise ValueError(f"pct_start must be in (0, 1), got {pct_start}")

    t1 = pct_start * total_steps
    if t >= total_steps:
        return mom_max
    if t <= t1:
        # Mirror of the LR ramp: high -> low.
        cos = math.cos(math.pi * t / t1)
        return mom_max - 0.5 * (mom_max - mom_min) * (1.0 - cos)
    progress = (t - t1) / (total_steps - t1)
    cos = math.cos(math.pi * progress)
    return mom_max - 0.5 * (mom_max - mom_min) * (1.0 + cos)


def schedule_curve(
    name: str,
    total_steps: int,
    **kwargs: float,
) -> list[float]:
    """Materialize a schedule over all steps ``0, 1, ..., total_steps - 1``.

    ``name`` is one of ``"cosine"``, ``"warmup_cosine"``, or ``"one_cycle"``.
    Remaining keyword arguments are forwarded to the corresponding function.
    Returns a list of ``total_steps`` learning rates.
    """
    total_steps = _check_horizon(total_steps)
    if name == "cosine":
        return [cosine_annealing(t, total_steps, **kwargs) for t in range(total_steps)]
    if name == "warmup_cosine":
        return [warmup_cosine(t, total_steps=total_steps, **kwargs) for t in range(total_steps)]
    if name == "one_cycle":
        return [one_cycle(t, total_steps, **kwargs) for t in range(total_steps)]
    raise ValueError(
        f"unknown schedule name {name!r}; expected 'cosine', 'warmup_cosine', or 'one_cycle'"
    )
