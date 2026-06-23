"""RMSProp adaptive-learning-rate optimizer from scratch.

A small, well-validated reference implementation of RMSProp (root mean square
propagation), the per-coordinate adaptive optimizer introduced by Geoffrey Hinton.
The public API mirrors the Julia (``AIInAction.Ch195Rmsprop``) and Rust
(``aiinaction::ch195_rmsprop``) implementations one-to-one; the cross-language
parity tests assert that all three agree to within floating-point tolerance on
shared fixtures.

The optimizer maintains, per parameter coordinate ``i``, an exponential moving
average of the squared gradient,

    v_t,i = beta * v_{t-1,i} + (1 - beta) * g_t,i ** 2,    v_0 = 0,

and updates each coordinate with its own effective step size,

    theta_{t+1,i} = theta_t,i - eta * g_t,i / (sqrt(v_t,i) + eps).

This module implements the plain (uncentered, no-momentum) RMSProp update with the
epsilon placed *outside* the square root, matching the description in the chapter.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["RMSPropState", "init_state", "rmsprop_step", "minimize"]

Vector = Sequence[float]
GradFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class RMSPropState:
    """Immutable state of an RMSProp optimizer.

    Attributes
    ----------
    params:
        Current parameter vector, shape ``(d,)``.
    v:
        Exponential moving average of squared gradients, shape ``(d,)``. Starts at
        the zero vector.
    lr:
        Global learning rate ``eta`` (positive).
    beta:
        Decay rate of the squared-gradient EMA, in ``[0, 1)``. Common default 0.9.
    eps:
        Numerical-stability constant added to ``sqrt(v)`` in the denominator
        (positive). Common default 1e-8.
    step_count:
        Number of update steps applied so far.
    """

    params: NDArray[np.float64]
    v: NDArray[np.float64]
    lr: float
    beta: float
    eps: float
    step_count: int

    @property
    def n_params(self) -> int:
        """Dimensionality of the parameter vector."""
        return int(self.params.shape[0])


def _as_vector(x: Vector, name: str) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got array with {arr.ndim} dimension(s)")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (nan or inf)")
    return arr


def _validate_hparams(lr: float, beta: float, eps: float) -> tuple[float, float, float]:
    lr = float(lr)
    beta = float(beta)
    eps = float(eps)
    if not lr > 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    if not eps > 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    return lr, beta, eps


def init_state(
    params: Vector,
    *,
    lr: float = 1e-2,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> RMSPropState:
    """Initialize an RMSProp state from a starting parameter vector.

    Parameters
    ----------
    params:
        Initial parameter vector of shape ``(d,)`` with ``d >= 1`` finite entries.
    lr:
        Global learning rate ``eta``; must be positive.
    beta:
        Squared-gradient EMA decay in ``[0, 1)``.
    eps:
        Positive numerical-stability constant.

    Returns
    -------
    RMSPropState
        State with ``v`` initialized to zeros and ``step_count = 0``.

    Examples
    --------
    >>> s = init_state([1.0, -2.0], lr=0.1)
    >>> s.n_params
    2
    >>> bool((s.v == 0.0).all())
    True
    """
    p = _as_vector(params, "params")
    lr, beta, eps = _validate_hparams(lr, beta, eps)
    return RMSPropState(
        params=p,
        v=np.zeros_like(p),
        lr=lr,
        beta=beta,
        eps=eps,
        step_count=0,
    )


def rmsprop_step(state: RMSPropState, grad: Vector) -> RMSPropState:
    """Apply one RMSProp update and return the new state.

    Performs, coordinate-wise,

        v <- beta * v + (1 - beta) * grad ** 2
        params <- params - lr * grad / (sqrt(v) + eps)

    Parameters
    ----------
    state:
        Current optimizer state.
    grad:
        Gradient ``g_t`` at the current parameters, shape ``(d,)``, matching
        ``state.n_params``. Must be finite.

    Returns
    -------
    RMSPropState
        New state with updated ``params``, ``v`` and an incremented step count.
    """
    g = _as_vector(grad, "grad")
    if g.shape[0] != state.n_params:
        raise ValueError(
            f"grad has {g.shape[0]} entries but state has {state.n_params} parameters"
        )
    v_new = state.beta * state.v + (1.0 - state.beta) * g * g
    params_new = state.params - state.lr * g / (np.sqrt(v_new) + state.eps)
    return RMSPropState(
        params=params_new,
        v=v_new,
        lr=state.lr,
        beta=state.beta,
        eps=state.eps,
        step_count=state.step_count + 1,
    )


def minimize(
    grad_fn: GradFn,
    params0: Vector,
    n_steps: int,
    *,
    lr: float = 1e-2,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> RMSPropState:
    """Run RMSProp for ``n_steps`` iterations on the objective with gradient ``grad_fn``.

    Parameters
    ----------
    grad_fn:
        Callable mapping the current parameter vector to the gradient at that point.
        It must return a finite vector of the same length as ``params0``.
    params0:
        Initial parameter vector, shape ``(d,)``.
    n_steps:
        Number of update steps to perform; must be a non-negative integer.
    lr, beta, eps:
        Hyperparameters forwarded to :func:`init_state`.

    Returns
    -------
    RMSPropState
        The final state after ``n_steps`` updates.

    Examples
    --------
    Minimize the separable quadratic ``f(x) = 0.5 * sum(c_i * x_i**2)`` whose
    gradient is ``c * x``; the minimizer is the origin.

    >>> import numpy as np
    >>> c = np.array([1.0, 9.0])
    >>> s = minimize(lambda x: c * x, [1.0, 1.0], 200, lr=0.05)
    >>> bool(np.all(np.abs(s.params) < 1e-2))
    True
    """
    if not isinstance(n_steps, (int, np.integer)):
        raise ValueError(f"n_steps must be an integer, got {type(n_steps).__name__}")
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")

    state = init_state(params0, lr=lr, beta=beta, eps=eps)
    for _ in range(n_steps):
        g = np.asarray(grad_fn(state.params), dtype=np.float64)
        state = rmsprop_step(state, g)
    return state
