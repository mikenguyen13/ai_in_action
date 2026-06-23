"""AdamW: Adam with decoupled weight decay, from scratch.

A small, well-validated reference implementation of the AdamW optimizer
(Loshchilov and Hutter, 2019). The public API mirrors the Julia
(`AIInAction.Ch197Adamw`) and Rust (`aiinaction::ch197_adamw`) implementations
one-to-one; the cross-language parity tests assert that all three agree to within
floating-point tolerance on shared fixtures.

The update rule for parameter vector ``theta`` at step ``t`` with gradient
``g`` is:

1. First moment   ``m = beta1 * m + (1 - beta1) * g``
2. Second moment  ``v = beta2 * v + (1 - beta2) * g * g``
3. Bias correct   ``mhat = m / (1 - beta1**t)``, ``vhat = v / (1 - beta2**t)``
4. Update         ``theta = theta - lr * (mhat / (sqrt(vhat) + eps) + wd * theta)``

The key difference from L2-regularized Adam is step 4: the weight-decay term
``wd * theta`` is applied directly to the parameters, *outside* the adaptive
preconditioner ``sqrt(vhat) + eps``, rather than being folded into the gradient
``g``. This decoupling makes the learning rate and weight decay roughly
orthogonal hyperparameters.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["AdamWConfig", "AdamWState", "init_state", "adamw_step", "minimize"]

Vector = Sequence[float]
GradFn = Callable[[NDArray[np.float64]], Sequence[float]]


@dataclass(frozen=True)
class AdamWConfig:
    """Hyperparameters for the AdamW optimizer.

    Attributes
    ----------
    lr:
        Learning rate (step size) ``alpha``. Must be positive.
    beta1:
        Exponential decay rate for the first moment, in ``[0, 1)``.
    beta2:
        Exponential decay rate for the second moment, in ``[0, 1)``.
    eps:
        Numerical floor added to ``sqrt(vhat)`` in the denominator. Must be
        positive.
    weight_decay:
        Decoupled weight-decay coefficient ``lambda``. Must be non-negative.
        ``0.0`` recovers plain Adam.
    """

    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if not (self.lr > 0.0):
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not (0.0 <= self.beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}")
        if not (0.0 <= self.beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}")
        if not (self.eps > 0.0):
            raise ValueError(f"eps must be positive, got {self.eps}")
        if not (self.weight_decay >= 0.0):
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")


@dataclass
class AdamWState:
    """Mutable optimizer state for one parameter vector.

    Attributes
    ----------
    m:
        First-moment estimate, same shape as the parameters.
    v:
        Second-moment estimate, same shape as the parameters.
    t:
        Number of update steps taken so far (starts at 0).
    """

    m: NDArray[np.float64]
    v: NDArray[np.float64]
    t: int = 0


def init_state(n_params: int) -> AdamWState:
    """Create a zero-initialized optimizer state for ``n_params`` parameters.

    Parameters
    ----------
    n_params:
        Number of parameters (length of the parameter vector). Must be >= 1.

    Examples
    --------
    >>> s = init_state(3)
    >>> s.t
    0
    >>> s.m.tolist()
    [0.0, 0.0, 0.0]
    """
    if not isinstance(n_params, (int, np.integer)):
        raise ValueError(f"n_params must be an integer, got {type(n_params).__name__}")
    n_params = int(n_params)
    if n_params < 1:
        raise ValueError(f"n_params must be >= 1, got {n_params}")
    return AdamWState(m=np.zeros(n_params, dtype=np.float64), v=np.zeros(n_params, dtype=np.float64), t=0)


def _as_vector(x: Vector, name: str, expected_len: int | None = None) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got array with {arr.ndim} dimension(s)")
    if arr.size < 1:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (nan or inf)")
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"{name} has length {arr.size} but expected {expected_len}")
    return arr


def adamw_step(
    params: Vector,
    grad: Vector,
    state: AdamWState,
    config: AdamWConfig,
) -> NDArray[np.float64]:
    """Apply one AdamW update in place on ``state`` and return the new parameters.

    The state's ``m``, ``v``, and ``t`` fields are advanced one step. The
    returned array is a fresh vector of updated parameters; ``params`` itself is
    not modified.

    Parameters
    ----------
    params:
        Current parameter vector of length ``d``.
    grad:
        Gradient of the loss with respect to ``params``, length ``d``.
    state:
        Optimizer state whose ``m`` and ``v`` have length ``d``.
    config:
        Hyperparameters.

    Returns
    -------
    numpy.ndarray
        The updated parameter vector of length ``d``.

    Examples
    --------
    >>> cfg = AdamWConfig(lr=0.1, weight_decay=0.01)
    >>> st = init_state(2)
    >>> theta = adamw_step([1.0, -2.0], [0.5, -1.0], st, cfg)
    >>> [round(float(x), 6) for x in theta]
    [0.899, -1.898]
    """
    theta = _as_vector(params, "params")
    d = theta.size
    g = _as_vector(grad, "grad", expected_len=d)
    if state.m.shape != theta.shape or state.v.shape != theta.shape:
        raise ValueError(
            f"state has shape m={state.m.shape}, v={state.v.shape} but params has length {d}"
        )

    b1 = config.beta1
    b2 = config.beta2
    state.t += 1
    t = state.t

    state.m = b1 * state.m + (1.0 - b1) * g
    state.v = b2 * state.v + (1.0 - b2) * (g * g)

    mhat = state.m / (1.0 - b1**t)
    vhat = state.v / (1.0 - b2**t)

    adaptive = mhat / (np.sqrt(vhat) + config.eps)
    return theta - config.lr * (adaptive + config.weight_decay * theta)


def minimize(
    grad_fn: GradFn,
    x0: Vector,
    config: AdamWConfig,
    n_steps: int,
) -> NDArray[np.float64]:
    """Run AdamW for ``n_steps`` iterations starting from ``x0``.

    Parameters
    ----------
    grad_fn:
        Callable mapping a parameter vector to its gradient vector (same length).
    x0:
        Initial parameter vector.
    config:
        Hyperparameters.
    n_steps:
        Number of update steps to perform. Must be >= 1.

    Returns
    -------
    numpy.ndarray
        The parameter vector after ``n_steps`` updates.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([3.0, 1.0]); target = np.array([2.0, -1.0])
    >>> grad = lambda x: A * (x - target)
    >>> x = minimize(grad, [0.0, 0.0], AdamWConfig(lr=0.1), 200)
    >>> bool(np.allclose(x, target, atol=1e-3))
    True
    """
    if not isinstance(n_steps, (int, np.integer)):
        raise ValueError(f"n_steps must be an integer, got {type(n_steps).__name__}")
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    x = _as_vector(x0, "x0")
    state = init_state(x.size)
    for _ in range(n_steps):
        g = _as_vector(grad_fn(x), "grad_fn(x)", expected_len=x.size)
        x = adamw_step(x, g, state, config)
    return x
