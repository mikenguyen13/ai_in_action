"""AdaGrad adaptive learning rates from scratch.

A small, well-validated reference implementation of the diagonal AdaGrad optimizer
of Duchi, Hazan, and Singer (2011). The public API mirrors the Julia
(`AIInAction.Ch194Adagrad`) and Rust (`aiinaction::ch194_adagrad`) implementations
one-to-one; the cross-language parity tests assert that all three agree to within
floating-point tolerance on shared fixtures.

AdaGrad gives every coordinate its own learning rate. For each coordinate ``i`` it
accumulates the sum of squared gradients

    G[i] = sum_tau g[tau, i]**2

and takes the step

    theta[i] -= eta / (sqrt(G[i]) + eps) * g[i].

Because ``G[i]`` is nondecreasing, the effective per-coordinate rate
``eta / (sqrt(G[i]) + eps)`` only ever shrinks. This module exposes the single
update (:func:`adagrad_step`), a full minimizer over a user-supplied gradient
function (:func:`minimize`), and the per-coordinate effective learning rate
(:func:`effective_learning_rate`).

The optimizer is gradient-only and makes no assumption about the objective beyond
differentiability; the helpers :func:`quadratic_value` and :func:`quadratic_grad`
provide a separable convex test objective used by the examples and parity tests.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AdaGradState",
    "AdaGradResult",
    "init_state",
    "adagrad_step",
    "effective_learning_rate",
    "minimize",
    "quadratic_value",
    "quadratic_grad",
]

Vector = Sequence[float]
GradFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class AdaGradState:
    """Mutable optimizer state.

    Attributes
    ----------
    theta:
        Current parameter vector, shape ``(d,)``.
    accumulator:
        Per-coordinate sum of squared gradients ``G``, shape ``(d,)``. Starts at
        zero and is nondecreasing.
    learning_rate:
        Global base learning rate ``eta`` (positive).
    epsilon:
        Numerical-stability constant added to ``sqrt(G)`` in the denominator
        (positive).
    """

    theta: NDArray[np.float64]
    accumulator: NDArray[np.float64]
    learning_rate: float
    epsilon: float


@dataclass(frozen=True)
class AdaGradResult:
    """The outcome of a :func:`minimize` run.

    Attributes
    ----------
    theta:
        Final parameter vector, shape ``(d,)``.
    accumulator:
        Final gradient-square accumulator, shape ``(d,)``.
    n_steps:
        Number of update steps actually performed.
    grad_norm:
        Euclidean norm of the gradient at the final ``theta``.
    converged:
        ``True`` if the run stopped because ``grad_norm <= tol`` rather than by
        exhausting ``max_iter``.
    """

    theta: NDArray[np.float64]
    accumulator: NDArray[np.float64]
    n_steps: int
    grad_norm: float
    converged: bool


def _as_vector(v: Vector, name: str) -> NDArray[np.float64]:
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got array with {arr.ndim} dimension(s)")
    if arr.size < 1:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (nan or inf)")
    return arr


def _check_hyperparams(learning_rate: float, epsilon: float) -> None:
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be a positive finite number, got {learning_rate}")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be a positive finite number, got {epsilon}")


def init_state(theta0: Vector, *, learning_rate: float = 0.1, epsilon: float = 1e-8) -> AdaGradState:
    """Create a fresh :class:`AdaGradState` with a zero accumulator.

    Parameters
    ----------
    theta0:
        Initial parameter vector of length ``d >= 1`` with finite entries.
    learning_rate:
        Positive global base rate ``eta``.
    epsilon:
        Positive stabilizer added to ``sqrt(G)``.

    Examples
    --------
    >>> s = init_state([0.0, 0.0], learning_rate=0.5)
    >>> s.accumulator.tolist()
    [0.0, 0.0]
    """
    theta = _as_vector(theta0, "theta0")
    _check_hyperparams(learning_rate, epsilon)
    return AdaGradState(
        theta=theta.copy(),
        accumulator=np.zeros_like(theta),
        learning_rate=float(learning_rate),
        epsilon=float(epsilon),
    )


def effective_learning_rate(state: AdaGradState) -> NDArray[np.float64]:
    """Return the current per-coordinate effective learning rate.

    This is ``eta / (sqrt(G) + eps)`` evaluated at the state's current
    accumulator, shape ``(d,)``. At a freshly initialized state (``G = 0``) every
    entry equals ``eta / eps``.
    """
    return state.learning_rate / (np.sqrt(state.accumulator) + state.epsilon)


def adagrad_step(state: AdaGradState, grad: Vector) -> AdaGradState:
    """Apply one in-place AdaGrad update for the given gradient.

    Accumulates ``G += grad**2`` and steps
    ``theta -= eta / (sqrt(G) + eps) * grad``. Mutates and returns ``state`` so
    callers can chain updates in a loop.

    Parameters
    ----------
    state:
        Optimizer state to update.
    grad:
        Gradient vector, same length as ``state.theta``, with finite entries.

    Raises
    ------
    ValueError
        If ``grad`` has the wrong length or contains non-finite values.
    """
    g = _as_vector(grad, "grad")
    if g.shape != state.theta.shape:
        raise ValueError(
            f"grad has length {g.size} but theta has length {state.theta.size}"
        )
    state.accumulator = state.accumulator + g * g
    step = state.learning_rate / (np.sqrt(state.accumulator) + state.epsilon)
    state.theta = state.theta - step * g
    return state


def minimize(
    grad_fn: GradFn,
    theta0: Vector,
    *,
    learning_rate: float = 0.1,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> AdaGradResult:
    """Minimize an objective by AdaGrad using only its gradient.

    Repeatedly evaluates ``grad_fn(theta)`` and applies :func:`adagrad_step` until
    the gradient norm drops to ``tol`` or ``max_iter`` steps have been taken.

    Parameters
    ----------
    grad_fn:
        Callable mapping a parameter vector to its gradient (same shape). Must
        return finite values.
    theta0:
        Initial parameter vector.
    learning_rate, epsilon:
        AdaGrad hyperparameters; see :func:`init_state`.
    max_iter:
        Maximum number of update steps; must be a positive integer.
    tol:
        Nonnegative gradient-norm tolerance for early stopping. Set to ``0`` to
        always run the full ``max_iter`` steps.

    Returns
    -------
    AdaGradResult
        Final parameters, accumulator, step count, final gradient norm, and a
        convergence flag.

    Examples
    --------
    >>> import numpy as np
    >>> g = lambda th: 2.0 * (th - np.array([1.0, -2.0]))
    >>> res = minimize(g, [0.0, 0.0], learning_rate=0.5, max_iter=5000)
    >>> bool(np.allclose(res.theta, [1.0, -2.0], atol=1e-3))
    True
    """
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f"tol must be a nonnegative finite number, got {tol}")
    max_iter = int(max_iter)

    state = init_state(theta0, learning_rate=learning_rate, epsilon=epsilon)
    d = state.theta.size

    grad_norm = float("inf")
    steps = 0
    converged = False
    for _ in range(max_iter):
        g = np.asarray(grad_fn(state.theta), dtype=np.float64)
        if g.shape != (d,):
            raise ValueError(
                f"grad_fn returned shape {g.shape} but expected ({d},)"
            )
        if not np.all(np.isfinite(g)):
            raise ValueError("grad_fn returned non-finite values (nan or inf)")
        grad_norm = float(np.sqrt(np.sum(g * g)))
        if grad_norm <= tol:
            converged = True
            break
        adagrad_step(state, g)
        steps += 1

    return AdaGradResult(
        theta=state.theta,
        accumulator=state.accumulator,
        n_steps=steps,
        grad_norm=grad_norm,
        converged=converged,
    )


def quadratic_grad(theta: Vector, a: Vector, b: Vector) -> NDArray[np.float64]:
    """Gradient of the separable quadratic ``f(theta) = 0.5 * sum a_i (theta_i - b_i)^2``.

    The gradient is ``a_i * (theta_i - b_i)``. The coefficients ``a`` (positive
    curvatures) and ``b`` (the minimizer) must match ``theta`` in length.
    """
    th = _as_vector(theta, "theta")
    av = _as_vector(a, "a")
    bv = _as_vector(b, "b")
    if not (th.shape == av.shape == bv.shape):
        raise ValueError(
            f"theta, a, b must share length; got {th.size}, {av.size}, {bv.size}"
        )
    if np.any(av <= 0.0):
        raise ValueError("curvatures a must be strictly positive")
    return av * (th - bv)


def quadratic_value(theta: Vector, a: Vector, b: Vector) -> float:
    """Value of ``f(theta) = 0.5 * sum a_i (theta_i - b_i)^2`` (the test objective)."""
    th = _as_vector(theta, "theta")
    av = _as_vector(a, "a")
    bv = _as_vector(b, "b")
    if not (th.shape == av.shape == bv.shape):
        raise ValueError(
            f"theta, a, b must share length; got {th.size}, {av.size}, {bv.size}"
        )
    if np.any(av <= 0.0):
        raise ValueError("curvatures a must be strictly positive")
    return float(0.5 * np.sum(av * (th - bv) ** 2))
