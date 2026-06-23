"""Momentum (heavy-ball) gradient descent from scratch.

A small, well-validated reference implementation of Polyak's heavy-ball momentum
optimizer. The public API mirrors the Julia (`AIInAction.Ch192Momentum`) and Rust
(`aiinaction::ch192_momentum`) implementations one-to-one; the cross-language
parity tests assert that all three agree to within floating-point tolerance on
shared fixtures.

The optimizer maintains a velocity vector ``v`` that is an exponentially weighted
accumulation of past gradients and steps the parameters along it:

.. code-block:: text

    v <- beta * v + grad(theta)
    theta <- theta - alpha * v

Setting ``beta = 0`` recovers plain gradient descent (``v_t = grad(theta_t)``). As
``beta`` approaches one, the velocity retains more of its history and the iterate
behaves like a massive object that resists abrupt changes in direction.

The driver :func:`minimize` is generic in the objective: it takes a callable that
returns the gradient at a point, so it works for any differentiable function. A
convenience builder :func:`quadratic_gradient` constructs the gradient of the
quadratic ``f(theta) = 1/2 (theta - b)^T H (theta - b)`` used in the chapter and
tests.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MomentumResult",
    "momentum_step",
    "minimize",
    "quadratic_gradient",
    "optimal_beta",
]

Vector = Sequence[float]
GradFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class MomentumResult:
    """The outcome of a momentum optimization run.

    Attributes
    ----------
    theta:
        Final parameter vector, shape ``(d,)``.
    velocity:
        Final velocity vector ``v``, shape ``(d,)``.
    n_iter:
        Number of update steps actually performed (``<= max_iter``).
    converged:
        ``True`` if the run stopped because ``||grad|| <= tol`` rather than by
        exhausting ``max_iter``.
    grad_norm:
        Euclidean norm of the gradient at ``theta`` on the final iteration.
    history:
        Objective-gradient-norm at the start of each performed step, shape
        ``(n_iter,)``. Useful for plotting convergence curves.
    """

    theta: NDArray[np.float64]
    velocity: NDArray[np.float64]
    n_iter: int
    converged: bool
    grad_norm: float
    history: NDArray[np.float64]

    @property
    def n_features(self) -> int:
        """Dimensionality of the parameter space."""
        return int(self.theta.shape[0])


def _as_vector(x: Vector, name: str) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got array with {arr.ndim} dimension(s)")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must have at least one entry")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (nan or inf)")
    return arr


def _check_hyperparams(alpha: float, beta: float) -> tuple[float, float]:
    a = float(alpha)
    b = float(beta)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError(f"alpha (learning rate) must be a positive finite number, got {alpha}")
    if not np.isfinite(b) or b < 0.0 or b >= 1.0:
        raise ValueError(f"beta (momentum) must be in [0, 1), got {beta}")
    return a, b


def momentum_step(
    theta: Vector,
    velocity: Vector,
    grad: Vector,
    alpha: float,
    beta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply a single heavy-ball update and return ``(new_theta, new_velocity)``.

    Implements

    .. math::

        v_{t} = \\beta\\, v_{t-1} + g_t, \\qquad
        \\theta_{t+1} = \\theta_t - \\alpha\\, v_t,

    where ``g_t`` is the supplied gradient. The three vectors must share a length.

    Examples
    --------
    >>> th, v = momentum_step([1.0], [0.0], [2.0], alpha=0.1, beta=0.9)
    >>> round(float(th[0]), 6), round(float(v[0]), 6)
    (0.8, 2.0)
    """
    a, b = _check_hyperparams(alpha, beta)
    th = _as_vector(theta, "theta")
    v = _as_vector(velocity, "velocity")
    g = _as_vector(grad, "grad")
    if not (th.shape[0] == v.shape[0] == g.shape[0]):
        raise ValueError(
            f"length mismatch: theta={th.shape[0]}, velocity={v.shape[0]}, grad={g.shape[0]}"
        )
    new_v = b * v + g
    new_theta = th - a * new_v
    return new_theta, new_v


def minimize(
    grad_fn: GradFn,
    theta0: Vector,
    *,
    alpha: float,
    beta: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> MomentumResult:
    """Minimize an objective with heavy-ball momentum from initial point ``theta0``.

    Parameters
    ----------
    grad_fn:
        Callable mapping a parameter vector to its gradient, both length ``d``.
    theta0:
        Initial parameter vector, shape ``(d,)``. Must be finite.
    alpha:
        Learning rate, a positive finite number.
    beta:
        Momentum coefficient in ``[0, 1)``. ``beta = 0`` is plain gradient descent.
    max_iter:
        Maximum number of update steps. Must be a positive integer.
    tol:
        Convergence tolerance on the Euclidean gradient norm. The loop stops as
        soon as ``||grad(theta)|| <= tol``. Must be non-negative.

    Returns
    -------
    MomentumResult
        Final iterate, velocity, iteration count, and convergence diagnostics.

    Examples
    --------
    >>> g = quadratic_gradient([[1.0]], [3.0])
    >>> r = minimize(g, [0.0], alpha=0.5, beta=0.0)
    >>> round(float(r.theta[0]), 6)
    3.0
    """
    a, b = _check_hyperparams(alpha, beta)
    theta = _as_vector(theta0, "theta0")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    max_iter = int(max_iter)
    tol_f = float(tol)
    if not np.isfinite(tol_f) or tol_f < 0.0:
        raise ValueError(f"tol must be a non-negative finite number, got {tol}")

    velocity = np.zeros_like(theta)
    history: list[float] = []
    grad_norm = float("inf")
    converged = False
    n_iter = 0

    for _ in range(max_iter):
        g = np.asarray(grad_fn(theta), dtype=np.float64)
        if g.shape != theta.shape:
            raise ValueError(
                f"grad_fn returned shape {g.shape} but theta has shape {theta.shape}"
            )
        if not np.all(np.isfinite(g)):
            raise ValueError("grad_fn returned non-finite values (nan or inf)")
        grad_norm = float(np.sqrt(np.sum(g * g)))
        history.append(grad_norm)
        if grad_norm <= tol_f:
            converged = True
            break
        velocity = b * velocity + g
        theta = theta - a * velocity
        n_iter += 1

    return MomentumResult(
        theta=theta,
        velocity=velocity,
        n_iter=n_iter,
        converged=converged,
        grad_norm=grad_norm,
        history=np.asarray(history, dtype=np.float64),
    )


def quadratic_gradient(H: Sequence[Sequence[float]], b: Vector) -> GradFn:
    """Build the gradient of ``f(theta) = 1/2 (theta - b)^T H (theta - b)``.

    The returned callable computes ``grad f(theta) = H (theta - b)``, whose unique
    stationary point is ``theta = b`` when ``H`` is positive definite. ``H`` must be
    a square ``d x d`` matrix and ``b`` a length-``d`` vector.
    """
    Hm = np.asarray(H, dtype=np.float64)
    if Hm.ndim != 2 or Hm.shape[0] != Hm.shape[1]:
        raise ValueError(f"H must be a square 2-D matrix, got shape {Hm.shape}")
    if not np.all(np.isfinite(Hm)):
        raise ValueError("H contains non-finite values (nan or inf)")
    bv = _as_vector(b, "b")
    if Hm.shape[0] != bv.shape[0]:
        raise ValueError(f"H is {Hm.shape[0]}x{Hm.shape[1]} but b has length {bv.shape[0]}")

    def grad(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        t = np.asarray(theta, dtype=np.float64)
        return Hm @ (t - bv)

    return grad


def optimal_beta(lambda_min: float, lambda_max: float) -> float:
    """Polyak's optimal momentum for a quadratic with the given curvature extremes.

    For ``f(theta) = 1/2 theta^T H theta`` with eigenvalues of ``H`` spanning
    ``[lambda_min, lambda_max]`` (both positive), the optimally tuned heavy-ball
    momentum coefficient is

    .. math::

        \\beta^\\star = \\left(
            \\frac{\\sqrt{\\lambda_{\\max}} - \\sqrt{\\lambda_{\\min}}}
                 {\\sqrt{\\lambda_{\\max}} + \\sqrt{\\lambda_{\\min}}}
        \\right)^2 .

    Examples
    --------
    >>> round(optimal_beta(1.0, 1.0), 6)
    0.0
    """
    lo = float(lambda_min)
    hi = float(lambda_max)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.0 or hi <= 0.0:
        raise ValueError("lambda_min and lambda_max must be positive finite numbers")
    if hi < lo:
        raise ValueError(f"lambda_max ({hi}) must be >= lambda_min ({lo})")
    r = (np.sqrt(hi) - np.sqrt(lo)) / (np.sqrt(hi) + np.sqrt(lo))
    return float(r * r)
