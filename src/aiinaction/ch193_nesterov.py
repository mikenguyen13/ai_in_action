"""Nesterov Accelerated Gradient (NAG) from scratch.

A small, well-validated reference implementation of Nesterov's accelerated
gradient method for smooth, unconstrained minimization. The public API mirrors the
Julia (`AIInAction.Ch193Nesterov`) and Rust (`aiinaction::ch193_nesterov`)
implementations one-to-one; the cross-language parity tests assert that all three
agree to within floating-point tolerance on shared fixtures.

Two schedules are provided, both written against a user-supplied gradient oracle:

1. ``nesterov_convex`` -- the two-sequence (FISTA-style) form used in the
   convergence analysis for smooth convex objectives. With step size ``1/L`` it
   maintains an iterate sequence ``x_k`` and an extrapolated lookahead sequence
   ``y_k``::

       x_{k+1} = y_k - eta * grad f(y_k)
       t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
       gamma   = (t_k - 1) / t_{k+1}
       y_{k+1} = x_{k+1} + gamma * (x_{k+1} - x_k)

   The momentum coefficient ``gamma_k`` grows toward one and drives the optimal
   ``O(1/k^2)`` rate.

2. ``nesterov_momentum`` -- the constant-momentum velocity form preferred for
   strongly convex problems::

       y_k     = x_k + beta * v_k
       v_{k+1} = beta * v_k - eta * grad f(y_k)
       x_{k+1} = x_k + v_{k+1}

   With ``beta = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)`` this attains the
   ``sqrt(kappa)`` accelerated linear rate.

Both routines are gradient-oracle based and make no assumption about the form of
``f`` beyond differentiability; the gradient is supplied as a callable that maps a
point (1-D array) to its gradient (1-D array of the same length).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["OptimizeResult", "nesterov_convex", "nesterov_momentum"]

Vector = Sequence[float]
GradFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class OptimizeResult:
    """The outcome of a Nesterov optimization run.

    Attributes
    ----------
    x:
        Best iterate found, shape ``(n,)``.
    n_iter:
        Number of iterations actually performed (``<= max_iter``). Equals the
        iteration at which the gradient-norm tolerance was met, or ``max_iter`` if
        the run exhausted its budget.
    grad_norm:
        Euclidean norm of the gradient at ``x``.
    converged:
        ``True`` if ``grad_norm <= tol`` was reached within the iteration budget.
    history:
        Gradient norm recorded after each iteration, length ``n_iter``. Useful for
        plotting convergence curves.
    """

    x: NDArray[np.float64]
    n_iter: int
    grad_norm: float
    converged: bool
    history: NDArray[np.float64]


def _prepare(x0: Vector, grad: GradFn) -> NDArray[np.float64]:
    arr = np.asarray(x0, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"x0 must be a 1-D vector, got array with {arr.ndim} dimension(s)")
    if arr.size < 1:
        raise ValueError("x0 must have at least one entry")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x0 contains non-finite values (nan or inf)")
    if not callable(grad):
        raise ValueError("grad must be a callable mapping a point to its gradient")
    return arr.copy()


def _eval_grad(grad: GradFn, y: NDArray[np.float64]) -> NDArray[np.float64]:
    g = np.asarray(grad(y), dtype=np.float64)
    if g.shape != y.shape:
        raise ValueError(
            f"gradient has shape {g.shape} but the point has shape {y.shape}; "
            "the gradient oracle must return a vector matching x0"
        )
    if not np.all(np.isfinite(g)):
        raise ValueError("gradient returned non-finite values (nan or inf)")
    return g


def nesterov_convex(
    grad: GradFn,
    x0: Vector,
    step_size: float,
    *,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> OptimizeResult:
    """Minimize a smooth convex function with the two-sequence Nesterov schedule.

    Parameters
    ----------
    grad:
        Gradient oracle ``grad(y) -> g`` returning a vector matching ``x0``.
    x0:
        Starting point, a 1-D vector of finite values.
    step_size:
        Constant step size ``eta``. For an ``L``-smooth objective the canonical
        choice is ``1/L``. Must be strictly positive.
    max_iter:
        Maximum number of iterations. Must be a positive integer.
    tol:
        Convergence tolerance on the Euclidean gradient norm. Must be non-negative.

    Returns
    -------
    OptimizeResult

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[3.0, 0.0], [0.0, 1.0]])
    >>> b = np.array([3.0, -1.0])
    >>> res = nesterov_convex(lambda x: A @ x - b, [0.0, 0.0], step_size=1/3)
    >>> bool(res.converged)
    True
    >>> np.allclose(res.x, [1.0, -1.0], atol=1e-6)
    True
    """
    x = _prepare(x0, grad)
    if not (step_size > 0.0) or not np.isfinite(step_size):
        raise ValueError(f"step_size must be a positive finite number, got {step_size}")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    if not (tol >= 0.0) or not np.isfinite(tol):
        raise ValueError(f"tol must be a non-negative finite number, got {tol}")

    max_iter = int(max_iter)
    eta = float(step_size)
    y = x.copy()
    t = 1.0
    history: list[float] = []
    converged = False
    n_iter = 0

    for k in range(1, max_iter + 1):
        n_iter = k
        g = _eval_grad(grad, y)
        x_next = y - eta * g
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        gamma = (t - 1.0) / t_next
        y = x_next + gamma * (x_next - x)
        x = x_next
        t = t_next

        gn = float(np.linalg.norm(_eval_grad(grad, x)))
        history.append(gn)
        if gn <= tol:
            converged = True
            break

    return OptimizeResult(
        x=x,
        n_iter=n_iter,
        grad_norm=history[-1] if history else float(np.linalg.norm(_eval_grad(grad, x))),
        converged=converged,
        history=np.asarray(history, dtype=np.float64),
    )


def nesterov_momentum(
    grad: GradFn,
    x0: Vector,
    step_size: float,
    momentum: float,
    *,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> OptimizeResult:
    """Minimize with the constant-momentum (velocity) Nesterov form.

    The lookahead point is ``y_k = x_k + beta * v_k`` and the gradient is taken
    there before the velocity is updated.

    Parameters
    ----------
    grad:
        Gradient oracle ``grad(y) -> g`` returning a vector matching ``x0``.
    x0:
        Starting point, a 1-D vector of finite values.
    step_size:
        Constant step size ``eta`` (``1/L`` is canonical). Must be strictly positive.
    momentum:
        Momentum coefficient ``beta`` in ``[0, 1)``. For a ``mu``-strongly convex,
        ``L``-smooth objective the accelerated choice is
        ``(sqrt(kappa) - 1) / (sqrt(kappa) + 1)`` with ``kappa = L / mu``.
    max_iter:
        Maximum number of iterations. Must be a positive integer.
    tol:
        Convergence tolerance on the Euclidean gradient norm. Must be non-negative.

    Returns
    -------
    OptimizeResult
    """
    x = _prepare(x0, grad)
    if not (step_size > 0.0) or not np.isfinite(step_size):
        raise ValueError(f"step_size must be a positive finite number, got {step_size}")
    if not (0.0 <= momentum < 1.0):
        raise ValueError(f"momentum must lie in [0, 1), got {momentum}")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    if not (tol >= 0.0) or not np.isfinite(tol):
        raise ValueError(f"tol must be a non-negative finite number, got {tol}")

    max_iter = int(max_iter)
    eta = float(step_size)
    beta = float(momentum)
    v = np.zeros_like(x)
    history: list[float] = []
    converged = False
    n_iter = 0

    for k in range(1, max_iter + 1):
        n_iter = k
        y = x + beta * v
        g = _eval_grad(grad, y)
        v = beta * v - eta * g
        x = x + v

        gn = float(np.linalg.norm(_eval_grad(grad, x)))
        history.append(gn)
        if gn <= tol:
            converged = True
            break

    return OptimizeResult(
        x=x,
        n_iter=n_iter,
        grad_norm=history[-1] if history else float(np.linalg.norm(_eval_grad(grad, x))),
        converged=converged,
        history=np.asarray(history, dtype=np.float64),
    )
