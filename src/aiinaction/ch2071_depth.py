"""Why depth is not free: signal propagation and depth diagnostics from scratch.

Reference implementations of the closed form and quadrature results derived in
the chapter "Why Deeper Is Not Always Better". The public API mirrors the Julia
(``AIInAction.Ch2071Depth``) and Rust (``aiinaction::ch2071_depth``)
implementations one to one, and shared numeric fixtures keep the three at
parity.

Contents
--------
* Mean field signal propagation for wide random networks (Poole et al. 2016,
  Schoenholz et al. 2017): the length map ``q -> sigma_w^2 E[phi(sqrt(q) z)^2]
  + sigma_b^2``, its fixed point ``q*``, the correlation map, the slope
  ``chi_1`` that separates the ordered and chaotic phases, and the correlation
  depth scale ``xi_c = -1 / log(chi_c)``.
* The ReLU correlation map in closed form (the normalized arc cosine kernel of
  Cho and Saul 2009), its iteration in plain and residual networks, the
  ``9 pi^2 / (2 l^2)`` collapse law, and the continuous depth limit reached
  with ``1 / sqrt(L)`` branch scaling.
* Residual stream variance growth ``q_L = q_0 (1 + alpha^2)^L``.
* Learning time of a scalar deep linear network under gradient flow, which
  grows like ``a_0^{-(N-2)}`` with depth ``N`` from small initialization
  (Saxe et al. 2014).
* Path statistics of residual networks viewed as ensembles (Veit et al.
  2016).
* Telgarsky's sawtooth construction, which shows what depth buys in
  expressivity.

Everything is deterministic and uses only the standard library, so the Julia
and Rust ports reproduce the same numbers to floating point tolerance.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

__all__ = [
    "ACTIVATIONS",
    "activation",
    "activation_derivative",
    "gaussian_expectation",
    "gaussian_expectation_2d",
    "length_map",
    "length_fixed_point",
    "correlation_map",
    "chi1",
    "correlation_depth_scale",
    "relu_correlation_map",
    "relu_correlation_trajectory",
    "relu_collapse_asymptotic",
    "relu_kernel_ode",
    "residual_variance",
    "deep_linear_learning_time",
    "deep_linear_time_asymptotic",
    "path_length_distribution",
    "effective_path_distribution",
    "sawtooth",
    "count_linear_pieces",
]

ACTIVATIONS = ("linear", "relu", "tanh")

# Quadrature defaults. Composite Simpson on [-LIM, LIM] with N (even) panels.
_LIM = 10.0
_N1 = 4000
_N2 = 400


def _check_activation(name: str) -> None:
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; expected one of {ACTIVATIONS}")


def activation(name: str, x: float) -> float:
    """Evaluate ``phi(x)`` for ``name`` in ``ACTIVATIONS``."""
    _check_activation(name)
    if name == "linear":
        return x
    if name == "relu":
        return x if x > 0.0 else 0.0
    return math.tanh(x)


def activation_derivative(name: str, x: float) -> float:
    """Evaluate ``phi'(x)``; the ReLU derivative at 0 is taken as 0."""
    _check_activation(name)
    if name == "linear":
        return 1.0
    if name == "relu":
        return 1.0 if x > 0.0 else 0.0
    t = math.tanh(x)
    return 1.0 - t * t


def _simpson_weights(n: int) -> list[float]:
    if n < 2 or n % 2:
        raise ValueError("number of Simpson panels must be an even integer >= 2")
    w = [2.0 if i % 2 == 0 else 4.0 for i in range(n + 1)]
    w[0] = w[n] = 1.0
    return w


def gaussian_expectation(
    f: Callable[[float], float], n: int = _N1, lim: float = _LIM
) -> float:
    """``E[f(z)]`` for ``z ~ N(0, 1)`` by composite Simpson quadrature.

    The integrand is ``f(z) * exp(-z^2 / 2) / sqrt(2 pi)`` on ``[-lim, lim]``;
    the neglected tails carry mass below ``1e-22`` for the default ``lim``.
    """
    w = _simpson_weights(n)
    h = 2.0 * lim / n
    c = 1.0 / math.sqrt(2.0 * math.pi)
    total = 0.0
    for i in range(n + 1):
        z = -lim + i * h
        total += w[i] * f(z) * math.exp(-0.5 * z * z)
    return total * h / 3.0 * c


def gaussian_expectation_2d(
    f: Callable[[float, float], float], n: int = _N2, lim: float = _LIM
) -> float:
    """``E[f(z1, z2)]`` for independent standard normals by tensor Simpson."""
    w = _simpson_weights(n)
    h = 2.0 * lim / n
    c = 1.0 / (2.0 * math.pi)
    nodes = [-lim + i * h for i in range(n + 1)]
    dens = [math.exp(-0.5 * z * z) for z in nodes]
    total = 0.0
    for i, z1 in enumerate(nodes):
        wi = w[i] * dens[i]
        row = 0.0
        for j, z2 in enumerate(nodes):
            row += w[j] * dens[j] * f(z1, z2)
        total += wi * row
    return total * (h / 3.0) ** 2 * c


def _check_variances(sigma_w2: float, sigma_b2: float) -> None:
    if sigma_w2 <= 0.0 or sigma_b2 < 0.0:
        raise ValueError("need sigma_w2 > 0 and sigma_b2 >= 0")


def length_map(q: float, sigma_w2: float, sigma_b2: float, phi: str) -> float:
    """One step of the mean field length map.

    ``q_{l+1} = sigma_w^2 E[phi(sqrt(q_l) z)^2] + sigma_b^2``, the variance of a
    preactivation in layer ``l + 1`` of a wide network whose weights have
    variance ``sigma_w^2 / n`` and biases variance ``sigma_b^2``.
    """
    _check_variances(sigma_w2, sigma_b2)
    if q < 0.0:
        raise ValueError("q must be nonnegative")
    s = math.sqrt(q)
    return sigma_w2 * gaussian_expectation(lambda z: activation(phi, s * z) ** 2) + sigma_b2


def length_fixed_point(
    sigma_w2: float,
    sigma_b2: float,
    phi: str,
    q0: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> float:
    """Iterate the length map from ``q0`` to its fixed point ``q*``.

    Raises ``RuntimeError`` if the map has not converged (for example ReLU with
    ``sigma_w^2 > 2``, where the variance grows without bound).
    """
    q = q0
    for _ in range(max_iter):
        q_next = length_map(q, sigma_w2, sigma_b2, phi)
        if not math.isfinite(q_next) or q_next > 1e12:
            raise RuntimeError("length map diverged; no finite fixed point")
        if abs(q_next - q) <= tol * max(1.0, q):
            return q_next
        q = q_next
    raise RuntimeError("length map did not converge")


def correlation_map(
    c: float, q: float, sigma_w2: float, sigma_b2: float, phi: str, n: int = _N2
) -> float:
    """One step of the mean field correlation map at fixed length ``q``.

    ``c_{l+1} = (sigma_w^2 E[phi(u1) phi(u2)] + sigma_b^2) / q`` with
    ``u1 = sqrt(q) z1`` and ``u2 = sqrt(q) (c z1 + sqrt(1 - c^2) z2)``. Pass
    ``q = q*`` so that both inputs sit at the length fixed point.
    """
    _check_variances(sigma_w2, sigma_b2)
    if not -1.0 <= c <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    if q <= 0.0:
        raise ValueError("q must be positive")
    s = math.sqrt(q)
    r = math.sqrt(max(0.0, 1.0 - c * c))
    e = gaussian_expectation_2d(
        lambda z1, z2: activation(phi, s * z1) * activation(phi, s * (c * z1 + r * z2)), n
    )
    return (sigma_w2 * e + sigma_b2) / q


def chi1(q: float, sigma_w2: float, phi: str) -> float:
    """Slope of the correlation map at ``c = 1``.

    ``chi_1 = sigma_w^2 E[phi'(sqrt(q*) z)^2]``. It is also the mean squared
    singular value of a layer Jacobian, so ``chi_1 < 1`` is the ordered
    (vanishing) phase and ``chi_1 > 1`` the chaotic (exploding) phase.
    """
    if sigma_w2 <= 0.0 or q < 0.0:
        raise ValueError("need sigma_w2 > 0 and q >= 0")
    s = math.sqrt(q)
    return sigma_w2 * gaussian_expectation(lambda z: activation_derivative(phi, s * z) ** 2)


def correlation_depth_scale(chi_c: float) -> float:
    """``xi_c = -1 / log(chi_c)``; infinite at criticality ``chi_c = 1``."""
    if chi_c <= 0.0:
        raise ValueError("chi_c must be positive")
    if chi_c == 1.0:
        return math.inf
    return -1.0 / math.log(chi_c)


def relu_correlation_map(c: float) -> float:
    """Closed form ReLU correlation map at criticality (``sigma_w^2 = 2``).

    ``f(c) = (sqrt(1 - c^2) + (pi - arccos c) c) / pi``, the normalized arc
    cosine kernel of degree one.
    """
    if not -1.0 <= c <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    return (math.sqrt(max(0.0, 1.0 - c * c)) + (math.pi - math.acos(c)) * c) / math.pi


def relu_correlation_trajectory(
    c0: float, depth: int, alpha: float | None = None
) -> list[float]:
    """Correlation between two inputs after each of ``depth`` layers.

    ``alpha = None`` is a plain critical ReLU network, ``c_{l+1} = f(c_l)``.
    A residual network ``x_{l+1} = x_l + alpha W relu(x_l)`` with
    ``W ~ N(0, 2 / n)`` instead follows
    ``c_{l+1} = (c_l + alpha^2 f(c_l)) / (1 + alpha^2)``. Returns
    ``[c_0, c_1, ..., c_depth]``.
    """
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    if alpha is not None and alpha < 0.0:
        raise ValueError("alpha must be nonnegative")
    out = [c0]
    c = c0
    for _ in range(depth):
        fc = relu_correlation_map(c)
        if alpha is None:
            c = fc
        else:
            a2 = alpha * alpha
            c = (c + a2 * fc) / (1.0 + a2)
        c = min(1.0, max(-1.0, c))
        out.append(c)
    return out


def relu_collapse_asymptotic(depth: int | float, alpha: float | None = None) -> float:
    """Asymptotic ``1 - c_l`` for large depth.

    Plain network: ``9 pi^2 / (2 l^2)``. Residual network with branch scale
    ``alpha``: ``9 pi^2 (1 + alpha^2)^2 / (2 alpha^4 l^2)``. Valid only while
    the value is small, i.e. when the correlation is already near one.
    """
    if depth <= 0:
        raise ValueError("depth must be positive")
    base = 9.0 * math.pi**2 / (2.0 * depth**2)
    if alpha is None:
        return base
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    a2 = alpha * alpha
    return base * (1.0 + a2) ** 2 / (a2 * a2)


def relu_kernel_ode(c0: float, t: float, steps: int = 10_000) -> float:
    """Integrate ``dc/dt = f(c) - c`` from ``c(0) = c0`` to time ``t`` (RK4).

    This is the continuous depth limit of a residual ReLU network with branch
    scale ``alpha = 1 / sqrt(L)``: after ``L`` blocks the correlation equals
    ``c(1)`` up to ``O(1 / L)``, independent of ``L``.
    """
    if steps <= 0 or t < 0.0:
        raise ValueError("need steps > 0 and t >= 0")

    def g(c: float) -> float:
        c = min(1.0, max(-1.0, c))
        return relu_correlation_map(c) - c

    h = t / steps
    c = c0
    for _ in range(steps):
        k1 = g(c)
        k2 = g(c + 0.5 * h * k1)
        k3 = g(c + 0.5 * h * k2)
        k4 = g(c + h * k3)
        c += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return c


def residual_variance(q0: float, depth: int, alpha: float) -> float:
    """Residual stream variance ``q_0 (1 + alpha^2)^depth`` for ReLU branches.

    Each block ``x + alpha W relu(x)`` with ``W ~ N(0, 2 / n)`` adds
    ``alpha^2 q`` of fresh variance. With ``alpha = 1`` the variance doubles
    every block; with ``alpha = 1 / sqrt(depth)`` it tends to ``q_0 e``.
    """
    if q0 < 0.0 or depth < 0 or alpha < 0.0:
        raise ValueError("need q0 >= 0, depth >= 0, alpha >= 0")
    return q0 * (1.0 + alpha * alpha) ** depth


def deep_linear_learning_time(
    depth: int, a0: float, s: float = 1.0, frac: float = 0.5, n: int = 4000
) -> float:
    """Gradient flow time for a scalar depth ``N`` linear net to reach ``frac * s``.

    The network computes ``u = a_1 a_2 ... a_N`` and minimizes
    ``(s - u)^2 / 2`` from the balanced start ``a_i = a0``. The product then
    obeys ``du/dt = N u^{2 - 2/N} (s - u)``, so the time is
    ``T = int_{u0}^{frac s} du / (N u^{2 - 2/N} (s - u))`` with
    ``u0 = a0^N``, evaluated by Simpson's rule in ``v = log u``.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if a0 <= 0.0 or s <= 0.0 or not 0.0 < frac < 1.0:
        raise ValueError("need a0 > 0, s > 0 and 0 < frac < 1")
    u0 = a0**depth
    u1 = frac * s
    if u0 >= u1:
        return 0.0
    v0, v1 = math.log(u0), math.log(u1)
    w = _simpson_weights(n)
    h = (v1 - v0) / n
    p = 2.0 - 2.0 / depth
    total = 0.0
    for i in range(n + 1):
        v = v0 + i * h
        u = math.exp(v)
        total += w[i] * u / (depth * u**p * (s - u))
    return total * h / 3.0


def deep_linear_time_asymptotic(depth: int, a0: float, s: float = 1.0) -> float:
    """Leading order learning time ``a0^{-(N-2)} / (s (N - 2))`` for ``N > 2``.

    It dominates when ``a0`` is small: the plateau before learning starts
    lengthens geometrically with depth.
    """
    if depth <= 2:
        raise ValueError("asymptotic formula needs depth > 2")
    if a0 <= 0.0 or s <= 0.0:
        raise ValueError("need a0 > 0 and s > 0")
    return a0 ** (-(depth - 2)) / (s * (depth - 2))


def path_length_distribution(n_blocks: int) -> list[float]:
    """Fraction of the ``2^n`` input to output paths that cross ``k`` branches.

    Unrolling ``n`` residual blocks gives ``C(n, k)`` paths through exactly
    ``k`` residual branches, a Binomial(n, 1/2) law.
    """
    if n_blocks < 0:
        raise ValueError("n_blocks must be nonnegative")
    total = 2.0**n_blocks
    return [math.comb(n_blocks, k) / total for k in range(n_blocks + 1)]


def effective_path_distribution(n_blocks: int, rho: float) -> list[float]:
    """Share of the gradient carried by paths of each length.

    If each residual branch scales a backpropagated signal by ``rho`` (and the
    shortcut by one), the paths of length ``k`` carry total weight
    ``C(n, k) rho^k``. Normalized, this is Binomial(n, rho / (1 + rho)), whose
    mean ``n rho / (1 + rho)`` is the effective depth.
    """
    if n_blocks < 0 or rho < 0.0:
        raise ValueError("need n_blocks >= 0 and rho >= 0")
    p = rho / (1.0 + rho)
    return [math.comb(n_blocks, k) * p**k * (1.0 - p) ** (n_blocks - k) for k in range(n_blocks + 1)]


def sawtooth(x: float, depth: int) -> float:
    """Telgarsky's construction: ``depth`` compositions of the tent map.

    The tent map ``m(x) = 2 relu(x) - 4 relu(x - 1/2)`` on ``[0, 1]`` is a two
    unit ReLU layer. Composing it ``L`` times yields ``2^L`` linear pieces with
    only ``2 L`` units, while a one hidden layer ReLU network needs ``2^L - 1``
    units to produce as many pieces.
    """
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    y = x
    for _ in range(depth):
        y = 2.0 * max(y, 0.0) - 4.0 * max(y - 0.5, 0.0)
    return y


def count_linear_pieces(ys: Sequence[float], tol: float = 1e-9) -> int:
    """Count maximal linear pieces of a function sampled on a uniform grid.

    Counts maximal runs of grid points whose second difference exceeds ``tol``
    in magnitude; the second difference is nonzero exactly at the grid cells
    that contain a breakpoint, so each run marks one breakpoint. Grid breakpoints must be
    separated by at least two cells for the count to be exact.
    """
    if len(ys) < 3:
        raise ValueError("need at least three samples")
    pieces = 1
    in_kink = False
    for i in range(1, len(ys) - 1):
        d2 = ys[i + 1] - 2.0 * ys[i] + ys[i - 1]
        kink = abs(d2) > tol
        if kink and not in_kink:
            pieces += 1
        in_kink = kink
    return pieces
