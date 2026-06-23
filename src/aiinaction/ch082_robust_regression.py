"""Linear regression extensions: robust (Huber), WLS, GLS, quantile, basis.

Small, dependency-light reference implementations with explicit input validation,
accompanying the "Linear Regression Extensions" chapter. They mirror the Julia
(`AIInAction.Ch082RobustRegression`) and Rust (`aiinaction::ch082_robust_regression`)
implementations one-to-one; the cross-language parity tests assert that all three
agree to within floating-point tolerance on the shared fixtures.

The core algorithm is Huber robust regression solved by iteratively reweighted
least squares (IRLS). The same weighted-normal-equations engine powers weighted
least squares (WLS), generalized least squares (GLS), and quantile regression, and
``vandermonde`` provides a polynomial basis expansion so the linear-in-parameters
machinery extends to nonlinear conditional means.

All solvers operate on a design matrix ``X`` given as a list/sequence of rows
(each row a sequence of floats) and a response vector ``y``. To keep the numerics
identical across Python, Julia, and Rust, the linear systems are solved here with a
plain Gaussian-elimination routine rather than a library factorization.
"""
from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "solve",
    "fit_ols",
    "fit_wls",
    "fit_gls",
    "fit_huber",
    "fit_quantile",
    "vandermonde",
    "predict",
    "HuberResult",
]

Matrix = Sequence[Sequence[float]]
Vector = Sequence[float]


# --------------------------------------------------------------------------- #
# Validation and small linear-algebra primitives                              #
# --------------------------------------------------------------------------- #
def _as_matrix(X: Matrix) -> list[list[float]]:
    rows = [[float(v) for v in row] for row in X]
    if not rows:
        raise ValueError("design matrix X must have at least one row")
    ncol = len(rows[0])
    if ncol == 0:
        raise ValueError("design matrix X must have at least one column")
    for i, row in enumerate(rows):
        if len(row) != ncol:
            raise ValueError(
                f"ragged design matrix: row 0 has {ncol} columns but row {i} has {len(row)}"
            )
    return rows


def _check_xy(X: Matrix, y: Vector) -> tuple[list[list[float]], list[float]]:
    rows = _as_matrix(X)
    yv = [float(v) for v in y]
    if len(rows) != len(yv):
        raise ValueError(
            f"length mismatch: X has {len(rows)} rows but y has {len(yv)} entries"
        )
    if len(rows) < len(rows[0]):
        raise ValueError(
            f"underdetermined system: {len(rows)} rows < {len(rows[0])} columns"
        )
    return rows, yv


def solve(A: Matrix, b: Vector) -> list[float]:
    """Solve the square linear system ``A x = b`` by Gaussian elimination.

    Uses partial pivoting for numerical stability. Raises ``ValueError`` if ``A``
    is not square, the dimensions are inconsistent, or the matrix is singular.

    >>> solve([[2.0, 0.0], [0.0, 4.0]], [2.0, 8.0])
    [1.0, 2.0]
    """
    a = [[float(v) for v in row] for row in A]
    rhs = [float(v) for v in b]
    n = len(a)
    if n == 0:
        raise ValueError("system must be non-empty")
    for row in a:
        if len(row) != n:
            raise ValueError(f"matrix must be square: got {len(row)} columns for {n} rows")
    if len(rhs) != n:
        raise ValueError(f"length mismatch: A is {n}x{n} but b has {len(rhs)} entries")

    # Forward elimination with partial pivoting.
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-14:
            raise ValueError("matrix is singular or nearly singular")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        inv = 1.0 / a[col][col]
        for r in range(col + 1, n):
            factor = a[r][col] * inv
            if factor != 0.0:
                for c in range(col, n):
                    a[r][c] -= factor * a[col][c]
                rhs[r] -= factor * rhs[col]

    # Back substitution.
    x = [0.0] * n
    for col in range(n - 1, -1, -1):
        s = rhs[col]
        for c in range(col + 1, n):
            s -= a[col][c] * x[c]
        x[col] = s / a[col][col]
    return x


def _matvec(X: Sequence[Sequence[float]], beta: Sequence[float]) -> list[float]:
    return [sum(xij * bj for xij, bj in zip(row, beta)) for row in X]


def predict(X: Matrix, beta: Vector) -> list[float]:
    """Linear prediction ``X @ beta``.

    Raises ``ValueError`` if the number of columns of ``X`` does not match the
    length of ``beta``.
    """
    rows = _as_matrix(X)
    bv = [float(v) for v in beta]
    if len(rows[0]) != len(bv):
        raise ValueError(
            f"shape mismatch: X has {len(rows[0])} columns but beta has {len(bv)} entries"
        )
    return _matvec(rows, bv)


def _weighted_normal_equations(
    X: Sequence[Sequence[float]], y: Sequence[float], w: Sequence[float]
) -> list[float]:
    """Solve ``(Xᵀ W X) β = Xᵀ W y`` for diagonal weights ``w``."""
    n = len(X)
    p = len(X[0])
    xtwx = [[0.0] * p for _ in range(p)]
    xtwy = [0.0] * p
    for i in range(n):
        wi = w[i]
        row = X[i]
        wyi = wi * y[i]
        for a in range(p):
            xa = row[a]
            xtwy[a] += xa * wyi
            wxa = wi * xa
            for b in range(a, p):
                xtwx[a][b] += wxa * row[b]
    for a in range(p):
        for b in range(a):
            xtwx[a][b] = xtwx[b][a]
    return solve(xtwx, xtwy)


# --------------------------------------------------------------------------- #
# Robust scale                                                                #
# --------------------------------------------------------------------------- #
def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _mad_scale(residuals: Sequence[float]) -> float:
    """Median absolute deviation scaled to estimate the Gaussian standard deviation."""
    med = _median(residuals)
    abs_dev = [abs(r - med) for r in residuals]
    return 1.4826 * _median(abs_dev)


# --------------------------------------------------------------------------- #
# Estimators                                                                  #
# --------------------------------------------------------------------------- #
def fit_ols(X: Matrix, y: Vector) -> list[float]:
    """Ordinary least squares: ``argmin_β ||y - Xβ||²``.

    Solves the normal equations ``XᵀX β = Xᵀy``.
    """
    rows, yv = _check_xy(X, y)
    w = [1.0] * len(rows)
    return _weighted_normal_equations(rows, yv, w)


def fit_wls(X: Matrix, y: Vector, weights: Vector) -> list[float]:
    """Weighted least squares: ``argmin_β Σ wᵢ (yᵢ - xᵢᵀβ)²``.

    ``weights`` are the diagonal of ``W`` (typically ``1 / Var(εᵢ)``) and must be
    non-negative and the same length as ``y``.
    """
    rows, yv = _check_xy(X, y)
    wv = [float(v) for v in weights]
    if len(wv) != len(rows):
        raise ValueError(
            f"length mismatch: X has {len(rows)} rows but weights has {len(wv)} entries"
        )
    if any(wi < 0.0 for wi in wv):
        raise ValueError("weights must be non-negative")
    return _weighted_normal_equations(rows, yv, wv)


def fit_gls(X: Matrix, y: Vector, cov: Matrix) -> list[float]:
    """Generalized least squares for a known error covariance ``cov`` (Ω).

    Minimizes ``(y - Xβ)ᵀ Ω⁻¹ (y - Xβ)`` with closed form
    ``β = (Xᵀ Ω⁻¹ X)⁻¹ Xᵀ Ω⁻¹ y``. ``cov`` must be a symmetric positive-definite
    ``n x n`` matrix where ``n`` is the number of observations.

    The covariance inverse is applied by solving linear systems against ``Ω`` (one
    per design column plus one for ``y``), avoiding an explicit matrix inverse.
    """
    rows, yv = _check_xy(X, y)
    n = len(rows)
    omega = [[float(v) for v in r] for r in cov]
    if len(omega) != n or any(len(r) != n for r in omega):
        raise ValueError(f"cov must be {n}x{n} to match {n} observations")
    p = len(rows[0])

    # Omega^{-1} y  and  Omega^{-1} X, computed columnwise.
    oinv_y = solve(omega, yv)
    oinv_cols: list[list[float]] = []
    for j in range(p):
        col = [rows[i][j] for i in range(n)]
        oinv_cols.append(solve(omega, col))

    # XtOiX = Xᵀ (Ω⁻¹ X)  and  XtOiy = Xᵀ (Ω⁻¹ y).
    xtoix = [[0.0] * p for _ in range(p)]
    xtoiy = [0.0] * p
    for a in range(p):
        col_a = [rows[i][a] for i in range(n)]
        xtoiy[a] = sum(col_a[i] * oinv_y[i] for i in range(n))
        for b in range(p):
            xtoix[a][b] = sum(col_a[i] * oinv_cols[b][i] for i in range(n))
    return solve(xtoix, xtoiy)


class HuberResult:
    """Result of a Huber IRLS fit.

    Attributes
    ----------
    coef:
        Estimated coefficients ``β``.
    scale:
        Final robust scale estimate (MAD of residuals).
    n_iter:
        Number of IRLS iterations performed.
    converged:
        Whether the coefficient change fell below ``tol`` before ``max_iter``.
    """

    __slots__ = ("coef", "scale", "n_iter", "converged")

    def __init__(self, coef: list[float], scale: float, n_iter: int, converged: bool):
        self.coef = coef
        self.scale = scale
        self.n_iter = n_iter
        self.converged = converged

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"HuberResult(coef={self.coef}, scale={self.scale!r}, "
            f"n_iter={self.n_iter}, converged={self.converged})"
        )


def fit_huber(
    X: Matrix,
    y: Vector,
    delta: float = 1.345,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> HuberResult:
    """Robust regression with the Huber loss via iteratively reweighted least squares.

    Minimizes ``Σ ρ_δ(rᵢ / s)`` where ``ρ_δ`` is the Huber loss, ``rᵢ`` is the
    residual, and ``s`` is a robust MAD scale re-estimated each iteration. Each
    step forms Huber weights ``wᵢ = min(1, δ s / |rᵢ|)`` and performs a weighted
    least squares update, which bounds the influence of large residuals.

    Parameters
    ----------
    delta:
        Huber tuning constant in scale units. ``1.345`` gives ~95% Gaussian
        efficiency while bounding influence. Must be positive.
    max_iter:
        Maximum IRLS iterations. Must be positive.
    tol:
        Convergence threshold on the max absolute change in coefficients.

    Returns a :class:`HuberResult`. The fit is seeded with OLS.
    """
    rows, yv = _check_xy(X, y)
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if tol < 0.0:
        raise ValueError(f"tol must be non-negative, got {tol}")

    beta = _weighted_normal_equations(rows, yv, [1.0] * len(rows))
    scale = 1.0
    converged = False
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        resid = [yi - ri for yi, ri in zip(yv, _matvec(rows, beta))]
        scale = _mad_scale(resid)
        if scale <= 1e-12:
            # Residuals already (near) exact; nothing left to downweight.
            converged = True
            break
        thresh = delta * scale
        weights = [min(1.0, thresh / abs(r)) if abs(r) > 1e-30 else 1.0 for r in resid]
        new_beta = _weighted_normal_equations(rows, yv, weights)
        change = max(abs(nb - b) for nb, b in zip(new_beta, beta))
        beta = new_beta
        if change < tol:
            converged = True
            break
    return HuberResult(coef=beta, scale=scale, n_iter=n_iter, converged=converged)


def fit_quantile(
    X: Matrix,
    y: Vector,
    tau: float = 0.5,
    max_iter: int = 200,
    tol: float = 1e-10,
    eps: float = 1e-6,
) -> list[float]:
    """Quantile regression at level ``tau`` via IRLS on the pinball loss.

    Minimizes ``Σ ρ_τ(yᵢ - xᵢᵀβ)`` where ``ρ_τ(r) = r(τ - 1[r<0])``. The
    non-smooth pinball loss is approximated by a Majorize-Minimize / IRLS scheme:
    each step assigns weight ``(τ or 1-τ) / max(|rᵢ|, eps)`` depending on the sign
    of the residual and performs a weighted least squares update. ``tau=0.5``
    recovers least-absolute-deviations (the conditional median).

    Parameters
    ----------
    tau:
        Quantile level in the open interval ``(0, 1)``.
    eps:
        Smoothing floor on ``|rᵢ|`` to keep weights finite near zero residuals.
    """
    rows, yv = _check_xy(X, y)
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in the open interval (0, 1), got {tau}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    beta = _weighted_normal_equations(rows, yv, [1.0] * len(rows))
    for _ in range(max_iter):
        resid = [yi - ri for yi, ri in zip(yv, _matvec(rows, beta))]
        weights = [
            (tau if r >= 0.0 else (1.0 - tau)) / max(abs(r), eps) for r in resid
        ]
        new_beta = _weighted_normal_equations(rows, yv, weights)
        change = max(abs(nb - b) for nb, b in zip(new_beta, beta))
        beta = new_beta
        if change < tol:
            break
    return beta


def vandermonde(x: Vector, degree: int, include_bias: bool = True) -> list[list[float]]:
    """Polynomial basis expansion (Vandermonde design matrix).

    Maps each scalar ``xᵢ`` to ``[1, xᵢ, xᵢ², ..., xᵢ^degree]`` (the leading ``1``
    is dropped when ``include_bias`` is ``False``). Feeding the result into any of
    the estimators above fits a model that is nonlinear in ``x`` yet linear in the
    coefficients.

    >>> vandermonde([0.0, 2.0], 2)
    [[1.0, 0.0, 0.0], [1.0, 2.0, 4.0]]
    """
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    xs = [float(v) for v in x]
    if not xs:
        raise ValueError("input x must be non-empty")
    start = 0 if include_bias else 1
    if not include_bias and degree == 0:
        raise ValueError("degree must be >= 1 when include_bias is False")
    return [[xi ** k for k in range(start, degree + 1)] for xi in xs]
