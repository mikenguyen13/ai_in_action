"""McNemar's test for comparing two classifiers on a shared test set.

A small, well-validated reference implementation of McNemar's paired test for the
disagreement between two classifiers. The public API mirrors the Julia
(``AIInAction.Ch165Mcnemar``) and Rust (``aiinaction::ch165_mcnemar``)
implementations one-to-one; the cross-language parity tests assert that all three
agree to within floating-point tolerance on shared fixtures.

Given two trained classifiers evaluated on the same held-out examples, build the
2x2 contingency table of correctness::

                       B correct   B wrong
        A correct          a           b
        A wrong            c           d

The concordant cells ``a`` and ``d`` carry no information about which model is
better. Only the discordant cells matter: ``b`` counts examples ``A`` got right
and ``B`` got wrong, and ``c`` the reverse. Under the null hypothesis that the two
models have equal error rates, a discordant example is equally likely to fall in
``b`` or ``c``. Conditioned on ``n = b + c``, the count ``b`` is therefore
``Binomial(n, 1/2)``.

Two test variants are provided:

* The chi-squared approximation with Edwards' continuity correction,
  ``chi2 = (|b - c| - 1)^2 / (b + c)``, referred to a chi-squared distribution
  with one degree of freedom. Reliable when ``b + c`` is reasonably large.
* The exact two-sided binomial test on ``min(b, c)`` against ``Binomial(b+c, 1/2)``,
  preferred when ``b + c`` is small (a common rule of thumb is below 25).

When ``exact`` is left as ``None`` the implementation chooses the exact test
whenever ``b + c < 25`` and the chi-squared approximation otherwise.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "ContingencyTable",
    "McNemarResult",
    "contingency_table",
    "mcnemar_test",
]


@dataclass(frozen=True)
class ContingencyTable:
    """The 2x2 correctness contingency table for two classifiers.

    Attributes
    ----------
    a:
        Count where both classifiers are correct.
    b:
        Count where ``A`` is correct and ``B`` is wrong (discordant).
    c:
        Count where ``A`` is wrong and ``B`` is correct (discordant).
    d:
        Count where both classifiers are wrong.
    """

    a: int
    b: int
    c: int
    d: int

    @property
    def n(self) -> int:
        """Total number of examples in the table."""
        return self.a + self.b + self.c + self.d

    @property
    def n_discordant(self) -> int:
        """Number of discordant examples ``b + c``."""
        return self.b + self.c


@dataclass(frozen=True)
class McNemarResult:
    """The outcome of McNemar's test.

    Attributes
    ----------
    statistic:
        The chi-squared statistic ``(|b - c| - correction)^2 / (b + c)`` for the
        chi-squared variant, or ``float(min(b, c))`` for the exact variant.
    p_value:
        Two-sided p-value under ``H_0`` of equal error rates.
    method:
        Either ``"chi2"`` or ``"exact"``.
    b:
        Discordant count favouring ``A``.
    c:
        Discordant count favouring ``B``.
    """

    statistic: float
    p_value: float
    method: str
    b: int
    c: int


def _check_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def contingency_table(
    correct_a: Sequence[bool], correct_b: Sequence[bool]
) -> ContingencyTable:
    """Build the 2x2 correctness table from two per-example correctness vectors.

    Parameters
    ----------
    correct_a, correct_b:
        Equal-length sequences of booleans (or 0/1 values). Entry ``i`` is truthy
        iff the corresponding classifier predicted example ``i`` correctly.

    Returns
    -------
    ContingencyTable

    Raises
    ------
    ValueError
        If the inputs differ in length or are empty.

    Examples
    --------
    >>> t = contingency_table([1, 1, 0, 0], [1, 0, 1, 0])
    >>> (t.a, t.b, t.c, t.d)
    (1, 1, 1, 1)
    """
    a_list = [bool(v) for v in correct_a]
    b_list = [bool(v) for v in correct_b]
    if len(a_list) != len(b_list):
        raise ValueError(
            f"length mismatch: len(correct_a)={len(a_list)} != len(correct_b)={len(b_list)}"
        )
    if not a_list:
        raise ValueError("inputs must be non-empty")

    a = b = c = d = 0
    for ca, cb in zip(a_list, b_list):
        if ca and cb:
            a += 1
        elif ca and not cb:
            b += 1
        elif not ca and cb:
            c += 1
        else:
            d += 1
    return ContingencyTable(a=a, b=b, c=c, d=d)


def _chi2_sf_1dof(x: float) -> float:
    """Survival function of the chi-squared distribution with one degree of freedom.

    For one degree of freedom, ``P(X > x) = erfc(sqrt(x / 2))`` exactly, which keeps
    this dependency-free and identical across languages.
    """
    if x <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def _binom_pmf_half(k: int, n: int) -> float:
    """Probability mass ``C(n, k) (1/2)^n`` evaluated in log space for stability."""
    log_coef = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.exp(log_coef - n * math.log(2.0))


def _exact_two_sided_p(b: int, c: int) -> float:
    """Two-sided exact binomial p-value for ``min(b, c)`` under ``Binomial(b+c, 1/2)``.

    Because the null binomial is symmetric about ``n/2``, the two-sided p-value is
    twice the lower tail ``P(X <= min(b, c))``, capped at one.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    lower_tail = sum(_binom_pmf_half(i, n) for i in range(0, k + 1))
    return min(1.0, 2.0 * lower_tail)


def mcnemar_test(
    b: int,
    c: int,
    *,
    exact: bool | None = None,
    correction: bool = True,
) -> McNemarResult:
    """Run McNemar's test on the discordant counts ``b`` and ``c``.

    Parameters
    ----------
    b:
        Number of examples the first classifier got right and the second wrong.
    c:
        Number of examples the first classifier got wrong and the second right.
    exact:
        Force the exact binomial test (``True``) or the chi-squared approximation
        (``False``). When ``None`` (the default), the exact test is used whenever
        ``b + c < 25`` and the chi-squared approximation otherwise.
    correction:
        Whether to apply Edwards' continuity correction to the chi-squared
        statistic. Ignored by the exact variant.

    Returns
    -------
    McNemarResult

    Raises
    ------
    ValueError
        If ``b`` or ``c`` is negative or non-integer, or if both are zero (the test
        is undefined when there are no discordant pairs).

    Examples
    --------
    >>> r = mcnemar_test(30, 15, exact=False)
    >>> round(r.statistic, 6)
    4.355556
    >>> round(r.p_value, 6)
    0.036888
    """
    b = _check_count("b", b)
    c = _check_count("c", c)
    n = b + c
    if n == 0:
        raise ValueError("McNemar's test is undefined when b + c = 0 (no discordant pairs)")

    use_exact = (n < 25) if exact is None else bool(exact)

    if use_exact:
        p = _exact_two_sided_p(b, c)
        return McNemarResult(
            statistic=float(min(b, c)), p_value=p, method="exact", b=b, c=c
        )

    delta = abs(b - c)
    if correction:
        delta = max(0.0, delta - 1.0)
    chi2 = (delta * delta) / n
    p = _chi2_sf_1dof(chi2)
    return McNemarResult(statistic=chi2, p_value=p, method="chi2", b=b, c=c)
