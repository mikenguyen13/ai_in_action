"""Easy Data Augmentation (EDA) for text classification.

Reference implementation of the four EDA operations from Wei and Zou (2019),
"EDA: Easy Data Augmentation Techniques for Boosting Performance on Text
Classification Tasks" (https://arxiv.org/abs/1901.11196):

* Synonym Replacement (SR)
* Random Insertion (RI)
* Random Swap (RS)
* Random Deletion (RD)

The implementation is dependency-free (numpy is optional and unused here) and
mirrors the Julia (`AIInAction.Ch071Eda`) and Rust (`aiinaction::ch071_eda`)
versions one-to-one. Cross-language parity requires bit-identical randomness, so
this module uses an explicit 32-bit linear congruential generator (the
"minstd"/Lehmer generator with the Park-Miller constant) rather than each
language's native RNG. Every random choice is derived from that generator in a
fixed order, which makes the three implementations agree exactly on shared,
seeded fixtures.

The synonym source is a small, fixed lookup table (``SYNONYMS``) instead of
WordNet, again so that all three languages produce identical output without an
external dependency.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = [
    "SYNONYMS",
    "Lcg",
    "tokenize",
    "synonym_replacement",
    "random_insertion",
    "random_swap",
    "random_deletion",
    "eda",
]

# Park-Miller minimal standard generator constants (modulus 2**31 - 1).
_LCG_MOD = 2147483647  # 2**31 - 1, a Mersenne prime
_LCG_MULT = 16807

# Fixed synonym table shared verbatim across the three language implementations.
# Keys are lowercase words; each maps to a deterministic, ordered candidate list.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "quick": ("fast", "rapid", "swift"),
    "fast": ("quick", "rapid", "speedy"),
    "happy": ("glad", "joyful", "cheerful"),
    "sad": ("unhappy", "gloomy", "downcast"),
    "big": ("large", "huge", "massive"),
    "small": ("tiny", "little", "compact"),
    "good": ("great", "fine", "decent"),
    "bad": ("poor", "awful", "lousy"),
    "smart": ("clever", "bright", "sharp"),
    "movie": ("film", "picture", "feature"),
    "car": ("automobile", "vehicle", "auto"),
    "house": ("home", "dwelling", "residence"),
}


class Lcg:
    """A 32-bit Park-Miller linear congruential generator.

    Deterministic across Python, Julia and Rust: the same seed yields the same
    stream of values. The state is always kept in ``[1, 2**31 - 2]``.

    Parameters
    ----------
    seed:
        A non-negative integer seed. ``seed`` is reduced modulo ``2**31 - 1``;
        a reduced value of ``0`` is remapped to ``1`` because the generator is
        undefined at zero.

    Raises
    ------
    ValueError
        If ``seed`` is negative.
    """

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        state = int(seed) % _LCG_MOD
        if state == 0:
            state = 1
        self._state = state

    def next_u32(self) -> int:
        """Advance the generator and return the new 31-bit state."""
        self._state = (self._state * _LCG_MULT) % _LCG_MOD
        return self._state

    def next_float(self) -> float:
        """Return the next value mapped to the half-open interval ``[0, 1)``."""
        return (self.next_u32() - 1) / (_LCG_MOD - 1)

    def randint(self, n: int) -> int:
        """Return an integer in ``[0, n)`` via modulo reduction.

        Raises
        ------
        ValueError
            If ``n`` is not positive.
        """
        if n <= 0:
            raise ValueError(f"randint bound must be positive, got {n}")
        return self.next_u32() % n


def tokenize(text: str) -> list[str]:
    """Split text into whitespace-delimited tokens.

    Raises
    ------
    ValueError
        If ``text`` is empty or contains only whitespace.
    """
    tokens = text.split()
    if not tokens:
        raise ValueError("text must contain at least one token")
    return tokens


def _synonyms_for(word: str, table: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    return tuple(table.get(word.lower(), ()))


def _has_any_synonym(tokens: Sequence[str], table: Mapping[str, Sequence[str]]) -> bool:
    return any(_synonyms_for(t, table) for t in tokens)


def synonym_replacement(
    tokens: Sequence[str],
    n: int,
    rng: Lcg,
    table: Mapping[str, Sequence[str]] = SYNONYMS,
) -> list[str]:
    """Replace up to ``n`` words with a synonym drawn from ``table``.

    Candidate words are those with at least one synonym. For each of ``n``
    attempts a random candidate position is chosen and a random synonym
    substituted. The same position may be chosen more than once. If no token
    has a synonym, the input is returned unchanged.

    Raises
    ------
    ValueError
        If ``tokens`` is empty or ``n`` is negative.
    """
    if not tokens:
        raise ValueError("tokens must be non-empty")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    out = list(tokens)
    candidate_idx = [i for i, t in enumerate(out) if _synonyms_for(t, table)]
    if not candidate_idx:
        return out
    for _ in range(n):
        pos = candidate_idx[rng.randint(len(candidate_idx))]
        cands = _synonyms_for(out[pos], table)
        out[pos] = cands[rng.randint(len(cands))]
    return out


def random_insertion(
    tokens: Sequence[str],
    n: int,
    rng: Lcg,
    table: Mapping[str, Sequence[str]] = SYNONYMS,
) -> list[str]:
    """Insert ``n`` synonyms of random words at random positions.

    Each insertion picks a random word that has a synonym, picks a random
    synonym of it, and inserts that synonym at a random position in the growing
    sequence. If no token has a synonym, the input is returned unchanged.

    Raises
    ------
    ValueError
        If ``tokens`` is empty or ``n`` is negative.
    """
    if not tokens:
        raise ValueError("tokens must be non-empty")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    out = list(tokens)
    if not _has_any_synonym(out, table):
        return out
    for _ in range(n):
        # Draw a word with a synonym (bounded retries keep this deterministic).
        word_synonyms: tuple[str, ...] = ()
        for _attempt in range(10):
            cand = out[rng.randint(len(out))]
            word_synonyms = _synonyms_for(cand, table)
            if word_synonyms:
                break
        if not word_synonyms:
            continue
        new_word = word_synonyms[rng.randint(len(word_synonyms))]
        insert_at = rng.randint(len(out) + 1)
        out.insert(insert_at, new_word)
    return out


def random_swap(tokens: Sequence[str], n: int, rng: Lcg) -> list[str]:
    """Swap two random tokens ``n`` times.

    Raises
    ------
    ValueError
        If ``tokens`` is empty or ``n`` is negative.
    """
    if not tokens:
        raise ValueError("tokens must be non-empty")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    out = list(tokens)
    if len(out) < 2:
        return out
    for _ in range(n):
        i = rng.randint(len(out))
        j = rng.randint(len(out))
        out[i], out[j] = out[j], out[i]
    return out


def random_deletion(tokens: Sequence[str], p: float, rng: Lcg) -> list[str]:
    """Delete each token independently with probability ``p``.

    If every token would be deleted, a single random token is kept so the output
    is never empty (matching the original EDA reference behaviour).

    Raises
    ------
    ValueError
        If ``tokens`` is empty or ``p`` is outside ``[0, 1]``.
    """
    if not tokens:
        raise ValueError("tokens must be non-empty")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    out = list(tokens)
    if len(out) == 1:
        return out
    kept: list[str] = []
    for t in out:
        if rng.next_float() >= p:
            kept.append(t)
    if not kept:
        kept = [out[rng.randint(len(out))]]
    return kept


def eda(
    text: str,
    *,
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    p_rd: float = 0.1,
    num_aug: int = 4,
    seed: int = 0,
    table: Mapping[str, Sequence[str]] = SYNONYMS,
) -> list[str]:
    """Generate ``num_aug`` augmented sentences from ``text``.

    The four operations are applied in a fixed round-robin (SR, RI, RS, RD) and
    the per-sentence operation count ``n`` is ``max(1, round(alpha * n_words))``,
    following the EDA paper. A single :class:`Lcg` seeded with ``seed`` drives
    every random decision, so the output is fully reproducible and identical
    across the Python, Julia and Rust implementations.

    Parameters
    ----------
    text:
        Input sentence (whitespace tokenised).
    alpha_sr, alpha_ri, alpha_rs:
        Fraction of words touched by SR / RI / RS respectively.
    p_rd:
        Per-word deletion probability for RD.
    num_aug:
        Number of augmented sentences to produce.
    seed:
        Seed for the deterministic generator.
    table:
        Synonym table; defaults to :data:`SYNONYMS`.

    Returns
    -------
    list[str]
        ``num_aug`` augmented sentences as joined strings.

    Raises
    ------
    ValueError
        If ``text`` is empty, ``num_aug`` is negative, any ``alpha`` is outside
        ``[0, 1]``, or ``p_rd`` is outside ``[0, 1]``.
    """
    if num_aug < 0:
        raise ValueError(f"num_aug must be non-negative, got {num_aug}")
    for name, a in (("alpha_sr", alpha_sr), ("alpha_ri", alpha_ri), ("alpha_rs", alpha_rs)):
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {a}")
    if not 0.0 <= p_rd <= 1.0:
        raise ValueError(f"p_rd must be in [0, 1], got {p_rd}")
    tokens = tokenize(text)
    n_words = len(tokens)
    rng = Lcg(seed)
    out: list[str] = []
    for i in range(num_aug):
        op = i % 4
        if op == 0:
            n = max(1, round(alpha_sr * n_words))
            aug = synonym_replacement(tokens, n, rng, table)
        elif op == 1:
            n = max(1, round(alpha_ri * n_words))
            aug = random_insertion(tokens, n, rng, table)
        elif op == 2:
            n = max(1, round(alpha_rs * n_words))
            aug = random_swap(tokens, n, rng)
        else:
            aug = random_deletion(tokens, p_rd, rng)
        out.append(" ".join(aug))
    return out
