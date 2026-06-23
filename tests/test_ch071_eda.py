"""Tests for aiinaction.ch071_eda, including the shared cross-language fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same LCG stream, the same per-operation outputs, and the same
``eda`` results, which is what keeps the three implementations at parity.
"""
from __future__ import annotations

import pytest

from aiinaction.ch071_eda import (
    Lcg,
    eda,
    random_deletion,
    random_insertion,
    random_swap,
    synonym_replacement,
    tokenize,
)

# --- Shared LCG fixtures (mirrored in Julia/Rust) ---------------------------
LCG_SEED = 42
LCG_U32 = [705894, 1126542223, 1579310009, 565444343, 807934826]
LCG_FLOAT = [0.000328707, 0.5245871018, 0.7354235321]
RANDINT_SEED = 7
RANDINT_10 = [9, 3, 6, 5, 9, 7]

# --- Shared text fixtures ---------------------------------------------------
SENTENCE = "the quick movie was good and fast"
TOKENS = ["the", "quick", "movie", "was", "good", "and", "fast"]

SR_EXPECTED = ["the", "quick", "feature", "was", "good", "and", "rapid"]
RI_EXPECTED = ["fine", "great", "the", "quick", "movie", "was", "good", "and", "fast"]
RS_EXPECTED = ["the", "quick", "movie", "good", "was", "and", "fast"]
RD_EXPECTED = ["quick", "was", "and"]

EDA_EXPECTED = [
    "the quick feature was good and fast",
    "the quick movie was good rapid and fast",
    "the quick movie was good and fast",
    "the quick movie was good and fast",
]

EDA_ALPHA_SENTENCE = "a sleek and surprisingly fast car"
EDA_ALPHA_EXPECTED = [
    "a sleek and surprisingly fast auto",
    "auto a sleek and surprisingly fast car",
    "car sleek and surprisingly fast a",
    "a sleek and surprisingly fast car",
    "a sleek and surprisingly fast automobile",
    "a sleek and surprisingly fast car speedy",
]


# --- LCG --------------------------------------------------------------------
def test_lcg_u32_stream():
    rng = Lcg(LCG_SEED)
    assert [rng.next_u32() for _ in range(5)] == LCG_U32


def test_lcg_float_stream():
    rng = Lcg(LCG_SEED)
    got = [rng.next_float() for _ in range(3)]
    for g, e in zip(got, LCG_FLOAT):
        assert g == pytest.approx(e, abs=1e-9)


def test_lcg_randint_stream():
    rng = Lcg(RANDINT_SEED)
    assert [rng.randint(10) for _ in range(6)] == RANDINT_10


def test_lcg_seed_zero_remaps():
    # seed 0 must not lock the generator at zero forever
    assert Lcg(0).next_u32() != 0


def test_lcg_negative_seed_raises():
    with pytest.raises(ValueError, match="non-negative"):
        Lcg(-1)


def test_lcg_randint_nonpositive_raises():
    with pytest.raises(ValueError, match="positive"):
        Lcg(1).randint(0)


# --- Tokenize ---------------------------------------------------------------
def test_tokenize_basic():
    assert tokenize(SENTENCE) == TOKENS


def test_tokenize_empty_raises():
    with pytest.raises(ValueError, match="at least one token"):
        tokenize("   ")


# --- Individual operations --------------------------------------------------
def test_synonym_replacement_fixture():
    assert synonym_replacement(TOKENS, 2, Lcg(1)) == SR_EXPECTED


def test_random_insertion_fixture():
    assert random_insertion(TOKENS, 2, Lcg(2)) == RI_EXPECTED


def test_random_swap_fixture():
    assert random_swap(TOKENS, 2, Lcg(3)) == RS_EXPECTED


def test_random_deletion_fixture():
    assert random_deletion(TOKENS, 0.3, Lcg(4)) == RD_EXPECTED


def test_synonym_replacement_no_candidates_identity():
    toks = ["xx", "yy", "zz"]
    assert synonym_replacement(toks, 3, Lcg(5)) == toks


def test_random_swap_single_token_identity():
    assert random_swap(["solo"], 5, Lcg(6)) == ["solo"]


def test_random_deletion_single_token_identity():
    assert random_deletion(["solo"], 1.0, Lcg(6)) == ["solo"]


def test_random_deletion_never_empty():
    # p = 1.0 would delete everything; one token must survive
    out = random_deletion(TOKENS, 1.0, Lcg(8))
    assert len(out) == 1


# --- eda orchestrator -------------------------------------------------------
def test_eda_fixture():
    assert eda(SENTENCE, seed=123, num_aug=4) == EDA_EXPECTED


def test_eda_alpha_fixture():
    out = eda(
        EDA_ALPHA_SENTENCE,
        seed=99,
        num_aug=6,
        alpha_sr=0.2,
        alpha_ri=0.2,
        alpha_rs=0.2,
        p_rd=0.2,
    )
    assert out == EDA_ALPHA_EXPECTED


def test_eda_reproducible():
    a = eda(SENTENCE, seed=7, num_aug=5)
    b = eda(SENTENCE, seed=7, num_aug=5)
    assert a == b


def test_eda_count():
    assert len(eda(SENTENCE, seed=1, num_aug=10)) == 10
    assert eda(SENTENCE, seed=1, num_aug=0) == []


# --- Validation -------------------------------------------------------------
def test_eda_empty_text_raises():
    with pytest.raises(ValueError, match="at least one token"):
        eda("   ", seed=1)


def test_eda_negative_num_aug_raises():
    with pytest.raises(ValueError, match="num_aug"):
        eda(SENTENCE, num_aug=-1)


def test_eda_bad_alpha_raises():
    with pytest.raises(ValueError, match="alpha_sr"):
        eda(SENTENCE, alpha_sr=1.5)


def test_eda_bad_prd_raises():
    with pytest.raises(ValueError, match="p_rd"):
        eda(SENTENCE, p_rd=-0.1)


@pytest.mark.parametrize(
    "fn", [synonym_replacement, random_insertion]
)
def test_ops_empty_tokens_raise(fn):
    with pytest.raises(ValueError, match="non-empty"):
        fn([], 1, Lcg(1))


def test_random_swap_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        random_swap([], 1, Lcg(1))


def test_random_deletion_bad_p_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        random_deletion(TOKENS, 2.0, Lcg(1))
