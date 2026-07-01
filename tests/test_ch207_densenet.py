"""Tests for aiinaction.ch207_densenet, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which is what keeps the three implementations at
parity. The channel/parameter arithmetic is exact integer math and needs no
randomness; the toy forward pass draws layer weights from the same 64-bit LCG
used elsewhere in this book, so the same seed produces the same weights (and
hence the same forward pass) in every language.
"""
from __future__ import annotations

import pytest

from aiinaction import ch207_densenet as dn

X0 = [1.0, 0.6, -0.3, 0.9]
EXPECTED_SIZES = [4, 7, 10, 13]
EXPECTED_OUT = [
    1.0, 0.6, -0.3, 0.9,
    0.6279286728409486, 0.11130697455664373, 0.08343734603269459,
    0.0, 0.9451360437126258, 1.4047640097915077,
    0.0, 0.0, 0.8720018399583354,
]


def test_lcg_uniform_sequence_matches_fixture():
    rng = dn.Lcg(0)
    got = [rng.next_uniform() for _ in range(4)]
    assert got == pytest.approx(
        [0.07820865487829387, 0.10169876029679303, 0.60532332262523347, 0.40121620369530075],
        abs=1e-12,
    )


def test_dense_block_channel_sizes():
    assert dn.dense_block_channel_sizes(c0=4, growth_rate=4, num_layers=3) == [4, 8, 12, 16]
    assert dn.dense_block_channel_sizes(c0=4, growth_rate=3, num_layers=3) == EXPECTED_SIZES


def test_dense_block_channel_sizes_rejects_invalid():
    with pytest.raises(ValueError, match="c0 must be >= 0"):
        dn.dense_block_channel_sizes(-1, 4, 3)
    with pytest.raises(ValueError, match="growth_rate must be >= 1"):
        dn.dense_block_channel_sizes(4, 0, 3)
    with pytest.raises(ValueError, match="num_layers must be >= 1"):
        dn.dense_block_channel_sizes(4, 4, 0)


def test_dense_block_param_count_matches_fixture():
    assert dn.dense_block_param_count(4, 4, 3, bn_size=4) == 2112
    assert dn.dense_block_param_count(8, 12, 4, bn_size=4) == 25728
    assert dn.dense_block_param_count(64, 32, 6, bn_size=4) == 331776


def test_transition_output_channels_matches_fixture():
    assert dn.transition_output_channels(64, 0.5) == 32
    assert dn.transition_output_channels(128, 0.5) == 64


def test_transition_output_channels_rejects_invalid():
    with pytest.raises(ValueError, match="theta must satisfy"):
        dn.transition_output_channels(64, 0.0)
    with pytest.raises(ValueError, match="theta must satisfy"):
        dn.transition_output_channels(64, 1.5)


def test_plain_block_param_count_matches_fixture():
    assert dn.plain_block_param_count(16, 16, 3) == 6912
    assert dn.plain_block_param_count(64, 64, 6) == 221184


def test_dense_connectivity_is_more_parameter_efficient_than_plain():
    # Same total output width; dense connectivity uses far fewer parameters
    # than a plain stack of matched width -- the claim behind Section 4.2.
    dense = dn.dense_block_param_count(16, 12, 8, bn_size=4)
    plain = dn.plain_block_param_count(16, 16 + 12 * 8, 8)
    assert dense < plain


def test_dense_block_forward_matches_fixture():
    out, sizes = dn.dense_block_forward(X0, growth_rate=3, num_layers=3, seed=2)
    assert sizes == EXPECTED_SIZES
    assert out.tolist() == pytest.approx(EXPECTED_OUT, abs=1e-9)


def test_dense_block_forward_channel_trace_matches_theory():
    # The observed feature-vector length at every step must equal the
    # theoretical channel-size trace, for any growth rate / depth / seed.
    out, sizes = dn.dense_block_forward([1.0, -1.0, 0.5, 0.5], growth_rate=2, num_layers=2, seed=0)
    assert sizes == dn.dense_block_channel_sizes(4, 2, 2)
    assert len(out) == sizes[-1]


def test_dense_block_forward_rejects_invalid():
    with pytest.raises(ValueError, match="non-empty"):
        dn.dense_block_forward([], growth_rate=2, num_layers=2, seed=0)
    with pytest.raises(ValueError, match="non-finite"):
        dn.dense_block_forward([1.0, float("nan")], growth_rate=2, num_layers=2, seed=0)


def test_densenet_variant_param_totals_match_fixture():
    assert dn.densenet_dense_param_total("121") == 6_860_800
    assert dn.densenet_dense_param_total("169") == 12_316_672
    assert dn.densenet_dense_param_total("201") == 17_854_464
    assert dn.densenet_dense_param_total("264") == 30_240_768


def test_densenet_variant_param_totals_increase_with_depth():
    totals = [dn.densenet_dense_param_total(v) for v in ("121", "169", "201", "264")]
    assert totals == sorted(totals)


def test_densenet_dense_param_total_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unknown variant"):
        dn.densenet_dense_param_total("bogus")
