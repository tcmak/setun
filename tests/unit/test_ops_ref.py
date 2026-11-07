"""Unit tests for Task 1.4a reference ops and NUMERICS.md worked example."""

from __future__ import annotations

import numpy as np

from setun.ref.ops import (
    apply_scale,
    dot_bit_width,
    group_accumulate,
    group_sum_bit_width,
    lane_select_code,
    row_dot,
    ternary_dot,
)


def test_numerics_md_bit_width_formulas() -> None:
    assert dot_bit_width(lanes=4, act_bits=8) == 11
    assert dot_bit_width(lanes=16, act_bits=8) == 13
    assert dot_bit_width(lanes=64, act_bits=8) == 15
    assert group_sum_bit_width(group_size=64, act_bits=8) == 15
    assert group_sum_bit_width(group_size=128, act_bits=8) == 16
    assert group_sum_bit_width(group_size=256, act_bits=8) == 17


def test_numerics_md_worked_example() -> None:
    """Asserts the hand-computed worked example in docs/NUMERICS.md Section 3."""
    w = np.array([1, -1, 0, 1, -1, -1], dtype=np.int8)
    x = np.array([-128, 50, 99, 30, -20, 10], dtype=np.int8)
    scales = np.array([300], dtype=np.uint16)

    # Beat 0 and Beat 1 check
    assert ternary_dot(w[:4], x[:4]) == -148
    assert ternary_dot(np.array([-1, -1, 0, 0]), np.array([-20, 10, 0, 0])) == 10

    # Group accumulate and scale check
    p0 = group_accumulate(
        np.array([1, -1, 0, 1, -1, -1, 0, 0]),
        np.array([-128, 50, 99, 30, -20, 10, 0, 0]),
        lanes=4,
    )
    assert p0 == -138
    assert apply_scale(p0, 300, shift=0) == -41400
    assert apply_scale(p0, 300, shift=4) == -2587

    # Full ragged row_dot check (K=6, G=8, L=4)
    out = row_dot(w, x, scales, group_size=8, lanes=4, shift=4)
    assert out == -2587


def test_zero_vector_and_zero_scale() -> None:
    w = np.zeros(128, dtype=np.int8)
    x = np.full(128, -128, dtype=np.int8)
    assert ternary_dot(w, x) == 0
    assert row_dot(w, x, np.array([65535], dtype=np.uint16), group_size=128, lanes=64) == 0
    assert row_dot(np.ones(128, dtype=np.int8), x, np.array([0], dtype=np.uint16)) == 0


def test_minus_128_and_all_max_extremes() -> None:
    """Verifies -128 negation (+128) does not overflow 16-bit group sum at G=128."""
    assert lane_select_code(0b10, -128) == 128
    assert lane_select_code(0b01, -128) == -128
    assert lane_select_code(0b11, -128) == 0

    w_neg = np.full(128, -1, dtype=np.int8)
    w_pos = np.full(128, 1, dtype=np.int8)
    x_min = np.full(128, -128, dtype=np.int8)

    max_group = group_accumulate(w_neg, x_min, lanes=64)
    min_group = group_accumulate(w_pos, x_min, lanes=64)
    assert max_group == 16384  # Fits in signed 16-bit [-32768, 32767]
    assert min_group == -16384

    # Max scale 65535 * 16384 = 1,073,725,440 (fits in signed 32-bit product)
    assert apply_scale(max_group, 65535, shift=0) == 1_073_725_440


def test_ragged_k_padding_invariance() -> None:
    rng = np.random.default_rng(99)
    for k in [1, 7, 127, 129, 300]:
        g = 128
        n_groups = (k + g - 1) // g
        w = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=k)
        x = rng.integers(-128, 128, size=k, dtype=np.int8)
        s = rng.integers(1, 1000, size=n_groups, dtype=np.uint16)

        res_ragged = row_dot(w, x, s, group_size=g, lanes=64, shift=8)

        # Compare against explicitly zero-padded K_pad = n_groups * G
        w_pad = np.pad(w, (0, n_groups * g - k))
        x_pad = np.pad(x, (0, n_groups * g - k))
        res_padded = row_dot(w_pad, x_pad, s, group_size=g, lanes=64, shift=8)
        assert res_ragged == res_padded
