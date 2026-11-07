"""Golden integer reference operations for Season 1 (Task 1.4a)."""

from __future__ import annotations

import math

import numpy as np


def dot_bit_width(lanes: int, act_bits: int = 8) -> int:
    """Return signed bit width for an L-lane ternary dot product: ceil(log2(L * A_max + 1)) + 1."""
    a_max = 1 << (act_bits - 1)  # 128 for int8 (since -(-128) = +128)
    return math.ceil(math.log2(lanes * a_max + 1)) + 1


def group_sum_bit_width(group_size: int, act_bits: int = 8) -> int:
    """Return signed bit width for a group partial sum: ceil(log2(G * A_max + 1)) + 1."""
    a_max = 1 << (act_bits - 1)
    return math.ceil(math.log2(group_size * a_max + 1)) + 1


def lane_select_code(code: int, x: int) -> int:
    """Single-lane combinational reference: 2-bit code {00:0, 01:+1, 10:-1, 11:0} x signed int8."""
    if not (-128 <= x <= 127):
        raise ValueError(f"Activation {x} out of int8 range [-128, 127].")
    if code == 0b01:
        return int(x)
    if code == 0b10:
        return -int(x)
    # 0b00 and reserved 0b11 both evaluate to 0 in hardware per ADR-003
    return 0


def round_shift_right(val: int | np.ndarray, shift: int) -> int | np.ndarray:
    """Hardware-friendly round-half-up arithmetic right shift."""
    if shift < 0:
        raise ValueError(f"shift must be >= 0, got {shift}.")
    if shift == 0:
        return val
    bias = 1 << (shift - 1)
    return (val + bias) >> shift


def ternary_dot(trits: np.ndarray, activations: np.ndarray) -> int:
    """Compute a single L-lane ternary dot product in exact integer arithmetic."""
    w = np.asarray(trits, dtype=np.int64)
    x = np.asarray(activations, dtype=np.int64)
    if w.ndim != 1 or x.ndim != 1 or w.shape != x.shape:
        raise ValueError(f"Expected matching 1D vectors, got {w.shape} and {x.shape}.")
    if not np.isin(w, (-1, 0, 1)).all():
        raise ValueError("Trits must be in {-1, 0, +1}.")
    if np.any((x < -128) | (x > 127)):
        raise ValueError("Activations must fit in signed int8 [-128, 127].")
    return int(np.sum(w * x))


def group_accumulate(
    trits_group: np.ndarray,
    activations_group: np.ndarray,
    lanes: int | None = None,
) -> int:
    """Accumulate G/L beats of ternary_dot into a single group partial sum."""
    w = np.asarray(trits_group, dtype=np.int64)
    x = np.asarray(activations_group, dtype=np.int64)
    if w.ndim != 1 or x.ndim != 1 or w.shape != x.shape:
        raise ValueError(f"Expected matching 1D group vectors, got {w.shape} and {x.shape}.")
    g = w.shape[0]
    if lanes is None:
        lanes = g
    if g % lanes != 0:
        raise ValueError(f"Group size G={g} must be divisible by lanes L={lanes}.")

    partial = 0
    for b in range(0, g, lanes):
        partial += ternary_dot(w[b : b + lanes], x[b : b + lanes])
    return int(partial)


def apply_scale(group_partial_sum: int, scale_u16: int, shift: int = 0) -> int:
    """Multiply a signed group partial sum by a uint16 scale and optionally right-shift."""
    if not (0 <= scale_u16 <= 0xFFFF):
        raise ValueError(f"scale_u16 must be in [0, 65535], got {scale_u16}.")
    product = int(group_partial_sum) * int(scale_u16)
    return int(round_shift_right(product, shift))


def row_dot(
    row_trits: np.ndarray,
    activations: np.ndarray,
    row_scales: np.ndarray,
    group_size: int = 128,
    lanes: int = 64,
    shift: int = 0,
) -> int:
    """Compute one full output row of a grouped ternary GEMV (supports ragged K)."""
    w = np.asarray(row_trits, dtype=np.int64)
    x = np.asarray(activations, dtype=np.int64)
    s = np.asarray(row_scales, dtype=np.int64)

    if w.ndim != 1 or x.ndim != 1 or w.shape != x.shape:
        raise ValueError(f"Mismatched row/activation shapes: {w.shape} vs {x.shape}.")
    if group_size % lanes != 0:
        raise ValueError(f"group_size ({group_size}) must be a multiple of lanes ({lanes}).")

    k = w.shape[0]
    n_groups = (k + group_size - 1) // group_size
    if s.shape != (n_groups,):
        raise ValueError(
            f"Expected {n_groups} group scales for K={k}, G={group_size}, got {s.shape}."
        )
    if np.any((s < 0) | (s > 0xFFFF)):
        raise ValueError("Group scales must fit in uint16 [0, 65535].")

    k_pad = n_groups * group_size
    if k_pad != k:
        w_pad = np.zeros(k_pad, dtype=np.int64)
        x_pad = np.zeros(k_pad, dtype=np.int64)
        w_pad[:k] = w
        x_pad[:k] = x
    else:
        w_pad = w
        x_pad = x

    row_acc = 0
    for g_idx in range(n_groups):
        start = g_idx * group_size
        end = start + group_size
        p_g = group_accumulate(w_pad[start:end], x_pad[start:end], lanes=lanes)
        row_acc += apply_scale(p_g, int(s[g_idx]), shift=0)

    return int(round_shift_right(row_acc, shift))
