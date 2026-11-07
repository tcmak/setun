"""Unit tests for Task 1.3a internal trit format pack/unpack."""

from __future__ import annotations

import os

import numpy as np
import pytest

from setun.ref.format import (
    deserialize_packed_weights,
    pack_ternary_tensor,
    serialize_packed_weights,
    unpack_ternary_tensor,
    unpack_trits_1d,
)


def test_format_md_worked_example_byte_exact() -> None:
    """Asserts the exact 42-byte worked example in docs/FORMAT.md."""
    w = np.array([[1, -1, 0, 1, -1, -1]], dtype=np.int8)
    s = np.array([[0x0180]], dtype=np.uint16)
    packed = pack_ternary_tensor(w, s, group_size=64, shift=8)
    blob = serialize_packed_weights(packed)

    expected_header = bytes.fromhex(
        "53544e57"  # "STNW"
        "01"  # version = 1
        "08"  # shift = 8
        "4000"  # G = 64
        "01000000"  # N = 1
        "06000000"  # K = 6
        "01000000"  # n_groups = 1
        "00000000"  # reserved = 0
    )
    expected_trits = bytes.fromhex("490a" + "00" * 14)
    expected_scales = bytes.fromhex("8001")

    assert blob == expected_header + expected_trits + expected_scales

    restored = deserialize_packed_weights(blob)
    w_out, s_out, shift_out = unpack_ternary_tensor(restored)
    assert np.array_equal(w_out, w)
    assert np.array_equal(s_out, s)
    assert shift_out == 8


def test_reserved_trit_code_rejected_on_unpack() -> None:
    bad_byte = np.array([0b00_00_11_01], dtype=np.uint8)
    with pytest.raises(ValueError, match="reserved trit code 0b11"):
        unpack_trits_1d(bad_byte)


@pytest.mark.parametrize("group_size", [64, 128, 256])
@pytest.mark.parametrize("k", [1, 7, 63, 64, 65, 127, 128, 129, 512])
def test_ragged_and_aligned_shapes_roundtrip(group_size: int, k: int) -> None:
    rng = np.random.default_rng(1958 + group_size + k)
    n = 5
    n_groups = (k + group_size - 1) // group_size
    w = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n, k))
    s = rng.integers(0, 65536, size=(n, n_groups), dtype=np.uint16)
    packed = pack_ternary_tensor(w, s, group_size=group_size, shift=11)
    w2, s2, sh2 = unpack_ternary_tensor(
        deserialize_packed_weights(serialize_packed_weights(packed))
    )
    assert np.array_equal(w, w2)
    assert np.array_equal(s, s2)
    assert sh2 == 11


def test_random_tensors_roundtrip_sweep() -> None:
    """Runs 10^4 random tensors in CI (or 10^6 when SETUN_NIGHTLY=1)."""
    n_trials = 1_000_000 if os.environ.get("SETUN_NIGHTLY") == "1" else 10_000
    rng = np.random.default_rng(20261017)

    for i in range(n_trials):
        g = (64, 128, 256)[i % 3]
        n = int(rng.integers(1, 5))
        k = int(rng.integers(1, 260))
        n_groups = (k + g - 1) // g
        w = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n, k))
        s = rng.integers(0, 65536, size=(n, n_groups), dtype=np.uint16)
        packed = pack_ternary_tensor(w, s, group_size=g, shift=i & 0x1F)
        w_out, s_out, sh_out = unpack_ternary_tensor(packed)
        if not (np.array_equal(w, w_out) and np.array_equal(s, s_out) and sh_out == (i & 0x1F)):
            pytest.fail(f"Round-trip mismatch on trial {i} (N={n}, K={k}, G={g})")
