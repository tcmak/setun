"""Internal ternary weight format pack/unpack and binary codec (Task 1.3a, ADR-003, ADR-011)."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

VALID_GROUP_SIZES = (64, 128, 256)
HEADER_STRUCT = struct.Struct("<4sBBHIIII")
MAGIC = b"STNW"
FORMAT_VERSION = 1

# 2-bit trit codes per ADR-003: {00: 0, 01: +1, 10: -1, 11: reserved}
CODE_ZERO = 0b00
CODE_POS = 0b01
CODE_NEG = 0b10
CODE_RESERVED = 0b11


@dataclass(frozen=True)
class PackedTernaryTensor:
    """In-memory container for a packed grouped ternary matrix."""

    packed_trits: np.ndarray  # uint8, shape (N, n_groups, G // 4)
    scales: np.ndarray  # uint16, shape (N, n_groups)
    n_rows: int  # logical N
    n_cols: int  # logical K (may not be a multiple of group_size)
    group_size: int  # G in {64, 128, 256}
    shift: int = 0  # shared right-shift exponent (uint8)


def pack_trits_1d(trits: np.ndarray) -> np.ndarray:
    """Pack a 1D or ND array of trits in {-1, 0, +1} (last dim multiple of 4) LSB-first."""
    if not np.isin(trits, (-1, 0, 1)).all():
        raise ValueError("Ternary weights must be strictly in {-1, 0, +1}.")
    if trits.shape[-1] % 4 != 0:
        raise ValueError(f"Last dimension ({trits.shape[-1]}) must be a multiple of 4.")

    codes = np.zeros(trits.shape, dtype=np.uint8)
    codes[trits == 1] = CODE_POS
    codes[trits == -1] = CODE_NEG

    c0 = codes[..., 0::4]
    c1 = codes[..., 1::4] << 2
    c2 = codes[..., 2::4] << 4
    c3 = codes[..., 3::4] << 6
    return (c0 | c1 | c2 | c3).astype(np.uint8)


def unpack_trits_1d(packed: np.ndarray) -> np.ndarray:
    """Unpack LSB-first packed uint8 bytes into int8 trits in {-1, 0, +1}.

    Raises ValueError if the reserved code 0b11 is encountered (ADR-003).
    """
    packed_u8 = np.asarray(packed, dtype=np.uint8)
    out_shape = (*packed_u8.shape[:-1], packed_u8.shape[-1] * 4)
    codes = np.empty(out_shape, dtype=np.uint8)
    codes[..., 0::4] = (packed_u8 >> 0) & 0b11
    codes[..., 1::4] = (packed_u8 >> 2) & 0b11
    codes[..., 2::4] = (packed_u8 >> 4) & 0b11
    codes[..., 3::4] = (packed_u8 >> 6) & 0b11

    if np.any(codes == CODE_RESERVED):
        raise ValueError("Encountered reserved trit code 0b11 during unpack.")

    out = np.zeros(out_shape, dtype=np.int8)
    out[codes == CODE_POS] = 1
    out[codes == CODE_NEG] = -1
    return out


def pack_ternary_tensor(
    weights: np.ndarray,
    scales: np.ndarray,
    group_size: int = 128,
    shift: int = 0,
) -> PackedTernaryTensor:
    """Pack a 2D (N, K) ternary matrix and (N, ceil(K/G)) uint16 scales."""
    if group_size not in VALID_GROUP_SIZES:
        raise ValueError(f"group_size must be one of {VALID_GROUP_SIZES}, got {group_size}.")
    if not (0 <= shift <= 255):
        raise ValueError(f"shift must fit in uint8 [0, 255], got {shift}.")

    w = np.asarray(weights, dtype=np.int8)
    if w.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix (N, K), got shape {w.shape}.")
    n_rows, n_cols = w.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Weight matrix dimensions N and K must be > 0.")

    n_groups = (n_cols + group_size - 1) // group_size
    k_pad = n_groups * group_size

    s = np.asarray(scales, dtype=np.uint16)
    if s.shape != (n_rows, n_groups):
        raise ValueError(f"Expected scales shape {(n_rows, n_groups)}, got {s.shape}.")

    if k_pad != n_cols:
        w_padded = np.zeros((n_rows, k_pad), dtype=np.int8)
        w_padded[:, :n_cols] = w
    else:
        w_padded = w

    w_grouped = w_padded.reshape(n_rows, n_groups, group_size)
    packed_trits = pack_trits_1d(w_grouped)

    return PackedTernaryTensor(
        packed_trits=np.ascontiguousarray(packed_trits, dtype=np.uint8),
        scales=np.ascontiguousarray(s, dtype=np.uint16),
        n_rows=n_rows,
        n_cols=n_cols,
        group_size=group_size,
        shift=shift,
    )


def unpack_ternary_tensor(packed: PackedTernaryTensor) -> tuple[np.ndarray, np.ndarray, int]:
    """Unpack a PackedTernaryTensor back into (weights_int8[N, K], scales_uint16, shift)."""
    w_grouped = unpack_trits_1d(packed.packed_trits)
    w_padded = w_grouped.reshape(packed.n_rows, -1)

    if packed.n_cols < w_padded.shape[1]:
        tail = w_padded[:, packed.n_cols :]
        if np.any(tail != 0):
            raise ValueError("Non-zero padding trits found beyond logical K.")

    w = np.ascontiguousarray(w_padded[:, : packed.n_cols], dtype=np.int8)
    return w, packed.scales.copy(), packed.shift


def serialize_packed_weights(packed: PackedTernaryTensor) -> bytes:
    """Serialize a PackedTernaryTensor to a byte-exact binary blob."""
    n_groups = packed.scales.shape[1]
    header = HEADER_STRUCT.pack(
        MAGIC,
        FORMAT_VERSION,
        packed.shift,
        packed.group_size,
        packed.n_rows,
        packed.n_cols,
        n_groups,
        0,
    )
    trit_bytes = np.ascontiguousarray(packed.packed_trits, dtype=np.uint8).tobytes()
    scale_bytes = np.ascontiguousarray(packed.scales, dtype="<u2").tobytes()
    return header + trit_bytes + scale_bytes


def deserialize_packed_weights(blob: bytes) -> PackedTernaryTensor:
    """Deserialize a byte-exact binary blob into a PackedTernaryTensor."""
    if len(blob) < HEADER_STRUCT.size:
        raise ValueError("Blob too short to contain Setun weight header.")
    magic, ver, shift, g, n_rows, n_cols, n_groups, rsvd = HEADER_STRUCT.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"Invalid magic {magic!r}, expected {MAGIC!r}.")
    if ver != FORMAT_VERSION:
        raise ValueError(f"Unsupported format version {ver}.")
    if rsvd != 0:
        raise ValueError("Reserved header field must be 0.")
    if g not in VALID_GROUP_SIZES:
        raise ValueError(f"Invalid group_size {g} in header.")

    expected_groups = (n_cols + g - 1) // g
    if n_groups != expected_groups:
        raise ValueError(f"Header n_groups={n_groups} != ceil(K/G)={expected_groups}.")

    trits_nbytes = n_rows * n_groups * (g // 4)
    scales_nbytes = n_rows * n_groups * 2
    expected_len = HEADER_STRUCT.size + trits_nbytes + scales_nbytes
    if len(blob) != expected_len:
        raise ValueError(f"Blob size mismatch: got {len(blob)}, expected {expected_len}.")

    offset = HEADER_STRUCT.size
    packed_trits = np.frombuffer(blob, dtype=np.uint8, count=trits_nbytes, offset=offset).reshape(
        n_rows, n_groups, g // 4
    )
    offset += trits_nbytes
    scales = (
        np.frombuffer(blob, dtype="<u2", count=n_rows * n_groups, offset=offset)
        .astype(np.uint16)
        .reshape(n_rows, n_groups)
    )

    return PackedTernaryTensor(
        packed_trits=packed_trits.copy(),
        scales=scales.copy(),
        n_rows=n_rows,
        n_cols=n_cols,
        group_size=g,
        shift=shift,
    )
