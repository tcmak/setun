"""Canonical TTAPE v1 codec and internal ternary weight pack/unpack (ADR-003, ADR-011, ADR-014)."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

VALID_GROUP_SIZES = (64, 128, 256)
MAGIC = b"TTAP"
VERSION_MAJOR = 1
VERSION_MINOR = 0

FLAG_LINKS = 0x01
FLAG_MANIFEST = 0x02
FLAG_NON_CORE = 0x04
FLAG_RESERVED_MASK = 0xF8

TAG_WEIGHTS = 0x0001
TAG_SCALES = 0x0002
TAG_ARCH_DESCRIPTOR = 0x0003
TAG_MANIFEST = 0x0004
TAG_LINK_TABLE = 0x0005
TAG_GROUP_SCALES_U16 = 0x0006
TAG_FP32_AUX = 0x0007

LAYER_TOKEN_SHIFT = 1
LAYER_GLU_MUX = 2
LAYER_RMSNORM_SHIFT = 3
LAYER_TERNARY_DENSE = 4
LAYER_KV_ARENA = 5
LAYER_HOST_EMBEDDING = 6
LAYER_HOST_SOFTMAX_HEAD = 7

AUX_ROLE_EMBEDDING = 1
AUX_ROLE_RMSNORM = 2
AUX_ROLE_SOFTMAX = 3

HEADER_STRUCT = struct.Struct("<4sBBBBIIQ8s")
SECTION_HEADER_STRUCT = struct.Struct("<HHII")
ARCH_LAYER_STRUCT = struct.Struct("<HHHBBII")
GROUP_SCALE_REC_STRUCT = struct.Struct("<HHBBH")
FP32_AUX_REC_STRUCT = struct.Struct("<HBBI")

# 2-bit trit codes per ADR-003: {00: 0, 01: +1, 10: -1, 11: reserved}
CODE_ZERO = 0b00
CODE_POS = 0b01
CODE_NEG = 0b10
CODE_RESERVED = 0b11


class TtapeValidationError(ValueError):
    """Typed Goose-style diagnostic raised when a .ttape buffer violates spec/ttape-v1-draft.md."""

    def __init__(self, error_code: str, message: str, byte_offset: int = 0) -> None:
        self.error_code = error_code
        self.byte_offset = byte_offset
        super().__init__(
            f"format.validate_ttape [{error_code} @ offset 0x{byte_offset:x}]: {message}"
        )


@dataclass(frozen=True)
class PackedTernaryTensor:
    """In-memory container for a packed grouped ternary matrix."""

    packed_trits: np.ndarray  # uint8, shape (N, n_groups, G // 4)
    scales: np.ndarray  # uint16, shape (N, n_groups)
    n_rows: int  # logical N
    n_cols: int  # logical K (may not be a multiple of group_size)
    group_size: int  # G in {64, 128, 256}
    shift: int = 0  # shared right-shift exponent (uint8)


def _fnv1a_32(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _fnv1a_64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _header_fnv1a_32(hdr_bytes: bytes) -> int:
    return _fnv1a_32(hdr_bytes[:12] + hdr_bytes[16:32])


def _clog2_u16(n: int) -> int:
    stages = 0
    span = 1
    while span < n:
        span <<= 1
        stages += 1
    return stages


def pack_trits_1d(trits: np.ndarray) -> np.ndarray:
    """Pack a 1D or ND array of trits in {-1, 0, +1} (last dim multiple of 4) LSB-first."""
    if not np.isin(trits, (-1, 0, 1)).all():
        raise ValueError(
            "format.pack_trits_1d: ternary weights must be strictly in {-1, 0, +1}, "
            f"got unique values {np.unique(trits).tolist()}."
        )
    if trits.shape[-1] % 4 != 0:
        raise ValueError(
            f"format.pack_trits_1d: last dimension ({trits.shape[-1]}) must be a multiple of 4."
        )

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
        bad_idx = int(np.flatnonzero(codes == CODE_RESERVED)[0])
        raise ValueError(
            f"format.unpack_trits_1d: encountered reserved trit code 0b11 at lane index {bad_idx} "
            f"(byte offset 0x{bad_idx // 4:x})."
        )

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
        raise ValueError(
            f"format.pack_ternary_tensor: expected group_size in {VALID_GROUP_SIZES}, "
            f"got {group_size}."
        )
    if not (0 <= shift <= 255):
        raise ValueError(
            f"format.pack_ternary_tensor: shift must fit in uint8 [0, 255], got {shift}."
        )

    w = np.asarray(weights, dtype=np.int8)
    if w.ndim != 2:
        raise ValueError(
            f"format.pack_ternary_tensor: expected 2D weight matrix (N, K), got shape {w.shape}."
        )
    n_rows, n_cols = w.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError(
            f"format.pack_ternary_tensor: dimensions N={n_rows}, K={n_cols} must be > 0."
        )

    n_groups = (n_cols + group_size - 1) // group_size
    k_pad = n_groups * group_size

    s = np.asarray(scales, dtype=np.uint16)
    if s.shape != (n_rows, n_groups):
        raise ValueError(
            f"format.pack_ternary_tensor: expected scales shape {(n_rows, n_groups)}, "
            f"got {s.shape}."
        )

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
            raise ValueError(
                "format.unpack_ternary_tensor: non-zero padding trits found beyond logical "
                f"K={packed.n_cols} (padded width={w_padded.shape[1]})."
            )

    w = np.ascontiguousarray(w_padded[:, : packed.n_cols], dtype=np.int8)
    return w, packed.scales.copy(), packed.shift


def _audit_ttape(
    blob: bytes,
    reader_profile: str = "grouped",
) -> tuple[str, str, int, dict[int, bytes] | None]:
    """Audit a .ttape buffer per §4 precedence order of spec/ttape-v1-draft.md."""
    if reader_profile not in ("core", "grouped", "extended"):
        raise ValueError(
            "format.validate_ttape: expected reader_profile in "
            f"('core', 'grouped', 'extended'), got {reader_profile!r}."
        )

    if blob is None or len(blob) < HEADER_STRUCT.size:
        got_len = 0 if blob is None else len(blob)
        return (
            "TTAPE_ERR_NULL_OR_TRUNCATED",
            f"expected >= {HEADER_STRUCT.size} header bytes, got {got_len}",
            0,
            None,
        )

    magic, ver_maj, _ver_min, flags, sec_count, total_len, hdr_crc, artifact_id, reserved = (
        HEADER_STRUCT.unpack_from(blob, 0)
    )
    if magic != MAGIC:
        return (
            "TTAPE_ERR_BAD_MAGIC",
            f"expected magic {MAGIC!r} (0x50415454), got {magic!r}",
            0,
            None,
        )
    if ver_maj != VERSION_MAJOR:
        return (
            "TTAPE_ERR_UNSUPPORTED_VERSION",
            f"expected version_major={VERSION_MAJOR}, got {ver_maj}",
            4,
            None,
        )
    if any(b != 0 for b in reserved):
        return (
            "TTAPE_ERR_NONZERO_RESERVED",
            f"expected header reserved[8] all zero, got {reserved.hex()}",
            24,
            None,
        )
    computed_hdr_crc = _header_fnv1a_32(blob[:32])
    if computed_hdr_crc != hdr_crc:
        return (
            "TTAPE_ERR_CHECKSUM_MISMATCH",
            f"header_fnv1a expected 0x{computed_hdr_crc:08x}, got 0x{hdr_crc:08x}",
            12,
            None,
        )
    if (flags & FLAG_RESERVED_MASK) != 0:
        return (
            "TTAPE_ERR_UNKNOWN_FLAGS",
            f"header flags=0x{flags:02x} sets reserved bits (mask 0x{FLAG_RESERVED_MASK:02x})",
            6,
            None,
        )

    is_non_core_tape = (flags & FLAG_NON_CORE) != 0
    if reader_profile == "core" and is_non_core_tape:
        return (
            "TTAPE_ERR_UNSUPPORTED_PROFILE",
            f"reader_profile='core' rejects non-core flags=0x{flags:02x}",
            6,
            None,
        )
    if sec_count < 3:
        return (
            "TTAPE_ERR_SECTION_COUNT_MISMATCH",
            f"expected section_count >= 3, got {sec_count}",
            7,
            None,
        )
    if len(blob) - HEADER_STRUCT.size != total_len:
        return (
            "TTAPE_ERR_LENGTH_MISMATCH",
            f"expected file length 32 + {total_len} = {32 + total_len}, got {len(blob)}",
            8,
            None,
        )

    sec_stream = blob[HEADER_STRUCT.size :]
    computed_art_id = _fnv1a_64(sec_stream)
    if computed_art_id != artifact_id:
        return (
            "TTAPE_ERR_ARTIFACT_ID_MISMATCH",
            f"artifact_id expected 0x{computed_art_id:016x}, got 0x{artifact_id:016x}",
            16,
            None,
        )

    sec_map: dict[int, bytes] = {}
    sec_offsets: dict[int, int] = {}
    off = 0
    for _ in range(sec_count):
        abs_off = HEADER_STRUCT.size + off
        if total_len - off < SECTION_HEADER_STRUCT.size:
            return (
                "TTAPE_ERR_LENGTH_MISMATCH",
                f"truncated section header at byte offset 0x{abs_off:x}",
                abs_off,
                None,
            )
        tag, sec_res, slen, payload_crc = SECTION_HEADER_STRUCT.unpack_from(sec_stream, off)
        if sec_res != 0:
            return (
                "TTAPE_ERR_NONZERO_RESERVED",
                f"section 0x{tag:04x} reserved field expected 0, got {sec_res}",
                abs_off + 2,
                None,
            )
        rem = total_len - off - SECTION_HEADER_STRUCT.size
        if slen > rem:
            return (
                "TTAPE_ERR_LENGTH_MISMATCH",
                f"section 0x{tag:04x} length={slen} exceeds remaining {rem} bytes",
                abs_off + 4,
                None,
            )
        payload = sec_stream[
            off + SECTION_HEADER_STRUCT.size : off + SECTION_HEADER_STRUCT.size + slen
        ]
        computed_sec_crc = _fnv1a_32(payload)
        if computed_sec_crc != payload_crc:
            return (
                "TTAPE_ERR_CHECKSUM_MISMATCH",
                f"section 0x{tag:04x} payload_fnv1a expected 0x{computed_sec_crc:08x}, "
                f"got 0x{payload_crc:08x}",
                abs_off + 8,
                None,
            )
        if tag < TAG_WEIGHTS or tag > TAG_FP32_AUX:
            return (
                "TTAPE_ERR_UNKNOWN_TAG",
                f"unknown section tag 0x{tag:04x} outside 0x0001..0x0007",
                abs_off,
                None,
            )
        if tag in sec_map:
            return (
                "TTAPE_ERR_DUPLICATE_TAG",
                f"duplicate section tag 0x{tag:04x}",
                abs_off,
                None,
            )
        if tag in (TAG_GROUP_SCALES_U16, TAG_FP32_AUX) and not is_non_core_tape:
            return (
                "TTAPE_ERR_PROFILE_SECTION_FORBIDDEN",
                f"section 0x{tag:04x} forbidden when core profile flag bit 2 == 0",
                abs_off,
                None,
            )
        if tag == TAG_FP32_AUX and reader_profile != "extended":
            return (
                "TTAPE_ERR_UNSUPPORTED_PROFILE",
                f"section 0x0007 (fp32_aux) rejected by reader_profile={reader_profile!r}",
                abs_off,
                None,
            )
        sec_map[tag] = payload
        sec_offsets[tag] = abs_off + SECTION_HEADER_STRUCT.size
        off += SECTION_HEADER_STRUCT.size + slen

    if off != total_len:
        return (
            "TTAPE_ERR_LENGTH_MISMATCH",
            f"framed section bytes ({off}) != header total_length ({total_len})",
            HEADER_STRUCT.size + off,
            None,
        )

    has_base = TAG_WEIGHTS in sec_map and TAG_SCALES in sec_map and TAG_ARCH_DESCRIPTOR in sec_map
    manifest_ok = ((flags & FLAG_MANIFEST) != 0) == (TAG_MANIFEST in sec_map)
    links_ok = ((flags & FLAG_LINKS) != 0) == (TAG_LINK_TABLE in sec_map)
    non_core_ok = (not is_non_core_tape) or (
        TAG_GROUP_SCALES_U16 in sec_map or TAG_FP32_AUX in sec_map
    )
    if not (has_base and manifest_ok and links_ok and non_core_ok):
        return (
            "TTAPE_ERR_MISSING_REQUIRED_SECTION",
            f"missing mandatory section or flag/section mismatch (flags=0x{flags:02x}, "
            f"tags={[f'0x{t:04x}' for t in sorted(sec_map)]})",
            HEADER_STRUCT.size,
            None,
        )

    arch_buf = sec_map[TAG_ARCH_DESCRIPTOR]
    arch_base = sec_offsets[TAG_ARCH_DESCRIPTOR]
    weights_buf = sec_map[TAG_WEIGHTS]
    weights_base = sec_offsets[TAG_WEIGHTS]
    scales_buf = sec_map[TAG_SCALES]
    scales_base = sec_offsets[TAG_SCALES]

    if len(arch_buf) < 4:
        return (
            "TTAPE_ERR_MALFORMED_ARCH",
            f"section 0x0003 length={len(arch_buf)} < 4",
            arch_base,
            None,
        )
    n_layers, arch_res = struct.unpack_from("<HH", arch_buf, 0)
    if arch_res != 0:
        return (
            "TTAPE_ERR_NONZERO_RESERVED",
            f"section 0x0003 header reserved={arch_res}, expected 0",
            arch_base + 2,
            None,
        )
    if n_layers == 0 or len(arch_buf) != 4 + 16 * n_layers:
        return (
            "TTAPE_ERR_MALFORMED_ARCH",
            f"section 0x0003 n_layers={n_layers}, length={len(arch_buf)} != {4 + 16 * n_layers}",
            arch_base,
            None,
        )

    layers: list[tuple[int, int, int, int, int, int, int]] = []
    dense_layers: list[tuple[int, int, int]] = []
    expected_weight_bytes = 0

    for i in range(n_layers):
        l_off = arch_base + 4 + 16 * i
        kind, dim_in, dim_out, stages, l_res, lat, cap = ARCH_LAYER_STRUCT.unpack_from(
            arch_buf, 4 + 16 * i
        )
        if l_res != 0:
            return (
                "TTAPE_ERR_NONZERO_RESERVED",
                f"section 0x0003 layer[{i}] reserved={l_res}, expected 0",
                l_off + 7,
                None,
            )
        if kind < 1 or kind > 7 or dim_in == 0 or dim_out == 0 or cap == 0:
            return (
                "TTAPE_ERR_MALFORMED_ARCH",
                f"section 0x0003 layer[{i}] invalid kind={kind}, dim_in={dim_in}, "
                f"dim_out={dim_out}, context_cap={cap}",
                l_off,
                None,
            )
        if kind >= 6:
            if not is_non_core_tape:
                return (
                    "TTAPE_ERR_PROFILE_SECTION_FORBIDDEN",
                    f"section 0x0003 layer[{i}] host kind={kind} forbidden in core profile",
                    l_off,
                    None,
                )
            if reader_profile != "extended":
                return (
                    "TTAPE_ERR_UNSUPPORTED_PROFILE",
                    f"section 0x0003 layer[{i}] host kind={kind} unsupported by "
                    f"reader_profile={reader_profile!r}",
                    l_off,
                    None,
                )
        if kind == LAYER_TERNARY_DENSE:
            exp_stages = _clog2_u16(dim_in)
            if stages != exp_stages or lat < exp_stages:
                return (
                    "TTAPE_ERR_MALFORMED_ARCH",
                    f"section 0x0003 layer[{i}] expected reduce_stages={exp_stages} and "
                    f"latency>={exp_stages}, got stages={stages}, latency={lat}",
                    l_off + 6,
                    None,
                )
            expected_weight_bytes += dim_out * ((dim_in + 3) // 4)
            dense_layers.append((i, dim_in, dim_out))
        else:
            if stages != 0:
                return (
                    "TTAPE_ERR_MALFORMED_ARCH",
                    f"section 0x0003 non-dense layer[{i}] expected reduce_stages=0, got {stages}",
                    l_off + 6,
                    None,
                )
            if (
                kind in (LAYER_TOKEN_SHIFT, LAYER_GLU_MUX, LAYER_RMSNORM_SHIFT)
                and dim_in != dim_out
            ):
                return (
                    "TTAPE_ERR_MALFORMED_ARCH",
                    f"section 0x0003 layer[{i}] kind={kind} requires dim_in==dim_out, "
                    f"got {dim_in} vs {dim_out}",
                    l_off + 2,
                    None,
                )
        layers.append((kind, dim_in, dim_out, stages, l_res, lat, cap))

    if (
        len(dense_layers) == 0
        or len(weights_buf) != expected_weight_bytes
        or len(scales_buf) != 2 * len(dense_layers)
    ):
        return (
            "TTAPE_ERR_DIM_PAYLOAD_MISMATCH",
            f"section 0x0001/0x0002 byte lengths ({len(weights_buf)}, {len(scales_buf)}) "
            f"disagree with arch_descriptor ({expected_weight_bytes}, {2 * len(dense_layers)})",
            weights_base,
            None,
        )

    w_off = 0
    for _l_idx, dim_in, dim_out in dense_layers:
        row_bytes = (dim_in + 3) // 4
        total_lanes = row_bytes * 4
        for _r in range(dim_out):
            for c in range(total_lanes):
                byte_idx = w_off + (c >> 2)
                b = weights_buf[byte_idx]
                code = (b >> ((c & 3) * 2)) & 0x03
                if c < dim_in:
                    if code == CODE_RESERVED:
                        return (
                            "TTAPE_ERR_RESERVED_TERNARY_CODE",
                            f"section 0x0001 reserved code 0b11 at lane {c}",
                            weights_base + byte_idx,
                            None,
                        )
                elif code != CODE_ZERO:
                    return (
                        "TTAPE_ERR_NONZERO_PADDING",
                        f"section 0x0001 non-zero padding 0b{code:02b} at lane {c} >= {dim_in}",
                        weights_base + byte_idx,
                        None,
                    )
            w_off += row_bytes

    scale_exps: list[int] = []
    for s_idx in range(len(dense_layers)):
        (exp_val,) = struct.unpack_from("<h", scales_buf, 2 * s_idx)
        if exp_val < -30 or exp_val > 30:
            return (
                "TTAPE_ERR_INVALID_SCALE",
                f"section 0x0002 scale[{s_idx}]={exp_val} outside [-30, +30]",
                scales_base + 2 * s_idx,
                None,
            )
        scale_exps.append(exp_val)

    if TAG_MANIFEST in sec_map:
        m_buf = sec_map[TAG_MANIFEST]
        m_base = sec_offsets[TAG_MANIFEST]
        m_off = 0
        for f_idx in range(5):
            if len(m_buf) - m_off < 2:
                return (
                    "TTAPE_ERR_MALFORMED_MANIFEST",
                    f"section 0x0004 truncated field[{f_idx}] length prefix",
                    m_base + m_off,
                    None,
                )
            (flen,) = struct.unpack_from("<H", m_buf, m_off)
            if (
                flen == 0
                or flen > len(m_buf) - m_off - 2
                or (f_idx == 1 and flen != 32)
                or (f_idx == 4 and flen < 8)
            ):
                return (
                    "TTAPE_ERR_MALFORMED_MANIFEST",
                    f"section 0x0004 invalid field[{f_idx}] length={flen}",
                    m_base + m_off,
                    None,
                )
            m_off += 2 + flen
        if m_off != len(m_buf):
            return (
                "TTAPE_ERR_MALFORMED_MANIFEST",
                f"section 0x0004 trailing bytes ({len(m_buf) - m_off})",
                m_base + m_off,
                None,
            )

    if TAG_LINK_TABLE in sec_map:
        l_buf = sec_map[TAG_LINK_TABLE]
        l_base = sec_offsets[TAG_LINK_TABLE]
        if len(l_buf) == 0 or (len(l_buf) & 3) != 0:
            return (
                "TTAPE_ERR_INVALID_LINK",
                f"section 0x0005 length={len(l_buf)} must be positive multiple of 4",
                l_base,
                None,
            )
        for l_off in range(0, len(l_buf), 4):
            (delta,) = struct.unpack_from("<i", l_buf, l_off)
            target = l_off + delta
            if delta == 0 or (delta & 3) != 0 or target < 0 or target >= len(l_buf):
                return (
                    "TTAPE_ERR_INVALID_LINK",
                    f"section 0x0005 invalid self-relative link delta={delta} at offset {l_off}",
                    l_base + l_off,
                    None,
                )

    if TAG_GROUP_SCALES_U16 in sec_map:
        g_buf = sec_map[TAG_GROUP_SCALES_U16]
        g_base = sec_offsets[TAG_GROUP_SCALES_U16]
        if len(g_buf) < 4:
            return (
                "TTAPE_ERR_MALFORMED_GROUP_SCALES",
                f"section 0x0006 length={len(g_buf)} < 4",
                g_base,
                None,
            )
        tensor_count, g_res = struct.unpack_from("<HH", g_buf, 0)
        if g_res != 0:
            return (
                "TTAPE_ERR_NONZERO_RESERVED",
                f"section 0x0006 header reserved={g_res}, expected 0",
                g_base + 2,
                None,
            )
        if tensor_count != len(dense_layers):
            return (
                "TTAPE_ERR_MALFORMED_GROUP_SCALES",
                f"section 0x0006 tensor_count={tensor_count} != {len(dense_layers)} dense layers",
                g_base,
                None,
            )
        g_off = 4
        for d_idx, (exp_layer_idx, dim_in, dim_out) in enumerate(dense_layers):
            if len(g_buf) - g_off < GROUP_SCALE_REC_STRUCT.size:
                return (
                    "TTAPE_ERR_MALFORMED_GROUP_SCALES",
                    f"section 0x0006 truncated record[{d_idx}]",
                    g_base + g_off,
                    None,
                )
            layer_idx, group_size, shift, rec_res, n_groups = GROUP_SCALE_REC_STRUCT.unpack_from(
                g_buf, g_off
            )
            if rec_res != 0:
                return (
                    "TTAPE_ERR_NONZERO_RESERVED",
                    f"section 0x0006 record[{d_idx}] reserved={rec_res}, expected 0",
                    g_base + g_off + 5,
                    None,
                )
            if (
                layer_idx != exp_layer_idx
                or group_size not in VALID_GROUP_SIZES
                or shift > 30
                or scale_exps[d_idx] != -int(shift)
            ):
                return (
                    "TTAPE_ERR_MALFORMED_GROUP_SCALES",
                    f"section 0x0006 record[{d_idx}] invalid layer_index={layer_idx} "
                    f"(expected {exp_layer_idx}), group_size={group_size} "
                    f"(expected in {VALID_GROUP_SIZES}), or shift={shift} "
                    f"(0x0002 scale_exp={scale_exps[d_idx]})",
                    g_base + g_off,
                    None,
                )
            exp_groups = (dim_in + group_size - 1) // group_size
            if n_groups != exp_groups:
                return (
                    "TTAPE_ERR_MALFORMED_GROUP_SCALES",
                    f"section 0x0006 record[{d_idx}] n_groups={n_groups} != "
                    f"ceil({dim_in}/{group_size})={exp_groups}",
                    g_base + g_off + 6,
                    None,
                )
            scales_bytes = dim_out * n_groups * 2
            if len(g_buf) - g_off - GROUP_SCALE_REC_STRUCT.size < scales_bytes:
                return (
                    "TTAPE_ERR_MALFORMED_GROUP_SCALES",
                    f"section 0x0006 record[{d_idx}] truncated u16 scale payload",
                    g_base + g_off + GROUP_SCALE_REC_STRUCT.size,
                    None,
                )
            g_off += GROUP_SCALE_REC_STRUCT.size + scales_bytes
        if g_off != len(g_buf):
            return (
                "TTAPE_ERR_MALFORMED_GROUP_SCALES",
                f"section 0x0006 trailing bytes ({len(g_buf) - g_off})",
                g_base + g_off,
                None,
            )

    if TAG_FP32_AUX in sec_map:
        a_buf = sec_map[TAG_FP32_AUX]
        a_base = sec_offsets[TAG_FP32_AUX]
        if len(a_buf) < 4:
            return (
                "TTAPE_ERR_MALFORMED_FP32_AUX",
                f"section 0x0007 length={len(a_buf)} < 4",
                a_base,
                None,
            )
        tensor_count, a_res = struct.unpack_from("<HH", a_buf, 0)
        if a_res != 0:
            return (
                "TTAPE_ERR_NONZERO_RESERVED",
                f"section 0x0007 header reserved={a_res}, expected 0",
                a_base + 2,
                None,
            )
        if tensor_count == 0:
            return (
                "TTAPE_ERR_MALFORMED_FP32_AUX",
                "section 0x0007 tensor_count=0, expected > 0",
                a_base,
                None,
            )
        a_off = 4
        for t_idx in range(tensor_count):
            if len(a_buf) - a_off < FP32_AUX_REC_STRUCT.size:
                return (
                    "TTAPE_ERR_MALFORMED_FP32_AUX",
                    f"section 0x0007 truncated record[{t_idx}]",
                    a_base + a_off,
                    None,
                )
            layer_idx, aux_role, rec_res, elem_count = FP32_AUX_REC_STRUCT.unpack_from(a_buf, a_off)
            if rec_res != 0:
                return (
                    "TTAPE_ERR_NONZERO_RESERVED",
                    f"section 0x0007 record[{t_idx}] reserved={rec_res}, expected 0",
                    a_base + a_off + 3,
                    None,
                )
            if (
                layer_idx >= n_layers
                or aux_role < 1
                or aux_role > 3
                or elem_count == 0
                or elem_count > (len(a_buf) - a_off - FP32_AUX_REC_STRUCT.size) // 4
            ):
                return (
                    "TTAPE_ERR_MALFORMED_FP32_AUX",
                    f"section 0x0007 record[{t_idx}] invalid layer_index={layer_idx}, "
                    f"aux_role={aux_role}, elem_count={elem_count}",
                    a_base + a_off,
                    None,
                )
            target_kind, t_din, t_dout, *_ = layers[layer_idx]
            exp_kind = (
                LAYER_HOST_EMBEDDING
                if aux_role == AUX_ROLE_EMBEDDING
                else (
                    LAYER_RMSNORM_SHIFT if aux_role == AUX_ROLE_RMSNORM else LAYER_HOST_SOFTMAX_HEAD
                )
            )
            exp_elems = (
                t_din * t_dout
                if aux_role == AUX_ROLE_EMBEDDING
                else (t_din if aux_role == AUX_ROLE_RMSNORM else t_dout)
            )
            if target_kind != exp_kind or elem_count != exp_elems:
                return (
                    "TTAPE_ERR_MALFORMED_FP32_AUX",
                    f"section 0x0007 record[{t_idx}] target kind={target_kind}/elems={elem_count} "
                    f"mismatch expected kind={exp_kind}/elems={exp_elems}",
                    a_base + a_off,
                    None,
                )
            vals_off = a_off + FP32_AUX_REC_STRUCT.size
            for k in range(elem_count):
                (word,) = struct.unpack_from("<I", a_buf, vals_off + 4 * k)
                if (word & 0x7F800000) == 0x7F800000:
                    return (
                        "TTAPE_ERR_MALFORMED_FP32_AUX",
                        f"section 0x0007 record[{t_idx}] non-finite IEEE-754 word 0x{word:08x}",
                        a_base + vals_off + 4 * k,
                        None,
                    )
            a_off += FP32_AUX_REC_STRUCT.size + elem_count * 4
        if a_off != len(a_buf):
            return (
                "TTAPE_ERR_MALFORMED_FP32_AUX",
                f"section 0x0007 trailing bytes ({len(a_buf) - a_off})",
                a_base + a_off,
                None,
            )

    return ("TTAPE_OK", "ok", 0, sec_map)


def validate_ttape(blob: bytes, reader_profile: str = "grouped") -> str:
    """Validate a .ttape artifact and return its §4 TTAPE_* diagnostic symbol."""
    code, _msg, _off, _secs = _audit_ttape(blob, reader_profile=reader_profile)
    return code


def load_ttape_layers(
    blob: bytes,
    reader_profile: str = "grouped",
) -> list[PackedTernaryTensor]:
    """Validate a .ttape buffer and decode all ternary_dense layers into PackedTernaryTensors."""
    code, msg, off, sec_map = _audit_ttape(blob, reader_profile=reader_profile)
    if code != "TTAPE_OK" or sec_map is None:
        raise TtapeValidationError(code, msg, byte_offset=off)

    arch_buf = sec_map[TAG_ARCH_DESCRIPTOR]
    weights_buf = sec_map[TAG_WEIGHTS]
    scales_buf = sec_map[TAG_SCALES]
    n_layers, _ = struct.unpack_from("<HH", arch_buf, 0)

    dense_dims: list[tuple[int, int, int]] = []
    for i in range(n_layers):
        kind, dim_in, dim_out, *_ = ARCH_LAYER_STRUCT.unpack_from(arch_buf, 4 + 16 * i)
        if kind == LAYER_TERNARY_DENSE:
            dense_dims.append((i, dim_in, dim_out))

    group_meta: list[tuple[int, int, np.ndarray]] = []
    if TAG_GROUP_SCALES_U16 in sec_map:
        g_buf = sec_map[TAG_GROUP_SCALES_U16]
        g_off = 4
        for _l_idx, _dim_in, dim_out in dense_dims:
            _rec_lidx, group_size, shift, _res, n_groups = GROUP_SCALE_REC_STRUCT.unpack_from(
                g_buf, g_off
            )
            g_off += GROUP_SCALE_REC_STRUCT.size
            scales_len = dim_out * n_groups * 2
            scales_arr = (
                np.frombuffer(g_buf, dtype="<u2", count=dim_out * n_groups, offset=g_off)
                .astype(np.uint16)
                .reshape(dim_out, n_groups)
                .copy()
            )
            g_off += scales_len
            group_meta.append((group_size, shift, scales_arr))
    else:
        # TTAPE-core shift-only mapping (ADR-014): group_size=64
        for d_idx, (_l_idx, dim_in, dim_out) in enumerate(dense_dims):
            (exp_val,) = struct.unpack_from("<h", scales_buf, 2 * d_idx)
            if exp_val > 15:
                raise TtapeValidationError(
                    "TTAPE_ERR_INVALID_SCALE",
                    f"TTAPE-core positive scale_exp={exp_val} exceeds uint16 capacity [0, 15]",
                    byte_offset=2 * d_idx,
                )
            group_size = 64
            n_groups = (dim_in + group_size - 1) // group_size
            if exp_val <= 0:
                shift = -exp_val
                scales_arr = np.ones((dim_out, n_groups), dtype=np.uint16)
            else:
                shift = 0
                scales_arr = np.full((dim_out, n_groups), 1 << exp_val, dtype=np.uint16)
            group_meta.append((group_size, shift, scales_arr))

    out_tensors: list[PackedTernaryTensor] = []
    w_off = 0
    for (_l_idx, dim_in, dim_out), (group_size, shift, scales_arr) in zip(
        dense_dims, group_meta, strict=True
    ):
        row_bytes = (dim_in + 3) // 4
        layer_bytes = dim_out * row_bytes
        raw_rows = np.frombuffer(
            weights_buf, dtype=np.uint8, count=layer_bytes, offset=w_off
        ).reshape(dim_out, row_bytes)
        w_off += layer_bytes

        trits_row_padded = unpack_trits_1d(raw_rows)
        logical_trits = trits_row_padded[:, :dim_in]
        out_tensors.append(
            pack_ternary_tensor(logical_trits, scales_arr, group_size=group_size, shift=shift)
        )

    return out_tensors


def serialize_packed_weights(packed: PackedTernaryTensor, *, context_cap: int = 64) -> bytes:
    """Serialize a PackedTernaryTensor to a canonical single-layer TTAPE-grouped .ttape blob."""
    if not (0 <= packed.shift <= 30):
        raise ValueError(
            "format.serialize_packed_weights: TTAPE v1 shift must be in [0, 30], "
            f"got {packed.shift}."
        )
    if packed.group_size not in VALID_GROUP_SIZES:
        raise ValueError(
            f"format.serialize_packed_weights: expected group_size in {VALID_GROUP_SIZES}, "
            f"got {packed.group_size}."
        )
    if not (1 <= packed.n_rows <= 0xFFFF and 1 <= packed.n_cols <= 0xFFFF):
        raise ValueError(
            f"format.serialize_packed_weights: dimensions ({packed.n_rows}, {packed.n_cols}) "
            "must fit in uint16 [1, 65535]."
        )

    w_logical, scales_u16, shift = unpack_ternary_tensor(packed)
    n_rows, n_cols = w_logical.shape
    row_bytes = (n_cols + 3) // 4
    if row_bytes * 4 != n_cols:
        w_wire_pad = np.zeros((n_rows, row_bytes * 4), dtype=np.int8)
        w_wire_pad[:, :n_cols] = w_logical
    else:
        w_wire_pad = w_logical

    weights_payload = np.ascontiguousarray(pack_trits_1d(w_wire_pad), dtype=np.uint8).tobytes()
    scales_payload = struct.pack("<h", -shift)

    reduce_stages = _clog2_u16(n_cols)
    arch_payload = struct.pack("<HH", 1, 0) + ARCH_LAYER_STRUCT.pack(
        LAYER_TERNARY_DENSE,
        n_cols,
        n_rows,
        reduce_stages,
        0,
        reduce_stages,
        context_cap,
    )

    n_groups = scales_u16.shape[1]
    group_scales_payload = (
        struct.pack("<HH", 1, 0)
        + GROUP_SCALE_REC_STRUCT.pack(0, packed.group_size, shift, 0, n_groups)
        + np.ascontiguousarray(scales_u16, dtype="<u2").tobytes()
    )

    sections = (
        (TAG_WEIGHTS, weights_payload),
        (TAG_SCALES, scales_payload),
        (TAG_ARCH_DESCRIPTOR, arch_payload),
        (TAG_GROUP_SCALES_U16, group_scales_payload),
    )
    sec_blobs = bytearray()
    for tag, payload in sections:
        sec_blobs.extend(SECTION_HEADER_STRUCT.pack(tag, 0, len(payload), _fnv1a_32(payload)))
        sec_blobs.extend(payload)

    sec_bytes = bytes(sec_blobs)
    artifact_id = _fnv1a_64(sec_bytes)
    draft_hdr = HEADER_STRUCT.pack(
        MAGIC,
        VERSION_MAJOR,
        VERSION_MINOR,
        FLAG_NON_CORE,
        len(sections),
        len(sec_bytes),
        0,
        artifact_id,
        b"\x00" * 8,
    )
    hdr_crc = _header_fnv1a_32(draft_hdr)
    final_hdr = HEADER_STRUCT.pack(
        MAGIC,
        VERSION_MAJOR,
        VERSION_MINOR,
        FLAG_NON_CORE,
        len(sections),
        len(sec_bytes),
        hdr_crc,
        artifact_id,
        b"\x00" * 8,
    )
    return final_hdr + sec_bytes


def deserialize_packed_weights(blob: bytes) -> PackedTernaryTensor:
    """Deserialize a single-layer .ttape binary blob into a PackedTernaryTensor."""
    layers = load_ttape_layers(blob, reader_profile="grouped")
    if len(layers) != 1:
        raise TtapeValidationError(
            "TTAPE_ERR_MALFORMED_ARCH",
            f"deserialize_packed_weights expected 1 ternary_dense layer, got {len(layers)}",
            byte_offset=32,
        )
    return layers[0]
