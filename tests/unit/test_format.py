"""Unit tests for TTAPE v1 conformance and trit pack/unpack in setun.ref.format."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from setun.ref.format import (
    MAGIC,
    TtapeValidationError,
    deserialize_packed_weights,
    load_ttape_layers,
    pack_ternary_tensor,
    serialize_packed_weights,
    unpack_ternary_tensor,
    unpack_trits_1d,
    validate_ttape,
)
from setun.ref.ops import row_dot

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "third_party" / "TernaryTape" / "fixtures"


def test_ttape_grouped_worked_example_matches_golden_fixture() -> None:
    """Asserts serialize_packed_weights matches accept/grouped_with_u16_scales.ttape byte-exact."""
    assert MAGIC == b"TTAP"
    w = np.array([[1, -1, 0, 1, -1, -1]], dtype=np.int8)
    s = np.array([[0x0180]], dtype=np.uint16)
    packed = pack_ternary_tensor(w, s, group_size=64, shift=8)
    blob = serialize_packed_weights(packed)

    golden_path = FIXTURES_DIR / "accept" / "grouped_with_u16_scales.ttape"
    assert golden_path.exists(), f"Missing golden fixture {golden_path}"
    assert blob == golden_path.read_bytes()

    restored = deserialize_packed_weights(blob)
    w_out, s_out, shift_out = unpack_ternary_tensor(restored)
    assert np.array_equal(w_out, w)
    assert np.array_equal(s_out, s)
    assert shift_out == 8

    # End-to-end check with setun/ref/ops.py row_dot
    act = np.array([-128, 50, 99, 30, -20, 10], dtype=np.int8)
    got_dot = row_dot(w_out[0], act, s_out[0], group_size=64, lanes=16, shift=shift_out)
    # (-128 - 50 + 0 + 30 + 20 - 10) = -138; (-138 * 384 + 128) >> 8 = -52864 >> 8 = -207
    assert got_dot == ((-138 * 0x0180 + (1 << 7)) >> 8)


def test_canonical_ttape_accept_and_reject_corpus() -> None:
    """Validates all 21 accept/*.ttape and reject/*.ttape fixtures from third_party/TernaryTape."""
    spec_path = REPO_ROOT / "third_party" / "TernaryTape" / "spec" / "ttape-v1.0.md"
    assert spec_path.exists(), f"Expected frozen canonical spec at {spec_path}"
    assert "FROZEN v1.0" in spec_path.read_text(encoding="utf-8")

    accept_files = sorted((FIXTURES_DIR / "accept").glob("*.ttape"))
    reject_files = sorted((FIXTURES_DIR / "reject").glob("*.ttape"))
    assert len(accept_files) == 5
    assert len(reject_files) == 16

    for bin_path in [*accept_files, *reject_files]:
        meta = json.loads(bin_path.with_suffix(".json").read_text(encoding="utf-8"))
        blob = bin_path.read_bytes()
        expected_code = str(meta["error_code"])
        reader_profile = str(meta["reader_profile"])

        got_code = validate_ttape(blob, reader_profile=reader_profile)
        assert got_code == expected_code, (
            f"{bin_path.name}: expected {expected_code} (profile={reader_profile}), got {got_code}"
        )

        if meta["expected"] == "reject":
            with pytest.raises(TtapeValidationError) as exc_info:
                load_ttape_layers(blob, reader_profile=reader_profile)
            assert exc_info.value.error_code == expected_code

    # Setun default hardware loader profile ("grouped") accepts core + grouped, rejects extended
    core_min_blob = (FIXTURES_DIR / "accept" / "core_minimal_dot4.ttape").read_bytes()
    core_layers = load_ttape_layers(core_min_blob)  # default reader_profile="grouped"
    assert len(core_layers) == 1
    w_core, s_core, shift_core = unpack_ternary_tensor(core_layers[0])
    assert np.array_equal(w_core, np.array([[1, -1, 0, 1], [-1, 1, 1, 0]], dtype=np.int8))
    assert np.array_equal(s_core, np.array([[1], [1]], dtype=np.uint16))
    assert shift_core == 2

    act4 = np.array([12, 20, 28, 36], dtype=np.int8)
    assert row_dot(w_core[0], act4, s_core[0], group_size=64, lanes=16, shift=shift_core) == 7
    assert row_dot(w_core[1], act4, s_core[1], group_size=64, lanes=16, shift=shift_core) == 9

    ext_blob = (FIXTURES_DIR / "accept" / "extended_with_fp32_aux.ttape").read_bytes()
    assert validate_ttape(ext_blob, reader_profile="grouped") == "TTAPE_ERR_UNSUPPORTED_PROFILE"
    with pytest.raises(TtapeValidationError) as ext_exc:
        load_ttape_layers(ext_blob)  # default reader_profile="grouped"
    assert ext_exc.value.error_code == "TTAPE_ERR_UNSUPPORTED_PROFILE"

    # Exhaustive per-byte single-bit mutation check across all Setun-accepted golden tapes
    for name in (
        "core_minimal_dot4.ttape",
        "core_with_manifest_and_links.ttape",
        "grouped_with_u16_scales.ttape",
        "lux_chain_grouped_mlp.ttape",
    ):
        orig = (FIXTURES_DIR / "accept" / name).read_bytes()
        assert validate_ttape(orig, reader_profile="grouped") == "TTAPE_OK"
        for idx in range(len(orig)):
            mutated = bytearray(orig)
            mutated[idx] ^= 0x01
            code = validate_ttape(bytes(mutated), reader_profile="grouped")
            assert code != "TTAPE_OK", f"{name}: mutation at byte {idx} was not rejected"


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


@pytest.mark.slow
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


def _build_framed_ttape(
    sections: list[tuple[int, bytes, int]],
    *,
    flags: int = 0,
    magic: bytes = b"TTAP",
    version_major: int = 1,
    header_reserved: bytes = b"\x00" * 8,
    sec_count_override: int | None = None,
    total_len_override: int | None = None,
    trailing_section_bytes: bytes = b"",
    trailing_file_bytes: bytes = b"",
) -> bytes:
    """Construct a checksum-consistent .ttape buffer to test deep §4 structural validators."""

    from setun.ref.format import (
        HEADER_STRUCT,
        SECTION_HEADER_STRUCT,
        _fnv1a_32,
        _fnv1a_64,
        _header_fnv1a_32,
    )

    sec_buf = bytearray()
    for tag, payload, sec_res in sections:
        sec_buf.extend(SECTION_HEADER_STRUCT.pack(tag, sec_res, len(payload), _fnv1a_32(payload)))
        sec_buf.extend(payload)
    sec_buf.extend(trailing_section_bytes)
    sec_bytes = bytes(sec_buf)

    sec_count = len(sections) if sec_count_override is None else sec_count_override
    total_len = len(sec_bytes) if total_len_override is None else total_len_override
    art_id = _fnv1a_64(sec_bytes)

    draft = HEADER_STRUCT.pack(
        magic, version_major, 0, flags, sec_count, total_len, 0, art_id, header_reserved
    )
    hdr = HEADER_STRUCT.pack(
        magic,
        version_major,
        0,
        flags,
        sec_count,
        total_len,
        _header_fnv1a_32(draft),
        art_id,
        header_reserved,
    )
    return hdr + sec_bytes + trailing_file_bytes


def test_format_pack_unpack_and_serializer_negative_guards() -> None:
    """Cover input validation guards across pack/unpack/serialize/deserialize helpers."""
    from setun.ref.format import PackedTernaryTensor, pack_trits_1d

    with pytest.raises(ValueError, match="format.pack_trits_1d.*strictly in"):
        pack_trits_1d(np.array([1, 2, 0, -1], dtype=np.int8))
    with pytest.raises(ValueError, match="format.pack_trits_1d.*multiple of 4"):
        pack_trits_1d(np.array([1, 0, -1], dtype=np.int8))

    w_ok = np.ones((2, 64), dtype=np.int8)
    s_ok = np.ones((2, 1), dtype=np.uint16)
    with pytest.raises(ValueError, match="format.pack_ternary_tensor.*group_size"):
        pack_ternary_tensor(w_ok, s_ok, group_size=32)
    with pytest.raises(ValueError, match="format.pack_ternary_tensor.*shift"):
        pack_ternary_tensor(w_ok, s_ok, group_size=64, shift=-1)
    with pytest.raises(ValueError, match="format.pack_ternary_tensor.*shift"):
        pack_ternary_tensor(w_ok, s_ok, group_size=64, shift=256)
    with pytest.raises(ValueError, match="format.pack_ternary_tensor.*2D weight matrix"):
        pack_ternary_tensor(np.ones(64, dtype=np.int8), s_ok, group_size=64)
    with pytest.raises(ValueError, match="format.pack_ternary_tensor.*must be > 0"):
        pack_ternary_tensor(np.zeros((0, 64), dtype=np.int8), np.zeros((0, 1), dtype=np.uint16), 64)
    with pytest.raises(ValueError, match="format.pack_ternary_tensor.*scales shape"):
        pack_ternary_tensor(w_ok, np.ones((2, 2), dtype=np.uint16), group_size=64)

    # Corrupted in-memory non-zero tail padding beyond n_cols rejected on unpack
    ragged = pack_ternary_tensor(
        np.ones((1, 6), dtype=np.int8), np.ones((1, 1), dtype=np.uint16), 64
    )
    bad_trits = ragged.packed_trits.copy()
    bad_trits[0, 0, 15] = 0b01
    corrupted = PackedTernaryTensor(
        packed_trits=bad_trits,
        scales=ragged.scales,
        n_rows=ragged.n_rows,
        n_cols=ragged.n_cols,
        group_size=ragged.group_size,
        shift=ragged.shift,
    )
    with pytest.raises(ValueError, match="format.unpack_ternary_tensor.*non-zero padding"):
        unpack_ternary_tensor(corrupted)

    with pytest.raises(ValueError, match="format.validate_ttape.*reader_profile"):
        validate_ttape(b"", reader_profile="bogus")

    # serialize_packed_weights guards
    valid_packed = pack_ternary_tensor(w_ok, s_ok, group_size=64, shift=31)
    with pytest.raises(ValueError, match="format.serialize_packed_weights.*shift"):
        serialize_packed_weights(valid_packed)

    bad_grp = PackedTernaryTensor(valid_packed.packed_trits, s_ok, 2, 64, group_size=32, shift=0)
    with pytest.raises(ValueError, match="format.serialize_packed_weights.*group_size"):
        serialize_packed_weights(bad_grp)

    bad_dims = PackedTernaryTensor(valid_packed.packed_trits, s_ok, 0, 64, group_size=64, shift=0)
    with pytest.raises(ValueError, match="format.serialize_packed_weights.*uint16"):
        serialize_packed_weights(bad_dims)

    # deserialize_packed_weights rejects multi-layer .ttape
    lux_blob = (FIXTURES_DIR / "accept" / "lux_chain_grouped_mlp.ttape").read_bytes()
    with pytest.raises(TtapeValidationError) as exc_info:
        deserialize_packed_weights(lux_blob)
    assert exc_info.value.error_code == "TTAPE_ERR_MALFORMED_ARCH"


def test_ttape_core_positive_scale_exp_and_extended_profile_audit() -> None:
    """Cover TTAPE-core positive scale_exp (1..15 vs >15) and TTAPE-extended fp32_aux roles."""
    import struct

    from setun.ref.format import (
        ARCH_LAYER_STRUCT,
        AUX_ROLE_EMBEDDING,
        AUX_ROLE_RMSNORM,
        AUX_ROLE_SOFTMAX,
        FLAG_NON_CORE,
        FP32_AUX_REC_STRUCT,
        LAYER_HOST_EMBEDDING,
        LAYER_HOST_SOFTMAX_HEAD,
        LAYER_RMSNORM_SHIFT,
        LAYER_TERNARY_DENSE,
        TAG_ARCH_DESCRIPTOR,
        TAG_FP32_AUX,
        TAG_SCALES,
        TAG_WEIGHTS,
        pack_trits_1d,
    )

    w_bytes = pack_trits_1d(np.array([[1, -1, 0, 1]], dtype=np.int8)).tobytes()
    arch_1d = struct.pack("<HH", 1, 0) + ARCH_LAYER_STRUCT.pack(
        LAYER_TERNARY_DENSE, 4, 1, 2, 0, 2, 64
    )

    # 1. TTAPE-core with 0 < scale_exp <= 15 (exp=3 -> shift=0, scale_u16=8)
    core_pos = _build_framed_ttape(
        [
            (TAG_WEIGHTS, w_bytes, 0),
            (TAG_SCALES, struct.pack("<h", 3), 0),
            (TAG_ARCH_DESCRIPTOR, arch_1d, 0),
        ],
        flags=0,
    )
    assert validate_ttape(core_pos) == "TTAPE_OK"
    layers_pos = load_ttape_layers(core_pos)
    _, s_pos, sh_pos = unpack_ternary_tensor(layers_pos[0])
    assert sh_pos == 0 and int(s_pos[0, 0]) == 8

    # 2. TTAPE-core with scale_exp = 16 (valid wire TTAPE-core, exceeds Setun uint16 loader range)
    core_exp16 = _build_framed_ttape(
        [
            (TAG_WEIGHTS, w_bytes, 0),
            (TAG_SCALES, struct.pack("<h", 16), 0),
            (TAG_ARCH_DESCRIPTOR, arch_1d, 0),
        ],
        flags=0,
    )
    assert validate_ttape(core_exp16, reader_profile="core") == "TTAPE_OK"
    with pytest.raises(TtapeValidationError) as exc_16:
        load_ttape_layers(core_exp16)
    assert exc_16.value.error_code == "TTAPE_ERR_INVALID_SCALE"

    # 3. TTAPE-extended positive load (including AUX_ROLE_SOFTMAX on LAYER_HOST_SOFTMAX_HEAD)
    ext_fixture = (FIXTURES_DIR / "accept" / "extended_with_fp32_aux.ttape").read_bytes()
    ext_layers = load_ttape_layers(ext_fixture, reader_profile="extended")
    assert len(ext_layers) == 1

    arch_ext = (
        struct.pack("<HH", 4, 0)
        + ARCH_LAYER_STRUCT.pack(LAYER_HOST_EMBEDDING, 4, 2, 0, 0, 0, 64)
        + ARCH_LAYER_STRUCT.pack(LAYER_RMSNORM_SHIFT, 4, 4, 0, 0, 0, 64)
        + ARCH_LAYER_STRUCT.pack(LAYER_TERNARY_DENSE, 4, 4, 2, 0, 2, 64)
        + ARCH_LAYER_STRUCT.pack(LAYER_HOST_SOFTMAX_HEAD, 4, 3, 0, 0, 0, 64)
    )
    w_4x4 = pack_trits_1d(np.ones((4, 4), dtype=np.int8)).tobytes()
    fp32_payload = (
        struct.pack("<HH", 3, 0)
        + FP32_AUX_REC_STRUCT.pack(0, AUX_ROLE_EMBEDDING, 0, 8)
        + np.ones(8, dtype="<f4").tobytes()
        + FP32_AUX_REC_STRUCT.pack(1, AUX_ROLE_RMSNORM, 0, 4)
        + np.ones(4, dtype="<f4").tobytes()
        + FP32_AUX_REC_STRUCT.pack(3, AUX_ROLE_SOFTMAX, 0, 3)
        + np.ones(3, dtype="<f4").tobytes()
    )
    ext_softmax = _build_framed_ttape(
        [
            (TAG_WEIGHTS, w_4x4, 0),
            (TAG_SCALES, struct.pack("<h", -2), 0),
            (TAG_ARCH_DESCRIPTOR, arch_ext, 0),
            (TAG_FP32_AUX, fp32_payload, 0),
        ],
        flags=FLAG_NON_CORE,
    )
    assert validate_ttape(ext_softmax, reader_profile="extended") == "TTAPE_OK"
    assert len(load_ttape_layers(ext_softmax, reader_profile="extended")) == 1
    # Host layer kinds {6, 7} without 0x0007 are forbidden in core and unsupported in grouped
    no_aux_host = _build_framed_ttape(
        [
            (TAG_WEIGHTS, w_4x4, 0),
            (TAG_SCALES, struct.pack("<h", -2), 0),
            (TAG_ARCH_DESCRIPTOR, arch_ext, 0),
        ],
        flags=0,
    )
    assert (
        validate_ttape(no_aux_host, reader_profile="extended")
        == "TTAPE_ERR_PROFILE_SECTION_FORBIDDEN"
    )

    grouped_host_only = _build_framed_ttape(
        [
            (TAG_WEIGHTS, w_4x4, 0),
            (TAG_SCALES, struct.pack("<h", -2), 0),
            (TAG_ARCH_DESCRIPTOR, arch_ext, 0),
            (
                0x0006,
                struct.pack("<HH", 1, 0)
                + struct.pack("<HHBBH", 2, 64, 2, 0, 1)
                + np.ones(4, dtype="<u2").tobytes(),
                0,
            ),
        ],
        flags=FLAG_NON_CORE,
    )
    assert (
        validate_ttape(grouped_host_only, reader_profile="grouped")
        == "TTAPE_ERR_UNSUPPORTED_PROFILE"
    )


def test_ttape_section4_exhaustive_structural_error_branches() -> None:
    """Exercise every §4 _audit_ttape structural error branch with valid FNV-1a checksums."""
    import struct

    from setun.ref.format import (
        ARCH_LAYER_STRUCT,
        FLAG_LINKS,
        FLAG_MANIFEST,
        FLAG_NON_CORE,
        FP32_AUX_REC_STRUCT,
        GROUP_SCALE_REC_STRUCT,
        LAYER_HOST_EMBEDDING,
        LAYER_RMSNORM_SHIFT,
        LAYER_TERNARY_DENSE,
        SECTION_HEADER_STRUCT,
        TAG_ARCH_DESCRIPTOR,
        TAG_FP32_AUX,
        TAG_GROUP_SCALES_U16,
        TAG_LINK_TABLE,
        TAG_MANIFEST,
        TAG_SCALES,
        TAG_WEIGHTS,
        _fnv1a_32,
        pack_trits_1d,
    )

    w1 = pack_trits_1d(np.array([[1, -1, 0, 1]], dtype=np.int8)).tobytes()
    s1 = struct.pack("<h", -2)
    arch1 = struct.pack("<HH", 1, 0) + ARCH_LAYER_STRUCT.pack(
        LAYER_TERNARY_DENSE, 4, 1, 2, 0, 2, 64
    )
    base_secs = [(TAG_WEIGHTS, w1, 0), (TAG_SCALES, s1, 0), (TAG_ARCH_DESCRIPTOR, arch1, 0)]

    # Header & framing errors
    assert validate_ttape(b"short") == "TTAPE_ERR_NULL_OR_TRUNCATED"
    assert (
        validate_ttape(_build_framed_ttape(base_secs, header_reserved=b"\x01" + b"\x00" * 7))
        == "TTAPE_ERR_NONZERO_RESERVED"
    )
    assert (
        validate_ttape(_build_framed_ttape(base_secs[:2], sec_count_override=2))
        == "TTAPE_ERR_SECTION_COUNT_MISMATCH"
    )
    assert (
        validate_ttape(_build_framed_ttape(base_secs, trailing_file_bytes=b"\x00"))
        == "TTAPE_ERR_LENGTH_MISMATCH"
    )
    assert (
        validate_ttape(
            _build_framed_ttape(base_secs, sec_count_override=4, trailing_section_bytes=b"abcd")
        )
        == "TTAPE_ERR_LENGTH_MISMATCH"
    )
    assert (
        validate_ttape(_build_framed_ttape([(TAG_WEIGHTS, w1, 1), *base_secs[1:]]))
        == "TTAPE_ERR_NONZERO_RESERVED"
    )
    # Section slen > rem
    bad_slen_sec = SECTION_HEADER_STRUCT.pack(TAG_WEIGHTS, 0, 9999, _fnv1a_32(w1)) + w1
    bad_slen_tape = _build_framed_ttape(
        base_secs[:2], sec_count_override=3, trailing_section_bytes=bad_slen_sec
    )
    assert validate_ttape(bad_slen_tape) == "TTAPE_ERR_LENGTH_MISMATCH"

    # Duplicate tag & trailing bytes after sec_count sections
    dup_tape = _build_framed_ttape([*base_secs, (TAG_WEIGHTS, w1, 0)])
    assert validate_ttape(dup_tape) == "TTAPE_ERR_DUPLICATE_TAG"
    trail_sec_tape = _build_framed_ttape(base_secs, trailing_section_bytes=b"\x00" * 4)
    assert validate_ttape(trail_sec_tape) == "TTAPE_ERR_LENGTH_MISMATCH"

    # Arch descriptor (0x0003) branches
    for bad_arch, exp_code in [
        (b"\x01\x00", "TTAPE_ERR_MALFORMED_ARCH"),
        (struct.pack("<HH", 1, 1) + b"\x00" * 16, "TTAPE_ERR_NONZERO_RESERVED"),
        (struct.pack("<HH", 0, 0), "TTAPE_ERR_MALFORMED_ARCH"),
        (struct.pack("<HH", 1, 0) + b"\x00" * 8, "TTAPE_ERR_MALFORMED_ARCH"),
        (
            struct.pack("<HH", 1, 0)
            + ARCH_LAYER_STRUCT.pack(LAYER_TERNARY_DENSE, 4, 1, 2, 1, 2, 64),
            "TTAPE_ERR_NONZERO_RESERVED",
        ),
        (
            struct.pack("<HH", 1, 0) + ARCH_LAYER_STRUCT.pack(0, 4, 1, 0, 0, 0, 64),
            "TTAPE_ERR_MALFORMED_ARCH",
        ),
        (
            struct.pack("<HH", 1, 0)
            + ARCH_LAYER_STRUCT.pack(LAYER_TERNARY_DENSE, 4, 1, 9, 0, 9, 64),
            "TTAPE_ERR_MALFORMED_ARCH",
        ),
        (
            struct.pack("<HH", 1, 0)
            + ARCH_LAYER_STRUCT.pack(LAYER_RMSNORM_SHIFT, 4, 4, 1, 0, 0, 64),
            "TTAPE_ERR_MALFORMED_ARCH",
        ),
        (
            struct.pack("<HH", 1, 0)
            + ARCH_LAYER_STRUCT.pack(LAYER_RMSNORM_SHIFT, 4, 8, 0, 0, 0, 64),
            "TTAPE_ERR_MALFORMED_ARCH",
        ),
        (
            struct.pack("<HH", 1, 0)
            + ARCH_LAYER_STRUCT.pack(LAYER_RMSNORM_SHIFT, 4, 4, 0, 0, 0, 64),
            "TTAPE_ERR_DIM_PAYLOAD_MISMATCH",
        ),
    ]:
        t = _build_framed_ttape(
            [(TAG_WEIGHTS, w1, 0), (TAG_SCALES, s1, 0), (TAG_ARCH_DESCRIPTOR, bad_arch, 0)]
        )
        assert validate_ttape(t) == exp_code

    # Scale out of [-30, +30]
    bad_scale_t = _build_framed_ttape(
        [
            (TAG_WEIGHTS, w1, 0),
            (TAG_SCALES, struct.pack("<h", 31), 0),
            (TAG_ARCH_DESCRIPTOR, arch1, 0),
        ]
    )
    assert validate_ttape(bad_scale_t) == "TTAPE_ERR_INVALID_SCALE"

    # Manifest (0x0004) branches
    valid_m_fields = [b"prod", b"R" * 32, b"alg", b"prm", b"H" * 8]
    valid_m = b"".join(struct.pack("<H", len(f)) + f for f in valid_m_fields)
    for bad_m in [
        b"\x01",  # truncated prefix
        struct.pack("<H", 0),  # zero length field
        valid_m + b"\x00",  # trailing byte
    ]:
        mt = _build_framed_ttape([*base_secs, (TAG_MANIFEST, bad_m, 0)], flags=FLAG_MANIFEST)
        assert validate_ttape(mt) == "TTAPE_ERR_MALFORMED_MANIFEST"

    # Link table (0x0005) branches
    for bad_lnk in [
        b"\x01\x02",  # not multiple of 4
        struct.pack("<i", 0),  # self-link delta == 0
        struct.pack("<ii", 2, -4),  # unaligned delta
        struct.pack("<i", 8),  # out-of-bounds target
    ]:
        lt = _build_framed_ttape([*base_secs, (TAG_LINK_TABLE, bad_lnk, 0)], flags=FLAG_LINKS)
        assert validate_ttape(lt) == "TTAPE_ERR_INVALID_LINK"

    # Group scales u16 (0x0006) branches
    for bad_g, exp_g_code in [
        (b"\x01\x00", "TTAPE_ERR_MALFORMED_GROUP_SCALES"),
        (struct.pack("<HH", 1, 1), "TTAPE_ERR_NONZERO_RESERVED"),
        (struct.pack("<HH", 2, 0), "TTAPE_ERR_MALFORMED_GROUP_SCALES"),
        (struct.pack("<HH", 1, 0) + b"\x00\x01", "TTAPE_ERR_MALFORMED_GROUP_SCALES"),
        (
            struct.pack("<HH", 1, 0) + GROUP_SCALE_REC_STRUCT.pack(0, 64, 2, 1, 1) + b"\x01\x00",
            "TTAPE_ERR_NONZERO_RESERVED",
        ),
        (
            struct.pack("<HH", 1, 0)
            + GROUP_SCALE_REC_STRUCT.pack(0, 64, 2, 0, 9)
            + b"\x01\x00" * 9,
            "TTAPE_ERR_MALFORMED_GROUP_SCALES",
        ),
        (
            struct.pack("<HH", 1, 0) + GROUP_SCALE_REC_STRUCT.pack(0, 64, 2, 0, 1),
            "TTAPE_ERR_MALFORMED_GROUP_SCALES",
        ),
        (
            struct.pack("<HH", 1, 0)
            + GROUP_SCALE_REC_STRUCT.pack(0, 64, 2, 0, 1)
            + b"\x01\x00\xff",
            "TTAPE_ERR_MALFORMED_GROUP_SCALES",
        ),
    ]:
        gt = _build_framed_ttape(
            [*base_secs, (TAG_GROUP_SCALES_U16, bad_g, 0)], flags=FLAG_NON_CORE
        )
        assert validate_ttape(gt) == exp_g_code

    # FP32 aux (0x0007) branches under reader_profile="extended"
    arch_aux = (
        struct.pack("<HH", 2, 0)
        + ARCH_LAYER_STRUCT.pack(LAYER_HOST_EMBEDDING, 4, 2, 0, 0, 0, 64)
        + ARCH_LAYER_STRUCT.pack(LAYER_TERNARY_DENSE, 4, 1, 2, 0, 2, 64)
    )
    aux_base = [(TAG_WEIGHTS, w1, 0), (TAG_SCALES, s1, 0), (TAG_ARCH_DESCRIPTOR, arch_aux, 0)]
    for bad_a, exp_a_code in [
        (b"\x01\x00", "TTAPE_ERR_MALFORMED_FP32_AUX"),
        (struct.pack("<HH", 1, 1), "TTAPE_ERR_NONZERO_RESERVED"),
        (struct.pack("<HH", 0, 0), "TTAPE_ERR_MALFORMED_FP32_AUX"),
        (struct.pack("<HH", 1, 0) + b"\x00\x01", "TTAPE_ERR_MALFORMED_FP32_AUX"),
        (
            struct.pack("<HH", 1, 0) + FP32_AUX_REC_STRUCT.pack(0, 1, 1, 8) + b"\x00" * 32,
            "TTAPE_ERR_NONZERO_RESERVED",
        ),
        (
            struct.pack("<HH", 1, 0) + FP32_AUX_REC_STRUCT.pack(9, 1, 0, 8) + b"\x00" * 32,
            "TTAPE_ERR_MALFORMED_FP32_AUX",
        ),
        (
            struct.pack("<HH", 1, 0) + FP32_AUX_REC_STRUCT.pack(0, 2, 0, 4) + b"\x00" * 16,
            "TTAPE_ERR_MALFORMED_FP32_AUX",
        ),
        (
            struct.pack("<HH", 1, 0) + FP32_AUX_REC_STRUCT.pack(0, 1, 0, 8) + b"\x00" * 36,
            "TTAPE_ERR_MALFORMED_FP32_AUX",
        ),
    ]:
        at = _build_framed_ttape([*aux_base, (TAG_FP32_AUX, bad_a, 0)], flags=FLAG_NON_CORE)
        assert validate_ttape(at, reader_profile="extended") == exp_a_code
