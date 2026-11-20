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
    """Validates all 20 accept/*.ttape and reject/*.ttape fixtures from third_party/TernaryTape."""
    accept_files = sorted((FIXTURES_DIR / "accept").glob("*.ttape"))
    reject_files = sorted((FIXTURES_DIR / "reject").glob("*.ttape"))
    assert len(accept_files) == 4
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
