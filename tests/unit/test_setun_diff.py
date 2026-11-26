"""Tests for Task 3.5: Trace container, 1-bit GEMV fault localization, and `setun diff` CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from setun.cli import main as cli_main
from setun.ref.m0 import PRESET_CONFIGS, build_toy_m0_model, generate_greedy, serialize_m0_ttape
from setun.runtime.m0_runner import HybridM0Runner, open_m0_backend
from setun.runtime.trace import (
    TraceEntry,
    TraceRecorder,
    deserialize_trace,
    diff_traces,
    format_diff_report,
    load_trace,
    save_trace,
    serialize_trace,
)


def test_trace_codec_roundtrip_and_validation_guards(tmp_path: Path) -> None:
    """Verify STRC v1 binary trace round-trip, FNV-1a-64 hashes, and corrupt-blob guards."""
    rec = TraceRecorder(metadata={"backend": "ref", "model": "m0-tiny"})
    rec("embed", np.array([1.0, -2.5, 3.25, 0.0], dtype=np.float32))
    rec("layers.0.q_proj", np.arange(8, dtype=np.int32).reshape(2, 4))
    rec("embed", np.array([0.5, 1.5, -0.5, 2.0], dtype=np.float32))

    assert len(rec.entries) == 3
    assert rec.entries[0].step == 0 and rec.entries[0].op_id == "step0.embed"
    assert rec.entries[1].step == 0 and rec.entries[1].op_id == "step0.layers.0.q_proj"
    assert rec.entries[2].step == 1 and rec.entries[2].op_id == "step1.embed"
    assert len(rec.entries[0].fnv1a64) == 16

    trace_path = tmp_path / "run.strc"
    save_trace(trace_path, rec.entries, metadata=rec.metadata)
    loaded, meta = load_trace(str(trace_path))
    assert meta == {"backend": "ref", "model": "m0-tiny"}
    assert len(loaded) == 3
    for orig, rest in zip(rec.entries, loaded, strict=True):
        assert orig.op_id == rest.op_id
        assert orig.shape == rest.shape
        assert orig.dtype == rest.dtype
        assert orig.fnv1a64 == rest.fnv1a64
        assert np.array_equal(orig.tensor, rest.tensor)

    # Also support load_trace directly from bytes
    loaded_from_bytes, _ = load_trace(trace_path.read_bytes())
    assert len(loaded_from_bytes) == 3

    # Non-'embed' initial op starts cleanly at step 0
    mlp_rec = TraceRecorder()
    mlp_rec("layers.0", np.array([1, -1], dtype=np.int64))
    assert mlp_rec.entries[0].step == 0 and mlp_rec.entries[0].op_id == "step0.layers.0"

    # Validation guards on malformed STRC blobs
    valid_blob = serialize_trace(rec.entries)
    with pytest.raises(ValueError, match="truncated"):
        deserialize_trace(b"ST")
    with pytest.raises(ValueError, match="magic"):
        deserialize_trace(b"NOPE" + valid_blob[4:])

    bad_hdr_len = b"STRC" + (999_999).to_bytes(4, "little") + b"{}"
    with pytest.raises(ValueError, match="header"):
        deserialize_trace(bad_hdr_len)

    bad_ver = serialize_trace(rec.entries).replace(b'"version":1', b'"version":9')
    with pytest.raises(ValueError, match="version"):
        deserialize_trace(bad_ver)

    with pytest.raises(ValueError, match="payload out of bounds"):
        deserialize_trace(valid_blob[:-2])

    corrupted = bytearray(valid_blob)
    corrupted[-1] ^= 0x01
    with pytest.raises(ValueError, match="hash mismatch"):
        deserialize_trace(bytes(corrupted))


def test_gemv_1bit_fault_injection_localized_to_exact_op_and_row(tmp_path: Path) -> None:
    """Task 3.5 Gate: Inject 1-bit GEMV fault and localize exact op & row."""
    cfg = PRESET_CONFIGS["tiny"]
    model = build_toy_m0_model(cfg, seed=1958)
    prompt = [1, 42]

    # Reference golden trace via setun.ref.m0.generate_greedy
    ref_rec = TraceRecorder(metadata={"source": "ref"})
    ref_tokens, _, _ = generate_greedy(
        model,
        prompt_tokens=prompt,
        max_new_tokens=2,
        mode="integer",
        trace_hook=ref_rec,
    )

    # Faulty hybrid run: flip bit 4 of row 73 in layers.1.gate_proj on forward step 2
    target_step = 2
    target_op = "layers.1.gate_proj"
    target_row = 73

    def inject_1bit_fault(step: int, key: str, y_i32: np.ndarray) -> np.ndarray:
        if step == target_step and key == target_op:
            y_i32[target_row] ^= 1 << 4
        return y_i32

    fault_rec = TraceRecorder(metadata={"source": "fault_injected"})
    with open_m0_backend(model, "ref") as drv:
        runner = HybridM0Runner(model, drv, gemv_fault_hook=inject_1bit_fault)
        runner.generate_greedy(prompt_tokens=prompt, max_new_tokens=2, trace_hook=fault_rec)

    assert len(ref_tokens) == 2
    res = diff_traces(ref_rec.entries, fault_rec.entries, context=3)
    assert not res.matched
    assert res.reason == "value_mismatch"
    assert res.step == target_step
    assert res.op_name == target_op
    assert res.op_id == f"step{target_step}.{target_op}"
    assert res.first_flat_index == target_row
    assert res.first_multi_index == (target_row,)
    assert res.num_mismatched == 1
    assert len(res.prev_ops) == 3
    assert res.prev_ops[-1].endswith("layers.1.ffn_norm")
    assert len(res.next_ops) > 0

    report = format_diff_report(res)
    assert f"step{target_step}.{target_op}" in report
    assert f"index=({target_row},)" in report
    assert "layers.1.ffn_norm" in report


def test_diff_traces_structural_and_tolerance_modes() -> None:
    """Cover match, tolerance, sign-zero bits, int mismatch, and structural diffs."""
    e0 = TraceRecorder()
    e0("embed", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    e0("proj", np.array([10, 20, 30], dtype=np.int32))

    # Exact match
    ok = diff_traces(e0.entries, e0.entries)
    assert ok.matched and "MATCH" in format_diff_report(ok)

    # Tolerance pass vs exact fail
    e_eps = TraceRecorder()
    e_eps("embed", np.array([[1.0, 2.0001], [3.0, 4.0]], dtype=np.float32))
    e_eps("proj", np.array([10, 20, 30], dtype=np.int32))
    assert not diff_traces(e0.entries, e_eps.entries).matched
    assert diff_traces(e0.entries, e_eps.entries, atol=1e-3).matched

    # Integer mismatch formatting & 2D multi-index localization
    e_int = TraceRecorder()
    e_int("embed", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    e_int("proj", np.array([10, 99, 30], dtype=np.int32))
    r_int = diff_traces(e0.entries, e_int.entries)
    assert not r_int.matched and r_int.first_multi_index == (1,)
    assert "20" in format_diff_report(r_int) and "99" in format_diff_report(r_int)

    r_2d = diff_traces(e0.entries, e_eps.entries)
    assert r_2d.first_multi_index == (0, 1)

    # -0.0 vs +0.0 bit-exact detection when atol==0 and rtol==0
    e_pos0 = TraceRecorder()
    e_neg0 = TraceRecorder()
    e_pos0("embed", np.array([0.0], dtype=np.float32))
    e_neg0("embed", np.array([-0.0], dtype=np.float32))
    assert not diff_traces(e_pos0.entries, e_neg0.entries).matched

    # Structural mismatches: length, op_id, shape, dtype
    r_len = diff_traces(e0.entries, e0.entries[:1])
    assert not r_len.matched and r_len.reason == "length_mismatch"
    assert "length_mismatch" in format_diff_report(r_len)

    e_op = [
        TraceEntry(
            seq=0,
            step=0,
            op_id="step0.other",
            op_name="other",
            shape=(2, 2),
            dtype="<f4",
            fnv1a64=e0.entries[0].fnv1a64,
            tensor=e0.entries[0].tensor,
        )
    ]
    r_op = diff_traces(e0.entries[:1], e_op)
    assert not r_op.matched and r_op.reason == "op_id_mismatch"
    assert "op_id_mismatch" in format_diff_report(r_op)

    e_shape = TraceRecorder()
    e_shape("embed", np.ones((4,), dtype=np.float32))
    r_shape = diff_traces(e0.entries[:1], e_shape.entries)
    assert not r_shape.matched and r_shape.reason == "shape_mismatch"
    assert "shape_mismatch" in format_diff_report(r_shape)

    e_dtype = TraceRecorder()
    e_dtype("embed", np.array([[1, 2], [3, 4]], dtype=np.int32))
    r_dtype = diff_traces(e0.entries[:1], e_dtype.entries)
    assert not r_dtype.matched and r_dtype.reason == "dtype_mismatch"
    assert "dtype_mismatch" in format_diff_report(r_dtype)


def test_cli_run_trace_and_diff_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify `setun run --trace` + `setun diff` CLI return codes (0 match, 1 mismatch) & guards."""
    model = build_toy_m0_model(PRESET_CONFIGS["tiny"], seed=1958)
    tape_path = tmp_path / "m0.ttape"
    tape_path.write_bytes(serialize_m0_ttape(model))

    trace_a = tmp_path / "trace_a.strc"
    trace_b = tmp_path / "trace_b.strc"

    rc_run_a = cli_main(
        ["run", str(tape_path), "--backend", "ref", "--tokens", "2", "--trace", str(trace_a)]
    )
    rc_run_b = cli_main(
        ["run", str(tape_path), "--backend", "ref", "--tokens", "2", "--trace", str(trace_b)]
    )
    assert rc_run_a == 0 and rc_run_b == 0
    capsys.readouterr()

    # Identical traces -> exit code 0
    rc_eq = cli_main(["diff", str(trace_a), str(trace_b)])
    assert rc_eq == 0
    out_eq = capsys.readouterr().out
    assert "MATCH" in out_eq

    # Divergent trace (different prompt) -> exit code 1
    trace_c = tmp_path / "trace_c.strc"
    cli_main(
        [
            "run",
            str(tape_path),
            "--backend",
            "ref",
            "--tokens",
            "2",
            "--prompt",
            "2",
            "--trace",
            str(trace_c),
        ]
    )
    capsys.readouterr()

    rc_diff = cli_main(["diff", str(trace_a), str(trace_c), "--context", "2"])
    assert rc_diff == 1
    out_diff = capsys.readouterr().out
    assert "MISMATCH" in out_diff and "step0.embed" in out_diff

    # CLI validation guards
    with pytest.raises(ValueError, match="--atol must be >= 0"):
        cli_main(["diff", str(trace_a), str(trace_b), "--atol", "-0.1"])
    with pytest.raises(ValueError, match="--rtol must be >= 0"):
        cli_main(["diff", str(trace_a), str(trace_b), "--rtol", "-0.1"])
    with pytest.raises(ValueError, match="--context must be >= 0"):
        cli_main(["diff", str(trace_a), str(trace_b), "--context", "-1"])
