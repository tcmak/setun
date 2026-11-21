"""End-to-end LiteX SoC pipeline composition & deterministic CSR map tests (Task C1.1c)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from litex.soc.integration import export

from setun.ref.format import load_ttape_layers, pack_ternary_tensor, serialize_packed_weights
from setun.ref.m0 import forward_ternary_chain, ternary_matvec_integer
from setun.soc import DEFAULT_CSR_HEADER, DEFAULT_CSR_JSON, TernaryAccel, export_soc_csr_artifacts
from targets.sim import SetunSimSoC
from tests.sim_helper import run_migen_sim

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "third_party" / "TernaryTape" / "fixtures" / "accept"
REPORTS_DIR = REPO_ROOT / "reports"


def _to_signed_i32(word_u32: int) -> int:
    val = int(word_u32) & 0xFFFFFFFF
    return val - 0x100000000 if (val & 0x80000000) else val


def _pack_activation_words_i8(acts_i8: np.ndarray, k_pad: int) -> list[int]:
    padded = np.zeros(k_pad, dtype=np.int8)
    padded[: len(acts_i8)] = np.asarray(acts_i8, dtype=np.int8)
    raw = padded.view(np.uint8)
    words: list[int] = []
    for off in range(0, k_pad, 4):
        words.append(int.from_bytes(raw[off : off + 4].tobytes(), byteorder="little"))
    return words


def _prepare_layer_stream_words(
    layer,
    beat_bits: int,
) -> tuple[list[tuple[int, int]], list[int], int]:
    """Slice PackedTernaryTensor bytes into beat_bits words + row_last flags and flat scales."""
    bytes_per_word = beat_bits // 8
    n_rows = layer.n_rows
    n_groups = layer.scales.shape[1]
    k_pad = n_groups * layer.group_size
    row_bytes = k_pad // 4
    if row_bytes % bytes_per_word != 0:
        raise ValueError(
            f"row_bytes ({row_bytes}) must be divisible by bytes_per_word ({bytes_per_word})"
        )
    words_per_row = row_bytes // bytes_per_word

    trit_words: list[tuple[int, int]] = []
    for r in range(n_rows):
        row_flat = layer.packed_trits[r].reshape(-1)
        for w_idx in range(words_per_row):
            chunk = row_flat[w_idx * bytes_per_word : (w_idx + 1) * bytes_per_word]
            word_val = int.from_bytes(chunk.tobytes(), byteorder="little")
            is_row_last = 1 if (w_idx == words_per_row - 1) else 0
            trit_words.append((word_val, is_row_last))

    flat_scales = [int(s) for s in layer.scales.reshape(-1)]
    return trit_words, flat_scales, k_pad


def _build_soc_with_csr_map(
    *,
    lanes: int,
    group_size: int,
    beat_bits: int,
) -> tuple[SetunSimSoC, dict[str, int], dict[str, dict[str, int]]]:
    soc = SetunSimSoC(
        with_accel_stub=True,
        with_ternary_accel=True,
        accel_lanes=lanes,
        accel_group_size=group_size,
        accel_beat_bits=beat_bits,
        expose_host_wb=True,
        use_external_crg=False,
        cpu_type=None,
        integrated_rom_size=0,
        integrated_main_ram_size=0x1000,
    )
    soc.finalize()
    csr_json_str = export.get_csr_json(
        soc=soc,
        csr_regions=soc.csr.regions,
        constants=soc.constants,
        mem_regions=soc.mem_regions,
    )
    csr_data = json.loads(csr_json_str)
    addrs = {name: int(meta["addr"]) for name, meta in csr_data["csr_registers"].items()}
    return soc, addrs, csr_data


def _run_ttape_layers_on_soc(
    layers: list,
    x_init: np.ndarray,
    *,
    lanes: int,
    group_size: int,
    beat_bits: int,
    seed: int,
) -> list[np.ndarray]:
    soc, addrs, _ = _build_soc_with_csr_map(
        lanes=lanes,
        group_size=group_size,
        beat_bits=beat_bits,
    )
    accel: TernaryAccel = soc.ternary_accel
    wb = soc.host_wb
    rng = np.random.default_rng(seed)

    ctrl_addr = addrs["ternary_accel_ctrl"]
    cfg_addr = addrs["ternary_accel_cfg"]
    info_addr = addrs["ternary_accel_info"]
    status_addr = addrs["ternary_accel_status"]
    act_idx_addr = addrs["ternary_accel_act_word_idx"]
    act_data_addr = addrs["ternary_accel_act_word_data"]
    out_idx_addr = addrs["ternary_accel_out_row_idx"]
    out_data_addr = addrs["ternary_accel_out_row_data"]

    layer_outputs: list[np.ndarray] = []

    def wb_read(byte_addr: int):
        val = yield from wb.read(byte_addr >> 2)
        return val

    def wb_write(byte_addr: int, data: int):
        yield from wb.write(byte_addr >> 2, data & 0xFFFFFFFF)

    def tb():
        # Verify hardware geometry CSR matches configured (lanes, group_size, beat_bits)
        got_info = yield from wb_read(info_addr)
        exp_info = (lanes & 0xFF) | ((group_size & 0xFFF) << 8) | ((beat_bits & 0xFFF) << 20)
        assert got_info == exp_info, (
            f"[test_soc_pipeline:TernaryAccel] info CSR mismatch: "
            f"expected 0x{exp_info:08x}, got 0x{got_info:08x}"
        )

        cur_x = np.asarray(x_init, dtype=np.int8)
        sim_cyc = 0

        for l_idx, layer in enumerate(layers):
            trit_words, flat_scales, k_pad = _prepare_layer_stream_words(layer, beat_bits)
            act_words = _pack_activation_words_i8(cur_x, k_pad)

            # Clear state & keep enabled (bit 0 = enable, bit 1 = clear pulse)
            yield from wb_write(ctrl_addr, 0b11)
            # Configure shift (bits 0..5) and target num_rows (bits 8..23)
            cfg_word = (int(layer.shift) & 0x3F) | ((int(layer.n_rows) & 0xFFFF) << 8)
            yield from wb_write(cfg_addr, cfg_word)

            # Stage quantized int8 activations over Wishbone CSRs
            yield from wb_write(act_idx_addr, 0)
            for w_val in act_words:
                yield from wb_write(act_data_addr, w_val)

            # Stream packed .ttape trits & group scales with independent random bubbles
            # AND a mid-stream CSR enable=0 stall pulse to verify backpressure propagation.
            w_ptr = 0
            s_ptr = 0
            trit_bubble_rem = 2  # start with multi-cycle bubble
            scale_bubble_rem = 1  # start with 1-cycle bubble
            did_mid_stall = False

            while w_ptr < len(trit_words) or s_ptr < len(flat_scales):
                if not did_mid_stall and len(trit_words) >= 2 and w_ptr == len(trit_words) // 2:
                    did_mid_stall = True
                    # Lower upstream valids during the multi-cycle Wishbone CSR write, then assert
                    # valid inputs while ctrl.enable=0 to verify backpressure propagation.
                    yield accel.trit_valid.eq(0)
                    yield accel.scale_valid.eq(0)
                    yield
                    yield from wb_write(ctrl_addr, 0b00)
                    if w_ptr < len(trit_words):
                        yield accel.trit_data.eq(trit_words[w_ptr][0])
                        yield accel.trit_last.eq(trit_words[w_ptr][1])
                        yield accel.trit_valid.eq(1)
                    if s_ptr < len(flat_scales):
                        yield accel.scale_data.eq(flat_scales[s_ptr])
                        yield accel.scale_valid.eq(1)
                    yield
                    sim_cyc += 1
                    stalled_tr = yield accel.trit_ready
                    stalled_sr = yield accel.scale_ready
                    assert stalled_tr == 0 and stalled_sr == 0, (
                        f"[test_soc_pipeline:TernaryAccel] backpressure invariant violated at "
                        f"cycle {sim_cyc} (layer={l_idx}): expected trit_ready=0, scale_ready=0 "
                        f"when ctrl.enable=0, got ({stalled_tr}, {stalled_sr})"
                    )
                    yield accel.trit_valid.eq(0)
                    yield accel.scale_valid.eq(0)
                    yield from wb_write(ctrl_addr, 0b01)

                drive_t_valid = 0
                if w_ptr < len(trit_words):
                    if trit_bubble_rem > 0:
                        trit_bubble_rem -= 1
                        yield accel.trit_data.eq((1 << beat_bits) - 1)  # adversarial 0b11 noise
                        yield accel.trit_last.eq(1)
                        yield accel.trit_valid.eq(0)
                    else:
                        drive_t_valid = 1
                        yield accel.trit_data.eq(trit_words[w_ptr][0])
                        yield accel.trit_last.eq(trit_words[w_ptr][1])
                        yield accel.trit_valid.eq(1)
                else:
                    yield accel.trit_valid.eq(0)
                    yield accel.trit_last.eq(0)

                drive_s_valid = 0
                if s_ptr < len(flat_scales):
                    if scale_bubble_rem > 0:
                        scale_bubble_rem -= 1
                        yield accel.scale_data.eq(0xFFFF)
                        yield accel.scale_valid.eq(0)
                    else:
                        drive_s_valid = 1
                        yield accel.scale_data.eq(flat_scales[s_ptr])
                        yield accel.scale_valid.eq(1)
                else:
                    yield accel.scale_valid.eq(0)

                yield
                sim_cyc += 1

                # Note Migen Simulator.run() order: sync ran on rising edge before generator,
                # and combinational ready reflects current settled signals after commit.
                # Wait one settle step or check handshake cleanly:
                t_ready = yield accel.trit_ready
                s_ready = yield accel.scale_ready
                if drive_t_valid and t_ready:
                    w_ptr += 1
                    trit_bubble_rem = int(rng.choice([0, 0, 1, 2, 4]))
                elif drive_t_valid and not t_ready and rng.random() < 0.35:
                    # Inject intra-word (mid-slice) trit_valid=0 bubble before final slice!
                    trit_bubble_rem = int(rng.choice([1, 2]))
                if drive_s_valid and s_ready:
                    s_ptr += 1
                    scale_bubble_rem = int(rng.choice([0, 1, 3]))
                elif drive_s_valid and not s_ready and rng.random() < 0.35:
                    # Inject mid-group scale_valid=0 bubble between beats of the same group!
                    scale_bubble_rem = int(rng.choice([1, 3]))

            yield accel.trit_valid.eq(0)
            yield accel.trit_last.eq(0)
            yield accel.scale_valid.eq(0)

            # Poll status.busy (bit 0) with a tight bound while R-row tile drains at 1 row/cycle
            status_val = 0
            for _poll in range(16):
                status_val = yield from wb_read(status_addr)
                if (status_val & 0x1) == 0:
                    break
            done_bit = (status_val >> 1) & 0x1
            rows_done = (status_val >> 16) & 0xFFFF
            assert done_bit == 1 and rows_done == layer.n_rows, (
                f"[test_soc_pipeline:TernaryAccel] completion status invariant violated at "
                f"layer {l_idx} (cycle {sim_cyc}): expected done=1, rows_completed={layer.n_rows}, "
                f"got status=0x{status_val:08x} (done={done_bit}, rows_completed={rows_done})"
            )

            got_vec = np.zeros(layer.n_rows, dtype=np.int64)
            for r in range(layer.n_rows):
                yield from wb_write(out_idx_addr, r)
                raw_out = yield from wb_read(out_data_addr)
                got_vec[r] = _to_signed_i32(raw_out)

            layer_outputs.append(got_vec)
            if l_idx + 1 < len(layers):
                cur_x = got_vec.astype(np.int8)

    run_migen_sim(soc, tb)
    return layer_outputs


def test_soc_csr_artifacts_deterministic_and_synced() -> None:
    """Task C1.1c: reports/csr.json and reports/csr.h regenerate byte-identically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        j1, h1 = tmp / "csr1.json", tmp / "csr1.h"
        j2, h2 = tmp / "csr2.json", tmp / "csr2.h"
        export_soc_csr_artifacts(json_path=j1, header_path=h1)
        export_soc_csr_artifacts(json_path=j2, header_path=h2)

        j1_txt, h1_txt = j1.read_text(encoding="utf-8"), h1.read_text(encoding="utf-8")
        assert j1_txt == j2.read_text(encoding="utf-8"), (
            "[test_soc_pipeline:export_soc_csr_artifacts] csr.json export is non-deterministic"
        )
        assert h1_txt == h2.read_text(encoding="utf-8"), (
            "[test_soc_pipeline:export_soc_csr_artifacts] csr.h export is non-deterministic"
        )

        committed_json = REPORTS_DIR / "csr.json"
        committed_h = REPORTS_DIR / "csr.h"
        assert committed_json.exists() and committed_h.exists(), (
            "[test_soc_pipeline] expected committed reports/csr.json and reports/csr.h"
        )
        assert committed_json.read_text(encoding="utf-8") == j1_txt, (
            "[test_soc_pipeline] reports/csr.json is out of sync with deterministic SoC export"
        )
        assert committed_h.read_text(encoding="utf-8") == h1_txt, (
            "[test_soc_pipeline] reports/csr.h is out of sync with deterministic SoC export"
        )

        parsed = json.loads(j1_txt)
        assert parsed["cores"].get("ternary_accel") == "TernaryAccel"
        assert "CSR_TERNARY_ACCEL_BASE" in h1_txt


@pytest.mark.slow
def test_soc_ttape_lux_chain_and_core_composition_with_bubbles() -> None:
    """Task C1.1c: Stream canonical .ttape fixtures with random bubbles vs setun/ref/m0.py."""
    # 1. Multi-layer TTAPE-grouped Lux.Chain MLP (8 -> 4 -> 2)
    lux_blob = (FIXTURES_DIR / "lux_chain_grouped_mlp.ttape").read_bytes()
    lux_meta = json.loads((FIXTURES_DIR / "lux_chain_grouped_mlp.json").read_text(encoding="utf-8"))
    lux_layers = load_ttape_layers(lux_blob, reader_profile="grouped")
    x_lux = np.asarray(lux_meta["input_vector"], dtype=np.int8)

    ref_traces: dict[str, np.ndarray] = {}
    ref_out = forward_ternary_chain(
        lux_layers,
        x_lux,
        lanes=16,
        mode="integer",
        trace_hook=lambda k, v: ref_traces.__setitem__(k, v),
    )

    hw_outs = _run_ttape_layers_on_soc(
        lux_layers,
        x_lux,
        lanes=16,
        group_size=64,
        beat_bits=64,
        seed=1958,
    )
    assert np.array_equal(hw_outs[0], ref_traces["layers.0"]), (
        f"[test_soc_pipeline:lux_chain] layer 0 mismatch vs m0.py: "
        f"expected {ref_traces['layers.0'].tolist()}, got {hw_outs[0].tolist()}"
    )
    assert np.array_equal(hw_outs[1], ref_out), (
        f"[test_soc_pipeline:lux_chain] final output mismatch vs m0.py: "
        f"expected {ref_out.tolist()}, got {hw_outs[1].tolist()}"
    )

    # 2. TTAPE-core shift-only fixture (core_minimal_dot4.ttape, scale_exp=-2 -> shift=2)
    core_blob = (FIXTURES_DIR / "core_minimal_dot4.ttape").read_bytes()
    core_layers = load_ttape_layers(core_blob, reader_profile="grouped")
    x_core = np.array([-128, 64, -32, 127], dtype=np.int8)
    ref_core = forward_ternary_chain(core_layers, x_core, lanes=16, mode="integer")
    hw_core = _run_ttape_layers_on_soc(
        core_layers,
        x_core,
        lanes=16,
        group_size=64,
        beat_bits=64,
        seed=2026,
    )
    assert np.array_equal(hw_core[0], ref_core), (
        f"[test_soc_pipeline:core_minimal_dot4] output mismatch vs m0.py: "
        f"expected {ref_core.tolist()}, got {hw_core[0].tolist()}"
    )


@pytest.mark.slow
def test_soc_m0_multi_group_ttape_composition_with_bubbles() -> None:
    """Task C1.1c: Canonical L=64, G=128, beat_bits=256 SoC on multi-group M0 .ttape vs m0.py."""
    rng = np.random.default_rng(19581120)
    n_rows, k, group_size, lanes, shift = 6, 256, 128, 64, 8
    weights = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n_rows, k))
    scales = rng.integers(64, 2048, size=(n_rows, k // group_size), dtype=np.uint16)
    # Force row 0 = -128 corner and row 1 = odd round-half-up tie exercise
    weights[0].fill(-1)
    scales[0].fill(512)

    packed = pack_ternary_tensor(weights, scales, group_size=group_size, shift=shift)
    ttape_blob = serialize_packed_weights(packed)
    reloaded_layers = load_ttape_layers(ttape_blob, reader_profile="grouped")

    x_in = rng.integers(-128, 128, size=k, dtype=np.int8)
    x_in[:64] = -128

    expected_m0 = forward_ternary_chain(reloaded_layers, x_in, lanes=lanes, mode="integer")
    expected_matvec = ternary_matvec_integer(reloaded_layers[0], x_in, lanes=lanes)
    assert np.array_equal(expected_m0, expected_matvec)

    hw_outs = _run_ttape_layers_on_soc(
        reloaded_layers,
        x_in,
        lanes=lanes,
        group_size=group_size,
        beat_bits=256,
        seed=31415,
    )
    assert np.array_equal(hw_outs[0], expected_m0), (
        f"[test_soc_pipeline:m0_multi_group] output mismatch vs m0.py: "
        f"expected {expected_m0.tolist()}, got {hw_outs[0].tolist()}"
    )


def test_ternary_accel_constructor_and_csr_check_guards(tmp_path: Path) -> None:
    """Cover TernaryAccel parameter validation and export_soc_csr_artifacts(check=True)."""
    import pytest

    with pytest.raises(ValueError, match="act_bits == 8"):
        TernaryAccel(act_bits=16)
    with pytest.raises(ValueError, match="multiple of 4"):
        TernaryAccel(lanes=2)
    with pytest.raises(ValueError, match="multiple of 4"):
        TernaryAccel(lanes=6)
    with pytest.raises(ValueError, match="max_k"):
        TernaryAccel(lanes=16, group_size=64, beat_bits=32, max_k=32)
    with pytest.raises(ValueError, match="max_k"):
        TernaryAccel(lanes=16, group_size=64, beat_bits=32, max_k=96)
    with pytest.raises(ValueError, match="max_rows"):
        TernaryAccel(max_rows=0)

    # check=True succeeds against synced committed artifacts
    j_txt, h_txt = export_soc_csr_artifacts(
        json_path=DEFAULT_CSR_JSON, header_path=DEFAULT_CSR_HEADER, check=True
    )
    assert "ternary_accel" in j_txt and "CSR_TERNARY_ACCEL_BASE" in h_txt

    # check=True fails Goose-style on missing or drifted csr.json / csr.h
    with pytest.raises(RuntimeError, match="deterministic CSR invariant violated"):
        export_soc_csr_artifacts(
            json_path=tmp_path / "missing_csr.json", header_path=DEFAULT_CSR_HEADER, check=True
        )

    bad_h = tmp_path / "drifted_csr.h"
    bad_h.write_text("// drifted\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="deterministic CSR invariant violated"):
        export_soc_csr_artifacts(json_path=DEFAULT_CSR_JSON, header_path=bad_h, check=True)


def test_soc_result_hi_lo_48bit_sign_extension_and_act_ptr_wrap() -> None:
    """Verify 48-bit result_hi/result_lo sign extension (>32-bit sums) and act_wr_ptr wrap."""
    accel = TernaryAccel(rows=1, lanes=4, group_size=16, beat_bits=16, max_k=16, max_rows=4)

    def tb():
        # Write starting at total_act_words - 1 (3) so act_wr_ptr wraps 3 -> 0 -> 1 -> 2
        yield accel.act_word_idx.storage.eq(accel.total_act_words - 1)
        yield accel.act_word_idx.re.eq(1)
        yield
        yield accel.act_word_idx.re.eq(0)

        for _ in range(accel.total_act_words):
            yield accel.act_word_data.storage.eq(0x80808080)  # four -128 int8 lanes per word
            yield accel.act_word_data.re.eq(1)
            yield
        yield accel.act_word_data.re.eq(0)

        # Pulse clear, enable=1, shift=0, num_rows=2
        yield accel.ctrl.fields.enable.eq(1)
        yield accel.ctrl.fields.clear.eq(1)
        yield accel.cfg.fields.shift.eq(0)
        yield accel.cfg.fields.num_rows.eq(2)
        yield
        yield accel.ctrl.fields.clear.eq(0)

        # 36 groups of G=16 (72 16-bit trit words = 144 beats > max_beats=4 -> tests beat wrap):
        # Row 0: w = -1 (0b10 -> 0xAAAA), x = -128 -> +2048/group * 65535 * 36 = +4,831,764,480
        # Row 1: w = +1 (0b01 -> 0x5555), x = -128 -> -2048/group * 65535 * 36 = -4,831,764,480
        exp_mag = 36 * 2048 * 65535
        for r_idx, (w_word, exp_48) in enumerate([(0xAAAA, exp_mag), (0x5555, -exp_mag)]):
            w_sent, s_sent = 0, 0
            while w_sent < 72 or s_sent < 36:
                if w_sent < 72:
                    yield accel.trit_data.eq(w_word)
                    yield accel.trit_last.eq(1 if w_sent == 71 else 0)
                    yield accel.trit_valid.eq(1)
                else:
                    yield accel.trit_valid.eq(0)
                if s_sent < 36:
                    yield accel.scale_data.eq(65535)
                    yield accel.scale_valid.eq(1)
                else:
                    yield accel.scale_valid.eq(0)
                yield
                if w_sent < 72 and (yield accel.trit_ready):
                    w_sent += 1
                if s_sent < 36 and (yield accel.scale_ready):
                    s_sent += 1

            yield accel.trit_valid.eq(0)
            yield accel.trit_last.eq(0)
            yield accel.scale_valid.eq(0)
            for _ in range(3):
                yield

            lo = yield accel.result_lo.status
            hi = _to_signed_i32((yield accel.result_hi.status))
            got_full = (hi << 32) | (lo & 0xFFFFFFFF)
            assert got_full == exp_48, (
                f"[test_soc_pipeline:result_hi_lo] 48-bit accumulator mismatch on row {r_idx}: "
                f"expected {exp_48} (hi={exp_48 >> 32}, lo={exp_48 & 0xFFFFFFFF:#x}), "
                f"got {got_full} (hi={hi}, lo={lo:#x})"
            )

    run_migen_sim(accel, tb)


def test_targets_sim_cli_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify targets/sim.py main() --export-csr-reports and --no-compile-gateware build."""
    import sys

    import targets.sim as sim_mod

    called_export: list[bool] = []
    monkeypatch.setattr(sim_mod, "export_soc_csr_artifacts", lambda: called_export.append(True))
    monkeypatch.setattr(sys, "argv", ["targets.sim", "--export-csr-reports"])
    sim_mod.main()
    assert called_export == [True]
    assert "[sim-csr]" in capsys.readouterr().out

    out_dir = tmp_path / "sim_gateware"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "targets.sim",
            "--output-dir",
            str(out_dir),
            "--with-accel-stub",
            "--no-ternary-accel",
            "--no-compile-gateware",
        ],
    )
    sim_mod.main()
    csr_path = out_dir / "csr.json"
    assert csr_path.exists()
    csr_meta = json.loads(csr_path.read_text(encoding="utf-8"))
    assert "accel_stub_id" in csr_meta["csr_registers"]
    assert "ternary_accel_ctrl" not in csr_meta["csr_registers"]
    assert "[sim-smoke]" in capsys.readouterr().out

    soc_wb = SetunSimSoC(
        expose_host_wb=True,
        use_external_crg=False,
        cpu_type=None,
        integrated_rom_size=0,
        integrated_main_ram_size=0x1000,
    )
    assert hasattr(soc_wb, "host_wb") and not hasattr(soc_wb, "crg")


def test_csr_doc_and_3_1a_autocsr_offset_stability() -> None:
    """Task 3.1b: Verify docs/CSR.md covers all TernaryAccel CSRs and 3.1a offsets stay fixed."""
    csr_doc = (REPO_ROOT / "docs" / "CSR.md").read_text(encoding="utf-8")
    _, addrs, csr_data = _build_soc_with_csr_map(lanes=64, group_size=128, beat_bits=256)

    base = addrs["ternary_accel_ctrl"]
    # All 10 Task 3.1a CSRs must remain at exact offsets +0x00 .. +0x24
    expected_3_1a = [
        ("ternary_accel_ctrl", 0x00),
        ("ternary_accel_cfg", 0x04),
        ("ternary_accel_info", 0x08),
        ("ternary_accel_status", 0x0C),
        ("ternary_accel_act_word_idx", 0x10),
        ("ternary_accel_act_word_data", 0x14),
        ("ternary_accel_result_lo", 0x18),
        ("ternary_accel_result_hi", 0x1C),
        ("ternary_accel_out_row_idx", 0x20),
        ("ternary_accel_out_row_data", 0x24),
    ]
    for reg_name, rel_off in expected_3_1a:
        got_off = addrs[reg_name] - base
        assert got_off == rel_off, (
            f"[test_soc_pipeline:offset_stability] 3.1a CSR '{reg_name}' drifted: "
            f"expected offset 0x{rel_off:02x}, got 0x{got_off:02x}"
        )

    # All new 3.1b CSRs must exist after +0x24 and be documented in docs/CSR.md
    new_3_1b = [
        "tile_info",
        "buf_caps",
        "dim_k",
        "weight_addr",
        "scale_addr",
        "act_addr",
        "out_addr",
        "act_scale",
        "req_cfg",
        "perf_cycles",
        "perf_bus_beats",
        "perf_sat_rows",
    ]
    for short_name in new_3_1b:
        full_name = f"ternary_accel_{short_name}"
        assert full_name in addrs, f"Missing 3.1b CSR '{full_name}' in csr.json"
        assert addrs[full_name] > base + 0x24, f"CSR '{full_name}' placed before 3.1a boundary"
        assert f"`{short_name}`" in csr_doc, f"CSR '{short_name}' undocumented in docs/CSR.md"

    # Default SetunSimSoC TernaryAccel must instantiate rows=4, max_k=2048, max_rows=2048
    soc_default = SetunSimSoC(
        with_accel_stub=False,
        with_ternary_accel=True,
        use_external_crg=False,
        cpu_type=None,
        integrated_rom_size=0,
        integrated_main_ram_size=0x1000,
    )
    assert soc_default.ternary_accel.rows == 4
    assert soc_default.ternary_accel.max_k == 2048
    assert soc_default.ternary_accel.max_rows == 2048
    assert "ternary_accel" in csr_data["cores"]


@pytest.mark.slow
def test_soc_3_1b_start_multirow_requantize_and_illegal_params() -> None:
    """Task 3.1b: Test ctrl.start multi-row GEMV, Requantize saturation, and non-hanging errors."""
    from setun.ref.ops import requantize_vec_int8, tile_gemv

    soc = SetunSimSoC(
        with_accel_stub=False,
        with_ternary_accel=True,
        accel_rows=4,
        accel_lanes=16,
        accel_group_size=64,
        accel_beat_bits=64,
        accel_max_k=256,
        accel_max_rows=32,
        expose_host_wb=True,
        use_external_crg=False,
        cpu_type=None,
        integrated_rom_size=0,
        integrated_main_ram_size=0x1000,
    )
    soc.finalize()
    csr_json = json.loads(
        export.get_csr_json(
            soc=soc,
            csr_regions=soc.csr.regions,
            constants=soc.constants,
            mem_regions=soc.mem_regions,
        )
    )
    addrs = {k: int(v["addr"]) for k, v in csr_json["csr_registers"].items()}
    accel: TernaryAccel = soc.ternary_accel
    wb = soc.host_wb

    rng = np.random.default_rng(20260312)
    n_rows, k_cols, group_size, lanes, rows, shift = 7, 128, 64, 16, 4, 5
    w = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n_rows, k_cols))
    x = rng.integers(-128, 128, size=k_cols, dtype=np.int8)
    s = rng.integers(100, 3000, size=(n_rows, k_cols // group_size), dtype=np.uint16)
    layer = pack_ternary_tensor(w, s, group_size=group_size, shift=shift)

    trit_words, flat_scales, k_pad = _prepare_layer_stream_words(layer, beat_bits=64)
    act_words = _pack_activation_words_i8(x, k_pad)
    exp_raw = tile_gemv(w, x, s, group_size=group_size, lanes=lanes, rows=rows, shift=shift)
    req_scale, req_shift = 512, 10
    exp_req_i8, exp_req_sat = requantize_vec_int8(exp_raw, req_scale, req_shift, symmetric=True)

    def wb_read(byte_addr: int):
        return (yield from wb.read(byte_addr >> 2))

    def wb_write(byte_addr: int, data: int):
        yield from wb.write(byte_addr >> 2, data & 0xFFFFFFFF)

    def _stream_weights():
        w_ptr, s_ptr = 0, 0
        while w_ptr < len(trit_words) or s_ptr < len(flat_scales):
            yield accel.trit_data.eq(trit_words[w_ptr][0] if w_ptr < len(trit_words) else 0)
            yield accel.trit_last.eq(trit_words[w_ptr][1] if w_ptr < len(trit_words) else 0)
            yield accel.trit_valid.eq(1 if w_ptr < len(trit_words) else 0)
            yield accel.scale_data.eq(flat_scales[s_ptr] if s_ptr < len(flat_scales) else 0)
            yield accel.scale_valid.eq(1 if s_ptr < len(flat_scales) else 0)
            yield
            if w_ptr < len(trit_words) and (yield accel.trit_ready):
                w_ptr += 1
            if s_ptr < len(flat_scales) and (yield accel.scale_ready):
                s_ptr += 1
        yield accel.trit_valid.eq(0)
        yield accel.trit_last.eq(0)
        yield accel.scale_valid.eq(0)

    def tb():
        # 1. Check tile_info & buf_caps 16-bit fields (Refinement #1)
        t_info = yield from wb_read(addrs["ternary_accel_tile_info"])
        b_caps = yield from wb_read(addrs["ternary_accel_buf_caps"])
        assert t_info == (rows | (lanes << 8) | (group_size << 16))
        assert b_caps == (256 | (32 << 16))

        # Stage activations into synchronous BRAM once
        yield from wb_write(addrs["ternary_accel_act_word_idx"], 0)
        for aw in act_words:
            yield from wb_write(addrs["ternary_accel_act_word_data"], aw)

        # 2. Exercise all 4 illegal parameter classes via ctrl.start (must NEVER hang!)
        illegal_cases = [
            # (cfg_n, dim_k_val, dim_g_val, w_addr, s_addr, a_addr, o_addr, exp_err_bit)
            (7, 0, 64, 0, 0, 0, 0, 0x1),  # K == 0 -> ERR_K_INVALID
            (7, 300, 64, 0, 0, 0, 0, 0x1),  # K > max_k (256) -> ERR_K_INVALID
            (0, 128, 64, 0, 0, 0, 0, 0x2),  # N == 0 -> ERR_N_INVALID
            (33, 128, 64, 0, 0, 0, 0, 0x2),  # N > max_rows (32) -> ERR_N_INVALID
            (7, 128, 128, 0, 0, 0, 0, 0x4),  # G != hw_group_size (64) -> ERR_G_MISMATCH
            (7, 128, 64, 0x40000002, 0, 0, 0, 0x8),  # misaligned weight_addr -> ERR_ADDR_ALIGN
            (7, 128, 64, 0, 0, 0, 0x40000001, 0x8),  # misaligned out_addr -> ERR_ADDR_ALIGN
        ]
        for c_n, c_k, c_g, wa, sa, aa, oa, exp_err in illegal_cases:
            yield from wb_write(addrs["ternary_accel_cfg"], shift | (c_n << 8))
            yield from wb_write(addrs["ternary_accel_dim_k"], c_k | (c_g << 16))
            yield from wb_write(addrs["ternary_accel_weight_addr"], wa)
            yield from wb_write(addrs["ternary_accel_scale_addr"], sa)
            yield from wb_write(addrs["ternary_accel_act_addr"], aa)
            yield from wb_write(addrs["ternary_accel_out_addr"], oa)
            # Pulse ctrl.start (bit 2) + enable (bit 0) -> 0b101
            yield from wb_write(addrs["ternary_accel_ctrl"], 0b00101)
            st = yield from wb_read(addrs["ternary_accel_status"])
            busy = st & 0x1
            done = (st >> 1) & 0x1
            err = (st >> 4) & 0x1
            err_code = (st >> 8) & 0xF
            assert err == 1 and busy == 0 and done == 0 and (err_code & exp_err) == exp_err, (
                f"[test_soc_pipeline:illegal_params] expected err=1, busy=0, done=0, "
                f"err_code&0x{exp_err:x} on ({c_n}, {c_k}, {c_g}, 0x{wa:x}, 0x{oa:x}), "
                f"got status=0x{st:08x}"
            )

        # 3. Sticky error recovery WITHOUT SoC reset (Refinement #5): launch valid raw GEMV
        yield from wb_write(addrs["ternary_accel_weight_addr"], 0)
        yield from wb_write(addrs["ternary_accel_out_addr"], 0)
        yield from wb_write(addrs["ternary_accel_cfg"], shift | (n_rows << 8))
        yield from wb_write(addrs["ternary_accel_dim_k"], k_cols | (group_size << 16))
        yield from wb_write(addrs["ternary_accel_act_scale"], 0x3F800000)
        yield from wb_write(addrs["ternary_accel_ctrl"], 0b00101)  # enable=1, start=1

        st_after_start = yield from wb_read(addrs["ternary_accel_status"])
        assert ((st_after_start >> 4) & 0x1) == 0 and ((st_after_start >> 8) & 0xF) == 0

        yield from _stream_weights()
        for _ in range(12):
            yield

        st_done = yield from wb_read(addrs["ternary_accel_status"])
        assert ((st_done >> 1) & 0x1) == 1 and ((st_done >> 16) & 0xFFFF) == n_rows
        p_cyc = yield from wb_read(addrs["ternary_accel_perf_cycles"])
        p_beats = yield from wb_read(addrs["ternary_accel_perf_bus_beats"])
        assert p_cyc > 0 and p_beats == len(trit_words)

        for r in range(n_rows):
            yield from wb_write(addrs["ternary_accel_out_row_idx"], r)
            got_r = _to_signed_i32((yield from wb_read(addrs["ternary_accel_out_row_data"])))
            assert got_r == int(exp_raw[r]), (
                f"[test_soc_pipeline:start_raw] row {r} mismatch: "
                f"expected {int(exp_raw[r])}, got {got_r}"
            )

        # 4. Launch with Requantize enabled + symmetric=1 (Refinement #4 saturation telemetry)
        yield from wb_write(addrs["ternary_accel_req_cfg"], req_scale | (req_shift << 16))
        # enable=1 (b0), start=1 (b2), req_enable=1 (b3), req_sym=1 (b4) -> 0b11101
        yield from wb_write(addrs["ternary_accel_ctrl"], 0b11101)
        yield from _stream_weights()
        for _ in range(16):
            yield

        st_req = yield from wb_read(addrs["ternary_accel_status"])
        sat_flag = (st_req >> 5) & 0x1
        sat_cnt = yield from wb_read(addrs["ternary_accel_perf_sat_rows"])
        exp_sat_cnt = int(np.sum(exp_req_sat))
        assert sat_flag == (1 if exp_sat_cnt > 0 else 0)
        assert sat_cnt == exp_sat_cnt

        for r in range(n_rows):
            yield from wb_write(addrs["ternary_accel_out_row_idx"], r)
            got_q = _to_signed_i32((yield from wb_read(addrs["ternary_accel_out_row_data"])))
            assert got_q == int(exp_req_i8[r]), (
                f"[test_soc_pipeline:start_req] row {r} mismatch: "
                f"expected {int(exp_req_i8[r])}, got {got_q}"
            )

    run_migen_sim(soc, tb)
