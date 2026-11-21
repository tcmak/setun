"""End-to-end LiteX SoC pipeline composition & deterministic CSR map tests (Task C1.1c)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from litex.soc.integration import export

from setun.ref.format import load_ttape_layers, pack_ternary_tensor, serialize_packed_weights
from setun.ref.m0 import forward_ternary_chain, ternary_matvec_integer
from setun.soc import TernaryAccel, export_soc_csr_artifacts
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
            for _ in range(4):
                yield

            status_val = yield from wb_read(status_addr)
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
