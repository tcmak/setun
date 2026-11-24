"""Unit and Verilator tests for SimSoC burst DMA & packed int8 output (Tasks 3.2 & 3.2b)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from litex.soc.integration import export
from migen import ClockDomain, Module, Signal

from setun.ref.format import pack_ternary_tensor
from setun.ref.ops import requantize_vec_int8, tile_gemv
from targets.sim import SetunSimSoC
from tests.sim_helper import run_cocotb_verilator, run_migen_sim

MAIN_RAM_BASE = 0x40000000


def _to_signed_i32(word_u32: int) -> int:
    val = int(word_u32) & 0xFFFFFFFF
    return val - 0x100000000 if (val & 0x80000000) else val


def _to_signed_i8(byte_u8: int) -> int:
    val = int(byte_u8) & 0xFF
    return val - 0x100 if (val & 0x80) else val


def _pack_ram_arena(blobs: list[bytes]) -> tuple[bytes, list[int]]:
    """Pack byte blobs sequentially at 16-byte aligned offsets for main_ram init."""
    buf = bytearray()
    offsets: list[int] = []
    for b in blobs:
        while len(buf) % 16 != 0:
            buf.append(0)
        offsets.append(len(buf))
        buf.extend(b)
    while len(buf) % 4 != 0:
        buf.append(0)
    return bytes(buf), offsets


def test_soc_dma_ram_init_elaboration_and_cli_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fast (<2s) elaboration & CLI coverage for SetunSimSoC ram_init (Path & Sequence[int])."""
    import targets.sim as sim_mod

    ram_bin = tmp_path / "fast_preload.bin"
    ram_bin.write_bytes(b"\x01\x02\x03\x04\x05")  # non-multiple-of-4 exercises pad path

    soc_path = SetunSimSoC(
        with_accel_stub=False,
        with_ternary_accel=True,
        accel_rows=2,
        accel_lanes=16,
        accel_group_size=64,
        accel_beat_bits=64,
        accel_max_k=128,
        accel_max_rows=64,
        ram_init=ram_bin,
        expose_host_wb=True,
        use_external_crg=False,
        cpu_type=None,
        integrated_rom_size=0,
        integrated_main_ram_size=0x4000,
    )
    soc_path.finalize()

    soc_seq = SetunSimSoC(
        with_accel_stub=False,
        with_ternary_accel=True,
        accel_rows=1,
        accel_lanes=16,
        accel_group_size=64,
        accel_beat_bits=32,
        accel_max_k=64,
        accel_max_rows=16,
        ram_init=[0x11223344, 0x55667788],
        expose_host_wb=False,
        use_external_crg=False,
        cpu_type=None,
        integrated_rom_size=0,
        integrated_main_ram_size=0x4000,
    )
    soc_seq.finalize()

    out_dir = tmp_path / "sim_ram_init_fast_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "targets.sim",
            "--output-dir",
            str(out_dir),
            "--ram-init",
            str(ram_bin),
            "--no-compile-gateware",
        ],
    )
    sim_mod.main()
    assert (out_dir / "csr.json").exists()


@pytest.mark.slow
def test_soc_dma_ram_init_migen_and_cli_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fast check: --ram-init preload, 64x64 burst DMA, 3.2b packed int8, & partial ptr error."""
    import targets.sim as sim_mod

    rng = np.random.default_rng(20260320)
    n_rows, k_cols, group_size, lanes, rows, shift = 64, 64, 64, 16, 4, 6
    w = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n_rows, k_cols))
    x = rng.integers(-128, 128, size=k_cols, dtype=np.int8)
    s = rng.integers(64, 2048, size=(n_rows, k_cols // group_size), dtype=np.uint16)
    packed = pack_ternary_tensor(w, s, group_size=group_size, shift=shift)

    w_bytes = packed.packed_trits.tobytes()
    s_bytes = np.ascontiguousarray(packed.scales, dtype="<u2").tobytes()
    a_bytes = x.view(np.uint8).tobytes()
    out_placeholder = bytes(n_rows * 4)

    arena_bytes, (w_off, s_off, a_off, o_off) = _pack_ram_arena(
        [w_bytes, s_bytes, a_bytes, out_placeholder]
    )
    ram_bin = tmp_path / "preload.bin"
    ram_bin.write_bytes(arena_bytes)

    exp_raw = tile_gemv(w, x, s, group_size=group_size, lanes=lanes, rows=rows, shift=shift)
    exp_req, _ = requantize_vec_int8(exp_raw, 256, 8, symmetric=False)

    soc = SetunSimSoC(
        with_accel_stub=False,
        with_ternary_accel=True,
        accel_rows=rows,
        accel_lanes=lanes,
        accel_group_size=group_size,
        accel_beat_bits=64,
        accel_max_k=256,
        accel_max_rows=128,
        ram_init=ram_bin,
        expose_host_wb=True,
        use_external_crg=False,
        cpu_type=None,
        integrated_rom_size=0,
        integrated_main_ram_size=0x10000,
    )
    soc.finalize()
    assert hasattr(soc.ternary_accel, "dma_wb")

    csr_data = json.loads(
        export.get_csr_json(
            soc=soc,
            csr_regions=soc.csr.regions,
            constants=soc.constants,
            mem_regions=soc.mem_regions,
        )
    )
    addrs = {k: int(v["addr"]) for k, v in csr_data["csr_registers"].items()}
    wb = soc.host_wb

    def wb_read(byte_addr: int):
        return (yield from wb.read(byte_addr >> 2))

    def wb_write(byte_addr: int, data: int):
        yield from wb.write(byte_addr >> 2, data & 0xFFFFFFFF)

    def tb():
        # 1. Partial DMA address (weight_addr != 0, scale_addr == 0) must set error & never hang
        yield from wb_write(addrs["ternary_accel_cfg"], shift | (n_rows << 8))
        yield from wb_write(addrs["ternary_accel_dim_k"], k_cols | (group_size << 16))
        yield from wb_write(addrs["ternary_accel_weight_addr"], MAIN_RAM_BASE + w_off)
        yield from wb_write(addrs["ternary_accel_scale_addr"], 0)
        yield from wb_write(addrs["ternary_accel_act_addr"], 0)
        yield from wb_write(addrs["ternary_accel_out_addr"], 0)
        yield from wb_write(addrs["ternary_accel_ctrl"], 0b00101)
        st_err = yield from wb_read(addrs["ternary_accel_status"])
        assert ((st_err >> 4) & 0x1) == 1 and (st_err & 0x1) == 0
        assert ((st_err >> 8) & 0xF) == 0x8

        # 2. Valid 64x64 RAM-resident GEMV (out_addr=0 first to record exact read_beats)
        yield from wb_write(addrs["ternary_accel_scale_addr"], MAIN_RAM_BASE + s_off)
        yield from wb_write(addrs["ternary_accel_act_addr"], MAIN_RAM_BASE + a_off)
        yield from wb_write(addrs["ternary_accel_out_addr"], 0)
        yield from wb_write(addrs["ternary_accel_ctrl"], 0b00101)

        while True:
            for _ in range(64):
                yield
            st = yield from wb_read(addrs["ternary_accel_status"])
            if (st >> 1) & 0x1:
                break

        cyc = yield from wb_read(addrs["ternary_accel_perf_cycles"])
        read_beats = yield from wb_read(addrs["ternary_accel_perf_bus_beats"])
        assert read_beats / max(cyc, 1) >= 0.80, f"Expected >=80% bus util, got {read_beats}/{cyc}"

        for r in range(n_rows):
            yield from wb_write(addrs["ternary_accel_out_row_idx"], r)
            got_v = _to_signed_i32((yield from wb_read(addrs["ternary_accel_out_row_data"])))
            assert got_v == int(exp_raw[r])

        # 3. Task 3.2b: Launch with Requantize + out_addr -> must write 16 packed 32b words (64B)
        yield from wb_write(addrs["ternary_accel_out_addr"], MAIN_RAM_BASE + o_off)
        yield from wb_write(addrs["ternary_accel_req_cfg"], 256 | (8 << 16))
        yield from wb_write(addrs["ternary_accel_ctrl"], 0b01101)  # enable | start | req_enable

        while True:
            for _ in range(64):
                yield
            st2 = yield from wb_read(addrs["ternary_accel_status"])
            if (st2 >> 1) & 0x1:
                break

        total_beats_req = yield from wb_read(addrs["ternary_accel_perf_bus_beats"])
        assert total_beats_req == read_beats + (n_rows // 4), (
            f"[test_soc_dma:3.2b] expected {read_beats} + {n_rows // 4} packed write beats, "
            f"got {total_beats_req}"
        )

        out_bytes = bytearray()
        for w_idx in range(n_rows // 4):
            wd = yield from wb_read(MAIN_RAM_BASE + o_off + w_idx * 4)
            out_bytes.extend(int(wd).to_bytes(4, "little"))
        for r in range(n_rows):
            assert _to_signed_i8(out_bytes[r]) == int(exp_req[r]), (
                f"[test_soc_dma:3.2b] packed int8 mismatch at row {r}: "
                f"expected {int(exp_req[r])}, got {_to_signed_i8(out_bytes[r])}"
            )

    run_migen_sim(soc, tb)

    # 4. Verify targets.sim CLI --ram-init flag builds gateware netlist without error
    out_dir = tmp_path / "sim_ram_init_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "targets.sim",
            "--output-dir",
            str(out_dir),
            "--ram-init",
            str(ram_bin),
            "--no-compile-gateware",
        ],
    )
    sim_mod.main()
    assert (out_dir / "csr.json").exists()


@pytest.mark.slow
def test_soc_dma_3_2b_packed_i8_aligned_ragged_sentinels_and_chaining() -> None:
    """Task 3.2b: Verify aligned/ragged packed int8, sentinel masking, beats, & zero-copy chain."""
    rng = np.random.default_rng(20260322)
    max_n, k_cols, group_size, lanes, rows = 64, 64, 64, 16, 4
    shift0, req_scale0, req_shift0 = 5, 300, 8
    shift1, req_scale1, req_shift1 = 4, 220, 7

    # Layer 0 weights (up to 64x64; prefixes used for N in {1, 2, 3, 4, 5, 7, 13, 16, 64})
    w0 = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(max_n, k_cols))
    s0 = rng.integers(80, 1500, size=(max_n, 1), dtype=np.uint16)
    x0 = rng.integers(-128, 128, size=k_cols, dtype=np.int8)
    packed0 = pack_ternary_tensor(w0, s0, group_size=group_size, shift=shift0)

    # Layer 1 weights (16x64) consuming Layer 0's 64-byte packed int8 output directly as act_addr!
    n1 = 16
    w1 = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n1, k_cols))
    s1 = rng.integers(80, 1500, size=(n1, 1), dtype=np.uint16)
    packed1 = pack_ternary_tensor(w1, s1, group_size=group_size, shift=shift1)

    out0_buf = bytes(128)
    out1_buf = bytes(128)
    arena_bytes, (w0_off, s0_off, a0_off, w1_off, s1_off, o0_off, o1_off) = _pack_ram_arena(
        [
            packed0.packed_trits.tobytes(),
            np.ascontiguousarray(packed0.scales, dtype="<u2").tobytes(),
            x0.view(np.uint8).tobytes(),
            packed1.packed_trits.tobytes(),
            np.ascontiguousarray(packed1.scales, dtype="<u2").tobytes(),
            out0_buf,
            out1_buf,
        ]
    )
    init_words = [
        int.from_bytes(arena_bytes[i : i + 4], "little") for i in range(0, len(arena_bytes), 4)
    ]

    soc = SetunSimSoC(
        with_accel_stub=False,
        with_ternary_accel=True,
        accel_rows=rows,
        accel_lanes=lanes,
        accel_group_size=group_size,
        accel_beat_bits=64,
        accel_max_k=128,
        accel_max_rows=128,
        ram_init=init_words,
        expose_host_wb=True,
        use_external_crg=False,
        cpu_type=None,
        integrated_rom_size=0,
        integrated_main_ram_size=0x10000,
    )
    soc.finalize()
    csr_data = json.loads(
        export.get_csr_json(
            soc=soc,
            csr_regions=soc.csr.regions,
            constants=soc.constants,
            mem_regions=soc.mem_regions,
        )
    )
    addrs = {k: int(v["addr"]) for k, v in csr_data["csr_registers"].items()}
    wb = soc.host_wb

    def wb_read(byte_addr: int):
        return (yield from wb.read(byte_addr >> 2))

    def wb_write(byte_addr: int, data: int):
        yield from wb.write(byte_addr >> 2, data & 0xFFFFFFFF)

    sentinel_u32 = 0x3CC35AA5  # bytes: [0xA5, 0x5A, 0xC3, 0x3C]
    sentinel_pat = sentinel_u32.to_bytes(4, "little")

    test_row_counts = [1, 2, 3, 4, 5, 7, 13, 16, 64]

    def tb():
        for n_sub in test_row_counts:
            exp_raw_sub = tile_gemv(
                w0[:n_sub],
                x0,
                s0[:n_sub],
                group_size=group_size,
                lanes=lanes,
                rows=rows,
                shift=shift0,
            )
            exp_q_sub, _ = requantize_vec_int8(exp_raw_sub, req_scale0, req_shift0, symmetric=False)

            # Paint output region with sentinel words before launching GEMV
            n_check_words = ((n_sub + 3) // 4) + 2
            for wi in range(n_check_words):
                yield from wb_write(MAIN_RAM_BASE + o0_off + wi * 4, sentinel_u32)

            yield from wb_write(addrs["ternary_accel_cfg"], shift0 | (n_sub << 8))
            yield from wb_write(addrs["ternary_accel_dim_k"], k_cols | (group_size << 16))
            yield from wb_write(addrs["ternary_accel_weight_addr"], MAIN_RAM_BASE + w0_off)
            yield from wb_write(addrs["ternary_accel_scale_addr"], MAIN_RAM_BASE + s0_off)
            yield from wb_write(addrs["ternary_accel_act_addr"], MAIN_RAM_BASE + a0_off)
            yield from wb_write(addrs["ternary_accel_out_addr"], MAIN_RAM_BASE + o0_off)
            yield from wb_write(addrs["ternary_accel_req_cfg"], req_scale0 | (req_shift0 << 16))
            yield from wb_write(addrs["ternary_accel_ctrl"], 0b01101)  # enable | start | req_enable

            while True:
                for _ in range(16):
                    yield
                st = yield from wb_read(addrs["ternary_accel_status"])
                if (st >> 1) & 0x1:
                    break

            # Expected exact DMA beats:
            # act = 64/4 = 16 words; scales = ceil(n_sub / 2) words; trits = n_sub * 4 words;
            # out = ceil(n_sub / 4) packed words!
            exp_read_beats = 16 + ((n_sub + 1) // 2) + (n_sub * 4)
            exp_out_beats = (n_sub + 3) // 4
            got_beats = yield from wb_read(addrs["ternary_accel_perf_bus_beats"])
            assert got_beats == exp_read_beats + exp_out_beats, (
                f"[3.2b:beats] N={n_sub}: expected {exp_read_beats} + {exp_out_beats} = "
                f"{exp_read_beats + exp_out_beats} beats, got {got_beats}"
            )

            # Read back memory bytes and verify [0:n_sub) == exp_q_sub AND [n_sub:end) == sentinels!
            mem_bytes = bytearray()
            for wi in range(n_check_words):
                wd = yield from wb_read(MAIN_RAM_BASE + o0_off + wi * 4)
                mem_bytes.extend(int(wd).to_bytes(4, "little"))

            for r in range(n_sub):
                assert _to_signed_i8(mem_bytes[r]) == int(exp_q_sub[r]), (
                    f"[3.2b:data] N={n_sub} row={r}: expected {int(exp_q_sub[r])}, "
                    f"got {_to_signed_i8(mem_bytes[r])}"
                )
                # Also verify on-chip CSR debug port still returns per-row signed int32!
                yield from wb_write(addrs["ternary_accel_out_row_idx"], r)
                csr_r = _to_signed_i32((yield from wb_read(addrs["ternary_accel_out_row_data"])))
                assert csr_r == int(exp_q_sub[r])

            exp_sentinels = sentinel_pat * n_check_words
            for b_idx in range(n_sub, len(mem_bytes)):
                assert mem_bytes[b_idx] == exp_sentinels[b_idx], (
                    f"[3.2b:sentinel] N={n_sub} byte offset {b_idx} clobbered: "
                    f"expected 0x{exp_sentinels[b_idx]:02x}, got 0x{mem_bytes[b_idx]:02x}"
                )

        # Zero-copy 2-layer chain: final loop ran N=64 into o0_off; feed o0_off as Layer 1 act_addr!
        exp_l0_raw = tile_gemv(
            w0, x0, s0, group_size=group_size, lanes=lanes, rows=rows, shift=shift0
        )
        exp_l0_i8, _ = requantize_vec_int8(exp_l0_raw, req_scale0, req_shift0, symmetric=False)
        exp_l1_raw = tile_gemv(
            w1, exp_l0_i8, s1, group_size=group_size, lanes=lanes, rows=rows, shift=shift1
        )
        exp_l1_i8, _ = requantize_vec_int8(exp_l1_raw, req_scale1, req_shift1, symmetric=False)

        # Layer 1a: req_enable=0 (wide int32 output) reading act_addr = o0_off directly
        yield from wb_write(addrs["ternary_accel_cfg"], shift1 | (n1 << 8))
        yield from wb_write(addrs["ternary_accel_dim_k"], k_cols | (group_size << 16))
        yield from wb_write(addrs["ternary_accel_weight_addr"], MAIN_RAM_BASE + w1_off)
        yield from wb_write(addrs["ternary_accel_scale_addr"], MAIN_RAM_BASE + s1_off)
        yield from wb_write(addrs["ternary_accel_act_addr"], MAIN_RAM_BASE + o0_off)
        yield from wb_write(addrs["ternary_accel_out_addr"], MAIN_RAM_BASE + o1_off)
        yield from wb_write(addrs["ternary_accel_ctrl"], 0b00101)  # req_enable=0

        while True:
            for _ in range(16):
                yield
            if ((yield from wb_read(addrs["ternary_accel_status"])) >> 1) & 0x1:
                break

        for r in range(n1):
            got_w = _to_signed_i32((yield from wb_read(MAIN_RAM_BASE + o1_off + r * 4)))
            assert got_w == int(exp_l1_raw[r])

        # Layer 1b: req_enable=1 (packed int8 output) reading act_addr = o0_off directly
        yield from wb_write(addrs["ternary_accel_req_cfg"], req_scale1 | (req_shift1 << 16))
        yield from wb_write(addrs["ternary_accel_ctrl"], 0b01101)  # req_enable=1

        while True:
            for _ in range(16):
                yield
            if ((yield from wb_read(addrs["ternary_accel_status"])) >> 1) & 0x1:
                break

        l1_bytes = bytearray()
        for wi in range(n1 // 4):
            wd = yield from wb_read(MAIN_RAM_BASE + o1_off + wi * 4)
            l1_bytes.extend(int(wd).to_bytes(4, "little"))
        for r in range(n1):
            assert _to_signed_i8(l1_bytes[r]) == int(exp_l1_i8[r])

    run_migen_sim(soc, tb)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_soc_dma_verilator_gemv_64_256_2048_and_bus_util(tmp_path: Path) -> None:
    """Tasks 3.2 & 3.2b: Verilator sweep across {64,256,2048}^2, N=1024 wide, & ragged packed i8."""
    rng = np.random.default_rng(19580302)
    rows, lanes, group_size, beat_bits = 4, 64, 128, 256

    max_n, max_k = 2048, 2048
    w_big = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(max_n, max_k))
    x_big = rng.integers(-128, 128, size=max_k, dtype=np.int8)
    s_big = rng.integers(32, 1024, size=(max_n, max_k // group_size), dtype=np.uint16)

    blobs: list[bytes] = []
    case_specs: list[dict[str, object]] = []

    k_arenas: dict[int, tuple[int, int, int]] = {}
    for k_cols in (64, 256, 2048):
        n_for_k = 2048
        n_grp = (k_cols + group_size - 1) // group_size
        packed_k = pack_ternary_tensor(
            w_big[:n_for_k, :k_cols],
            s_big[:n_for_k, :n_grp],
            group_size=group_size,
            shift=7,
        )
        x_pad = np.zeros(n_grp * group_size, dtype=np.int8)
        x_pad[:k_cols] = x_big[:k_cols]
        idx_w = len(blobs)
        blobs.append(packed_k.packed_trits.tobytes())
        idx_s = len(blobs)
        blobs.append(np.ascontiguousarray(packed_k.scales, dtype="<u2").tobytes())
        idx_a = len(blobs)
        blobs.append(x_pad.view(np.uint8).tobytes())
        k_arenas[k_cols] = (idx_w, idx_s, idx_a)

    idx_out = len(blobs)
    blobs.append(bytes(2048 * 4))

    arena_bytes, offsets = _pack_ram_arena(blobs)
    ram_bin = tmp_path / "soc_dma_arena.bin"
    ram_bin.write_bytes(arena_bytes)
    assert len(arena_bytes) < 0x400000

    for n_rows in (64, 256, 2048):
        for k_cols in (64, 256, 2048):
            n_grp = (k_cols + group_size - 1) // group_size
            shift = 7
            exp = tile_gemv(
                w_big[:n_rows, :k_cols],
                x_big[:k_cols],
                s_big[:n_rows, :n_grp],
                group_size=group_size,
                lanes=lanes,
                rows=rows,
                shift=shift,
            )
            iw, isc, ia = k_arenas[k_cols]
            use_out_dma = n_rows == k_cols
            case_specs.append(
                {
                    "n_rows": n_rows,
                    "k_cols": k_cols,
                    "group_size": group_size,
                    "shift": shift,
                    "weight_addr": MAIN_RAM_BASE + offsets[iw],
                    "scale_addr": MAIN_RAM_BASE + offsets[isc],
                    "act_addr": MAIN_RAM_BASE + offsets[ia],
                    "out_addr": (MAIN_RAM_BASE + offsets[idx_out]) if use_out_dma else 0,
                    "req_enable": False,
                    "check_util": True,
                    "expected": [int(v) for v in exp.tolist()],
                }
            )

    # Explicit N=1024 (>512) wide req_enable=0 DMA test proving zero 512-entry wrap
    iw64, isc64, ia64 = k_arenas[64]
    exp_1024 = tile_gemv(
        w_big[:1024, :64],
        x_big[:64],
        s_big[:1024, :1],
        group_size=group_size,
        lanes=lanes,
        rows=rows,
        shift=7,
    )
    case_specs.append(
        {
            "n_rows": 1024,
            "k_cols": 64,
            "group_size": group_size,
            "shift": 7,
            "weight_addr": MAIN_RAM_BASE + offsets[iw64],
            "scale_addr": MAIN_RAM_BASE + offsets[isc64],
            "act_addr": MAIN_RAM_BASE + offsets[ia64],
            "out_addr": MAIN_RAM_BASE + offsets[idx_out],
            "req_enable": False,
            "check_util": True,
            "expected": [int(v) for v in exp_1024.tolist()],
        }
    )

    # Task 3.2b Verilator checks: aligned N=64 & ragged N=13 packed int8 + sentinel preservation
    for n_pack, check_u in [(64, True), (13, False)]:
        exp_r = tile_gemv(
            w_big[:n_pack, :256],
            x_big[:256],
            s_big[:n_pack, :2],
            group_size=group_size,
            lanes=lanes,
            rows=rows,
            shift=7,
        )
        exp_q, _ = requantize_vec_int8(exp_r, 256, 8, symmetric=False)
        iw256, isc256, ia256 = k_arenas[256]
        case_specs.append(
            {
                "n_rows": n_pack,
                "k_cols": 256,
                "group_size": group_size,
                "shift": 7,
                "weight_addr": MAIN_RAM_BASE + offsets[iw256],
                "scale_addr": MAIN_RAM_BASE + offsets[isc256],
                "act_addr": MAIN_RAM_BASE + offsets[ia256],
                "out_addr": MAIN_RAM_BASE + offsets[idx_out],
                "req_enable": True,
                "req_scale": 256,
                "req_shift": 8,
                "check_util": check_u,
                "expected": [int(v) for v in exp_q.tolist()],
            }
        )

    class _SimSoCDmaTop(Module):
        def __init__(self) -> None:
            self.clock_domains.cd_sys = ClockDomain("sys")
            soc = SetunSimSoC(
                with_accel_stub=False,
                with_ternary_accel=True,
                accel_rows=rows,
                accel_lanes=lanes,
                accel_group_size=group_size,
                accel_beat_bits=beat_bits,
                accel_max_k=2048,
                accel_max_rows=2048,
                ram_init=ram_bin,
                expose_host_wb=True,
                use_external_crg=False,
                cpu_type=None,
                integrated_rom_size=0,
                integrated_main_ram_size=0x400000,
            )
            soc.finalize()
            self.submodules.soc = soc
            self.soc_ref = soc

            wb = soc.host_wb
            self.wb_adr = Signal(30, name="wb_adr")
            self.wb_dat_w = Signal(32, name="wb_dat_w")
            self.wb_dat_r = Signal(32, name="wb_dat_r")
            self.wb_sel = Signal(4, name="wb_sel")
            self.wb_we = Signal(name="wb_we")
            self.wb_cyc = Signal(name="wb_cyc")
            self.wb_stb = Signal(name="wb_stb")
            self.wb_ack = Signal(name="wb_ack")

            self.comb += [
                wb.adr.eq(self.wb_adr),
                wb.dat_w.eq(self.wb_dat_w),
                self.wb_dat_r.eq(wb.dat_r),
                wb.sel.eq(self.wb_sel),
                wb.we.eq(self.wb_we),
                wb.cyc.eq(self.wb_cyc),
                wb.stb.eq(self.wb_stb),
                self.wb_ack.eq(wb.ack),
            ]
            self.ios = {
                self.cd_sys.clk,
                self.cd_sys.rst,
                self.wb_adr,
                self.wb_dat_w,
                self.wb_dat_r,
                self.wb_sel,
                self.wb_we,
                self.wb_cyc,
                self.wb_stb,
                self.wb_ack,
            }

    top = _SimSoCDmaTop()
    csr_json = json.loads(
        export.get_csr_json(
            soc=top.soc_ref,
            csr_regions=top.soc_ref.csr.regions,
            constants=top.soc_ref.constants,
            mem_regions=top.soc_ref.mem_regions,
        )
    )
    csr_addrs = {k: int(v["addr"]) for k, v in csr_json["csr_registers"].items()}

    spec_file = tmp_path / "dma_spec.json"
    spec_file.write_text(
        json.dumps({"csr_addrs": csr_addrs, "cases": case_specs}),
        encoding="utf-8",
    )

    with patch.dict(os.environ, {"SETUN_DMA_SPEC_JSON": str(spec_file)}):
        run_cocotb_verilator(
            dut=top,
            ios=top.ios,
            test_module="unit.cocotb_soc_dma_tb",
            toplevel="sim_soc_dma_top",
            testcase="tb_soc_dma_gemv_sweep",
            seed=1958,
        )
