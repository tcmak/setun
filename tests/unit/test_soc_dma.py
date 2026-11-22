"""Red-first unit and Verilator tests for SimSoC Wishbone burst DMA (Task 3.2)."""

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


def _align4(offset: int) -> int:
    return (offset + 3) & ~3


def _to_signed_i32(word_u32: int) -> int:
    val = int(word_u32) & 0xFFFFFFFF
    return val - 0x100000000 if (val & 0x80000000) else val


def _pack_ram_arena(
    blobs: list[bytes],
) -> tuple[bytes, list[int]]:
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


def test_soc_dma_ram_init_migen_and_cli_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fast check: --ram-init preload, 64x64 burst DMA + out_addr writeback, & partial ptr error."""
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

        # 2. Valid 64x64 RAM-resident GEMV (with out_addr=0 first to check >85% read burst util)
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
        beats = yield from wb_read(addrs["ternary_accel_perf_bus_beats"])
        assert beats / max(cyc, 1) >= 0.80, f"Expected >=80% bus util, got {beats}/{cyc}"

        for r in range(n_rows):
            yield from wb_write(addrs["ternary_accel_out_row_idx"], r)
            got_v = _to_signed_i32((yield from wb_read(addrs["ternary_accel_out_row_data"])))
            assert got_v == int(exp_raw[r])

        # 3. Launch with Requantize + out_addr writeback to main_ram
        yield from wb_write(addrs["ternary_accel_out_addr"], MAIN_RAM_BASE + o_off)
        yield from wb_write(addrs["ternary_accel_req_cfg"], 256 | (8 << 16))
        yield from wb_write(addrs["ternary_accel_ctrl"], 0b01101)  # enable | start | req_enable

        while True:
            for _ in range(64):
                yield
            st2 = yield from wb_read(addrs["ternary_accel_status"])
            if (st2 >> 1) & 0x1:
                break

        for r in range(n_rows):
            got_ram = _to_signed_i32((yield from wb_read(MAIN_RAM_BASE + o_off + r * 4)))
            assert got_ram == int(exp_req[r])

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
@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_soc_dma_verilator_gemv_64_256_2048_and_bus_util(tmp_path: Path) -> None:
    """Task 3.2 Accept: Bit-exact DMA GEMV and >=80% bus util for N, K in {64, 256, 2048}."""
    rng = np.random.default_rng(19580302)
    rows, lanes, group_size, beat_bits = 4, 64, 128, 256

    # Allocate one shared 2048x2048 weight/scale/act arena in main_ram + output buffer,
    # slicing distinct row/group sub-windows across all 9 (N, K) in {64, 256, 2048}^2!
    max_n, max_k = 2048, 2048
    w_big = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(max_n, max_k))
    x_big = rng.integers(-128, 128, size=max_k, dtype=np.int8)
    s_big = rng.integers(32, 1024, size=(max_n, max_k // group_size), dtype=np.uint16)

    blobs: list[bytes] = []
    case_specs: list[dict[str, object]] = []

    # Pack exact .ttape slices for each K in {64, 256, 2048} (since row stride depends on K!)
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
    assert len(arena_bytes) < 0x400000, f"Arena {len(arena_bytes)} exceeds 4 MiB main_ram"

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
            # Test out_addr DMA writeback on diagonal shapes and BRAM readback on off-diagonals
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
                    "expected": [int(v) for v in exp.tolist()],
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
