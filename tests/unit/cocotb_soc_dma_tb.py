"""Cocotb + Verilator testbench for SimSoC burst DMA and packed int8 (Tasks 3.2/3.2b)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


def _to_signed_i32(val: int) -> int:
    u = int(val) & 0xFFFFFFFF
    return u - 0x100000000 if (u & 0x80000000) else u


def _to_signed_i8(val: int) -> int:
    u = int(val) & 0xFF
    return u - 0x100 if (u & 0x80) else u


async def _wb_write(dut, byte_addr: int, data: int) -> None:
    dut.wb_adr.value = (byte_addr >> 2) & 0x3FFFFFFF
    dut.wb_dat_w.value = int(data) & 0xFFFFFFFF
    dut.wb_sel.value = 0xF
    dut.wb_we.value = 1
    dut.wb_cyc.value = 1
    dut.wb_stb.value = 1
    while True:
        await RisingEdge(dut.sys_clk)
        if int(dut.wb_ack.value) == 1:
            break
    dut.wb_cyc.value = 0
    dut.wb_stb.value = 0
    dut.wb_we.value = 0


async def _wb_read(dut, byte_addr: int) -> int:
    dut.wb_adr.value = (byte_addr >> 2) & 0x3FFFFFFF
    dut.wb_sel.value = 0xF
    dut.wb_we.value = 0
    dut.wb_cyc.value = 1
    dut.wb_stb.value = 1
    while True:
        await RisingEdge(dut.sys_clk)
        if int(dut.wb_ack.value) == 1:
            val = int(dut.wb_dat_r.value)
            break
    dut.wb_cyc.value = 0
    dut.wb_stb.value = 0
    return val


@cocotb.test()
async def tb_soc_dma_gemv_sweep(dut) -> None:
    """Verify bit-exact RAM-resident GEMV, >=80% bus util, N>512 wide, & 3.2b packed int8."""
    spec_path = Path(os.environ["SETUN_DMA_SPEC_JSON"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    addrs: dict[str, int] = spec["csr_addrs"]
    cases: list[dict[str, object]] = spec["cases"]
    results_out: list[dict[str, object]] = []

    cocotb.start_soon(Clock(dut.sys_clk, 10, units="ns").start())
    dut.sys_rst.value = 1
    dut.wb_adr.value = 0
    dut.wb_dat_w.value = 0
    dut.wb_sel.value = 0
    dut.wb_we.value = 0
    dut.wb_cyc.value = 0
    dut.wb_stb.value = 0
    for _ in range(4):
        await RisingEdge(dut.sys_clk)
    dut.sys_rst.value = 0
    for _ in range(2):
        await RisingEdge(dut.sys_clk)

    sentinel_word = 0x3CC35AA5

    for case in cases:
        n_rows = int(case["n_rows"])
        k_cols = int(case["k_cols"])
        g_size = int(case["group_size"])
        shift = int(case["shift"])
        w_addr = int(case["weight_addr"])
        s_addr = int(case["scale_addr"])
        a_addr = int(case["act_addr"])
        o_addr = int(case["out_addr"])
        req_en = bool(case.get("req_enable", False))
        req_scale = int(case.get("req_scale", 1))
        req_shift = int(case.get("req_shift", 0))
        check_util = bool(case.get("check_util", True))
        expected = [int(v) for v in case["expected"]]

        if req_en and o_addr != 0:
            n_init_words = ((n_rows + 3) // 4) + 2
            for w_i in range(n_init_words):
                await _wb_write(dut, o_addr + w_i * 4, sentinel_word)

        await _wb_write(dut, addrs["ternary_accel_cfg"], (shift & 0x3F) | (n_rows << 8))
        await _wb_write(dut, addrs["ternary_accel_dim_k"], (k_cols & 0xFFFF) | (g_size << 16))
        await _wb_write(dut, addrs["ternary_accel_weight_addr"], w_addr)
        await _wb_write(dut, addrs["ternary_accel_scale_addr"], s_addr)
        await _wb_write(dut, addrs["ternary_accel_act_addr"], a_addr)
        await _wb_write(dut, addrs["ternary_accel_out_addr"], o_addr)
        if req_en:
            await _wb_write(
                dut, addrs["ternary_accel_req_cfg"], (req_scale & 0xFFFF) | (req_shift << 16)
            )

        ctrl_val = 0b01101 if req_en else 0b00101
        await _wb_write(dut, addrs["ternary_accel_ctrl"], ctrl_val)

        polls = 0
        while True:
            for _ in range(256):
                await RisingEdge(dut.sys_clk)
            st = await _wb_read(dut, addrs["ternary_accel_status"])
            err = (st >> 4) & 0x1
            done = (st >> 1) & 0x1
            assert err == 0, f"[soc_dma] unexpected error status=0x{st:08x} on ({n_rows},{k_cols})"
            if done == 1:
                break
            polls += 1
            if polls > 4000:
                raise AssertionError(
                    f"[soc_dma] timeout waiting for done on ({n_rows},{k_cols}), st=0x{st:08x}"
                )

        p_cyc = await _wb_read(dut, addrs["ternary_accel_perf_cycles"])
        p_beats = await _wb_read(dut, addrs["ternary_accel_perf_bus_beats"])
        util = p_beats / max(p_cyc, 1)
        results_out.append(
            {
                "n_rows": n_rows,
                "k_cols": k_cols,
                "req_enable": req_en,
                "out_dma": o_addr != 0,
                "beats": p_beats,
                "cycles": p_cyc,
                "bus_util_pct": round(util * 100.0, 2),
            }
        )
        if check_util:
            assert util >= 0.80, (
                f"[soc_dma] bus utilization below 80% on N={n_rows}, K={k_cols}: "
                f"got {util * 100.0:.2f}% ({p_beats}/{p_cyc})"
            )

        if req_en and o_addr != 0:
            n_words = ((n_rows + 3) // 4) + 2
            raw_bytes = bytearray()
            for w_i in range(n_words):
                wd = await _wb_read(dut, o_addr + w_i * 4)
                raw_bytes.extend(wd.to_bytes(4, "little"))
            for r in range(n_rows):
                got_q = _to_signed_i8(raw_bytes[r])
                if got_q != expected[r]:
                    raise AssertionError(
                        f"[soc_dma:packed_i8] mismatch on N={n_rows}, K={k_cols} row={r}: "
                        f"expected {expected[r]}, got {got_q}"
                    )
            sent_bytes = sentinel_word.to_bytes(4, "little") * n_words
            for b_pos in range(n_rows, len(raw_bytes)):
                if raw_bytes[b_pos] != sent_bytes[b_pos]:
                    raise AssertionError(
                        f"[soc_dma:sentinel] byte clobbered at offset {b_pos} (N={n_rows}): "
                        f"expected 0x{sent_bytes[b_pos]:02x}, got 0x{raw_bytes[b_pos]:02x}"
                    )
        else:
            for r in range(n_rows):
                if o_addr != 0:
                    raw_w = await _wb_read(dut, o_addr + r * 4)
                else:
                    await _wb_write(dut, addrs["ternary_accel_out_row_idx"], r)
                    raw_w = await _wb_read(dut, addrs["ternary_accel_out_row_data"])
                got_v = _to_signed_i32(raw_w)
                if got_v != expected[r]:
                    raise AssertionError(
                        f"[soc_dma] mismatch on N={n_rows}, K={k_cols} row={r}: "
                        f"expected {expected[r]}, got {got_v}"
                    )

    spec_path.with_suffix(".results.json").write_text(
        json.dumps(results_out, indent=2) + "\n", encoding="utf-8"
    )
