"""Red-first unit and Verilator cocotb tests for Requantize (Task 2.5, NUMERICS.md §4)."""

from __future__ import annotations

import os
import random

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from setun.ref.ops import requantize_int8, requantize_vec_int8, tile_gemv
from tests.sim_helper import run_cocotb_verilator, run_migen_sim
from tests.unit.test_tile import _prepare_tile_stripes


def _to_signed(val: int, bits: int) -> int:
    mask = (1 << bits) - 1
    v = val & mask
    if v & (1 << (bits - 1)):
        v -= 1 << bits
    return v


def _to_unsigned(val: int, bits: int) -> int:
    return int(val) & ((1 << bits) - 1)


# ---------------------------------------------------------------------------
# 1. Golden Reference Unit Tests (setun/ref/ops.py::requantize_int8)
# ---------------------------------------------------------------------------


def test_requantize_ref_corners_and_validation() -> None:
    """Verify 65-bit overflow safety, shift==0 bypass, ties, zero, and sym/asym saturation."""
    min_i48 = -(1 << 47)
    max_i48 = (1 << 47) - 1

    # Zero input or zero scale -> 0, never saturated
    assert requantize_int8(0, 65535, 0) == (0, False)
    assert requantize_int8(max_i48, 0, 0) == (0, False)
    assert requantize_int8(min_i48, 0, 62, symmetric=True) == (0, False)

    # shift == 0 bypass (no 1 << -1 bias) and exact clamp boundaries
    assert requantize_int8(127, 1, 0) == (127, False)
    assert requantize_int8(128, 1, 0) == (127, True)
    assert requantize_int8(-128, 1, 0, symmetric=False) == (-128, False)
    assert requantize_int8(-129, 1, 0, symmetric=False) == (-128, True)
    assert requantize_int8(-127, 1, 0, symmetric=True) == (-127, False)
    assert requantize_int8(-128, 1, 0, symmetric=True) == (-127, True)

    # Round-half-up ties (shift = 1 -> +0.5 rounds toward +inf)
    assert requantize_int8(1, 1, 1) == (1, False)  # +0.5 -> +1
    assert requantize_int8(3, 1, 1) == (2, False)  # +1.5 -> +2
    assert requantize_int8(-1, 1, 1) == (0, False)  # -0.5 -> 0
    assert requantize_int8(-3, 1, 1) == (-1, False)  # -1.5 -> -1
    assert requantize_int8(255, 1, 1) == (127, True)  # +127.5 -> +128 -> sat +127
    assert requantize_int8(-257, 1, 1, symmetric=False) == (-128, False)  # -128.5 -> -128
    assert requantize_int8(-258, 1, 1, symmetric=False) == (-128, True)  # -129.0 -> sat -128

    # Extreme 65-bit signed intermediate sum: (2^47 - 1) * 65535 + (1 << 61) >> 62
    # Must NOT overflow 64-bit signed into negative!
    q_pos, sat_pos = requantize_int8(max_i48, 65535, 62)
    assert q_pos == 2 and sat_pos is False
    q_neg, sat_neg = requantize_int8(min_i48, 65535, 62)
    assert q_neg == -2 and sat_neg is False

    # Extreme saturation at shift = 47, scale = 65535: ±65535 -> ±127 / -128
    assert requantize_int8(max_i48, 65535, 47) == (127, True)
    assert requantize_int8(min_i48, 65535, 47, symmetric=False) == (-128, True)
    assert requantize_int8(min_i48, 65535, 47, symmetric=True) == (-127, True)

    # Goose-style diagnostic validation checks (both upper and lower negative bounds)
    for bad_val, bad_scale, bad_shift in [
        (max_i48 + 1, 1, 0),
        (min_i48 - 1, 1, 0),
        (0, 65536, 0),
        (0, -1, 0),
        (0, 1, 63),
        (0, 1, -1),
    ]:
        with pytest.raises(ValueError, match="requantize_int8"):
            requantize_int8(bad_val, bad_scale, bad_shift)

    with pytest.raises(ValueError, match="requantize_vec_int8"):
        requantize_vec_int8(np.zeros((2, 4), dtype=np.int64), 1, 0)


def test_requantize_constructor_negative_guards() -> None:
    """Verify Requantize rejects invalid bit widths at elaboration."""
    from setun.accelerators.requantize import Requantize

    with pytest.raises(ValueError, match="Requantize"):
        Requantize(in_bits=48, out_bits=1)
    with pytest.raises(ValueError, match="Requantize"):
        Requantize(in_bits=8, out_bits=16)


def test_requantize_migen_sim_bubbles_and_backpressure() -> None:
    """Fast in-process Migen simulation of Requantize covering corners, bubbles, and stalls."""
    from setun.accelerators.requantize import Requantize

    dut = Requantize(in_bits=48, out_bits=8)
    dut._fragment.clock_domains.clear()

    vec = [0, 127, 128, -128, -129, 255, -257, (1 << 47) - 1, -(1 << 47)]

    def tb():
        for scale_u16, shift, sym in [(1, 0, 0), (1, 1, 1), (65535, 62, 0)]:
            yield dut.clear.eq(1)
            yield dut.sink.valid.eq(0)
            yield dut.source.ready.eq(0)
            yield
            yield dut.clear.eq(0)
            yield dut.scale_u16.eq(scale_u16)
            yield dut.shift.eq(shift)
            yield dut.symmetric.eq(sym)
            yield

            in_ptr = 0
            held = False
            got: list[tuple[int, int, int, int]] = []
            for cyc in range(1, 160):
                if len(got) == len(vec):
                    break
                drive_v = in_ptr < len(vec) and (held or (cyc % 4 not in (0, 1)))
                drive_r = 1 if (cyc % 3 != 0) else 0
                if drive_v:
                    yield dut.sink.result.eq(_to_unsigned(vec[in_ptr], 48))
                    yield dut.sink.row_idx.eq(in_ptr)
                    yield dut.sink.last.eq(1 if in_ptr == len(vec) - 1 else 0)
                    yield dut.sink.valid.eq(1)
                    held = True
                else:
                    yield dut.sink.valid.eq(0)
                yield dut.source.ready.eq(drive_r)
                yield

                if drive_v and (yield dut.sink.ready):
                    in_ptr += 1
                    held = False
                if (yield dut.source.valid) and drive_r:
                    got.append(
                        (
                            (yield dut.source.row_idx),
                            _to_signed((yield dut.source.result), 8),
                            (yield dut.source.saturated),
                            (yield dut.source.last),
                        )
                    )

            assert len(got) == len(vec), (
                f"[test_requantize:Requantize] expected {len(vec)} outputs, got {len(got)}"
            )
            for idx, (g_idx, g_q, g_sat, g_last) in enumerate(got):
                exp_q, exp_sat_b = requantize_int8(vec[idx], scale_u16, shift, symmetric=bool(sym))
                exp_tup = (idx, exp_q, int(exp_sat_b), 1 if idx == len(vec) - 1 else 0)
                assert (g_idx, g_q, g_sat, g_last) == exp_tup, (
                    f"[test_requantize:Requantize] mismatch at item {idx} "
                    f"(scale={scale_u16}, shift={shift}, sym={sym}): "
                    f"expected {exp_tup}, got {(g_idx, g_q, g_sat, g_last)}"
                )

    run_migen_sim(dut, tb)


# ---------------------------------------------------------------------------
# 2. Standalone Requantize Cocotb Testbench (Corners + II=1 + Skid Backpressure)
# ---------------------------------------------------------------------------


@cocotb.test()
async def tb_requantize_corners_ii1_and_backpressure(dut) -> None:
    """Test all shift/scale/±2^47 corners, II=1 throughput, and 2-stage skid stalls."""
    seed = int(os.environ.get("TEST_SEED", "1958"))
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    cocotb.start_soon(Clock(dut.sys_clk, 10, units="ns").start())
    dut.sys_rst.value = 1
    dut.clear.value = 0
    dut.scale_u16.value = 1
    dut.shift.value = 0
    dut.symmetric.value = 0
    dut.sink_valid.value = 0
    dut.sink_last.value = 0
    dut.sink_result.value = 0
    dut.sink_row_idx.value = 0
    dut.source_ready.value = 0

    for _ in range(3):
        await RisingEdge(dut.sys_clk)
    dut.sys_rst.value = 0
    await RisingEdge(dut.sys_clk)

    min_i48 = -(1 << 47)
    max_i48 = (1 << 47) - 1

    # Part A: Sweep all mandatory (scale_u16, shift, symmetric) configurations under random stalls
    configs = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
        (65535, 0, 0),
        (65535, 15, 0),
        (65535, 31, 1),
        (65535, 47, 0),
        (65535, 62, 0),
        (65535, 62, 1),
        (65535, 63, 0),
    ]

    directed_vals = [
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        126,
        127,
        128,
        129,
        -126,
        -127,
        -128,
        -129,
        254,
        255,
        256,
        -255,
        -256,
        -257,
        -258,
        (1 << 31) - 1,
        -(1 << 31),
        max_i48,
        max_i48 - 1,
        min_i48,
        min_i48 + 1,
    ]

    for scale_u16, shift, sym in configs:
        rand_vals = rng.integers(min_i48, max_i48 + 1, size=48, dtype=np.int64).tolist()
        vec = directed_vals + rand_vals
        n_items = len(vec)

        dut.clear.value = 1
        dut.sink_valid.value = 0
        dut.source_ready.value = 0
        await RisingEdge(dut.sys_clk)
        dut.clear.value = 0
        dut.scale_u16.value = scale_u16
        dut.shift.value = shift
        dut.symmetric.value = sym
        await RisingEdge(dut.sys_clk)

        in_ptr = 0
        held_sink = False
        got: list[tuple[int, int, int, int]] = []
        guard = 0

        while len(got) < n_items:
            guard += 1
            if guard > n_items * 16 + 100:
                raise AssertionError(
                    f"[test_requantize.py:tb_requantize] timeout in cfg "
                    f"(scale={scale_u16}, shift={shift}, sym={sym}) at cycle={guard}: "
                    f"got {len(got)}/{n_items} items."
                )

            if in_ptr < n_items:
                if held_sink or py_rng.random() < 0.70:
                    dut.sink_result.value = _to_unsigned(vec[in_ptr], 48)
                    dut.sink_row_idx.value = in_ptr
                    dut.sink_last.value = 1 if (in_ptr == n_items - 1) else 0
                    dut.sink_valid.value = 1
                    held_sink = True
                else:
                    dut.sink_valid.value = 0
            else:
                dut.sink_valid.value = 0
                held_sink = False

            # Exercise multi-cycle downstream stalls as well as single-cycle toggles
            dut.source_ready.value = 1 if (py_rng.random() < 0.62) else 0

            await RisingEdge(dut.sys_clk)

            if int(dut.sink_valid.value) == 1 and int(dut.sink_ready.value) == 1:
                in_ptr += 1
                held_sink = False

            if int(dut.source_valid.value) == 1 and int(dut.source_ready.value) == 1:
                q_out = _to_signed(int(dut.source_result.value), 8)
                r_idx = int(dut.source_row_idx.value)
                sat_f = int(dut.source_saturated.value)
                last_f = int(dut.source_last.value)
                got.append((r_idx, q_out, sat_f, last_f))

        for idx, (got_idx, got_q, got_sat, got_last) in enumerate(got):
            exp_q, exp_sat_bool = requantize_int8(
                vec[idx], scale_u16, min(shift, 62), symmetric=bool(sym)
            )
            exp_sat = 1 if exp_sat_bool else 0
            exp_last = 1 if (idx == n_items - 1) else 0
            if (got_idx, got_q, got_sat, got_last) != (idx, exp_q, exp_sat, exp_last):
                raise AssertionError(
                    f"[test_requantize.py:tb_requantize] mismatch at item {idx} "
                    f"(val={vec[idx]}, scale={scale_u16}, shift={shift}, sym={sym}): "
                    f"expected (idx={idx}, q={exp_q}, sat={exp_sat}, last={exp_last}), "
                    f"got (idx={got_idx}, q={got_q}, sat={got_sat}, last={got_last})."
                )

    # Part B: Verify continuous II = 1 line-rate throughput (N=512 beats in <= N + 3 cycles)
    n_stream = 512
    line_vals = rng.integers(-50000, 50000, size=n_stream, dtype=np.int64).tolist()
    dut.clear.value = 1
    dut.sink_valid.value = 0
    dut.source_ready.value = 0
    await RisingEdge(dut.sys_clk)
    dut.clear.value = 0
    dut.scale_u16.value = 317
    dut.shift.value = 14
    dut.symmetric.value = 0
    await RisingEdge(dut.sys_clk)

    in_ptr = 0
    out_cnt = 0
    cycles = 0
    dut.source_ready.value = 1
    while out_cnt < n_stream:
        if in_ptr < n_stream:
            dut.sink_result.value = _to_unsigned(line_vals[in_ptr], 48)
            dut.sink_row_idx.value = in_ptr
            dut.sink_last.value = 1 if (in_ptr == n_stream - 1) else 0
            dut.sink_valid.value = 1
        else:
            dut.sink_valid.value = 0

        await RisingEdge(dut.sys_clk)
        cycles += 1

        if int(dut.sink_valid.value) == 1 and int(dut.sink_ready.value) == 1:
            in_ptr += 1
        if int(dut.source_valid.value) == 1 and int(dut.source_ready.value) == 1:
            exp_q, exp_sat_b = requantize_int8(line_vals[out_cnt], 317, 14)
            got_q = _to_signed(int(dut.source_result.value), 8)
            got_sat = int(dut.source_saturated.value)
            if got_q != exp_q or got_sat != int(exp_sat_b):
                raise AssertionError(
                    f"[test_requantize.py:tb_requantize] II=1 stream mismatch at {out_cnt}: "
                    f"expected (q={exp_q}, sat={int(exp_sat_b)}), got (q={got_q}, sat={got_sat})."
                )
            out_cnt += 1

    # 2-stage pipeline takes exactly n_stream + 2 cycles from first valid beat to final drain
    if cycles > n_stream + 3:
        raise AssertionError(
            f"[test_requantize.py:tb_requantize] II=1 throughput invariant violated: "
            f"expected <= {n_stream + 3} cycles for {n_stream} beats, took {cycles} cycles."
        )


# ---------------------------------------------------------------------------
# 3. End-to-End Composition Cocotb Testbench: TernaryTile -> Requantize
# ---------------------------------------------------------------------------


@cocotb.test()
async def tb_tile_to_requantize_composition(dut) -> None:
    """Verify TernaryTile.source -> Requantize.sink composition under bubbles & backpressure."""
    rows = int(os.environ["SETUN_TILE_ROWS"])
    lanes = int(os.environ["SETUN_TILE_LANES"])
    group_size = int(os.environ["SETUN_TILE_GROUP"])
    seed = int(os.environ.get("TEST_SEED", "1958"))

    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    cocotb.start_soon(Clock(dut.sys_clk, 10, units="ns").start())
    dut.sys_rst.value = 1
    dut.clear.value = 0
    dut.tile_shift.value = 0
    dut.num_rows.value = 0
    dut.req_scale_u16.value = 1
    dut.req_shift.value = 0
    dut.req_symmetric.value = 0
    dut.act_wr_en.value = 0
    dut.act_wr_addr.value = 0
    dut.act_wr_data.value = 0
    dut.sink_valid.value = 0
    dut.sink_last.value = 0
    dut.sink_trits.value = 0
    dut.sink_scales.value = 0
    dut.source_ready.value = 0

    for _ in range(3):
        await RisingEdge(dut.sys_clk)
    dut.sys_rst.value = 0
    await RisingEdge(dut.sys_clk)

    scenarios = [
        # Ragged N=13, K=128, asymmetric [-128, 127] with heavy saturation & ties
        ("ragged_13x128_asym", 13, 128, 4, 219, 13, 0),
        # M0 shape 64x64, symmetric [-127, 127]
        ("m0_64x64_sym", 64, 64, 6, 409, 12, 1),
    ]

    for name, n_rows, k_cols, tile_shift, req_scale, req_shift, req_sym in scenarios:
        n_groups = (k_cols + group_size - 1) // group_size
        w = rng.choice([-1, 0, 1], size=(n_rows, k_cols)).astype(np.int8)
        x = rng.integers(-128, 128, size=(k_cols,), dtype=np.int8)
        s = rng.integers(64, 2048, size=(n_rows, n_groups), dtype=np.uint16)

        tile_out = tile_gemv(
            w, x, s, group_size=group_size, lanes=lanes, rows=rows, shift=tile_shift
        )
        exp_q_arr, exp_sat_arr = requantize_vec_int8(
            tile_out, req_scale, req_shift, symmetric=bool(req_sym)
        )
        act_slices, stream_beats = _prepare_tile_stripes(
            w, x, s, rows=rows, lanes=lanes, group_size=group_size
        )

        dut.clear.value = 1
        dut.sink_valid.value = 0
        dut.source_ready.value = 0
        await RisingEdge(dut.sys_clk)
        dut.clear.value = 0
        dut.num_rows.value = n_rows
        dut.tile_shift.value = tile_shift
        dut.req_scale_u16.value = req_scale
        dut.req_shift.value = req_shift
        dut.req_symmetric.value = req_sym
        await RisingEdge(dut.sys_clk)

        for addr_i, slice_val in enumerate(act_slices):
            dut.act_wr_addr.value = addr_i
            dut.act_wr_data.value = slice_val
            dut.act_wr_en.value = 1
            await RisingEdge(dut.sys_clk)
        dut.act_wr_en.value = 0
        await RisingEdge(dut.sys_clk)

        in_ptr = 0
        held_sink = False
        got_rows: list[tuple[int, int, int, int]] = []
        guard = 0
        max_cycles = len(stream_beats) * (rows + 16) + 300

        while len(got_rows) < n_rows:
            guard += 1
            if guard > max_cycles:
                raise AssertionError(
                    f"[test_requantize.py:tb_tile_to_requantize_composition] timeout in "
                    f"'{name}' at cycle={guard}: got {len(got_rows)}/{n_rows} rows."
                )

            if in_ptr < len(stream_beats):
                if held_sink or py_rng.random() < 0.72:
                    t_word, s_word, last_f = stream_beats[in_ptr]
                    dut.sink_trits.value = t_word
                    dut.sink_scales.value = s_word
                    dut.sink_last.value = last_f
                    dut.sink_valid.value = 1
                    held_sink = True
                else:
                    dut.sink_valid.value = 0
            else:
                dut.sink_valid.value = 0
                held_sink = False

            dut.source_ready.value = 1 if (py_rng.random() < 0.60) else 0

            await RisingEdge(dut.sys_clk)

            if int(dut.sink_valid.value) == 1 and int(dut.sink_ready.value) == 1:
                in_ptr += 1
                held_sink = False

            if int(dut.source_valid.value) == 1 and int(dut.source_ready.value) == 1:
                got_rows.append(
                    (
                        int(dut.source_row_idx.value),
                        _to_signed(int(dut.source_result.value), 8),
                        int(dut.source_saturated.value),
                        int(dut.source_last.value),
                    )
                )

        # Ensure pipeline stays quiet after final source_last
        dut.sink_valid.value = 0
        dut.source_ready.value = 1
        for extra in range(8):
            await RisingEdge(dut.sys_clk)
            if int(dut.source_valid.value) != 0:
                raise AssertionError(
                    f"[test_requantize.py:tb_tile_to_requantize_composition] spurious beat after "
                    f"source_last in '{name}' at post-drain cycle={extra}."
                )

        for idx, (got_idx, got_q, got_sat, got_last) in enumerate(got_rows):
            exp_q = int(exp_q_arr[idx])
            exp_sat = int(exp_sat_arr[idx])
            exp_last = 1 if (idx == n_rows - 1) else 0
            if (got_idx, got_q, got_sat, got_last) != (idx, exp_q, exp_sat, exp_last):
                raise AssertionError(
                    f"[test_requantize.py:tb_tile_to_requantize_composition] mismatch in '{name}' "
                    f"at row {idx}: expected (idx={idx}, q={exp_q}, sat={exp_sat}, "
                    f"last={exp_last}), got (idx={got_idx}, q={got_q}, sat={got_sat}, "
                    f"last={got_last})."
                )


def _build_requantize_sim_wrapper():
    from migen import ClockDomain, Module, Signal

    from setun.accelerators.requantize import Requantize

    class _ReqTop(Module):
        def __init__(self) -> None:
            self.clock_domains.cd_sys = ClockDomain("sys")
            req = Requantize(in_bits=48, out_bits=8)
            if hasattr(req, "_fragment"):
                req._fragment.clock_domains.clear()
            self.submodules.req = req

            self.clear = req.clear
            self.scale_u16 = req.scale_u16
            self.shift = req.shift
            self.symmetric = req.symmetric

            self.sink_valid = Signal(name="sink_valid")
            self.sink_ready = Signal(name="sink_ready")
            self.sink_last = Signal(name="sink_last")
            self.sink_result = Signal(48, name="sink_result")
            self.sink_row_idx = Signal(16, name="sink_row_idx")

            self.source_valid = Signal(name="source_valid")
            self.source_ready = Signal(name="source_ready")
            self.source_last = Signal(name="source_last")
            self.source_result = Signal(8, name="source_result")
            self.source_row_idx = Signal(16, name="source_row_idx")
            self.source_saturated = Signal(name="source_saturated")

            self.comb += [
                req.sink.valid.eq(self.sink_valid),
                self.sink_ready.eq(req.sink.ready),
                req.sink.last.eq(self.sink_last),
                req.sink.result.eq(self.sink_result),
                req.sink.row_idx.eq(self.sink_row_idx),
                self.source_valid.eq(req.source.valid),
                req.source.ready.eq(self.source_ready),
                self.source_last.eq(req.source.last),
                self.source_result.eq(req.source.result),
                self.source_row_idx.eq(req.source.row_idx),
                self.source_saturated.eq(req.source.saturated),
            ]

            self.ios = {
                self.cd_sys.clk,
                self.cd_sys.rst,
                self.clear,
                self.scale_u16,
                self.shift,
                self.symmetric,
                self.sink_valid,
                self.sink_ready,
                self.sink_last,
                self.sink_result,
                self.sink_row_idx,
                self.source_valid,
                self.source_ready,
                self.source_last,
                self.source_result,
                self.source_row_idx,
                self.source_saturated,
            }

    return _ReqTop()


def _build_tile_req_composition_wrapper(rows: int = 4, lanes: int = 16, group_size: int = 64):
    from migen import ClockDomain, Module, Signal

    from setun.accelerators.requantize import Requantize
    from setun.accelerators.tile import TernaryTile

    class _TileReqTop(Module):
        def __init__(self) -> None:
            self.clock_domains.cd_sys = ClockDomain("sys")
            tile = TernaryTile(
                rows=rows, lanes=lanes, group_size=group_size, act_bits=8, acc_bits=48, max_k=256
            )
            req = Requantize(in_bits=48, out_bits=8)
            if hasattr(tile, "_fragment"):
                tile._fragment.clock_domains.clear()
            if hasattr(req, "_fragment"):
                req._fragment.clock_domains.clear()
            self.submodules.tile = tile
            self.submodules.req = req

            self.clear = Signal(name="clear")
            self.tile_shift = Signal(6, name="tile_shift")
            self.num_rows = tile.num_rows
            self.req_scale_u16 = Signal(16, name="req_scale_u16")
            self.req_shift = Signal(6, name="req_shift")
            self.req_symmetric = Signal(name="req_symmetric")

            self.act_wr_addr = tile.act_wr_addr
            self.act_wr_data = tile.act_wr_data
            self.act_wr_en = tile.act_wr_en

            self.sink_valid = Signal(name="sink_valid")
            self.sink_ready = Signal(name="sink_ready")
            self.sink_last = Signal(name="sink_last")
            self.sink_trits = Signal(rows * 2 * lanes, name="sink_trits")
            self.sink_scales = Signal(rows * 16, name="sink_scales")

            self.source_valid = Signal(name="source_valid")
            self.source_ready = Signal(name="source_ready")
            self.source_last = Signal(name="source_last")
            self.source_result = Signal(8, name="source_result")
            self.source_row_idx = Signal(16, name="source_row_idx")
            self.source_saturated = Signal(name="source_saturated")

            # Direct LiteX stream.Endpoint connection: tile.source -> req.sink
            self.comb += [
                tile.clear.eq(self.clear),
                req.clear.eq(self.clear),
                tile.shift.eq(self.tile_shift),
                req.scale_u16.eq(self.req_scale_u16),
                req.shift.eq(self.req_shift),
                req.symmetric.eq(self.req_symmetric),
                tile.sink.valid.eq(self.sink_valid),
                self.sink_ready.eq(tile.sink.ready),
                tile.sink.last.eq(self.sink_last),
                tile.sink.trits.eq(self.sink_trits),
                tile.sink.scales.eq(self.sink_scales),
                tile.source.connect(req.sink),
                self.source_valid.eq(req.source.valid),
                req.source.ready.eq(self.source_ready),
                self.source_last.eq(req.source.last),
                self.source_result.eq(req.source.result),
                self.source_row_idx.eq(req.source.row_idx),
                self.source_saturated.eq(req.source.saturated),
            ]

            self.ios = {
                self.cd_sys.clk,
                self.cd_sys.rst,
                self.clear,
                self.tile_shift,
                self.num_rows,
                self.req_scale_u16,
                self.req_shift,
                self.req_symmetric,
                self.act_wr_addr,
                self.act_wr_data,
                self.act_wr_en,
                self.sink_valid,
                self.sink_ready,
                self.sink_last,
                self.sink_trits,
                self.sink_scales,
                self.source_valid,
                self.source_ready,
                self.source_last,
                self.source_result,
                self.source_row_idx,
                self.source_saturated,
            }

    return _TileReqTop()


@pytest.mark.slow
def test_requantize_rtl_corners_ii1_and_backpressure() -> None:
    """Verify Requantize RTL across ±2^47/shift/scale corners, II=1, and random stalls."""
    dut = _build_requantize_sim_wrapper()
    run_cocotb_verilator(
        dut=dut,
        ios=dut.ios,
        test_module="test_requantize",
        toplevel="requantize_top",
        testcase="tb_requantize_corners_ii1_and_backpressure",
    )


@pytest.mark.slow
def test_tile_to_requantize_end_to_end() -> None:
    """Verify TernaryTile -> Requantize composition via stream.Endpoint.connect()."""
    dut = _build_tile_req_composition_wrapper(rows=4, lanes=16, group_size=64)
    env_vars = {
        "SETUN_TILE_ROWS": "4",
        "SETUN_TILE_LANES": "16",
        "SETUN_TILE_GROUP": "64",
    }
    saved = {k: os.environ.get(k) for k in env_vars}
    os.environ.update(env_vars)
    try:
        run_cocotb_verilator(
            dut=dut,
            ios=dut.ios,
            test_module="test_requantize",
            toplevel="tile_req_comp_top",
            testcase="tb_tile_to_requantize_composition",
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
