"""Red-first unit and Verilator cocotb tests for TernaryTile (Task 2.4, ADR-015)."""

from __future__ import annotations

import math
import os
import random

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from setun.ref.format import pack_trits_1d
from setun.ref.ops import row_dot, tile_gemv
from tests.sim_helper import run_cocotb_verilator


def _pack_acts_slice_le(acts_slice: np.ndarray) -> int:
    """Pack L signed int8 activations LSB-first into an (8 * L)-bit unsigned integer."""
    raw = np.asarray(acts_slice, dtype=np.int8).tobytes()
    return int.from_bytes(raw, byteorder="little", signed=False)


def _prepare_tile_stripes(
    weights: np.ndarray,
    activations: np.ndarray,
    scales: np.ndarray,
    rows: int,
    lanes: int,
    group_size: int,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Pack activations and R-row lockstep (trits, scales, last) beats for TernaryTile."""
    n_rows, k_cols = weights.shape
    n_groups = (k_cols + group_size - 1) // group_size
    k_pad = n_groups * group_size
    beats_per_stripe = k_pad // lanes
    beats_per_group = group_size // lanes
    n_stripes = (n_rows + rows - 1) // rows
    n_pad = n_stripes * rows

    w_pad = np.zeros((n_pad, k_pad), dtype=np.int8)
    w_pad[:n_rows, :k_cols] = weights

    x_pad = np.zeros(k_pad, dtype=np.int8)
    x_pad[:k_cols] = activations

    s_pad = np.zeros((n_pad, n_groups), dtype=np.uint16)
    s_pad[:n_rows, :] = scales
    # Fill padded tail rows with non-zero trits/scales to prove hardware suppresses row_idx >= N
    if n_pad > n_rows:
        w_pad[n_rows:, :] = 1
        s_pad[n_rows:, :] = 0x7FFF

    act_slices = [
        _pack_acts_slice_le(x_pad[b * lanes : (b + 1) * lanes]) for b in range(beats_per_stripe)
    ]

    row_trit_bits = 2 * lanes
    trit_bytes_per_row_beat = row_trit_bits // 8
    packed_bytes = pack_trits_1d(w_pad.reshape(n_pad, beats_per_stripe, lanes))

    stream_beats: list[tuple[int, int, int]] = []
    for st in range(n_stripes):
        r_base = st * rows
        for b in range(beats_per_stripe):
            g_idx = b // beats_per_group
            trits_word = 0
            scales_word = 0
            for r_off in range(rows):
                r_bytes = packed_bytes[r_base + r_off, b].tobytes()
                assert len(r_bytes) == trit_bytes_per_row_beat
                r_tval = int.from_bytes(r_bytes, byteorder="little", signed=False)
                r_sval = int(s_pad[r_base + r_off, g_idx]) & 0xFFFF
                trits_word |= r_tval << (r_off * row_trit_bits)
                scales_word |= r_sval << (r_off * 16)
            last_flag = 1 if (b == beats_per_stripe - 1) else 0
            stream_beats.append((trits_word, scales_word, last_flag))

    return act_slices, stream_beats


def _to_signed(val: int, bits: int) -> int:
    mask = (1 << bits) - 1
    v = val & mask
    if v & (1 << (bits - 1)):
        v -= 1 << bits
    return v


# ---------------------------------------------------------------------------
# Golden Reference Unit Tests (setun/ref/ops.py::tile_gemv)
# ---------------------------------------------------------------------------


def test_tile_gemv_ref_parity_and_validation() -> None:
    """Verify tile_gemv matches row_dot across ragged N, ragged K, and shift corners."""
    rng = np.random.default_rng(19580204)
    for n_rows, k_cols, group_size, lanes, rows, shift in [
        (7, 129, 64, 16, 4, 0),
        (64, 64, 64, 16, 4, 5),
        (13, 256, 128, 64, 8, 12),
    ]:
        n_groups = math.ceil(k_cols / group_size)
        w = rng.choice([-1, 0, 1], size=(n_rows, k_cols)).astype(np.int8)
        x = rng.integers(-128, 128, size=(k_cols,), dtype=np.int8)
        s = rng.integers(0, 65536, size=(n_rows, n_groups), dtype=np.uint16)

        got = tile_gemv(w, x, s, group_size=group_size, lanes=lanes, rows=rows, shift=shift)
        expected = np.array(
            [
                row_dot(w[r], x, s[r], group_size=group_size, lanes=lanes, shift=shift)
                for r in range(n_rows)
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(got, expected)

    w_ok = np.zeros((4, 64), dtype=np.int8)
    x_ok = np.zeros(64, dtype=np.int8)
    s_ok = np.zeros((4, 1), dtype=np.uint16)

    for bad_kwargs in [
        {"weights": w_ok, "activations": x_ok, "scales": np.zeros((4, 2), dtype=np.uint16)},
        {"weights": w_ok, "activations": x_ok, "scales": s_ok, "rows": 0},
        {"weights": w_ok, "activations": x_ok, "scales": s_ok, "group_size": 64, "lanes": 24},
        {"weights": w_ok, "activations": x_ok, "scales": s_ok, "shift": -1},
        {"weights": w_ok, "activations": x_ok, "scales": s_ok, "shift": 31},
        {"weights": np.zeros(64, dtype=np.int8), "activations": x_ok, "scales": s_ok},
        {"weights": np.zeros((0, 64), dtype=np.int8), "activations": x_ok, "scales": s_ok},
        {"weights": w_ok, "activations": np.zeros(32, dtype=np.int8), "scales": s_ok},
    ]:
        with pytest.raises(ValueError, match="tile_gemv"):
            tile_gemv(**bad_kwargs)


def test_ternary_tile_constructor_negative_guards() -> None:
    """Verify TernaryTile rejects invalid rows, lanes/group_size, and max_k at elaboration."""
    from setun.accelerators.tile import TernaryTile

    with pytest.raises(ValueError, match="TernaryTile"):
        TernaryTile(rows=0, lanes=16, group_size=64, max_k=128)
    with pytest.raises(ValueError, match="TernaryTile"):
        TernaryTile(rows=4, lanes=24, group_size=64, max_k=128)
    with pytest.raises(ValueError, match="TernaryTile"):
        TernaryTile(rows=4, lanes=16, group_size=64, max_k=32)
    with pytest.raises(ValueError, match="TernaryTile"):
        TernaryTile(rows=4, lanes=16, group_size=64, max_k=72)


# ---------------------------------------------------------------------------
# Cocotb Testbenches for setun.accelerators.tile.TernaryTile
# ---------------------------------------------------------------------------


@cocotb.test()
async def tb_tile_matrix_suite(dut) -> None:
    """Exercise ragged N, ragged K, K==L short stripes, bubbles, backpressure, and num_rows==0."""
    rows = int(os.environ["SETUN_TILE_ROWS"])
    lanes = int(os.environ["SETUN_TILE_LANES"])
    group_size = int(os.environ["SETUN_TILE_GROUP"])
    acc_bits = int(os.environ.get("SETUN_TILE_ACC_BITS", "48"))
    seed = int(os.environ.get("TEST_SEED", "1958"))

    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    cocotb.start_soon(Clock(dut.sys_clk, 10, units="ns").start())
    dut.sys_rst.value = 1
    dut.clear.value = 0
    dut.shift.value = 0
    dut.num_rows.value = 0
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

    # Guardrail check (ADR-015): num_rows == 0 must reject/stall sink and never assert source_valid
    dut.num_rows.value = 0
    dut.sink_valid.value = 1
    dut.sink_last.value = 1
    dut.source_ready.value = 1
    for cyc in range(6):
        await RisingEdge(dut.sys_clk)
        if int(dut.sink_ready.value) != 0 or int(dut.source_valid.value) != 0:
            raise AssertionError(
                f"[test_tile.py:tb_tile_matrix_suite] ADR-015 num_rows==0 guard violated at "
                f"cycle={cyc}: expected sink_ready=0 and source_valid=0, got "
                f"sink_ready={int(dut.sink_ready.value)}, "
                f"source_valid={int(dut.source_valid.value)}."
            )
    dut.sink_valid.value = 0
    dut.sink_last.value = 0
    await RisingEdge(dut.sys_clk)

    # Directed + randomized scenarios:
    # (name, n_rows, k_cols, shift, random_bubbles_and_bp)
    cases = [
        # Short stripe K == G == L (1 beat/stripe < R=4) back-to-back with no bubbles:
        # forces sink_ready backpressure while shadow buffer drains 4 rows!
        ("short_k_eq_l_back_to_back", 12, lanes, 0, False),
        # Short stripe K == L with ragged N % R != 0 and random stalls
        ("short_k_eq_l_ragged_bp", 7, lanes, 3, True),
        # Ragged N = 1 (single row on multi-row tile: source.last on row 0)
        ("single_row_n1", 1, group_size * 2, 0, True),
        # Ragged N ∈ {3, 5, 13} and ragged K = group_size * 2 + 1
        ("ragged_n3_ragged_k", 3, group_size * 2 + 1, 7, True),
        ("ragged_n5_exact_k", 5, group_size * 3, 11, True),
        ("ragged_n13_shift15", 13, group_size * 2, 15, True),
        # M0 linear shapes (64x64, 192x64, 128x128)
        ("m0_qkv_192x64", 192, max(64, group_size), 8, False),
        ("m0_mlp_128x128", 128, max(128, group_size), 9, True),
    ]

    for case_name, n_rows, k_cols, shift, use_bp in cases:
        if k_cols % group_size == 0 or case_name == "ragged_n3_ragged_k":
            eff_group = lanes if case_name.startswith("short_k_eq_l") else group_size
            if eff_group != group_size:
                # Only run K == L case when compiled tile has group_size == lanes, else K = G
                k_cols = group_size
        n_groups = math.ceil(k_cols / group_size)
        w = rng.choice([-1, 0, 1], size=(n_rows, k_cols)).astype(np.int8)
        x = rng.integers(-128, 128, size=(k_cols,), dtype=np.int8)
        # Include extreme activation corners (-128, +127) on first case
        if case_name == "single_row_n1":
            x[0::2] = -128
            w[:, 0::2] = -1
        s = rng.integers(0, 65536, size=(n_rows, n_groups), dtype=np.uint16)

        golden = tile_gemv(w, x, s, group_size=group_size, lanes=lanes, rows=rows, shift=shift)
        act_slices, stream_beats = _prepare_tile_stripes(
            w, x, s, rows=rows, lanes=lanes, group_size=group_size
        )

        # Pulse clear and configure layer
        dut.clear.value = 1
        dut.sink_valid.value = 0
        dut.source_ready.value = 0
        await RisingEdge(dut.sys_clk)
        dut.clear.value = 0
        dut.num_rows.value = n_rows
        dut.shift.value = shift
        await RisingEdge(dut.sys_clk)

        # Write activation slices into tile BRAM/LUTRAM
        for addr_i, slice_val in enumerate(act_slices):
            dut.act_wr_addr.value = addr_i
            dut.act_wr_data.value = slice_val
            dut.act_wr_en.value = 1
            await RisingEdge(dut.sys_clk)
        dut.act_wr_en.value = 0
        await RisingEdge(dut.sys_clk)

        in_ptr = 0
        got_rows: list[tuple[int, int, int]] = []
        held_sink = False
        cycle_guard = 0
        max_cycles = len(stream_beats) * (rows + 12) + 200

        while len(got_rows) < n_rows:
            cycle_guard += 1
            if cycle_guard > max_cycles:
                raise AssertionError(
                    f"[test_tile.py:tb_tile_matrix_suite] timeout in case '{case_name}' at "
                    f"cycle={cycle_guard}: received {len(got_rows)}/{n_rows} rows "
                    f"(in_ptr={in_ptr}/{len(stream_beats)})."
                )

            # AXI-Stream / LiteX valid stability: once sink_valid asserts, hold until sink_ready
            if in_ptr < len(stream_beats):
                if held_sink or (not use_bp) or (py_rng.random() < 0.72):
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

            dut.source_ready.value = 1 if ((not use_bp) or (py_rng.random() < 0.68)) else 0

            await RisingEdge(dut.sys_clk)

            if int(dut.sink_valid.value) == 1 and int(dut.sink_ready.value) == 1:
                in_ptr += 1
                held_sink = False

            if int(dut.source_valid.value) == 1 and int(dut.source_ready.value) == 1:
                r_idx = int(dut.source_row_idx.value)
                r_val = _to_signed(int(dut.source_result.value), acc_bits)
                r_last = int(dut.source_last.value)
                got_rows.append((r_idx, r_val, r_last))

        # Drain check: ensure no spurious extra / padded rows emerge after source.last
        dut.sink_valid.value = 0
        dut.source_ready.value = 1
        for extra_cyc in range(rows + 4):
            await RisingEdge(dut.sys_clk)
            if int(dut.source_valid.value) != 0:
                raise AssertionError(
                    f"[test_tile.py:tb_tile_matrix_suite] spurious extra row emitted after "
                    f"completion in case '{case_name}' at post-drain cycle={extra_cyc}: "
                    f"row_idx={int(dut.source_row_idx.value)}."
                )

        for exp_idx, (got_idx, got_val, got_last) in enumerate(got_rows):
            exp_last = 1 if (exp_idx == n_rows - 1) else 0
            exp_val = int(golden[exp_idx])
            if got_idx != exp_idx or got_last != exp_last or got_val != exp_val:
                raise AssertionError(
                    f"[test_tile.py:tb_tile_matrix_suite] mismatch in case '{case_name}' "
                    f"at output #{exp_idx}: expected (row_idx={exp_idx}, result={exp_val}, "
                    f"last={exp_last}), got (row_idx={got_idx}, result={got_val}, last={got_last})."
                )


@cocotb.test()
async def tb_tile_m1_2048_utilization(dut) -> None:
    """Verify M1 (N=2048, K=2048) bit-exact GEMV and >= 95% steady-state MAC utilization."""
    rows = int(os.environ["SETUN_TILE_ROWS"])
    lanes = int(os.environ["SETUN_TILE_LANES"])
    group_size = int(os.environ["SETUN_TILE_GROUP"])
    acc_bits = int(os.environ.get("SETUN_TILE_ACC_BITS", "48"))
    n_rows = int(os.environ.get("SETUN_M1_N", "2048"))
    k_cols = int(os.environ.get("SETUN_M1_K", "2048"))

    rng = np.random.default_rng(20260204)
    n_groups = k_cols // group_size
    w = rng.choice([-1, 0, 1], size=(n_rows, k_cols)).astype(np.int8)
    x = rng.integers(-128, 128, size=(k_cols,), dtype=np.int8)
    s = rng.integers(1, 4096, size=(n_rows, n_groups), dtype=np.uint16)
    shift = 10

    golden = tile_gemv(w, x, s, group_size=group_size, lanes=lanes, rows=rows, shift=shift)
    act_slices, stream_beats = _prepare_tile_stripes(
        w, x, s, rows=rows, lanes=lanes, group_size=group_size
    )

    cocotb.start_soon(Clock(dut.sys_clk, 10, units="ns").start())
    dut.sys_rst.value = 1
    dut.clear.value = 0
    dut.shift.value = shift
    dut.num_rows.value = n_rows
    dut.act_wr_en.value = 0
    dut.act_wr_addr.value = 0
    dut.act_wr_data.value = 0
    dut.sink_valid.value = 0
    dut.sink_last.value = 0
    dut.sink_trits.value = 0
    dut.sink_scales.value = 0
    dut.source_ready.value = 1

    for _ in range(3):
        await RisingEdge(dut.sys_clk)
    dut.sys_rst.value = 0
    await RisingEdge(dut.sys_clk)

    for addr_i, slice_val in enumerate(act_slices):
        dut.act_wr_addr.value = addr_i
        dut.act_wr_data.value = slice_val
        dut.act_wr_en.value = 1
        await RisingEdge(dut.sys_clk)
    dut.act_wr_en.value = 0
    await RisingEdge(dut.sys_clk)

    in_ptr = 0
    out_cnt = 0
    active_cycles = 0
    total_beats = len(stream_beats)

    while out_cnt < n_rows:
        if in_ptr < total_beats:
            t_word, s_word, last_f = stream_beats[in_ptr]
            dut.sink_trits.value = t_word
            dut.sink_scales.value = s_word
            dut.sink_last.value = last_f
            dut.sink_valid.value = 1
        else:
            dut.sink_valid.value = 0

        await RisingEdge(dut.sys_clk)
        active_cycles += 1

        if int(dut.sink_valid.value) == 1 and int(dut.sink_ready.value) == 1:
            in_ptr += 1

        if int(dut.source_valid.value) == 1 and int(dut.source_ready.value) == 1:
            r_idx = int(dut.source_row_idx.value)
            r_val = _to_signed(int(dut.source_result.value), acc_bits)
            r_last = int(dut.source_last.value)
            exp_last = 1 if (out_cnt == n_rows - 1) else 0
            exp_val = int(golden[out_cnt])
            if r_idx != out_cnt or r_val != exp_val or r_last != exp_last:
                raise AssertionError(
                    f"[test_tile.py:tb_tile_m1_2048_utilization] M1 GEMV mismatch at row "
                    f"{out_cnt}: expected (idx={out_cnt}, val={exp_val}, last={exp_last}), "
                    f"got (idx={r_idx}, val={r_val}, last={r_last})."
                )
            out_cnt += 1

    ideal_macs = n_rows * k_cols
    peak_macs = active_cycles * rows * lanes
    utilization = ideal_macs / peak_macs
    cocotb.log.info(
        f"[TernaryTile Utilization] shape=({n_rows}x{k_cols}), R={rows}, L={lanes}, "
        f"G={group_size}: {active_cycles} cycles, utilization={utilization * 100.0:.2f}% "
        f"({ideal_macs / active_cycles:.2f} / {rows * lanes} MACs/cycle)"
    )
    if utilization < 0.95:
        raise AssertionError(
            f"[test_tile.py:tb_tile_m1_2048_utilization] steady-state utilization below 95% "
            f"gate: expected >= 0.9500, got {utilization:.4f} ({active_cycles} cycles)."
        )


def _build_tile_sim_wrapper(
    rows: int,
    lanes: int,
    group_size: int,
    max_k: int,
    acc_bits: int = 48,
):
    """Instantiate TernaryTile and expose flat top-level Verilog signals for Verilator."""
    from migen import ClockDomain, Module, Signal

    from setun.accelerators.tile import TernaryTile

    class _TileTop(Module):
        def __init__(self) -> None:
            self.clock_domains.cd_sys = ClockDomain("sys")
            tile = TernaryTile(
                rows=rows,
                lanes=lanes,
                group_size=group_size,
                act_bits=8,
                acc_bits=acc_bits,
                max_k=max_k,
            )
            if hasattr(tile, "_fragment"):
                tile._fragment.clock_domains.clear()
            self.submodules.tile = tile

            self.clear = tile.clear
            self.shift = tile.shift
            self.num_rows = tile.num_rows
            self.act_wr_addr = tile.act_wr_addr
            self.act_wr_data = tile.act_wr_data
            self.act_wr_en = tile.act_wr_en

            # Explicit named top-level ports wired to LiteX stream.Endpoint sink & source
            self.sink_valid = Signal(name="sink_valid")
            self.sink_ready = Signal(name="sink_ready")
            self.sink_last = Signal(name="sink_last")
            self.sink_trits = Signal(rows * 2 * lanes, name="sink_trits")
            self.sink_scales = Signal(rows * 16, name="sink_scales")

            self.source_valid = Signal(name="source_valid")
            self.source_ready = Signal(name="source_ready")
            self.source_last = Signal(name="source_last")
            self.source_result = Signal(acc_bits, name="source_result")
            self.source_row_idx = Signal(16, name="source_row_idx")

            self.comb += [
                tile.sink.valid.eq(self.sink_valid),
                self.sink_ready.eq(tile.sink.ready),
                tile.sink.last.eq(self.sink_last),
                tile.sink.trits.eq(self.sink_trits),
                tile.sink.scales.eq(self.sink_scales),
                self.source_valid.eq(tile.source.valid),
                tile.source.ready.eq(self.source_ready),
                self.source_last.eq(tile.source.last),
                self.source_result.eq(tile.source.result),
                self.source_row_idx.eq(tile.source.row_idx),
            ]

            self.ios = {
                self.cd_sys.clk,
                self.cd_sys.rst,
                self.clear,
                self.shift,
                self.num_rows,
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
            }

    return _TileTop()


def _run_tile_verilator(
    rows: int,
    lanes: int,
    group_size: int,
    max_k: int,
    testcase: str,
    toplevel: str,
    extra: dict[str, str] | None = None,
) -> None:
    dut = _build_tile_sim_wrapper(rows=rows, lanes=lanes, group_size=group_size, max_k=max_k)
    env_vars = {
        "SETUN_TILE_ROWS": str(rows),
        "SETUN_TILE_LANES": str(lanes),
        "SETUN_TILE_GROUP": str(group_size),
        "SETUN_TILE_ACC_BITS": "48",
    }
    if extra:
        env_vars.update(extra)

    saved = {k: os.environ.get(k) for k in env_vars}
    os.environ.update(env_vars)
    try:
        run_cocotb_verilator(
            dut=dut,
            ios=dut.ios,
            test_module="test_tile",
            toplevel=toplevel,
            testcase=testcase,
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_ternary_tile_short_k_eq_l_and_ragged_matrix() -> None:
    """Verify K == L backpressure stalls, ragged N/K, bubbles, and num_rows == 0 guard."""
    _run_tile_verilator(
        rows=4,
        lanes=16,
        group_size=16,
        max_k=512,
        testcase="tb_tile_matrix_suite",
        toplevel="ternary_tile_r4_l16_g16",
    )


def test_ternary_tile_m0_shapes_and_ragged_tails() -> None:
    """Verify M0 shapes (G=64, L=16, R=4) with random bubbles, backpressure, and ragged N."""
    _run_tile_verilator(
        rows=4,
        lanes=16,
        group_size=64,
        max_k=512,
        testcase="tb_tile_matrix_suite",
        toplevel="ternary_tile_r4_l16_g64",
    )


def test_ternary_tile_m1_2048x2048_and_utilization() -> None:
    """Verify M1 (N=2048, K=2048, R=4, L=64, G=128) bit-exactness and >= 95% utilization."""
    _run_tile_verilator(
        rows=4,
        lanes=64,
        group_size=128,
        max_k=2048,
        testcase="tb_tile_m1_2048_utilization",
        toplevel="ternary_tile_r4_l64_g128_m1",
        extra={"SETUN_M1_N": "2048", "SETUN_M1_K": "2048"},
    )
