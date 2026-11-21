"""Randomized PR and nightly regression suite via cocotb + Verilator (Tasks 2.8 & 2.8b)."""

from __future__ import annotations

import os
import shutil
from unittest.mock import patch

import numpy as np
import pytest
from migen import ClockDomain, Module, Signal

from setun.accelerators.group_scale import GroupAccumulateScale
from setun.accelerators.requantize import Requantize
from setun.accelerators.ternary_dot import TernaryDot
from setun.accelerators.tile import TernaryTile
from setun.ref.ops import requantize_vec_int8, tile_gemv
from tests.sim_helper import run_cocotb_verilator
from tests.unit.cocotb_regression_tb import (
    _precompute_requantize_batch,
    _precompute_tile_batch,
)

NIGHTLY = os.environ.get("SETUN_NIGHTLY", "0") == "1"
DOT_CASES_PER_CFG = 100_000 if NIGHTLY else 1_000
ROW_CASES_PER_CFG = 100_000 if NIGHTLY else 500
TILE_ROWS_PER_CFG = 100_000 if NIGHTLY else 512
REQ_CASES_PER_CFG = 100_000 if NIGHTLY else 1_000

DOT_LANES = [16, 64, 128] if NIGHTLY else [64]
ROW_CFGS = (
    [(128, 128, 64, 0), (129, 128, 64, 4), (256, 64, 16, 6)] if NIGHTLY else [(129, 128, 64, 4)]
)
TILE_CFGS = (
    [(4, 16, 64, 65), (4, 64, 128, 129), (8, 64, 128, 256)] if NIGHTLY else [(4, 16, 64, 65)]
)


def _base_seed() -> int:
    return int(os.environ.get("TEST_SEED", os.environ.get("RANDOM_SEED", "1958")))


def _build_tile_reg_wrapper(rows: int, lanes: int, group_size: int, max_k: int) -> Module:
    """Expose flat top-level Verilog ports around TernaryTile for cocotb VPI."""

    class _TileRegTop(Module):
        def __init__(self) -> None:
            self.clock_domains.cd_sys = ClockDomain("sys")
            tile = TernaryTile(
                rows=rows,
                lanes=lanes,
                group_size=group_size,
                act_bits=8,
                acc_bits=48,
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

            self.sink_valid = Signal(name="sink_valid")
            self.sink_ready = Signal(name="sink_ready")
            self.sink_last = Signal(name="sink_last")
            self.sink_trits = Signal(rows * 2 * lanes, name="sink_trits")
            self.sink_scales = Signal(rows * 16, name="sink_scales")

            self.source_valid = Signal(name="source_valid")
            self.source_ready = Signal(name="source_ready")
            self.source_last = Signal(name="source_last")
            self.source_result = Signal(48, name="source_result")
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

    return _TileRegTop()


def _build_requantize_reg_wrapper() -> Module:
    """Expose flat top-level Verilog ports around Requantize for cocotb VPI."""

    class _ReqRegTop(Module):
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

    return _ReqRegTop()


def test_regression_precompute_tile_and_requantize_parity() -> None:
    """Fast in-process check that vectorized regression generators match golden ops.py."""
    rng = np.random.default_rng(19580208)
    w, x, s, act_slices, beats, expected = _precompute_tile_batch(
        rng, n_rows=11, k=65, rows=4, lanes=16, group_size=64, shift=7
    )
    ref = tile_gemv(w, x, s, group_size=64, lanes=16, rows=4, shift=7)
    np.testing.assert_array_equal(expected, ref)
    assert len(act_slices) == 8  # ceil(65/64)*64 / 16 = 8 slices
    assert len(beats) == 3 * 8  # ceil(11/4)=3 stripes * 8 beats/stripe

    for scale_u16, shift, sym in [(65535, 62, False), (413, 13, True), (1, 0, False)]:
        vals, u48_words, exp_q, exp_sat = _precompute_requantize_batch(
            rng, n_items=64, scale_u16=scale_u16, shift=shift, symmetric=sym
        )
        ref_q, ref_sat = requantize_vec_int8(vals, scale_u16, shift, symmetric=sym)
        np.testing.assert_array_equal(exp_q, ref_q)
        np.testing.assert_array_equal(exp_sat, ref_sat)
        assert len(u48_words) == 64

    # Exercise wrapper builders in-process as well
    tile_top = _build_tile_reg_wrapper(rows=4, lanes=16, group_size=64, max_k=128)
    req_top = _build_requantize_reg_wrapper()
    assert len(tile_top.ios) == 18
    assert len(req_top.ios) == 17


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
@pytest.mark.parametrize("lanes", DOT_LANES)
def test_regression_sweep_ternary_dot(lanes: int) -> None:
    """Randomized parity sweep on TernaryDot (1,000 PR / 100,000 nightly per config)."""
    seed = _base_seed()
    dut = TernaryDot(lanes=lanes, act_bits=8, pipeline_depth=0)
    env_overrides = {
        "SETUN_LANES": str(lanes),
        "SETUN_CASES": str(DOT_CASES_PER_CFG),
    }
    with patch.dict(os.environ, env_overrides):
        run_cocotb_verilator(
            dut=dut,
            ios=dut.ios,
            test_module="unit.cocotb_regression_tb",
            toplevel=f"ternary_dot_reg_{lanes}",
            testcase="tb_regression_ternary_dot",
            seed=seed,
        )


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
@pytest.mark.parametrize("k,group_size,lanes,shift", ROW_CFGS)
def test_regression_sweep_group_accumulate_scale(
    k: int,
    group_size: int,
    lanes: int,
    shift: int,
) -> None:
    """Randomized parity sweep on GroupAccumulateScale (PR vs nightly >= 10^5)."""
    seed = _base_seed()
    dut = GroupAccumulateScale(
        group_size=group_size,
        lanes=lanes,
        act_bits=8,
        shift=shift,
    )
    env_overrides = {
        "SETUN_K": str(k),
        "SETUN_G": str(group_size),
        "SETUN_LANES": str(lanes),
        "SETUN_SHIFT": str(shift),
        "SETUN_CASES": str(ROW_CASES_PER_CFG),
    }
    with patch.dict(os.environ, env_overrides):
        run_cocotb_verilator(
            dut=dut,
            ios=dut.ios,
            test_module="unit.cocotb_regression_tb",
            toplevel=f"group_scale_reg_{k}_{group_size}_{lanes}",
            testcase="tb_regression_group_scale",
            seed=seed,
        )


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
@pytest.mark.parametrize("rows,lanes,group_size,k", TILE_CFGS)
def test_regression_sweep_ternary_tile(
    rows: int,
    lanes: int,
    group_size: int,
    k: int,
) -> None:
    """Randomized parity sweep on TernaryTile (512 PR / 100,000 nightly rows per config)."""
    seed = _base_seed()
    max_k = ((k + group_size - 1) // group_size) * group_size
    dut = _build_tile_reg_wrapper(
        rows=rows, lanes=lanes, group_size=group_size, max_k=max(max_k, group_size * 2)
    )
    env_overrides = {
        "SETUN_ROWS": str(rows),
        "SETUN_LANES": str(lanes),
        "SETUN_G": str(group_size),
        "SETUN_K": str(k),
        "SETUN_CASES": str(TILE_ROWS_PER_CFG),
    }
    with patch.dict(os.environ, env_overrides):
        run_cocotb_verilator(
            dut=dut,
            ios=dut.ios,
            test_module="unit.cocotb_regression_tb",
            toplevel=f"ternary_tile_reg_r{rows}_l{lanes}_g{group_size}_k{k}",
            testcase="tb_regression_ternary_tile",
            seed=seed,
        )


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_regression_sweep_requantize() -> None:
    """Randomized parity sweep on Requantize (1,000 PR / 100,000 nightly cases)."""
    seed = _base_seed()
    dut = _build_requantize_reg_wrapper()
    env_overrides = {
        "SETUN_CASES": str(REQ_CASES_PER_CFG),
    }
    with patch.dict(os.environ, env_overrides):
        run_cocotb_verilator(
            dut=dut,
            ios=dut.ios,
            test_module="unit.cocotb_regression_tb",
            toplevel="requantize_reg_top",
            testcase="tb_regression_requantize",
            seed=seed,
        )
