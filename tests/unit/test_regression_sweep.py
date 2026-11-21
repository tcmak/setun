"""Randomized PR and nightly regression suite via cocotb + Verilator (Task 2.8)."""

from __future__ import annotations

import os
import shutil
from unittest.mock import patch

import pytest

from setun.accelerators.group_scale import GroupAccumulateScale
from setun.accelerators.ternary_dot import TernaryDot
from tests.sim_helper import run_cocotb_verilator

NIGHTLY = os.environ.get("SETUN_NIGHTLY", "0") == "1"
DOT_CASES_PER_CFG = 100_000 if NIGHTLY else 1_000
ROW_CASES_PER_CFG = 100_000 if NIGHTLY else 500

DOT_LANES = [16, 64, 128] if NIGHTLY else [64]
ROW_CFGS = (
    [(128, 128, 64, 0), (129, 128, 64, 4), (256, 64, 16, 6)] if NIGHTLY else [(129, 128, 64, 4)]
)


def _base_seed() -> int:
    return int(os.environ.get("TEST_SEED", os.environ.get("RANDOM_SEED", "1958")))


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
