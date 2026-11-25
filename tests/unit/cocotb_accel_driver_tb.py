"""Cocotb testbench exercising AccelDriver over CocotbWishboneBackend (Task 3.3a)."""

from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from unit.test_accel_driver import run_shared_driver_verification

from setun.runtime.driver import AsyncAccelDriver, CocotbWishboneBackend


@cocotb.test()
async def tb_accel_driver_cocotb_direct(dut) -> None:
    """Run the shared AccelDriver verification script on cocotb-direct host_wb."""
    cocotb.start_soon(Clock(dut.sys_clk, 10, units="ns").start())
    dut.sys_rst.value = 1
    dut.host_wb_adr.value = 0
    dut.host_wb_dat_w.value = 0
    dut.host_wb_sel.value = 0
    dut.host_wb_cyc.value = 0
    dut.host_wb_stb.value = 0
    dut.host_wb_we.value = 0
    dut.host_wb_cti.value = 0
    dut.host_wb_bte.value = 0
    await ClockCycles(dut.sys_clk, 5)
    dut.sys_rst.value = 0
    await ClockCycles(dut.sys_clk, 2)

    backend = CocotbWishboneBackend(dut, clk_name="sys_clk", prefix="host_wb")
    drv = AsyncAccelDriver(backend)
    await run_shared_driver_verification(drv)
