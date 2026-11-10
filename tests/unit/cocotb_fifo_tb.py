"""Cocotb testbench for Migen SyncFIFO (Task 0.5b)."""

from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def tb_sync_fifo_pass(dut) -> None:
    """Verify write, full backpressure, read, and empty flags on SyncFIFO."""
    clock = Clock(dut.sys_clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Synchronous reset
    dut.sys_rst.value = 1
    dut.we.value = 0
    dut.re.value = 0
    dut.din.value = 0
    for _ in range(2):
        await RisingEdge(dut.sys_clk)
    dut.sys_rst.value = 0
    await RisingEdge(dut.sys_clk)

    # Initially empty and writable
    assert int(dut.readable.value) == 0
    assert int(dut.writable.value) == 1

    # Push 4 items (depth=4)
    payload = [0x11, 0x22, 0x33, 0x44]
    for val in payload:
        assert int(dut.writable.value) == 1
        dut.din.value = val
        dut.we.value = 1
        await RisingEdge(dut.sys_clk)
    dut.we.value = 0
    await RisingEdge(dut.sys_clk)

    # FIFO should now be full (writable == 0) and readable (readable == 1)
    assert int(dut.writable.value) == 0
    assert int(dut.readable.value) == 1

    # Pop all 4 items in FIFO order
    popped = []
    for _ in payload:
        assert int(dut.readable.value) == 1
        popped.append(int(dut.dout.value))
        dut.re.value = 1
        await RisingEdge(dut.sys_clk)
        dut.re.value = 0
        await RisingEdge(dut.sys_clk)

    assert popped == payload
    assert int(dut.readable.value) == 0
    assert int(dut.writable.value) == 1
