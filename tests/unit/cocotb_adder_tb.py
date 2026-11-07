"""cocotb coroutines for the Task 0.5a example adder test."""

from __future__ import annotations

import os
import random

import cocotb
from cocotb.triggers import Timer


@cocotb.test()
async def tb_comb_adder_pass(dut) -> None:
    seed = int(os.environ.get("TEST_SEED", "1958"))
    rng = random.Random(seed)
    for _ in range(50):
        a = rng.randint(0, 255)
        b = rng.randint(0, 255)
        dut.a.value = a
        dut.b.value = b
        await Timer(1, units="ns")
        got = int(dut.y.value)
        assert got == a + b, f"seed={seed}: {a} + {b} != {got}"


@cocotb.test()
async def tb_comb_adder_deliberate_fail(dut) -> None:
    dut.a.value = 10
    dut.b.value = 20
    await Timer(1, units="ns")
    # Deliberate failure to verify seed logging and FST retention
    assert int(dut.y.value) == 999, "Deliberate failure for harness check"
