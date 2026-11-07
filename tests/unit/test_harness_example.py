"""Unit tests for the Migen + cocotb + Verilator test helper (Task 0.5a)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from migen import Module, Signal

from tests.sim_helper import run_cocotb_verilator, run_migen_sim


class _ExampleCombAdder(Module):
    """Tiny test-only Migen adder (kept in tests/ per Rule 8)."""

    def __init__(self, width: int = 8) -> None:
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.y = Signal(width + 1, name="y")
        self.ios = {self.a, self.b, self.y}
        self.comb += self.y.eq(self.a + self.b)


def test_migen_native_sim_adder() -> None:
    dut = _ExampleCombAdder(width=8)

    def _tb():
        for a, b in [(0, 0), (15, 27), (255, 255)]:
            yield dut.a.eq(a)
            yield dut.b.eq(b)
            yield
            assert (yield dut.y) == a + b

    run_migen_sim(dut, _tb())


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_cocotb_verilator_adder_passes() -> None:
    dut = _ExampleCombAdder(width=8)
    run_cocotb_verilator(
        dut=dut,
        ios=dut.ios,
        test_module="unit.cocotb_adder_tb",
        toplevel="example_comb_adder",
        testcase="tb_comb_adder_pass",
        seed=4242,
    )


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_cocotb_verilator_failure_logs_seed_and_keeps_fst() -> None:
    dut = _ExampleCombAdder(width=8)
    with pytest.raises(RuntimeError) as exc_info:
        run_cocotb_verilator(
            dut=dut,
            ios=dut.ios,
            test_module="unit.cocotb_adder_tb",
            toplevel="example_comb_adder",
            testcase="tb_comb_adder_deliberate_fail",
            seed=7777,
        )

    err_text = str(exc_info.value)
    assert "TEST_SEED=7777" in err_text
    m = re.search(r"Waveform FST\s*:\s*(\S+\.fst)", err_text)
    assert m is not None, f"Expected FST path in error message, got:\n{err_text}"
    fst_path = Path(m.group(1))
    assert fst_path.exists(), f"Expected FST file at {fst_path}"
    shutil.rmtree(fst_path.parent, ignore_errors=True)
