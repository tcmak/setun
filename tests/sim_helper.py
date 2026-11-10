"""Migen -> Verilog -> cocotb + Verilator test harness (Task 0.5a)."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path

from migen import Module, Signal
from migen.fhdl import verilog
from migen.sim import run_simulation


def run_migen_sim(
    dut: Module,
    testbench: Generator | Callable,
    vcd_name: str | None = None,
) -> None:
    """Run Migen's built-in Python simulator for tiny combinational/sequential checks."""
    gen = testbench() if callable(testbench) else testbench
    run_simulation(dut, gen, vcd_name=vcd_name)


def run_cocotb_verilator(
    dut: Module,
    ios: set[Signal],
    test_module: str,
    toplevel: str = "top",
    testcase: str | None = None,
    seed: int | None = None,
) -> Path | None:
    """Convert a Migen module to Verilog and run it under cocotb + Verilator.

    On failure (or if KEEP_WAVES=1), retains the build directory and prints
    both the random seed (TEST_SEED) and the path to the generated FST waveform.
    """
    from cocotb_test.simulator import run as cocotb_run

    resolved_seed = (
        seed
        if seed is not None
        else int(os.environ.get("TEST_SEED", os.environ.get("RANDOM_SEED", "1958")))
    )
    keep_waves = os.environ.get("KEEP_WAVES", "0") == "1"

    tests_dir = Path(__file__).resolve().parent
    repo_root = tests_dir.parent
    artifacts_root = repo_root / "sim_build"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    sim_dir = Path(tempfile.mkdtemp(prefix=f"{toplevel}_{resolved_seed}_", dir=artifacts_root))

    verilog_path = sim_dir / f"{toplevel}.v"
    v_out = verilog.convert(dut, ios=ios, name=toplevel)
    v_out.write(str(verilog_path))

    extra_env = {
        "RANDOM_SEED": str(resolved_seed),
        "TEST_SEED": str(resolved_seed),
    }

    try:
        cocotb_run(
            verilog_sources=[str(verilog_path)],
            toplevel=toplevel,
            module=test_module,
            testcase=testcase,
            python_search=[
                str(repo_root),
                str(tests_dir),
                str(tests_dir / "unit"),
            ],
            simulator="verilator",
            sim_build=str(sim_dir),
            compile_args=[
                "--trace-fst",
                "--trace-structs",
                "-Wno-fatal",
                "-Wno-INITIALDLY",
            ],
            plus_args=["--trace"],
            waves=True,
            seed=resolved_seed,
            extra_env=extra_env,
        )
    except (Exception, SystemExit) as exc:
        fst_files = list(sim_dir.rglob("*.fst"))
        fst_hint = str(fst_files[0]) if fst_files else f"(check {sim_dir})"
        msg = (
            f"\n[cocotb FAILURE] TEST_SEED={resolved_seed}\n"
            f"  Waveform FST : {fst_hint}\n"
            f"  Reproduce    : TEST_SEED={resolved_seed} pytest -k {testcase or test_module}"
        )
        raise RuntimeError(msg) from exc

    if not keep_waves:
        shutil.rmtree(sim_dir, ignore_errors=True)
        return None
    return sim_dir
