"""Migen -> Verilog -> cocotb + Verilator test harness with build cache (Tasks 0.5a & 0.6)."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path

from migen import Module, Signal
from migen.fhdl import verilog
from migen.sim import run_simulation

_BASE_VERILATOR_ARGS = [
    "-Wno-fatal",
    "-Wno-INITIALDLY",
    "-Wno-WIDTHEXPAND",
    "-Wno-WIDTHTRUNC",
]
_TRACE_VERILATOR_ARGS = [
    "--trace-fst",
    "--trace-structs",
]


def run_migen_sim(
    dut: Module,
    testbench: Generator | Callable,
    vcd_name: str | None = None,
) -> None:
    """Run Migen's built-in Python simulator for tiny combinational/sequential checks."""
    gen = testbench() if callable(testbench) else testbench
    run_simulation(dut, gen, vcd_name=vcd_name)


def _write_verilog_bundle(sim_dir: Path, toplevel: str, v_out) -> Path:
    """Write Verilog top and any $readmemh .init files into sim_dir if not already cached."""
    verilog_path = sim_dir / f"{toplevel}.v"
    if not verilog_path.exists():
        verilog_path.write_text(v_out.main_source, encoding="utf-8")
    for data_name, data_content in v_out.data_files.items():
        data_path = sim_dir / data_name
        if not data_path.exists():
            data_path.write_text(data_content, encoding="utf-8")
    return verilog_path


def run_cocotb_verilator(
    dut: Module,
    ios: set[Signal],
    test_module: str,
    toplevel: str = "top",
    testcase: str | None = None,
    seed: int | None = None,
) -> Path | None:
    """Convert a Migen module to Verilog and run it under cocotb + Verilator.

    Fast path: compiles without FST trace instrumentation in a worker-isolated content-hash
    cache directory (`sim_build/.verilator_cache/<worker>/<toplevel>_<sha256>/`).
    On failure (or if KEEP_WAVES=1): produces an FST waveform and prints TEST_SEED.
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

    v_out = verilog.convert(dut, ios=ios, name=toplevel)
    py_search = [str(repo_root), str(tests_dir), str(tests_dir / "unit")]
    extra_env = {
        "RANDOM_SEED": str(resolved_seed),
        "TEST_SEED": str(resolved_seed),
    }

    def _invoke(sim_dir: Path, *, traced: bool) -> None:
        v_path = _write_verilog_bundle(sim_dir, toplevel, v_out)
        c_args = list(_BASE_VERILATOR_ARGS) + (_TRACE_VERILATOR_ARGS if traced else [])
        p_args = ["--trace"] if traced else []
        cocotb_run(
            verilog_sources=[str(v_path)],
            toplevel=toplevel,
            module=test_module,
            testcase=testcase,
            python_search=py_search,
            simulator="verilator",
            sim_build=str(sim_dir),
            compile_args=c_args,
            make_args=["-j1"],
            plus_args=p_args,
            waves=traced,
            seed=resolved_seed,
            extra_env=extra_env,
        )

    if keep_waves:
        wave_dir = Path(tempfile.mkdtemp(prefix=f"{toplevel}_{resolved_seed}_", dir=artifacts_root))
        try:
            _invoke(wave_dir, traced=True)
        except (Exception, SystemExit) as exc:
            fst_files = list(wave_dir.rglob("*.fst"))
            fst_hint = str(fst_files[0]) if fst_files else f"(check {wave_dir})"
            raise RuntimeError(
                f"\n[cocotb FAILURE] TEST_SEED={resolved_seed}\n"
                f"  Waveform FST : {fst_hint}\n"
                f"  Reproduce    : TEST_SEED={resolved_seed} pytest -k {testcase or test_module}"
            ) from exc
        return wave_dir

    # Fast cached untraced execution (worker-scoped to prevent xdist races)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    digest = hashlib.sha256()
    digest.update(toplevel.encode("utf-8") + b"\0" + v_out.main_source.encode("utf-8"))
    for d_name, d_body in sorted(v_out.data_files.items()):
        digest.update(b"\0" + d_name.encode("utf-8") + b"\0" + d_body.encode("utf-8"))
    cache_dir = (
        artifacts_root / ".verilator_cache" / worker / f"{toplevel}_{digest.hexdigest()[:16]}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        _invoke(cache_dir, traced=False)
    except (Exception, SystemExit) as exc:
        fail_dir = Path(
            tempfile.mkdtemp(prefix=f"{toplevel}_{resolved_seed}_fail_", dir=artifacts_root)
        )
        try:
            _invoke(fail_dir, traced=True)
        except (Exception, SystemExit):
            pass
        fst_files = list(fail_dir.rglob("*.fst"))
        fst_hint = str(fst_files[0]) if fst_files else f"(check {fail_dir})"
        msg = (
            f"\n[cocotb FAILURE] TEST_SEED={resolved_seed}\n"
            f"  Waveform FST : {fst_hint}\n"
            f"  Reproduce    : TEST_SEED={resolved_seed} pytest -k {testcase or test_module}"
        )
        raise RuntimeError(msg) from exc

    return None
