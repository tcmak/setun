"""Unit tests for Task 3.6: WishboneWaitJitter, rand_reset launcher, and sim_m0."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from litex.soc.interconnect import wishbone
from migen import Module

import scripts.sim_m0 as sim_m0_mod
import setun.runtime.driver as driver_mod
from scripts.sim_m0 import main as sim_m0_main
from scripts.sim_m0 import run_sim_m0
from targets.sim import SetunSimSoC, WishboneWaitJitter
from tests.sim_helper import run_migen_sim


def test_wishbone_wait_jitter_migen_and_sim_soc() -> None:
    """Verify WishboneWaitJitter injects bounded stalls while preserving R/W data."""

    class _Harness(Module):
        def __init__(self) -> None:
            inner = wishbone.Interface(
                data_width=32, address_width=32, addressing="word", bursting=False
            )
            self.submodules.sram = wishbone.SRAM(64, bus=inner)
            self.submodules.jitter = WishboneWaitJitter(inner, seed=1958)

    dut = _Harness()
    lats: list[int] = []
    got: list[int] = []
    words = [0xCAFEBABE, 0x01020304, 0xA5A55A5A, 0xFFFFFFFF]

    def _tb():
        bus = dut.jitter.bus
        for idx, val in enumerate(words):
            yield bus.adr.eq(idx)
            yield bus.dat_w.eq(val)
            yield bus.sel.eq(0xF)
            yield bus.we.eq(1)
            yield bus.cyc.eq(1)
            yield bus.stb.eq(1)
            yield
            lat = 1
            while (yield bus.ack) == 0:
                lat += 1
                yield
            lats.append(lat)
            yield bus.cyc.eq(0)
            yield bus.stb.eq(0)
            yield bus.we.eq(0)
            yield

        for idx in range(len(words)):
            yield bus.adr.eq(idx)
            yield bus.cyc.eq(1)
            yield bus.stb.eq(1)
            yield
            while (yield bus.ack) == 0:
                yield
            got.append((yield bus.dat_r))
            yield bus.cyc.eq(0)
            yield bus.stb.eq(0)
            yield

    run_migen_sim(dut, _tb)
    assert got == words and max(lats) > 1

    soc = SetunSimSoC(
        cpu_type=None,
        integrated_rom_size=0,
        integrated_main_ram_size=0x1000,
        ram_wait_jitter_seed=1958,
    )
    assert hasattr(soc, "ram_jitter") and "main_ram" in soc.bus.regions


def test_litex_sim_launcher_rand_reset_plumbing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _open_litex_sim_driver emits rand_reset_init.cpp and SETUN_SIM_SEED."""
    csr_sample = (driver_mod.REPO_ROOT / "reports" / "csr.json").read_text("utf-8")
    stub_worker = "unit_stub_rr"
    stub_root = driver_mod.REPO_ROOT / "sim_build" / ".litex_sim_cache" / stub_worker
    shutil.rmtree(stub_root, ignore_errors=True)
    vsim_env: dict[str, str] = {}

    class _StubSoc:
        clk_freq = 1_000_000

        def __init__(self, **_kw) -> None:
            pass

    class _StubBuilder:
        def __init__(self, _soc, output_dir: str, csr_json: str, **_kw) -> None:
            self.gw = Path(output_dir) / "gateware"
            self.csr = Path(csr_json)

        def build(self, **_kw) -> None:
            self.gw.mkdir(parents=True, exist_ok=True)
            (self.gw / "variables.mak").write_text("INC_DIR =\n", encoding="utf-8")
            (self.gw / "build_sim.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            self.csr.write_text(csr_sample, encoding="utf-8")

    def _fake_run(_cmd, cwd: str, **_kw) -> None:
        obj = Path(cwd) / "obj_dir"
        obj.mkdir(parents=True, exist_ok=True)
        (obj / "Vsim").write_text("bin", encoding="utf-8")

    class _StubProc:
        def __init__(self, _args, env=None, **_kw) -> None:
            if env is not None:
                vsim_env.update(env)

        def poll(self):
            return None

        def terminate(self) -> None:
            pass

        def wait(self, **_kw) -> int:
            return 0

    class _StubClient:
        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

        def read(self, _a: int, length: int | None = None):
            return 0 if length is None else [0] * length

    import litex.soc.integration.builder as b_mod
    import litex.tools.litex_client as lc_mod

    import targets.sim as t_mod

    try:
        monkeypatch.setenv("PYTEST_XDIST_WORKER", stub_worker)
        monkeypatch.setattr(t_mod, "SetunSimSoC", _StubSoc)
        monkeypatch.setattr(b_mod, "Builder", _StubBuilder)
        monkeypatch.setattr(driver_mod.subprocess, "run", _fake_run)
        monkeypatch.setattr(driver_mod.subprocess, "Popen", _StubProc)
        monkeypatch.setattr(driver_mod, "_wait_tcp_listening", lambda *_a, **_k: None)
        monkeypatch.setattr(lc_mod, "RemoteClient", lambda **_k: _StubClient())

        with driver_mod.open_backend("sim", rand_reset_seed=2026, ram_wait_jitter_seed=77) as drv:
            assert drv.read_csr("status") == 0

        assert vsim_env.get("SETUN_SIM_SEED") == "2026"
        cpps = list(stub_root.rglob("rand_reset_init.cpp"))
        assert len(cpps) == 1
        assert "--x-initial unique" in (cpps[0].parent / "variables.mak").read_text()
    finally:
        shutil.rmtree(stub_root, ignore_errors=True)


def test_sim_m0_script_artifacts_cli_and_mismatch_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise scripts/sim_m0.py artifact output, CLI main(), and divergence error."""
    model_path = driver_mod.REPO_ROOT / "models" / "m0_toy.ttape"
    out_dir = tmp_path / "artifacts"

    summary = run_sim_m0(
        model_path, tokens=2, prompt="1", seed=1958, out_dir=out_dir, sim_backend="ref"
    )
    assert summary["matched"] is True
    for name in ("ref.strc", "sim.strc", "ref_perf.log", "sim_perf.log", "diff_report.txt"):
        assert (out_dir / name).exists()
    assert "[sim-m0] PASS" in capsys.readouterr().out

    orig = sim_m0_mod._invoke_cli
    monkeypatch.setattr(
        sim_m0_mod,
        "_invoke_cli",
        lambda av: orig(["ref" if a == "sim" else a for a in av]),
    )
    assert (
        sim_m0_main(["--model", str(model_path), "--tokens", "2", "--out-dir", str(tmp_path / "c")])
        == 0
    )

    calls = 0

    def _bad(av: list[str]) -> tuple[int, str, str]:
        nonlocal calls
        calls += 1
        if av[0] == "diff":
            return 1, "[setun diff] MISMATCH", ""
        return 0, ("1 2\n" if calls == 1 else "1 9\n"), "perf\n"

    monkeypatch.setattr(sim_m0_mod, "_invoke_cli", _bad)
    with pytest.raises(RuntimeError, match="MISMATCH"):
        run_sim_m0(model_path, tokens=2, seed=99, out_dir=tmp_path / "bad")
