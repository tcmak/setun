"""RED-first unit & dual-sim-backend verification tests for AccelDriver (Task 3.3a)."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from setun.ref.format import pack_ternary_tensor
from setun.ref.m0 import ternary_matvec_integer
from setun.ref.ops import requantize_int8
from setun.runtime.driver import (
    VALID_BACKENDS,
    AccelDriver,
    AccelHardwareError,
    AsyncAccelDriver,
    PerfMetrics,
    RefBusBackend,
    add_backend_argument,
    open_backend,
)
from targets.sim import SetunSimSoC
from tests.sim_helper import run_cocotb_verilator


async def run_shared_driver_verification(drv: AsyncAccelDriver) -> None:
    """Canonical verification workload that MUST pass identically on ALL backends (Task 3.3a).

    Exercises:
      1. read_csr capability discovery (tile_info, buf_caps) & write_csr/read_csr round-trip
      2. dma_write + launch + wait + dma_read in wide int32 mode vs. ternary_matvec_integer
      3. dma_write + launch + wait + dma_read in packed int8 mode (ragged N=5, sentinel check)
      4. perf() telemetry (cycles, bus_beats, bus_utilization, mb_per_sec)
      5. Hardware legality error detection (ERR_ADDR_ALIGN) and sticky recovery without reset
    """
    tile_info = await drv.read_csr("tile_info")
    hw_lanes = (tile_info >> 8) & 0xFF
    hw_group_size = (tile_info >> 16) & 0xFFFF
    assert hw_lanes > 0 and hw_group_size >= hw_lanes

    # 1. Scratchpad CSR round-trip
    await drv.write_csr("ctrl_scratch", 0xCAFE_1958)
    assert (await drv.read_csr("ctrl_scratch")) == 0xCAFE_1958

    # Build deterministic ternary projection matching discovered hw_group_size
    rng = np.random.default_rng(3301)
    k_cols = hw_group_size
    shift = 6

    # 2. Wide int32 DMA GEMV (N = 8, req_enable = False)
    n_wide = 8
    w_wide = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n_wide, k_cols))
    s_wide = rng.integers(16, 200, size=(n_wide, 1), dtype=np.uint16)
    proj_wide = pack_ternary_tensor(w_wide, s_wide, group_size=hw_group_size, shift=shift)
    x_i8 = rng.integers(-100, 100, size=(k_cols,), dtype=np.int8)
    expected_wide = ternary_matvec_integer(proj_wide, x_i8, lanes=hw_lanes).astype(np.int32)

    base = drv.main_ram_base
    w_addr = base + 0x0000
    s_addr = base + 0x1000
    a_addr = base + 0x2000
    o_addr = base + 0x3000

    await drv.dma_write(w_addr, proj_wide.packed_trits)
    await drv.dma_write(s_addr, proj_wide.scales.astype("<u2").tobytes())
    await drv.dma_write(a_addr, x_i8.tobytes())

    await drv.launch(
        num_rows=n_wide,
        k_cols=k_cols,
        group_size=hw_group_size,
        shift=shift,
        weight_addr=w_addr,
        scale_addr=s_addr,
        act_addr=a_addr,
        out_addr=o_addr,
        act_scale=0.125,
        req_enable=False,
    )
    rows_done = await drv.wait()
    assert rows_done == n_wide

    raw_wide = await drv.dma_read(o_addr, n_wide * 4)
    got_wide = np.frombuffer(raw_wide, dtype="<i4")
    np.testing.assert_array_equal(got_wide, expected_wide)

    perf_wide: PerfMetrics = await drv.perf(clk_freq_hz=100e6)
    assert perf_wide.cycles > 0
    assert perf_wide.bus_beats > 0
    assert 0.0 < perf_wide.bus_utilization <= 1.0
    assert perf_wide.mb_per_sec > 0.0

    # 3. Packed int8 DMA GEMV with ragged N = 5 and sentinel preservation
    n_rag = 5
    w_rag = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n_rag, k_cols))
    s_rag = rng.integers(32, 300, size=(n_rag, 1), dtype=np.uint16)
    proj_rag = pack_ternary_tensor(w_rag, s_rag, group_size=hw_group_size, shift=shift)
    wide_rag = ternary_matvec_integer(proj_rag, x_i8, lanes=hw_lanes)
    req_scale = 96
    req_shift = 8
    req_pairs = [
        requantize_int8(int(v), scale_u16=req_scale, shift=req_shift, symmetric=True)
        for v in wide_rag
    ]
    expected_i8 = np.array([p[0] for p in req_pairs], dtype=np.int8)
    exp_sat = np.array([p[1] for p in req_pairs], dtype=bool)

    sentinel = bytes([0xA5, 0x5A, 0xC3, 0x3C, 0xF0, 0x0F, 0x99, 0x66])
    o_rag_addr = base + 0x3400
    await drv.dma_write(o_rag_addr, sentinel)
    await drv.dma_write(w_addr, proj_rag.packed_trits)
    await drv.dma_write(s_addr, proj_rag.scales.astype("<u2").tobytes())

    await drv.launch(
        num_rows=n_rag,
        k_cols=k_cols,
        group_size=hw_group_size,
        shift=shift,
        weight_addr=w_addr,
        scale_addr=s_addr,
        act_addr=a_addr,
        out_addr=o_rag_addr,
        req_enable=True,
        req_sym=True,
        req_scale_u16=req_scale,
        req_shift=req_shift,
    )
    assert (await drv.wait()) == n_rag

    full_8 = await drv.dma_read(o_rag_addr, 8)
    got_i8 = np.frombuffer(full_8[:n_rag], dtype=np.int8)
    np.testing.assert_array_equal(got_i8, expected_i8)
    assert full_8[n_rag:] == sentinel[n_rag:], (
        f"Ragged tail sentinel clobbered: expected {sentinel[n_rag:]!r}, got {full_8[n_rag:]!r}"
    )
    perf_rag = await drv.perf()
    assert perf_rag.sat_rows == int(np.sum(exp_sat))

    # 4. Hardware legality check (unaligned weight_addr -> ERR_ADDR_ALIGN) + sticky recovery
    await drv.write_csr("weight_addr", w_addr + 1)
    await drv.write_csr("scale_addr", s_addr)
    await drv.write_csr("act_addr", a_addr)
    await drv.write_csr("out_addr", o_addr)
    await drv.write_csr("cfg", (n_wide << 8) | shift)
    await drv.write_csr("dim_k", (hw_group_size << 16) | k_cols)
    await drv.write_csr("ctrl", 0x5)  # enable | start

    with pytest.raises(AccelHardwareError, match="ERR_ADDR_ALIGN"):
        await drv.wait()

    # Recover cleanly without SoC reset by launching valid wide GEMV again
    await drv.dma_write(w_addr, proj_wide.packed_trits)
    await drv.dma_write(s_addr, proj_wide.scales.astype("<u2").tobytes())
    await drv.launch(
        num_rows=n_wide,
        k_cols=k_cols,
        group_size=hw_group_size,
        shift=shift,
        weight_addr=w_addr,
        scale_addr=s_addr,
        act_addr=a_addr,
        out_addr=o_addr,
    )
    assert (await drv.wait()) == n_wide
    recovered = np.frombuffer(await drv.dma_read(o_addr, n_wide * 4), dtype="<i4")
    np.testing.assert_array_equal(recovered, expected_wide)


def test_accel_driver_ref_backend_and_guards() -> None:
    """Fast (<1s) test of AccelDriver over RefBusBackend, CLI flags, and Goose-style guards."""
    # 1. Run canonical async suite over RefBusBackend
    ref_bus = RefBusBackend(rows=4, lanes=16, group_size=64, beat_bits=64)
    asyncio.run(run_shared_driver_verification(AsyncAccelDriver(ref_bus)))

    # 2. Synchronous open_backend("ref") facade & run_gemv helper
    with open_backend("ref", rows=4, lanes=16, group_size=64) as drv:
        assert isinstance(drv, AccelDriver)
        drv.write_csr("ctrl_scratch", 0x12345678)
        assert drv.read_csr("ctrl_scratch") == 0x12345678
        assert struct.unpack("<f", struct.pack("<I", 0x3E000000))[0] == 0.125

        # Validation guards (Goose-style diagnostics)
        with pytest.raises(KeyError, match="unknown CSR"):
            drv.read_csr("nonexistent_reg")
        with pytest.raises(PermissionError, match="read-only"):
            drv.write_csr("status", 0)
        with pytest.raises(ValueError, match="32-bit unsigned"):
            drv.write_csr("ctrl_scratch", -1)
        with pytest.raises(ValueError, match="4-byte aligned"):
            drv.dma_write(drv.main_ram_base + 2, b"abcd")
        with pytest.raises(ValueError, match="4-byte aligned"):
            drv.dma_read(drv.main_ram_base + 1, 4)
        with pytest.raises(ValueError, match="out of main_ram bounds"):
            drv.dma_write(0x1000, b"abcd")
        with pytest.raises(ValueError, match="out of main_ram bounds"):
            drv.dma_read(drv.main_ram_base + drv.main_ram_size, 4)

    # 3. CLI --backend flag & board stub (Task 3.3b)
    parser = argparse.ArgumentParser()
    add_backend_argument(parser)
    for b_name in VALID_BACKENDS:
        parsed = parser.parse_args(["--backend", b_name])
        assert parsed.backend == b_name

    with pytest.raises(NotImplementedError, match="3.3b"):
        with open_backend("board"):
            pass

    with pytest.raises(ValueError, match="Unsupported backend"):
        with open_backend("invalid_backend"):
            pass


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_accel_driver_cocotb_direct_backend() -> None:
    """Run the shared AccelDriver verification script on cocotb-direct SetunSimSoC."""
    soc = SetunSimSoC(
        with_accel_stub=True,
        with_ternary_accel=True,
        accel_rows=4,
        accel_lanes=16,
        accel_group_size=64,
        accel_beat_bits=64,
        accel_max_k=256,
        accel_max_rows=64,
        expose_host_wb=True,
        use_external_crg=True,
        cpu_type=None,
        integrated_rom_size=0,
        integrated_main_ram_size=0x10000,
    )
    ios = {
        soc.crg.cd_sys.clk,
        soc.crg.cd_sys.rst,
        soc.host_wb.adr,
        soc.host_wb.dat_w,
        soc.host_wb.dat_r,
        soc.host_wb.sel,
        soc.host_wb.cyc,
        soc.host_wb.stb,
        soc.host_wb.we,
        soc.host_wb.cti,
        soc.host_wb.bte,
        soc.host_wb.ack,
    }
    soc.crg.cd_sys.clk.name_override = "sys_clk"
    soc.crg.cd_sys.rst.name_override = "sys_rst"
    for sig_name in ("adr", "dat_w", "dat_r", "sel", "cyc", "stb", "we", "cti", "bte", "ack"):
        getattr(soc.host_wb, sig_name).name_override = f"host_wb_{sig_name}"

    run_cocotb_verilator(
        dut=soc,
        ios=ios,
        test_module="unit.cocotb_accel_driver_tb",
        toplevel="setun_sim_soc_driver_tb",
        testcase="tb_accel_driver_cocotb_direct",
        seed=3301,
    )


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_accel_driver_etherbone_sim_backend() -> None:
    """Run the EXACT SAME verification script over litex_server + RemoteClient (--backend sim)."""
    with open_backend(
        "sim",
        rows=4,
        lanes=16,
        group_size=64,
        beat_bits=64,
        max_k=256,
        max_rows=64,
    ) as drv:
        asyncio.run(run_shared_driver_verification(drv.async_driver))


def test_cocotb_and_etherbone_backends_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise CocotbWishboneBackend and EtherboneBackend deterministically in-process."""
    import cocotb.triggers

    from setun.runtime.driver import CocotbWishboneBackend, EtherboneBackend

    class _Sig:
        def __init__(self, val: int = 0) -> None:
            self.value = val

    class _FakeDut:
        def __init__(self) -> None:
            self.sys_clk = _Sig(0)
            for suffix in ("adr", "dat_w", "dat_r", "sel", "cyc", "stb", "we", "cti", "bte", "ack"):
                setattr(self, f"host_wb_{suffix}", _Sig(0))

    dut = _FakeDut()
    mem: dict[int, int] = {}

    class _DoneAwaitable:
        def __init__(self, fn=None) -> None:
            self._fn = fn

        def __await__(self):
            if self._fn is not None:
                self._fn()
            if False:
                yield None
            return None

    def _on_rising_edge() -> None:
        if dut.host_wb_cyc.value == 1 and dut.host_wb_stb.value == 1:
            if dut.host_wb_ack.value == 0:
                # 1st edge is a wait state (ack=0 -> 1); 2nd ReadOnly observes ack=1
                dut.host_wb_ack.value = 1
                waddr = int(dut.host_wb_adr.value)
                if int(dut.host_wb_we.value) == 1:
                    mem[waddr] = int(dut.host_wb_dat_w.value) & 0xFFFF_FFFF
                else:
                    dut.host_wb_dat_r.value = mem.get(waddr, 0)
            else:
                dut.host_wb_ack.value = 0
        else:
            dut.host_wb_ack.value = 0

    monkeypatch.setattr(cocotb.triggers, "ReadOnly", lambda: _DoneAwaitable())
    monkeypatch.setattr(cocotb.triggers, "RisingEdge", lambda _clk: _DoneAwaitable(_on_rising_edge))

    wb = CocotbWishboneBackend(dut)

    async def _run_wb() -> list[int]:
        await wb.write_words(0x4000_0000, [0x1122_3344, 0xDEAD_BEEF])
        return await wb.read_words(0x4000_0000, 2)

    assert asyncio.run(_run_wb()) == [0x1122_3344, 0xDEAD_BEEF]

    # EtherboneBackend: >64-word chunking, scalar vs list read return, and poll_sleep_s
    eb_mem: dict[int, int] = {}

    class _FakeRemoteClient:
        def write(self, addr: int, datas) -> None:
            if isinstance(datas, int):
                eb_mem[addr] = datas & 0xFFFF_FFFF
            else:
                for idx, val in enumerate(datas):
                    eb_mem[addr + 4 * idx] = int(val) & 0xFFFF_FFFF

        def read(self, addr: int, length: int | None = None):
            if length is None or length == 1:
                return eb_mem.get(addr, 0)
            return [eb_mem.get(addr + 4 * idx, 0) for idx in range(length)]

    fake_client = _FakeRemoteClient()
    eb_sleep = EtherboneBackend(fake_client, poll_sleep_s=0.0001)
    eb_fast = EtherboneBackend(fake_client, poll_sleep_s=0.0)

    async def _run_eb() -> None:
        await eb_sleep.write32(0x1800, 0xCAFE_BABE)
        assert (await eb_sleep.read32(0x1800)) == 0xCAFE_BABE
        assert (await eb_fast.read32(0x1800)) == 0xCAFE_BABE
        words_65 = [(0x1000 + i) for i in range(65)]
        await eb_fast.write_words(0x4000_0000, words_65)
        assert (await eb_fast.read_words(0x4000_0000, 65)) == words_65

    asyncio.run(_run_eb())


def test_sync_driver_guards_and_ref_edge_cases() -> None:
    """Cover synchronous AccelDriver facade, error guards, timeouts, and RefBusBackend branches."""
    import argparse

    from setun.runtime.driver import (
        AccelHardwareError,
        AsyncAccelDriver,
        _run_immediate,
        add_backend_argument,
        open_backend,
    )

    parser = argparse.ArgumentParser()
    add_backend_argument(parser)
    assert parser.parse_args([]).backend == "ref"
    assert parser.parse_args(["--backend", "sim"]).backend == "sim"

    with pytest.raises(ValueError, match="Unsupported backend"):
        with open_backend("invalid"):
            pass

    with pytest.raises(NotImplementedError, match="Task 3.3b"):
        with open_backend("board"):
            pass

    with open_backend("ref", rows=4, lanes=16, group_size=64, beat_bits=64) as drv:
        base = drv.main_ram_base
        assert base == 0x4000_0000
        assert drv.main_ram_size > 0

        # Zero-cycle perf right after clear
        drv.write_csr("ctrl", 0x2)
        p0 = drv.perf(clk_freq_hz=50_000_000.0)
        assert p0.cycles == 0 and p0.bus_utilization == 0.0 and p0.mb_per_sec == 0.0

        # CSR validation & guard checks
        drv.write_csr("ctrl_scratch", 0xABCD_0123)
        assert drv.read_csr("ctrl_scratch") == 0xABCD_0123
        with pytest.raises(KeyError, match="unknown CSR"):
            drv.read_csr("does_not_exist")
        with pytest.raises(PermissionError, match="read-only"):
            drv.write_csr("tile_info", 0)
        with pytest.raises(ValueError, match="32-bit unsigned"):
            drv.write_csr("ctrl_scratch", -1)
        with pytest.raises(ValueError, match="32-bit unsigned"):
            drv.write_csr("ctrl_scratch", 0x1_0000_0000)

        # DMA alignment, bounds, empty, and ragged 3-byte round-trip
        with pytest.raises(ValueError, match="4-byte aligned"):
            drv.dma_write(base + 2, b"1234")
        with pytest.raises(ValueError, match="out of main_ram bounds"):
            drv.dma_write(base - 4, b"1234")
        with pytest.raises(ValueError, match="out of main_ram bounds"):
            drv.dma_read(base, -4)
        with pytest.raises(ValueError, match="out of main_ram bounds"):
            drv.dma_read(base + drv.main_ram_size, 4)

        drv.dma_write(base, b"")
        assert drv.dma_read(base, 0) == b""
        drv.dma_write(base, b"\xaa\xbb\xcc")
        assert drv.dma_read(base, 3) == b"\xaa\xbb\xcc"

        # Raw RefBusBackend read32/write32 on RAM & unmapped CSR
        ref_bus: RefBusBackend = drv.async_driver.backend  # type: ignore[assignment]
        asyncio.run(ref_bus.write32(base + 0x100, 0x0403_0201))
        assert asyncio.run(ref_bus.read32(base + 0x100)) == 0x0403_0201
        assert asyncio.run(ref_bus.read32(0x0000_FFFC)) == 0

        # Trigger all 4 legality error flags simultaneously (K=0, N=0, G!=64, unaligned ptr)
        drv.launch(
            num_rows=0,
            k_cols=0,
            group_size=32,
            shift=0,
            weight_addr=base + 1,
            scale_addr=0,
        )
        with pytest.raises(AccelHardwareError, match="ERR_K_INVALID"):
            drv.wait()
        drv.write_csr("ctrl", 0x2)

        # Launch with out_addr=0 in both wide mode and saturating req_enable mode
        w = np.ones((4, 64), dtype=np.int8)
        s = np.full((4, 1), 200, dtype=np.uint16)
        proj = pack_ternary_tensor(w, s, group_size=64, shift=0)
        x = np.full(64, 100, dtype=np.int8)
        drv.dma_write(base + 0x0000, proj.packed_trits)
        drv.dma_write(base + 0x1000, proj.scales.astype("<u2").tobytes())
        drv.dma_write(base + 0x2000, x.tobytes())

        drv.launch(
            num_rows=4,
            k_cols=64,
            group_size=64,
            shift=0,
            weight_addr=base + 0x0000,
            scale_addr=base + 0x1000,
            act_addr=base + 0x2000,
            out_addr=0,
            req_enable=False,
        )
        assert drv.wait() == 4

        drv.launch(
            num_rows=4,
            k_cols=64,
            group_size=64,
            shift=0,
            weight_addr=base + 0x0000,
            scale_addr=base + 0x1000,
            act_addr=base + 0x2000,
            out_addr=0,
            req_enable=True,
            req_sym=True,
            req_scale_u16=1,
            req_shift=0,
        )
        assert drv.wait() == 4
        p_sat = drv.perf()
        assert p_sat.req_saturated is True and p_sat.sat_rows == 4

    # wait() TimeoutError and UNKNOWN_ERR (err_code == 0 with error bit set)
    class _StuckBackend:
        def __init__(self, status_val: int) -> None:
            self.status_val = status_val

        async def read32(self, _addr: int) -> int:
            return self.status_val

        async def write32(self, _addr: int, _val: int) -> None:
            return None

        async def read_words(self, _addr: int, count: int) -> list[int]:
            return [0] * count

        async def write_words(self, _addr: int, _words) -> None:
            return None

    stuck_drv = AsyncAccelDriver(_StuckBackend((3 << 16) | 0x1))
    with pytest.raises(TimeoutError, match="rows_completed=3"):
        asyncio.run(stuck_drv.wait(max_polls=2))

    unk_err_drv = AsyncAccelDriver(_StuckBackend(1 << 4))
    with pytest.raises(AccelHardwareError, match="UNKNOWN_ERR"):
        asyncio.run(unk_err_drv.wait(max_polls=2))

    # _run_immediate guard when handed a genuinely yielding coroutine
    async def _yielding_coro() -> None:
        await asyncio.sleep(0)

    coro = _yielding_coro()
    try:
        with pytest.raises(RuntimeError, match="yielded unexpectedly"):
            _run_immediate(coro)
    finally:
        coro.close()


def test_sim_launcher_and_process_helpers_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise TCP port listener check, process termination, and cold/warm sim launcher."""
    import io
    import json
    import socket
    import subprocess

    import litex.soc.integration.builder as litex_builder
    import litex.tools.litex_client as litex_client_mod

    import setun.runtime.driver as drv_mod
    import targets.sim as sim_mod
    from setun.soc import DEFAULT_CSR_JSON

    # Instantiate SetunSimSoC(with_uartbone=True, ram_init=...) to cover targets/sim.py fast path
    ram_bin = tmp_path / "preload.bin"
    ram_bin.write_bytes(b"\x01\x02\x03\x04\x05")
    soc_ub = sim_mod.SetunSimSoC(
        with_accel_stub=True,
        with_ternary_accel=False,
        with_uartbone=True,
        ram_init=str(ram_bin),
        cpu_type=None,
        integrated_rom_size=0,
        integrated_main_ram_size=0x1000,
    )
    assert hasattr(soc_ub, "uartbone")

    ports = drv_mod._alloc_free_tcp_ports(2)
    assert len(ports) == 2 and ports[0] != ports[1]

    # _wait_tcp_listening: active listening socket, early exit, and timeout
    class _AliveProc:
        returncode = None
        stdout = io.StringIO("")

        def poll(self):
            return None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        bound_port = int(srv.getsockname()[1])
        drv_mod._wait_tcp_listening(bound_port, _AliveProc(), timeout_s=1.0)  # type: ignore[arg-type]

    class _DeadProc:
        returncode = 7
        stdout = io.StringIO("sim crashed")

        def poll(self):
            return 7

    with pytest.raises(RuntimeError, match="rc=7"):
        drv_mod._wait_tcp_listening(ports[0], _DeadProc(), timeout_s=0.5)  # type: ignore[arg-type]

    unbound_port = drv_mod._alloc_free_tcp_ports(1)[0]
    with pytest.raises(TimeoutError, match="timed out"):
        drv_mod._wait_tcp_listening(unbound_port, _AliveProc(), timeout_s=0.03)  # type: ignore[arg-type]

    # _terminate_proc: None, graceful terminate, and TimeoutExpired -> kill fallback
    drv_mod._terminate_proc(None)

    class _GracefulProc:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            return 0

    gp = _GracefulProc()
    drv_mod._terminate_proc(gp)  # type: ignore[arg-type]
    assert gp.terminated is True

    class _StubbornProc:
        def __init__(self) -> None:
            self.killed = False

        def poll(self):
            return None

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            if not self.killed:
                raise subprocess.TimeoutExpired(cmd="vsim", timeout=timeout or 3.0)
            return -9

    sp = _StubbornProc()
    drv_mod._terminate_proc(sp)  # type: ignore[arg-type]
    assert sp.killed is True

    # Mock cold + warm _open_litex_sim_driver via open_backend("sim", ...)
    fake_root = tmp_path / "repo"
    (fake_root / "targets").mkdir(parents=True)
    (fake_root / "setun" / "soc").mkdir(parents=True)
    (fake_root / "setun" / "accelerators").mkdir(parents=True)
    (fake_root / "targets" / "sim.py").write_text("# sim\n", encoding="utf-8")
    (fake_root / "setun" / "soc" / "ternary_accel.py").write_text("# soc\n", encoding="utf-8")
    (fake_root / "setun" / "accelerators" / "dot.py").write_text("# dot\n", encoding="utf-8")

    csr_payload = DEFAULT_CSR_JSON.read_text(encoding="utf-8")

    class _StubSoC:
        clk_freq = 1_000_000

        def __init__(self, **_kwargs) -> None:
            pass

    class _StubBuilder:
        def __init__(self, _soc, output_dir: str, csr_json: str, **_kwargs) -> None:
            self.output_dir = Path(output_dir)
            self.csr_json = Path(csr_json)

        def build(self, sim_config, run: bool = False) -> None:
            assert run is False
            gw = self.output_dir / "gateware"
            gw.mkdir(parents=True, exist_ok=True)
            self.csr_json.write_text(csr_payload, encoding="utf-8")
            (gw / "sim_config.js").write_text(
                json.dumps([{"module": "serial2tcp", "args": {"port": "1111"}}, {"timebase": 500}]),
                encoding="utf-8",
            )
            (gw / "build_sim.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    def _fake_run(cmd, cwd: str, env, check: bool) -> None:
        assert cmd == ["bash", "build_sim.sh"] and env["JOBS"] == "1" and check is True
        obj = Path(cwd) / "obj_dir"
        obj.mkdir(parents=True, exist_ok=True)
        (obj / "Vsim").write_text("bin", encoding="utf-8")

    class _StubPopen:
        def __init__(self, *_args, **_kwargs) -> None:
            self.returncode = 0
            self.stdout = io.StringIO("")

        def poll(self):
            return 0

    close_raises = {"flag": False}

    class _StubClient:
        def __init__(self, host: str, port: int, timeout: float) -> None:
            self.host = host
            self.port = port

        def open(self) -> None:
            pass

        def read(self, _addr: int, length: int | None = None):
            return 0xCAFE_1958 if length is None else [0] * length

        def write(self, _addr: int, _val) -> None:
            pass

        def close(self) -> None:
            if close_raises["flag"]:
                raise OSError("already closed")

    monkeypatch.setattr(drv_mod, "REPO_ROOT", fake_root)
    monkeypatch.setattr(sim_mod, "SetunSimSoC", _StubSoC)
    monkeypatch.setattr(litex_builder, "Builder", _StubBuilder)
    monkeypatch.setattr(litex_client_mod, "RemoteClient", _StubClient)
    monkeypatch.setattr(drv_mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(drv_mod.subprocess, "Popen", _StubPopen)
    monkeypatch.setattr(drv_mod, "_wait_tcp_listening", lambda *_a, **_kw: None)

    # 1. Cold build path
    with drv_mod.open_backend("sim", rows=4, lanes=16, group_size=64, ram_init=ram_bin) as sim_drv:
        assert sim_drv.read_csr("ctrl_scratch") == 0xCAFE_1958

    # 2. Warm cached path (updates sim_config.js serial2tcp port + tolerates client.close OSError)
    close_raises["flag"] = True
    with drv_mod.open_backend("sim", rows=4, lanes=16, group_size=64, ram_init=ram_bin) as sim_drv:
        assert sim_drv.read_csr("ctrl_scratch") == 0xCAFE_1958
