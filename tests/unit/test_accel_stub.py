"""Bus-level tests for AccelStub CSR peripheral in SetunSimSoC (Task 0.4)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from litex.soc.integration import export

from setun.accelerators.stub import SETUN_STUB_ID, SETUN_STUB_VERSION
from targets.sim import SetunSimSoC
from tests.sim_helper import run_migen_sim


def _build_soc_and_load_csr_json(tmpdir: str) -> tuple[SetunSimSoC, dict[str, int]]:
    """Finalize SetunSimSoC with AccelStub, export csr.json, and return CSR byte addresses."""
    csr_json_path = Path(tmpdir) / "csr.json"
    soc = SetunSimSoC(
        with_accel_stub=True,
        expose_host_wb=True,
        use_external_crg=False,
        cpu_type=None,
        integrated_rom_size=0,
        integrated_main_ram_size=0x1000,
    )
    soc.finalize()

    csr_json_content = export.get_csr_json(
        csr_regions=soc.csr.regions,
        constants=soc.constants,
        mem_regions=soc.mem_regions,
    )
    csr_json_path.write_text(csr_json_content, encoding="utf-8")

    assert csr_json_path.exists(), "Failed to export csr.json"
    csr_data = json.loads(csr_json_path.read_text(encoding="utf-8"))
    registers = csr_data["csr_registers"]

    return soc, {name: int(meta["addr"]) for name, meta in registers.items()}


def test_accel_stub_wishbone_bus_and_csr_json() -> None:
    """Verify ID, VERSION, COUNTER, and >= 100 random 32-bit SCRATCH patterns via csr.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        soc, csr_addrs = _build_soc_and_load_csr_json(tmpdir)

        # Ensure all 4 CSRs are present in exported csr.json (no hardcoded addresses)
        id_addr = csr_addrs["accel_stub_id"]
        ver_addr = csr_addrs["accel_stub_version"]
        scratch_addr = csr_addrs["accel_stub_scratch"]
        counter_addr = csr_addrs["accel_stub_counter"]

        rng = np.random.default_rng(1958)
        patterns = [int(x) for x in rng.integers(0, 1 << 32, size=128, dtype=np.uint32)]

        wb = soc.host_wb

        def wb_read(byte_addr: int):
            val = yield from wb.read(byte_addr >> 2)
            return val

        def wb_write(byte_addr: int, data: int):
            yield from wb.write(byte_addr >> 2, data)

        def tb():
            # 1. Read ID and VERSION through the real Wishbone -> CSR bridge
            got_id = yield from wb_read(id_addr)
            assert got_id == SETUN_STUB_ID, f"ID mismatch: {got_id:#x} != {SETUN_STUB_ID:#x}"

            got_ver = yield from wb_read(ver_addr)
            assert got_ver == SETUN_STUB_VERSION, (
                f"VERSION mismatch: {got_ver:#x} != {SETUN_STUB_VERSION:#x}"
            )

            # 2. Verify free-running COUNTER advances between bus reads
            c0 = yield from wb_read(counter_addr)
            c1 = yield from wb_read(counter_addr)
            assert c1 > c0 > 0, f"Expected COUNTER to advance: c0={c0}, c1={c1}"

            # 3. Verify >= 100 (128) random 32-bit patterns on SCRATCH
            for idx, pat in enumerate(patterns):
                yield from wb_write(scratch_addr, pat)
                readback = yield from wb_read(scratch_addr)
                assert readback == pat, (
                    f"SCRATCH mismatch at pattern {idx}: got {readback:#010x}, expected {pat:#010x}"
                )

        run_migen_sim(soc, tb)
