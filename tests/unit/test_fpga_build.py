"""TDD unit tests for reproducible FPGA builds and STA slack checks (Tasks 6.2 & 6.3)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from litex.soc.integration.builder import Builder

from scripts.fpga_build import (
    TimingClosureError,
    build_board,
    compute_de10nano_utilization_report,
    compute_ecp5_utilization_report,
)
from targets.de10nano import SetunDE10NanoSoC
from targets.ecpix5 import SetunECPIX5SoC


def test_ecpix5_and_de10nano_gateware_emit_and_csr_parity(tmp_path: Path) -> None:
    for name, cls in (("ecpix5", SetunECPIX5SoC), ("de10nano", SetunDE10NanoSoC)):
        out = tmp_path / name
        soc = cls(accel_rows=2, accel_lanes=16, accel_group_size=32, accel_max_k=128)
        assert soc.mem_map["main_ram"] == 0x40000000
        csr_path = out / "csr.json"
        Builder(
            soc,
            output_dir=str(out),
            csr_json=str(csr_path),
            compile_software=False,
            compile_gateware=False,
        ).build(run=False)
        gw = out / "gateware"
        assert any(gw.glob("*.v"))
        regs = json.loads(csr_path.read_text("utf-8"))["csr_registers"]
        for req in ("ternary_accel_ctrl", "ternary_accel_status", "ternary_accel_weight_addr"):
            assert req in regs


def test_ecp5_report_determinism_and_strict_wns() -> None:
    fake_pnr = {
        "utilization": {
            "TRELLIS_COMB": {"used": 41200, "available": 83640},
            "TRELLIS_FF": {"used": 8500, "available": 83640},
            "DP16KD": {"used": 24, "available": 208},
            "MULT18X18D": {"used": 0, "available": 156},
        },
        "fmax": {
            "$glbnet$main_crg_sys_clk": {"achieved": 112.5, "constraint": 100.0},
            "$glbnet$main_crg_sys2x_clk": {"achieved": 225.0, "constraint": 200.0},
        },
    }
    r1 = compute_ecp5_utilization_report(
        pnr_data=fake_pnr,
        target_mhz=100.0,
        seed=1,
        nodsp=True,
        bitstream_sha256="ab" * 32,
        critical_path_lines=["Max frequency for clock 'sys': 112.50 MHz"],
        strict_timing=True,
    )
    r2 = compute_ecp5_utilization_report(
        pnr_data=fake_pnr,
        target_mhz=100.0,
        seed=1,
        nodsp=True,
        bitstream_sha256="ab" * 32,
        critical_path_lines=["Max frequency for clock 'sys': 112.50 MHz"],
        strict_timing=True,
    )
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    assert r1["sys_fmax_mhz"] == pytest.approx(112.5)
    assert r1["sys_wns_ns"] == pytest.approx(10.0 - (1000.0 / 112.5), rel=1e-3)
    assert r1["dsps"] == 0

    slow_pnr = {
        **fake_pnr,
        "fmax": {"$glbnet$main_crg_sys_clk": {"achieved": 82.0, "constraint": 100.0}},
    }
    with pytest.raises(TimingClosureError):
        compute_ecp5_utilization_report(
            pnr_data=slow_pnr,
            target_mhz=100.0,
            seed=1,
            nodsp=True,
            bitstream_sha256="cd" * 32,
            critical_path_lines=[],
            strict_timing=True,
        )


def test_de10nano_report_determinism_and_zero_dsp() -> None:
    stat = {
        "design": {
            "num_cells_by_type": {
                "MISTRAL_ALUT4": 12400,
                "MISTRAL_ALUT5": 3100,
                "MISTRAL_FF": 6200,
                "MISTRAL_M10K": 18,
            }
        }
    }
    r1 = compute_de10nano_utilization_report(stat, target_mhz=100.0, nodsp=True)
    r2 = compute_de10nano_utilization_report(stat, target_mhz=100.0, nodsp=True)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    assert r1["aluts"] == 15500
    assert r1["m10k"] == 18
    assert r1["dsps"] == 0


@pytest.mark.skipif(
    os.environ.get("SETUN_NIGHTLY") != "1", reason="Full Yosys SoC synth in nightly"
)
def test_build_board_de10nano_reproducible_utilization(tmp_path: Path) -> None:
    r1 = build_board(
        "de10nano", tmp_path / "r1", rows=1, lanes=16, group_size=16, max_k=32, max_rows=16
    )
    r2 = build_board(
        "de10nano", tmp_path / "r2", rows=1, lanes=16, group_size=16, max_k=32, max_rows=16
    )
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    assert r1["aluts"] > 0 and r1["dsps"] == 0
