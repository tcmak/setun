"""Yosys + nextpnr-ecp5 resource & timing characterization on Lattice ECP5 and Intel Cyclone V."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from migen import ClockDomain, Module
from migen.fhdl import verilog

from setun.accelerators.group_scale import TernaryRow
from setun.accelerators.requantize import Requantize
from setun.accelerators.tile import TernaryTile
from setun.soc.ternary_accel import TernaryAccel

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_JSON = REPORTS_DIR / "resources.json"
DEFAULT_MD = REPORTS_DIR / "resources.md"
DEFAULT_BASELINE = REPORTS_DIR / "baseline.json"
SYNTH_CACHE_DIR = REPO_ROOT / ".cache" / "synth"
TOOLCHAIN_LOCK = REPO_ROOT / "toolchain.lock"
_ORIG_SUBPROCESS_RUN = subprocess.run

SUPPORTED_VENDORS = ("ecp5", "cyclonev")


def _synth_cache_enabled() -> bool:
    if os.environ.get("SETUN_DISABLE_SYNTH_CACHE") == "1":
        return False
    return (
        subprocess.run is _ORIG_SUBPROCESS_RUN
        or os.environ.get("SETUN_FORCE_SYNTH_CACHE_TEST") == "1"
    )


def _synth_cache_key(tag: str, verilog_src: str) -> Path:
    h = hashlib.sha256()
    if TOOLCHAIN_LOCK.exists():
        h.update(TOOLCHAIN_LOCK.read_bytes())
    h.update(b"\0" + tag.encode("utf-8") + b"\0" + verilog_src.encode("utf-8"))
    return SYNTH_CACHE_DIR / f"{h.hexdigest()}.json"


def _synth_cache_load(cache_file: Path) -> object | None:
    if _synth_cache_enabled() and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    return None


def _synth_cache_store(cache_file: Path, payload: object) -> None:
    if not _synth_cache_enabled():
        return
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(f".{os.getpid()}.tmp")
    tmp_file.write_text(json.dumps(payload), encoding="utf-8")
    tmp_file.replace(cache_file)


# Target board capacities across Lattice ECP5-85F and Intel Cyclone V (ADR-007, no AMD/Xilinx)
CANDIDATE_BOARDS = [
    {
        "name": "LambdaConcept ECPIX-5 (Lattice ECP5-85F, DDR3)",
        "vendor": "ecp5",
        "logic_unit": "LUT4",
        "logic_cap": 83_640,
        "dsps": 156,
        "mem_gbps": 1.6,
    },
    {
        "name": "Radiona ULX3S (Lattice ECP5-85F, SDRAM)",
        "vendor": "ecp5",
        "logic_unit": "LUT4",
        "logic_cap": 83_640,
        "dsps": 156,
        "mem_gbps": 0.3,
    },
    {
        "name": "Terasic DE10-Nano (Intel Cyclone V 5CSEBA6, HPS DDR3)",
        "vendor": "cyclonev",
        "logic_unit": "ALM",
        "logic_cap": 41_910,
        "dsps": 112,
        "mem_gbps": 3.2,
    },
    {
        "name": "Arrow DECA (Intel Cyclone V 5CSEMA6, DDR3)",
        "vendor": "cyclonev",
        "logic_unit": "ALM",
        "logic_cap": 41_910,
        "dsps": 112,
        "mem_gbps": 1.6,
    },
]

# Sweep configurations: (R_rows, L_lanes, G_group, A_act_bits)
SWEEP_CONFIGS = [
    (1, 16, 64, 8),
    (1, 16, 128, 8),
    (1, 64, 128, 8),
    (1, 64, 256, 8),
    (1, 128, 128, 8),
    (1, 64, 128, 16),
    (4, 64, 128, 8),
    (8, 64, 128, 8),
]


@dataclass
class SynthMetrics:
    rows: int
    lanes: int
    group_size: int
    act_bits: int
    macs_per_cycle: int
    luts: int
    ffs: int
    dsps: int
    luts_per_mac: float
    dsps_per_row: float
    nodsp: bool = True
    vendor: str = "ecp5"
    basis: str = "SYNTHESIZED (R=1)"


@dataclass
class VendorRowMetrics:
    vendor: str
    flow: str
    logic_unit: str
    logic_cells: int
    ffs: int
    dsps: int
    cells_per_mac: float
    fmax_mhz: float | None = None


@dataclass
class RequantizeStageMetrics:
    in_bits: int
    out_bits: int
    nodsp: bool
    ecp5_lut4: int
    ecp5_ffs: int
    ecp5_dsps: int
    ecp5_fmax_mhz: float
    cyclonev_aluts: int
    cyclonev_ffs: int
    cyclonev_dsps: int


@dataclass
class AccelBramMetrics:
    rows: int
    lanes: int
    group_size: int
    max_k: int
    max_rows: int
    nodsp: bool
    ecp5_lut4: int
    ecp5_ffs: int
    ecp5_dp16kd: int
    ecp5_dsps: int
    ecp5_fmax_mhz: float
    cyclonev_aluts: int
    cyclonev_ffs: int
    cyclonev_m10k: int
    cyclonev_dsps: int


def _resolve_nextpnr_ecp5() -> str:
    """Locate nextpnr-ecp5 (or yowasp-nextpnr-ecp5) in PATH or active virtualenv."""
    venv_bin = Path(sys.executable).resolve().parent
    for name in ("nextpnr-ecp5", "yowasp-nextpnr-ecp5"):
        found = shutil.which(name)
        if found:
            return found
        candidate = venv_bin / name
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        "[scripts/resource_report.py:_resolve_nextpnr_ecp5] toolchain invariant violated: "
        "expected 'nextpnr-ecp5' or 'yowasp-nextpnr-ecp5' executable, found MISSING."
    )


def _extract_pnr_fmax_mhz(report: dict[str, object], top_name: str, caller: str) -> float:
    """Extract and validate achieved clock Fmax (MHz) from a nextpnr-ecp5 JSON report."""
    fmax_map = report.get("fmax", {})
    if not isinstance(fmax_map, dict) or not fmax_map:
        raise RuntimeError(
            f"[scripts/resource_report.py:{caller}] nextpnr-ecp5 report for "
            f"{top_name} contained empty 'fmax' section."
        )
    clk_entry = fmax_map.get("sys_clk") or next(iter(fmax_map.values()))
    achieved = round(float(clk_entry["achieved"]), 2)
    if achieved <= 0.0:
        raise RuntimeError(
            f"[scripts/resource_report.py:{caller}] invalid measured ecp5_fmax_mhz "
            f"for {top_name}: expected > 0.0, got {achieved}."
        )
    return achieved


def _count_vendor_cells(cells: dict[str, int], vendor: str) -> tuple[int, int, int]:
    """Return (logic_cells, ffs, dsps) for Lattice ECP5 or Intel Cyclone V Yosys stat JSON."""
    if vendor == "ecp5":
        logic = sum(cnt for cell, cnt in cells.items() if cell.startswith("LUT4"))
        ffs = sum(cnt for cell, cnt in cells.items() if cell.startswith(("TRELLIS_FF", "FD")))
        dsps = sum(cnt for cell, cnt in cells.items() if "MULT18" in cell or "ALU54" in cell)
        return logic, ffs, dsps
    logic = sum(
        cnt for cell, cnt in cells.items() if any(k in cell for k in ("ALUT", "ALM", "LCELL"))
    )
    ffs = sum(
        cnt
        for cell, cnt in cells.items()
        if any(k in cell for k in ("MISTRAL_FF", "DFF", "REG")) or cell.startswith("FD")
    )
    dsps = sum(
        cnt for cell, cnt in cells.items() if any(k in cell for k in ("MISTRAL_MUL", "DSP", "MAC"))
    )
    return logic, ffs, dsps


def _measure_ecp5_fmax_mhz(
    lanes: int,
    group_size: int,
    act_bits: int,
    nodsp: bool = True,
    speed: int = 6,
) -> float:
    """Run Yosys synth_ecp5 + nextpnr-ecp5 place-and-route and return measured Fmax in MHz."""
    if shutil.which("yosys") is None:
        raise RuntimeError(
            "[scripts/resource_report.py:_measure_ecp5_fmax_mhz] yosys not found in PATH."
        )
    nextpnr_bin = _resolve_nextpnr_ecp5()
    dut = TernaryRow(group_size=group_size, lanes=lanes, act_bits=act_bits)
    mode_suffix = "_nodsp" if nodsp else "_dsp"
    top_name = f"ternary_row_ecp5_pnr_L{lanes}_G{group_size}_A{act_bits}{mode_suffix}"
    v_out = verilog.convert(dut, ios=dut.ios, name=top_name)
    ckey = _synth_cache_key(f"ecp5_fmax:{top_name}:s{speed}", v_out.main_source)
    cached_fmax = _synth_cache_load(ckey)
    if cached_fmax is not None:
        return float(cached_fmax)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        v_path = tmp / f"{top_name}.v"
        net_json = tmp / "netlist.json"
        rep_json = tmp / "pnr_report.json"

        v_out.write(str(v_path))

        nodsp_flag = " -nodsp" if nodsp else ""
        yosys_cmd = [
            "yosys",
            "-q",
            "-p",
            f"read_verilog {v_path}; synth_ecp5 -top {top_name}{nodsp_flag} -json {net_json}",
        ]
        subprocess.run(yosys_cmd, check=True)

        pnr_cmd = [
            nextpnr_bin,
            "--85k",
            "--package",
            "CABGA381",
            "--speed",
            str(speed),
            "--seed",
            "1",
            "--freq",
            "100",
            "--timing-allow-fail",
            "--out-of-context",
            "--json",
            "netlist.json",
            "--report",
            "pnr_report.json",
        ]
        subprocess.run(pnr_cmd, cwd=str(tmp), check=True, capture_output=True, text=True)
        report = json.loads(rep_json.read_text(encoding="utf-8"))

    fmax_map = report.get("fmax", {})
    if not fmax_map:
        raise RuntimeError(
            f"[scripts/resource_report.py:_measure_ecp5_fmax_mhz] nextpnr-ecp5 report for "
            f"{top_name} contained empty 'fmax' section."
        )
    clk_entry = fmax_map.get("sys_clk") or next(iter(fmax_map.values()))
    achieved = float(clk_entry["achieved"])
    if achieved <= 0.0:
        raise RuntimeError(
            f"[scripts/resource_report.py:_measure_ecp5_fmax_mhz] invalid achieved fmax_mhz "
            f"for {top_name}: expected > 0.0, got {achieved}."
        )
    res_fmax = round(achieved, 2)
    _synth_cache_store(ckey, res_fmax)
    return res_fmax


def _synthesize_single_row(
    lanes: int,
    group_size: int,
    act_bits: int,
    nodsp: bool = True,
    vendor: str = "ecp5",
) -> tuple[int, int, int]:
    """Run Yosys synthesis on one TernaryRow and return (logic_cells, ffs, dsps)."""
    if vendor not in SUPPORTED_VENDORS:
        raise ValueError(f"Unsupported synthesis vendor: {vendor} (supported: {SUPPORTED_VENDORS})")
    dut = TernaryRow(group_size=group_size, lanes=lanes, act_bits=act_bits)
    mode_suffix = "_nodsp" if nodsp else "_dsp"
    top_name = f"ternary_row_{vendor}_L{lanes}_G{group_size}_A{act_bits}{mode_suffix}"
    v_out = verilog.convert(dut, ios=dut.ios, name=top_name)
    ckey = _synth_cache_key(f"single_row:{top_name}", v_out.main_source)
    cached_row = _synth_cache_load(ckey)
    if isinstance(cached_row, list) and len(cached_row) == 3:
        return int(cached_row[0]), int(cached_row[1]), int(cached_row[2])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        v_path = tmp / f"{top_name}.v"
        json_path = tmp / "stat.json"

        v_out.write(str(v_path))

        nodsp_flag = " -nodsp" if nodsp else ""
        if vendor == "ecp5":
            synth_cmd = f"synth_ecp5 -top {top_name}{nodsp_flag}"
        else:
            synth_cmd = f"synth_intel_alm -family cyclonev -top {top_name}{nodsp_flag}"

        cmd = [
            "yosys",
            "-q",
            "-p",
            f"read_verilog {v_path}; {synth_cmd}; tee -q -o {json_path} stat -json",
        ]
        subprocess.run(cmd, check=True)
        stat_data = json.loads(json_path.read_text(encoding="utf-8"))

    cells: dict[str, int] = stat_data["modules"][f"\\{top_name}"]["num_cells_by_type"]
    row_counts = _count_vendor_cells(cells, vendor)
    _synth_cache_store(ckey, list(row_counts))
    return row_counts


def _synthesize_tile(
    rows: int,
    lanes: int,
    group_size: int,
    act_bits: int,
    nodsp: bool = True,
    vendor: str = "ecp5",
) -> tuple[int, int, int]:
    """Run Yosys synthesis on a multi-row TernaryTile and return (logic_cells, ffs, dsps)."""
    if vendor not in SUPPORTED_VENDORS:
        raise ValueError(f"Unsupported synthesis vendor: {vendor} (supported: {SUPPORTED_VENDORS})")
    dut = TernaryTile(
        rows=rows,
        lanes=lanes,
        group_size=group_size,
        act_bits=act_bits,
        max_k=max(2048, group_size),
    )
    mode_suffix = "_nodsp" if nodsp else "_dsp"
    top_name = f"ternary_tile_{vendor}_R{rows}_L{lanes}_G{group_size}_A{act_bits}{mode_suffix}"
    v_out = verilog.convert(dut, ios=dut.ios, name=top_name)
    ckey = _synth_cache_key(f"tile:{top_name}", v_out.main_source)
    cached_tile = _synth_cache_load(ckey)
    if isinstance(cached_tile, list) and len(cached_tile) == 3:
        return int(cached_tile[0]), int(cached_tile[1]), int(cached_tile[2])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        v_path = tmp / f"{top_name}.v"
        json_path = tmp / "stat.json"

        v_out.write(str(v_path))

        nodsp_flag = " -nodsp" if nodsp else ""
        if vendor == "ecp5":
            synth_cmd = f"synth_ecp5 -top {top_name}{nodsp_flag}"
        else:
            synth_cmd = f"synth_intel_alm -family cyclonev -top {top_name}{nodsp_flag}"

        cmd = [
            "yosys",
            "-q",
            "-p",
            f"read_verilog {v_path}; {synth_cmd}; tee -q -o {json_path} stat -json",
        ]
        subprocess.run(cmd, check=True)
        stat_data = json.loads(json_path.read_text(encoding="utf-8"))

    cells: dict[str, int] = stat_data["modules"][f"\\{top_name}"]["num_cells_by_type"]
    tile_counts = _count_vendor_cells(cells, vendor)
    _synth_cache_store(ckey, list(tile_counts))
    return tile_counts


def run_sweep(nodsp: bool = True) -> list[SynthMetrics]:
    """Synthesize (R, L, G, A) configs across Lattice ECP5 and Intel Cyclone V."""
    if shutil.which("yosys") is None:
        raise RuntimeError("yosys not found in PATH; run inside configured WSL2 environment.")

    row_cache: dict[tuple[str, int, int, int], tuple[int, int, int]] = {}
    tile_cache: dict[tuple[str, int, int, int, int], tuple[int, int, int]] = {}
    results: list[SynthMetrics] = []

    for vendor in SUPPORTED_VENDORS:
        for rows, lanes, group_size, act_bits in SWEEP_CONFIGS:
            if rows == 1:
                key = (vendor, lanes, group_size, act_bits)
                if key not in row_cache:
                    row_cache[key] = _synthesize_single_row(
                        lanes, group_size, act_bits, nodsp=nodsp, vendor=vendor
                    )
                total_luts, total_ffs, total_dsps = row_cache[key]
                basis = "SYNTHESIZED (TernaryRow)"
            else:
                tkey = (vendor, rows, lanes, group_size, act_bits)
                if tkey not in tile_cache:
                    tile_cache[tkey] = _synthesize_tile(
                        rows, lanes, group_size, act_bits, nodsp=nodsp, vendor=vendor
                    )
                total_luts, total_ffs, total_dsps = tile_cache[tkey]
                basis = f"SYNTHESIZED (TernaryTile R={rows})"

            macs = rows * lanes
            results.append(
                SynthMetrics(
                    rows=rows,
                    lanes=lanes,
                    group_size=group_size,
                    act_bits=act_bits,
                    macs_per_cycle=macs,
                    luts=total_luts,
                    ffs=total_ffs,
                    dsps=total_dsps,
                    luts_per_mac=round(total_luts / macs, 3),
                    dsps_per_row=round(total_dsps / rows, 2),
                    nodsp=nodsp,
                    vendor=vendor,
                    basis=basis,
                )
            )
    return results


def run_multivendor_reference(nodsp: bool = True) -> list[VendorRowMetrics]:
    """Characterize canonical L=64, G=128, A=8 row on ECP5 (+ nextpnr) and Cyclone V."""
    nodsp_flag = " -nodsp" if nodsp else ""
    specs = [
        (
            "ecp5",
            f"yosys synth_ecp5{nodsp_flag} + nextpnr-ecp5 (--85k -6)",
            "LUT4",
        ),
        (
            "cyclonev",
            f"yosys synth_intel_alm -family cyclonev{nodsp_flag}",
            "ALUT",
        ),
    ]
    out: list[VendorRowMetrics] = []
    for vendor, flow, unit in specs:
        logic, ffs, dsps = _synthesize_single_row(
            lanes=64, group_size=128, act_bits=8, nodsp=nodsp, vendor=vendor
        )
        fmax_mhz: float | None = None
        if vendor == "ecp5":
            fmax_mhz = _measure_ecp5_fmax_mhz(
                lanes=64, group_size=128, act_bits=8, nodsp=nodsp, speed=6
            )
        out.append(
            VendorRowMetrics(
                vendor=vendor,
                flow=flow,
                logic_unit=unit,
                logic_cells=logic,
                ffs=ffs,
                dsps=dsps,
                cells_per_mac=round(logic / 64.0, 3),
                fmax_mhz=fmax_mhz,
            )
        )
    return out


def run_requantize_characterization(
    in_bits: int = 48,
    out_bits: int = 8,
    nodsp: bool = True,
    speed: int = 6,
) -> RequantizeStageMetrics:
    """Synthesize Requantize on Lattice ECP5 (+ nextpnr-ecp5 Fmax) and Intel Cyclone V."""
    if shutil.which("yosys") is None:
        raise RuntimeError(
            "[scripts/resource_report.py:run_requantize_characterization] yosys not found in PATH."
        )
    nextpnr_bin = _resolve_nextpnr_ecp5()
    dut = Requantize(in_bits=in_bits, out_bits=out_bits)
    mode_suffix = "_nodsp" if nodsp else "_dsp"
    top_name = f"requantize_I{in_bits}_O{out_bits}{mode_suffix}"
    nodsp_flag = " -nodsp" if nodsp else ""

    v_out = verilog.convert(dut, ios=dut.ios, name=top_name)
    ckey = _synth_cache_key(f"requantize:{top_name}:s{speed}", v_out.main_source)
    cached_req = _synth_cache_load(ckey)
    if isinstance(cached_req, dict):
        return RequantizeStageMetrics(**cached_req)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        v_path = tmp / f"{top_name}.v"
        ecp5_net_json = tmp / "ecp5_netlist.json"
        ecp5_stat_json = tmp / "ecp5_stat.json"
        cycv_stat_json = tmp / "cycv_stat.json"
        pnr_json = tmp / "ecp5_pnr.json"

        v_out.write(str(v_path))

        # 1. Lattice ECP5 synthesis + nextpnr-ecp5 out-of-context place-and-route
        subprocess.run(
            [
                "yosys",
                "-q",
                "-p",
                f"read_verilog {v_path}; "
                f"synth_ecp5 -top {top_name}{nodsp_flag} -json {ecp5_net_json}; "
                f"tee -q -o {ecp5_stat_json} stat -json",
            ],
            check=True,
        )
        ecp5_cells: dict[str, int] = json.loads(ecp5_stat_json.read_text(encoding="utf-8"))[
            "modules"
        ][f"\\{top_name}"]["num_cells_by_type"]
        ecp5_lut4, ecp5_ffs, ecp5_dsps = _count_vendor_cells(ecp5_cells, "ecp5")

        # 2. Intel Cyclone V synthesis (synth-only)
        subprocess.run(
            [
                "yosys",
                "-q",
                "-p",
                f"read_verilog {v_path}; "
                f"synth_intel_alm -family cyclonev -top {top_name}{nodsp_flag}; "
                f"tee -q -o {cycv_stat_json} stat -json",
            ],
            check=True,
        )
        cycv_cells: dict[str, int] = json.loads(cycv_stat_json.read_text(encoding="utf-8"))[
            "modules"
        ][f"\\{top_name}"]["num_cells_by_type"]
        cyclonev_aluts, cyclonev_ffs, cyclonev_dsps = _count_vendor_cells(cycv_cells, "cyclonev")

        subprocess.run(
            [
                nextpnr_bin,
                "--85k",
                "--package",
                "CABGA381",
                "--speed",
                str(speed),
                "--seed",
                "1",
                "--freq",
                "100",
                "--timing-allow-fail",
                "--out-of-context",
                "--json",
                "ecp5_netlist.json",
                "--report",
                "ecp5_pnr.json",
            ],
            cwd=str(tmp),
            check=True,
            capture_output=True,
            text=True,
        )
        pnr_rep = json.loads(pnr_json.read_text(encoding="utf-8"))

    if nodsp and (ecp5_dsps != 0 or cyclonev_dsps != 0):
        raise RuntimeError(
            f"[scripts/resource_report.py:run_requantize_characterization] 0-DSP invariant "
            f"violated in -nodsp mode: expected ecp5_dsps=0 and cyclonev_dsps=0, got "
            f"ecp5_dsps={ecp5_dsps}, cyclonev_dsps={cyclonev_dsps}."
        )

    ecp5_fmax_mhz = _extract_pnr_fmax_mhz(pnr_rep, top_name, "run_requantize_characterization")

    req_metrics = RequantizeStageMetrics(
        in_bits=in_bits,
        out_bits=out_bits,
        nodsp=nodsp,
        ecp5_lut4=ecp5_lut4,
        ecp5_ffs=ecp5_ffs,
        ecp5_dsps=ecp5_dsps,
        ecp5_fmax_mhz=ecp5_fmax_mhz,
        cyclonev_aluts=cyclonev_aluts,
        cyclonev_ffs=cyclonev_ffs,
        cyclonev_dsps=cyclonev_dsps,
    )
    _synth_cache_store(ckey, asdict(req_metrics))
    return req_metrics


def run_accel_bram_characterization(
    rows: int = 4,
    lanes: int = 64,
    group_size: int = 128,
    beat_bits: int = 256,
    max_k: int = 2048,
    max_rows: int = 2048,
    nodsp: bool = True,
    speed: int = 6,
) -> AccelBramMetrics:
    """Synthesize TernaryAccel on ECP5 + Cyclone V, verify sync BRAMs, and measure ECP5 Fmax."""
    if shutil.which("yosys") is None:
        raise RuntimeError(
            "[scripts/resource_report.py:run_accel_bram_characterization] yosys not in PATH."
        )
    nextpnr_bin = _resolve_nextpnr_ecp5()

    class _AccelSynthTop(Module):
        def __init__(self) -> None:
            self.clock_domains.cd_sys = ClockDomain("sys")
            acc = TernaryAccel(
                rows=rows,
                lanes=lanes,
                group_size=group_size,
                beat_bits=beat_bits,
                max_k=max_k,
                max_rows=max_rows,
            )
            self.submodules.acc = acc
            self.ios = {
                self.cd_sys.clk,
                self.cd_sys.rst,
                acc.trit_data,
                acc.trit_last,
                acc.trit_valid,
                acc.trit_ready,
                acc.scale_data,
                acc.scale_valid,
                acc.scale_ready,
                acc.ctrl.storage,
                acc.ctrl.re,
                acc.ctrl.fields.enable,
                acc.ctrl.fields.clear,
                acc.ctrl.fields.start,
                acc.ctrl.fields.req_enable,
                acc.ctrl.fields.req_sym,
                acc.cfg.fields.shift,
                acc.cfg.fields.num_rows,
                acc.dim_k.fields.k_cols,
                acc.dim_k.fields.group_size,
                acc.req_cfg.fields.scale_u16,
                acc.req_cfg.fields.shift,
                acc.weight_addr.storage,
                acc.scale_addr.storage,
                acc.act_addr.storage,
                acc.out_addr.storage,
                acc.act_word_idx.storage,
                acc.act_word_idx.re,
                acc.act_word_data.storage,
                acc.act_word_data.re,
                acc.out_row_idx.storage,
                acc.out_row_data.status,
                acc.result_lo.status,
                acc.result_hi.status,
                acc.status.fields.busy,
                acc.status.fields.done,
                acc.status.fields.error,
                acc.status.fields.req_saturated,
                acc.status.fields.err_code,
                acc.status.fields.rows_completed,
                acc.perf_cycles.status,
                acc.perf_bus_beats.status,
                acc.perf_sat_rows.status,
                acc.dma_wb.adr,
                acc.dma_wb.dat_w,
                acc.dma_wb.dat_r,
                acc.dma_wb.sel,
                acc.dma_wb.cyc,
                acc.dma_wb.stb,
                acc.dma_wb.we,
                acc.dma_wb.cti,
                acc.dma_wb.bte,
                acc.dma_wb.ack,
            }

    dut = _AccelSynthTop()
    top_name = f"ternary_accel_R{rows}_L{lanes}_G{group_size}_K{max_k}"
    nodsp_flag = " -nodsp" if nodsp else ""
    v_out = verilog.convert(dut, ios=dut.ios, name=top_name)
    ckey = _synth_cache_key(
        f"accel:{top_name}:mr{max_rows}:b{beat_bits}:n{int(nodsp)}:s{speed}",
        v_out.main_source,
    )
    cached_acc = _synth_cache_load(ckey)
    if isinstance(cached_acc, dict):
        return AccelBramMetrics(**cached_acc)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        v_path = tmp / f"{top_name}.v"
        ecp5_net_json = tmp / "accel_ecp5_netlist.json"
        ecp5_json = tmp / "ecp5.json"
        cycv_json = tmp / "cycv.json"
        pnr_json = tmp / "accel_ecp5_pnr.json"

        v_out.write(str(v_path))

        subprocess.run(
            [
                "yosys",
                "-q",
                "-p",
                f"read_verilog {v_path}; "
                f"synth_ecp5 -top {top_name}{nodsp_flag} -json {ecp5_net_json}; "
                f"tee -q -o {ecp5_json} stat -json",
            ],
            check=True,
        )
        subprocess.run(
            [
                "yosys",
                "-q",
                "-p",
                f"read_verilog {v_path}; "
                f"synth_intel_alm -family cyclonev -top {top_name}{nodsp_flag}; "
                f"tee -q -o {cycv_json} stat -json",
            ],
            check=True,
        )
        subprocess.run(
            [
                nextpnr_bin,
                "--85k",
                "--package",
                "CABGA381",
                "--speed",
                str(speed),
                "--seed",
                "1",
                "--freq",
                "100",
                "--timing-allow-fail",
                "--out-of-context",
                "--json",
                "accel_ecp5_netlist.json",
                "--report",
                "accel_ecp5_pnr.json",
            ],
            cwd=str(tmp),
            check=True,
            capture_output=True,
            text=True,
        )

        ecp5_cells: dict[str, int] = json.loads(ecp5_json.read_text(encoding="utf-8"))["modules"][
            f"\\{top_name}"
        ]["num_cells_by_type"]
        cycv_cells: dict[str, int] = json.loads(cycv_json.read_text(encoding="utf-8"))["modules"][
            f"\\{top_name}"
        ]["num_cells_by_type"]
        pnr_rep = json.loads(pnr_json.read_text(encoding="utf-8"))

    ecp5_lut4, ecp5_ffs, ecp5_dsps = _count_vendor_cells(ecp5_cells, "ecp5")
    ecp5_dp16kd = sum(cnt for cell, cnt in ecp5_cells.items() if "16KD" in cell)

    cyclonev_aluts, cyclonev_ffs, cyclonev_dsps = _count_vendor_cells(cycv_cells, "cyclonev")
    cyclonev_m10k = sum(cnt for cell, cnt in cycv_cells.items() if "M10K" in cell)

    if nodsp and (ecp5_dsps != 0 or cyclonev_dsps != 0):
        raise RuntimeError(
            f"[scripts/resource_report.py:run_accel_bram_characterization] expected 0 DSPs in "
            f"-nodsp mode, got ecp5_dsps={ecp5_dsps}, cyclonev_dsps={cyclonev_dsps}."
        )
    if ecp5_dp16kd == 0 or cyclonev_m10k == 0:
        raise RuntimeError(
            f"[scripts/resource_report.py:run_accel_bram_characterization] synchronous BRAM "
            f"inference failed: got ecp5_dp16kd={ecp5_dp16kd}, cyclonev_m10k={cyclonev_m10k}."
        )

    ecp5_fmax_mhz = _extract_pnr_fmax_mhz(pnr_rep, top_name, "run_accel_bram_characterization")

    acc_metrics = AccelBramMetrics(
        rows=rows,
        lanes=lanes,
        group_size=group_size,
        max_k=max_k,
        max_rows=max_rows,
        nodsp=nodsp,
        ecp5_lut4=ecp5_lut4,
        ecp5_ffs=ecp5_ffs,
        ecp5_dp16kd=ecp5_dp16kd,
        ecp5_dsps=ecp5_dsps,
        ecp5_fmax_mhz=ecp5_fmax_mhz,
        cyclonev_aluts=cyclonev_aluts,
        cyclonev_ffs=cyclonev_ffs,
        cyclonev_m10k=cyclonev_m10k,
        cyclonev_dsps=cyclonev_dsps,
    )
    _synth_cache_store(ckey, asdict(acc_metrics))
    return acc_metrics


def resolve_governing_ecp5_fmax(
    row_fmax_mhz: float,
    requantize_stage: RequantizeStageMetrics | None = None,
    accel_soc: AccelBramMetrics | None = None,
) -> tuple[float, str]:
    """Return (governing_fmax_mhz, stage_label) as the strict min across active ECP5 stages."""
    stages: list[tuple[str, float]] = []
    if accel_soc is not None:
        stages.append(("TernaryAccel full top", accel_soc.ecp5_fmax_mhz))
    if requantize_stage is not None:
        stages.append(("Requantize 48b->8b stage", requantize_stage.ecp5_fmax_mhz))
    stages.append(("TernaryRow isolated stage", row_fmax_mhz))

    bottleneck_stage, bottleneck_fmax = min(stages, key=lambda item: item[1])
    return round(bottleneck_fmax, 2), bottleneck_stage


def render_markdown(
    metrics: list[SynthMetrics],
    vendor_rows: list[VendorRowMetrics],
    nodsp: bool = True,
    requantize_stage: RequantizeStageMetrics | None = None,
    accel_soc: AccelBramMetrics | None = None,
) -> str:
    """Format ECP5 + Cyclone V sweep, board capacity, and bandwidth analysis as Markdown."""
    title_suffix = " (0-DSP Default / -nodsp)" if nodsp else " (Opt-In 1 DSP/row Mode)"
    nodsp_flag = " -nodsp" if nodsp else ""
    lines = [
        f"# Setun Ternary Datapath Resource & Throughput Report{title_suffix} (ADR-007)",
        "",
        f"Primary non-Xilinx multi-vendor sweeps via Yosys (`synth_ecp5{nodsp_flag}` + "
        f"`nextpnr-ecp5 --85k -6` on Lattice ECP5-85F, and `synth_intel_alm -family cyclonev"
        f"{nodsp_flag}` on Intel Cyclone V).",
    ]

    ecp5_metrics = [m for m in metrics if m.vendor == "ecp5"]
    cycv_metrics = [m for m in metrics if m.vendor == "cyclonev"]

    if ecp5_metrics:
        lines.extend(
            [
                "",
                "## 1a. Lattice ECP5 Configuration Sweep (`R × L × G × A`, `synth_ecp5`)",
                "",
                "| R (rows) | L (lanes) | G | A (bits) | MACs/cycle | LUT4 | FFs | MULT18X18D "
                "| LUT4/MAC | DSPs/row | Basis |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for m in ecp5_metrics:
            lines.append(
                f"| {m.rows} | {m.lanes} | {m.group_size} | {m.act_bits} | {m.macs_per_cycle} | "
                f"{m.luts} | {m.ffs} | {m.dsps} | {m.luts_per_mac:.3f} | {m.dsps_per_row:.2f} | "
                f"{m.basis} |"
            )

    if cycv_metrics:
        lines.extend(
            [
                "",
                "## 1b. Intel Cyclone V Configuration Sweep (`R × L × G × A`, `synth_intel_alm`)",
                "",
                "| R (rows) | L (lanes) | G | A (bits) | MACs/cycle | ALUT | FFs | MUL18X18 "
                "| ALUT/MAC | DSPs/row | Basis |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for m in cycv_metrics:
            lines.append(
                f"| {m.rows} | {m.lanes} | {m.group_size} | {m.act_bits} | {m.macs_per_cycle} | "
                f"{m.luts} | {m.ffs} | {m.dsps} | {m.luts_per_mac:.3f} | {m.dsps_per_row:.2f} | "
                f"{m.basis} |"
            )

    lines.extend(
        [
            "",
            "## 2. Multi-Vendor Reference Row (`L=64, G=128, A=8`) for `ADR-007`",
            "",
            "| Vendor Target | Synthesis / PnR Flow | Logic Unit | Logic Cells | FFs | DSPs "
            "| Cells/MAC | Measured Fmax |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for vr in vendor_rows:
        fmax_str = f"{vr.fmax_mhz:.2f} MHz" if vr.fmax_mhz is not None else "synth-only"
        lines.append(
            f"| `{vr.vendor}` | `{vr.flow}` | {vr.logic_unit} | {vr.logic_cells} | "
            f"{vr.ffs} | {vr.dsps} | {vr.cells_per_mac:.3f} | {fmax_str} |"
        )

    vendor_map = {vr.vendor: vr for vr in vendor_rows}
    ref_ecp5 = vendor_map["ecp5"]
    if ref_ecp5.fmax_mhz is None or ref_ecp5.fmax_mhz <= 0.0:
        raise ValueError(
            "[scripts/resource_report.py:render_markdown] invariant violated: "
            f"expected measured ecp5 fmax_mhz > 0 from nextpnr-ecp5, got {ref_ecp5.fmax_mhz!r}."
        )
    row_fmax_mhz = ref_ecp5.fmax_mhz
    gov_fmax_mhz, gov_stage = resolve_governing_ecp5_fmax(
        row_fmax_mhz=row_fmax_mhz,
        requantize_stage=requantize_stage,
        accel_soc=accel_soc,
    )
    coeff_per_gbps = round(1000.0 / (0.25 * gov_fmax_mhz), 2)

    stage_summary_parts = [f"isolated `TernaryRow` = {row_fmax_mhz:.2f} MHz"]
    if requantize_stage is not None:
        stage_summary_parts.append(f"`Requantize` = {requantize_stage.ecp5_fmax_mhz:.2f} MHz")
    if accel_soc is not None:
        stage_summary_parts.append(f"full `TernaryAccel` = {accel_soc.ecp5_fmax_mhz:.2f} MHz")
    stage_summary_str = ", ".join(stage_summary_parts)

    lines.extend(
        [
            "",
            f"## 3. Candidate Board Fit & Memory Bandwidth Ceiling "
            f"(at {gov_fmax_mhz:.2f} MHz Governing ECP5 Fmax — `{gov_stage}`)",
            "",
            f"Reference ECP5 row (`L=64, G=128, A=8`): **{ref_ecp5.logic_cells} LUT4**, "
            f"**{ref_ecp5.dsps} MULT18X18D** ({ref_ecp5.cells_per_mac:.2f} LUT4/MAC). "
            f"System throughput on Lattice ECP5-85F (`LFE5U-85F-6BG381C`) is strictly governed by "
            f"**{gov_fmax_mhz:.2f} MHz** (`{gov_stage}`, `min` across measured pipeline stages: "
            f"{stage_summary_str}). ECP5 `nextpnr-ecp5` frequencies are **never** extrapolated to "
            "`synth-only` Intel Cyclone V targets.",
            "",
            "At 2 bits/trit (0.25 B/weight), an ECP5 memory interface supplying $B_{mem}$ GB/s "
            "feeds at most:",
            "",
            f"$$\\text{{Max useful MACs/cycle at }}{gov_fmax_mhz:.2f}\\text{{ MHz}} = "
            f"\\frac{{B_{{mem}} \\times 10^9}}{{0.25 \\times {gov_fmax_mhz:.2f} \\times 10^6}} "
            f"\\approx {coeff_per_gbps:.2f} \\times B_{{mem}}$$",
            "",
            "| Candidate Board | Logic Budget | DSP Budget | Usable BW (GB/s) "
            "| Max Rows (80% Logic, `[EXTRAPOLATED from R=1]`) | Fit MACs/cycle "
            "| BW-Matched MACs/cycle |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for b in CANDIDATE_BOARDS:
        vr = vendor_map[b["vendor"]]
        max_rows_logic = int((b["logic_cap"] * 0.80) // max(vr.logic_cells, 1))
        if vr.dsps > 0:
            max_rows_dsp = int(b["dsps"] // vr.dsps)
            fit_rows = min(max_rows_logic, max_rows_dsp)
        else:
            fit_rows = max_rows_logic
        fit_macs = fit_rows * 64
        if vr.fmax_mhz is not None:
            bw_macs = int(round((b["mem_gbps"] * 1e9) / (0.25 * gov_fmax_mhz * 1e6)))
            bw_cell = f"{bw_macs:,}"
        else:
            bw_cell = "N/A (synth-only, pending Quartus STA)"
        lines.append(
            f"| {b['name']} | {b['logic_cap']:,} {b['logic_unit']} | {b['dsps']} | "
            f"{b['mem_gbps']:.1f} | {fit_rows} | {fit_macs:,} | {bw_cell} |"
        )

    dsp_mode_desc = (
        "0 DSPs everywhere (default -nodsp LUT multipliers)"
        if nodsp
        else "1 hard 18x18 DSP per row (--use-dsp opt-in for u16 group scales)"
    )
    lines.extend(
        [
            "",
            "### Key Architectural Takeaways (Gate 2 / G1, `ADR-004`, & `ADR-007`)",
            (
                "- **Zero DSPs in `TernaryDot`:** All ternary MAC lanes "
                "are multiplier-free add/sub trees."
            ),
            f"- **Group Scale DSP Mode:** {dsp_mode_desc}.",
            f"- **Governing End-to-End ECP5 Timing (`nextpnr-ecp5`):** System throughput is "
            f"governed by **{gov_fmax_mhz:.2f} MHz** (`{gov_stage}`; {stage_summary_str}), never "
            "by isolated `TernaryRow` alone when downstream `Requantize` or full `TernaryAccel` "
            "measures lower (pipelining for >= 100 MHz board closure scheduled in Season 3 Task "
            "`6.3`). Multi-row (`R=4`, `R=8`) figures in Section 1 are directly synthesized from "
            "`TernaryTile`, while Section 3 board row ceilings remain `[EXTRAPOLATED from R=1]`.",
            "- **No Cross-Family Timing Extrapolation:** Intel Cyclone V (`DE10-Nano` / `DECA`) "
            "rows are synthesized via `synth_intel_alm` without open-source STA, so bandwidth-"
            "matched MAC targets remain `N/A (synth-only, pending Quartus STA)` until Quartus "
            "TimeQuest closure in Season 3.",
        ]
    )
    if requantize_stage is not None:
        lines.extend(
            [
                "",
                "## 4. Output Requantization Stage (`Requantize` 48-bit -> `int8`, Task 2.5)",
                "",
                "Shared 1-row/cycle 2-stage pipelined requantizer (`II = 1`, 1 instance per "
                "`TernaryTile` behind the serialized output mux):",
                "",
                "| Width (`in -> out`) | ECP5 LUT4 | ECP5 FFs | ECP5 MULT18X18D "
                "| Measured ECP5 Fmax | Cyclone V ALUT | Cyclone V FFs | Cyclone V MUL18X18 "
                "| Cyclone V Fmax |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
                f"| `{requantize_stage.in_bits}b -> {requantize_stage.out_bits}b` | "
                f"{requantize_stage.ecp5_lut4} | {requantize_stage.ecp5_ffs} | "
                f"{requantize_stage.ecp5_dsps} | {requantize_stage.ecp5_fmax_mhz:.2f} MHz | "
                f"{requantize_stage.cyclonev_aluts} | {requantize_stage.cyclonev_ffs} | "
                f"{requantize_stage.cyclonev_dsps} | synth-only |",
            ]
        )

    if accel_soc is not None:
        lines.extend(
            [
                "",
                "## 5. Multi-Row SoC, Sync BRAM Inference & Full-Top Timing (`TernaryAccel`)",
                "",
                f"Full `TernaryAccel(R={accel_soc.rows}, L={accel_soc.lanes}, "
                f"G={accel_soc.group_size}, max_k={accel_soc.max_k}, "
                f"max_rows={accel_soc.max_rows})` (`WeightUnpacker` + `(R-1)` sync-BRAM stripe "
                "banks + `TernaryTile` + `Requantize` + Wishbone burst DMA + sync scratchpad "
                "BRAMs):",
                "",
                "| Geometry | ECP5 LUT4 | ECP5 FFs | ECP5 BRAM (`DP16KD`) | ECP5 MULT18X18D "
                "| Measured ECP5 Fmax | Cyclone V ALUT | Cyclone V FFs | Cyclone V BRAM (`M10K`) "
                "| Cyclone V MUL18X18 | Cyclone V Fmax |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                f"| `R={accel_soc.rows}, L={accel_soc.lanes}, K={accel_soc.max_k}` | "
                f"{accel_soc.ecp5_lut4} | {accel_soc.ecp5_ffs} | {accel_soc.ecp5_dp16kd} | "
                f"{accel_soc.ecp5_dsps} | {accel_soc.ecp5_fmax_mhz:.2f} MHz | "
                f"{accel_soc.cyclonev_aluts} | {accel_soc.cyclonev_ffs} | "
                f"{accel_soc.cyclonev_m10k} | {accel_soc.cyclonev_dsps} | synth-only |",
            ]
        )

    return "\n".join(lines) + "\n"


def check_baseline(
    metrics: list[SynthMetrics],
    baseline_path: Path,
    max_regression: float = 0.10,
) -> None:
    """Fail if any baseline configuration's cells/MAC regresses beyond max_regression (10%)."""
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline file {baseline_path} not found. "
            "Run `python3 scripts/resource_report.py --update-baseline`."
        )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_map = {
        (
            row.get("vendor", "ecp5"),
            row["rows"],
            row["lanes"],
            row["group_size"],
            row["act_bits"],
        ): row["luts_per_mac"]
        for row in baseline["configs"]
    }

    regressions: list[str] = []
    for m in metrics:
        key = (m.vendor, m.rows, m.lanes, m.group_size, m.act_bits)
        if key not in base_map:
            continue
        base_val = base_map[key]
        limit = base_val * (1.0 + max_regression)
        if m.luts_per_mac > limit:
            pct = ((m.luts_per_mac - base_val) / base_val) * 100.0
            regressions.append(
                f"Config vendor={m.vendor},R={m.rows},L={m.lanes},G={m.group_size},"
                f"A={m.act_bits}: {m.luts_per_mac:.3f} cells/MAC vs baseline {base_val:.3f} "
                f"(+{pct:.1f}% > 10%)"
            )

    if regressions:
        raise RuntimeError("Resource regression detected:\n  " + "\n  ".join(regressions))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Setun Multi-Vendor Resource Report Generator (ECP5 + Cyclone V)",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite reports/baseline.json",
    )
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Verify <= 10% cells/MAC regression",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify deterministic reports/resources.{json,md} sync and <= 10% baseline gate",
    )
    parser.add_argument(
        "--use-dsp",
        action="store_true",
        help="Opt-in to 1 hard 18x18 DSP/row for group scaling (default is 0-DSP -nodsp mode)",
    )
    args = parser.parse_args(argv)

    nodsp = not args.use_dsp
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = run_sweep(nodsp=nodsp)
    vendor_rows = run_multivendor_reference(nodsp=nodsp)
    req_stage = run_requantize_characterization(nodsp=nodsp)
    accel_soc = run_accel_bram_characterization(nodsp=nodsp)
    ecp5_row = next(vr for vr in vendor_rows if vr.vendor == "ecp5")
    assert ecp5_row.fmax_mhz is not None
    gov_fmax_mhz, gov_stage = resolve_governing_ecp5_fmax(
        row_fmax_mhz=ecp5_row.fmax_mhz,
        requantize_stage=req_stage,
        accel_soc=accel_soc,
    )
    nodsp_flag = " -nodsp" if nodsp else ""
    payload = {
        "synth_flow": (
            f"yosys synth_ecp5{nodsp_flag} + nextpnr-ecp5 (--85k -6) & "
            f"synth_intel_alm -family cyclonev{nodsp_flag}"
        ),
        "nodsp": nodsp,
        "governing_ecp5_fmax_mhz": gov_fmax_mhz,
        "governing_ecp5_stage": gov_stage,
        "measured_row_ecp5_fmax_mhz": ecp5_row.fmax_mhz,
        "configs": [asdict(m) for m in metrics],
        "multivendor_ref": [asdict(vr) for vr in vendor_rows],
        "requantize_stage": asdict(req_stage),
        "ternary_accel_soc": asdict(accel_soc),
    }

    json_text = json.dumps(payload, indent=2) + "\n"
    md_text = render_markdown(
        metrics,
        vendor_rows,
        nodsp=nodsp,
        requantize_stage=req_stage,
        accel_soc=accel_soc,
    )

    if args.check:
        check_baseline(metrics, DEFAULT_BASELINE)
        if not DEFAULT_JSON.exists() or DEFAULT_JSON.read_text(encoding="utf-8") != json_text:
            raise RuntimeError(
                f"[scripts/resource_report.py:main] deterministic report invariant violated: "
                f"{DEFAULT_JSON} is missing or out of sync with measured synthesis/PnR output."
            )
        if not DEFAULT_MD.exists() or DEFAULT_MD.read_text(encoding="utf-8") != md_text:
            raise RuntimeError(
                f"[scripts/resource_report.py:main] deterministic report invariant violated: "
                f"{DEFAULT_MD} is missing or out of sync with measured synthesis/PnR output."
            )
        print(
            "[resource-report] Deterministic report check and <= 10% baseline gate passed "
            f"(governing ECP5 Fmax = {gov_fmax_mhz:.2f} MHz [{gov_stage}])."
        )
        return 0

    DEFAULT_JSON.write_text(json_text, encoding="utf-8")
    DEFAULT_MD.write_text(md_text, encoding="utf-8")
    print(f"[resource-report] Wrote {DEFAULT_MD} and {DEFAULT_JSON} (nodsp={nodsp})")

    if args.update_baseline:
        DEFAULT_BASELINE.write_text(json_text, encoding="utf-8")
        print(f"[resource-report] Updated baseline at {DEFAULT_BASELINE}")

    if args.check_baseline:
        check_baseline(metrics, DEFAULT_BASELINE)
        print("[resource-report] Baseline check passed (<= 10% cells/MAC delta).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
