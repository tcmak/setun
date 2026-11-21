"""Yosys multi-vendor LUT/ALM/DSP resource characterization and baseline check (Task 2.6)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from migen.fhdl import verilog

from setun.accelerators.group_scale import TernaryRow

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_JSON = REPORTS_DIR / "resources.json"
DEFAULT_MD = REPORTS_DIR / "resources.md"
DEFAULT_BASELINE = REPORTS_DIR / "baseline.json"

# Target board capacities across Xilinx XC7, Lattice ECP5, and Intel Cyclone V (ADR-007)
CANDIDATE_BOARDS = [
    {
        "name": "Artix-7 XC7A35T (Arty A7-35)",
        "vendor": "xc7",
        "logic_unit": "LUT6",
        "logic_cap": 20_800,
        "dsps": 90,
        "mem_gbps": 1.6,
    },
    {
        "name": "Artix-7 XC7A100T (Arty A7-100)",
        "vendor": "xc7",
        "logic_unit": "LUT6",
        "logic_cap": 63_400,
        "dsps": 240,
        "mem_gbps": 3.2,
    },
    {
        "name": "Lattice ECP5-85F (ULX3S)",
        "vendor": "ecp5",
        "logic_unit": "LUT4",
        "logic_cap": 83_640,
        "dsps": 156,
        "mem_gbps": 1.6,
    },
    {
        "name": "Intel Cyclone V 5CSEMA6 (Arrow DECA)",
        "vendor": "cyclonev",
        "logic_unit": "ALM",
        "logic_cap": 41_910,
        "dsps": 112,
        "mem_gbps": 3.2,
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
    vendor: str = "xc7"
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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        v_path = tmp / f"{top_name}.v"
        net_json = tmp / "netlist.json"
        rep_json = tmp / "pnr_report.json"

        v_out = verilog.convert(dut, ios=dut.ios, name=top_name)
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
    return round(achieved, 2)


def _synthesize_single_row(
    lanes: int,
    group_size: int,
    act_bits: int,
    nodsp: bool = True,
    vendor: str = "xc7",
) -> tuple[int, int, int]:
    """Run Yosys synthesis on one TernaryRow and return (logic_cells, ffs, dsps)."""
    dut = TernaryRow(group_size=group_size, lanes=lanes, act_bits=act_bits)
    mode_suffix = "_nodsp" if nodsp else "_dsp"
    top_name = f"ternary_row_{vendor}_L{lanes}_G{group_size}_A{act_bits}{mode_suffix}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        v_path = tmp / f"{top_name}.v"
        json_path = tmp / "stat.json"

        v_out = verilog.convert(dut, ios=dut.ios, name=top_name)
        v_out.write(str(v_path))

        nodsp_flag = " -nodsp" if nodsp else ""
        if vendor == "xc7":
            synth_cmd = f"synth_xilinx -family xc7 -top {top_name} -noiopad{nodsp_flag}"
        elif vendor == "ecp5":
            synth_cmd = f"synth_ecp5 -top {top_name}{nodsp_flag}"
        elif vendor == "cyclonev":
            synth_cmd = f"synth_intel_alm -family cyclonev -top {top_name}{nodsp_flag}"
        else:
            raise ValueError(f"Unsupported synthesis vendor: {vendor}")

        cmd = [
            "yosys",
            "-q",
            "-p",
            f"read_verilog {v_path}; {synth_cmd}; tee -q -o {json_path} stat -json",
        ]
        subprocess.run(cmd, check=True)
        stat_data = json.loads(json_path.read_text())

    mod_key = f"\\{top_name}"
    cells: dict[str, int] = stat_data["modules"][mod_key]["num_cells_by_type"]

    if vendor == "xc7":
        logic = sum(
            cnt
            for cell, cnt in cells.items()
            if cell in {"LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6"}
        )
        ffs = sum(cnt for cell, cnt in cells.items() if cell.startswith("FD"))
        dsps = sum(cnt for cell, cnt in cells.items() if "DSP" in cell)
    elif vendor == "ecp5":
        logic = sum(cnt for cell, cnt in cells.items() if cell.startswith("LUT4"))
        ffs = sum(cnt for cell, cnt in cells.items() if cell.startswith(("TRELLIS_FF", "FD")))
        dsps = sum(cnt for cell, cnt in cells.items() if "MULT18" in cell or "ALU" in cell)
    else:
        logic = sum(
            cnt
            for cell, cnt in cells.items()
            if any(k in cell for k in ("ALUT", "ALM", "LCELL", "LUT"))
        )
        ffs = sum(
            cnt
            for cell, cnt in cells.items()
            if any(k in cell for k in ("MISTRAL_FF", "DFF", "REG")) or cell.startswith("FD")
        )
        dsps = sum(
            cnt
            for cell, cnt in cells.items()
            if any(k in cell for k in ("MISTRAL_MUL", "DSP", "MAC"))
        )

    return logic, ffs, dsps


def run_sweep(nodsp: bool = True) -> list[SynthMetrics]:
    """Synthesize unique (L, G, A) rows on XC7 and compute metrics for all (R, L, G, A)."""
    if shutil.which("yosys") is None:
        raise RuntimeError("yosys not found in PATH; run inside configured WSL2 environment.")

    row_cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    results: list[SynthMetrics] = []

    for rows, lanes, group_size, act_bits in SWEEP_CONFIGS:
        key = (lanes, group_size, act_bits)
        if key not in row_cache:
            row_cache[key] = _synthesize_single_row(
                lanes, group_size, act_bits, nodsp=nodsp, vendor="xc7"
            )
        row_luts, row_ffs, row_dsps = row_cache[key]

        total_luts = rows * row_luts
        total_ffs = rows * row_ffs
        total_dsps = rows * row_dsps
        macs = rows * lanes
        basis = "SYNTHESIZED (R=1)" if rows == 1 else "[EXTRAPOLATED from R=1]"

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
                vendor="xc7",
                basis=basis,
            )
        )
    return results


def run_multivendor_reference(nodsp: bool = True) -> list[VendorRowMetrics]:
    """Characterize canonical L=64, G=128, A=8 row across XC7, ECP5 (+ nextpnr), and Cyclone V."""
    nodsp_flag = " -nodsp" if nodsp else ""
    specs = [
        ("xc7", f"yosys synth_xilinx -family xc7{nodsp_flag}", "LUT6"),
        (
            "ecp5",
            f"yosys synth_ecp5{nodsp_flag} + nextpnr-ecp5 (--85k -6)",
            "LUT4",
        ),
        ("cyclonev", f"yosys synth_intel_alm -family cyclonev{nodsp_flag}", "ALM/LCELL"),
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


def render_markdown(
    metrics: list[SynthMetrics],
    vendor_rows: list[VendorRowMetrics],
    nodsp: bool = True,
) -> str:
    """Format multi-vendor sweep, board capacity, and bandwidth analysis as Markdown."""
    title_suffix = " (0-DSP Default / -nodsp)" if nodsp else " (Opt-In 1 DSP/row Mode)"
    nodsp_flag = " -nodsp" if nodsp else ""
    lines = [
        f"# Setun Ternary Datapath Resource & Throughput Report{title_suffix} (Task 2.6 / C1.1b)",
        "",
        f"Primary XC7 sweep via Yosys (`synth_xilinx -family xc7 -noiopad{nodsp_flag}`), "
        "paired with real `nextpnr-ecp5` place-and-route timing on Lattice ECP5-85F.",
        "",
        "## 1. Xilinx 7-Series Configuration Sweep (`R × L × G × A`)",
        "",
        "| R (rows) | L (lanes) | G | A (bits) | MACs/cycle | LUT6 | FFs | DSP48E1 "
        "| LUTs/MAC | DSPs/row | Basis |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in metrics:
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
    ref_xc7 = vendor_map["xc7"]
    ref_ecp5 = vendor_map["ecp5"]
    if ref_ecp5.fmax_mhz is None or ref_ecp5.fmax_mhz <= 0.0:
        raise ValueError(
            "[scripts/resource_report.py:render_markdown] invariant violated: "
            f"expected measured ecp5 fmax_mhz > 0 from nextpnr-ecp5, got {ref_ecp5.fmax_mhz!r}."
        )
    fmax_mhz = ref_ecp5.fmax_mhz
    coeff_per_gbps = round(1000.0 / (0.25 * fmax_mhz), 2)

    lines.extend(
        [
            "",
            f"## 3. Candidate Board Fit & Memory Bandwidth Ceiling "
            f"(at {fmax_mhz:.2f} MHz Measured ECP5 Fmax)",
            "",
            f"Reference XC7 row (`L=64, G=128, A=8`): **{ref_xc7.logic_cells} LUT6**, "
            f"**{ref_xc7.dsps} DSP48E1** ({ref_xc7.cells_per_mac:.2f} LUTs/MAC). "
            f"Measured out-of-context `nextpnr-ecp5` (`LFE5U-85F-6BG381C`) clock frequency: "
            f"**{fmax_mhz:.2f} MHz**.",
            "",
            "At 2 bits/trit (0.25 B/weight), a memory interface supplying $B_{mem}$ GB/s "
            "feeds at most:",
            "",
            f"$$\\text{{Max useful MACs/cycle at }}{fmax_mhz:.2f}\\text{{ MHz}} = "
            f"\\frac{{B_{{mem}} \\times 10^9}}{{0.25 \\times {fmax_mhz:.2f} \\times 10^6}} "
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
        bw_macs = int(round((b["mem_gbps"] * 1e9) / (0.25 * fmax_mhz * 1e6)))
        lines.append(
            f"| {b['name']} | {b['logic_cap']:,} {b['logic_unit']} | {b['dsps']} | "
            f"{b['mem_gbps']:.1f} | {fit_rows} | {fit_macs:,} | {bw_macs:,} |"
        )

    dsp_mode_desc = (
        "0 DSPs everywhere (default -nodsp LUT multipliers)"
        if nodsp
        else "1 DSP48E1 per row (--use-dsp opt-in for u16 group scales)"
    )
    lines.extend(
        [
            "",
            "### Key Architectural Takeaways (Gate 2 / G1 & `ADR-004`)",
            (
                "- **Zero DSPs in `TernaryDot`:** All ternary MAC lanes "
                "are multiplier-free add/sub trees."
            ),
            f"- **Group Scale DSP Mode:** {dsp_mode_desc}.",
            f"- **Measured Single-Cycle Row Timing (`nextpnr-ecp5`):** Out-of-context place-and-"
            f"route on `LFE5U-85F-6BG381C` measures **{fmax_mhz:.2f} MHz** across the unpipelined "
            "`group_acc + dot_in -> scale_u16 multiply -> row_acc` critical path in "
            "`GroupAccumulateScale` (pipelining for >= 100 MHz board closure scheduled in "
            "Season 3 Task `6.3`). Multi-row (`R=4`, `R=8`) figures are `[EXTRAPOLATED from R=1]` "
            "pending `TernaryTile` (`Task 2.4`).",
            "- **Memory-bound regime:** Even on the smallest Artix-7 (`XC7A35T`) in 0-DSP mode, "
            "compute fabric fits more MACs/cycle than external DDR3/SPI weight bandwidth can feed "
            f"at {fmax_mhz:.2f} MHz—confirming that memory burst efficiency (`Phase 3.2`) "
            "governs throughput.",
        ]
    )
    return "\n".join(lines) + "\n"


def check_baseline(
    metrics: list[SynthMetrics],
    baseline_path: Path,
    max_regression: float = 0.10,
) -> None:
    """Fail if any baseline configuration's LUTs/MAC regresses beyond max_regression (10%)."""
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline file {baseline_path} not found. "
            "Run `python3 scripts/resource_report.py --update-baseline`."
        )
    baseline = json.loads(baseline_path.read_text())
    base_map = {
        (row["rows"], row["lanes"], row["group_size"], row["act_bits"]): row["luts_per_mac"]
        for row in baseline["configs"]
    }

    regressions: list[str] = []
    for m in metrics:
        key = (m.rows, m.lanes, m.group_size, m.act_bits)
        if key not in base_map:
            continue
        base_val = base_map[key]
        limit = base_val * (1.0 + max_regression)
        if m.luts_per_mac > limit:
            pct = ((m.luts_per_mac - base_val) / base_val) * 100.0
            regressions.append(
                f"Config R={m.rows},L={m.lanes},G={m.group_size},A={m.act_bits}: "
                f"{m.luts_per_mac:.3f} LUTs/MAC vs baseline {base_val:.3f} (+{pct:.1f}% > 10%)"
            )

    if regressions:
        raise RuntimeError("Resource regression detected:\n  " + "\n  ".join(regressions))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Setun Multi-Vendor Resource Report Generator",
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
        help="Verify <= 10% LUT/MAC regression",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify deterministic reports/resources.{json,md} sync and <= 10% baseline gate",
    )
    parser.add_argument(
        "--use-dsp",
        action="store_true",
        help="Opt-in to 1 DSP48E1/row for group scaling (default is 0-DSP -nodsp mode)",
    )
    args = parser.parse_args(argv)

    nodsp = not args.use_dsp
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = run_sweep(nodsp=nodsp)
    vendor_rows = run_multivendor_reference(nodsp=nodsp)
    ecp5_row = next(vr for vr in vendor_rows if vr.vendor == "ecp5")
    nodsp_flag = " -nodsp" if nodsp else ""
    payload = {
        "synth_flow": f"yosys synth_xilinx -family xc7 -noiopad{nodsp_flag}",
        "nodsp": nodsp,
        "measured_ecp5_fmax_mhz": ecp5_row.fmax_mhz,
        "configs": [asdict(m) for m in metrics],
        "multivendor_ref": [asdict(vr) for vr in vendor_rows],
    }

    json_text = json.dumps(payload, indent=2) + "\n"
    md_text = render_markdown(metrics, vendor_rows, nodsp=nodsp)

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
            f"(measured ECP5 Fmax = {ecp5_row.fmax_mhz:.2f} MHz)."
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
        print("[resource-report] Baseline check passed (<= 10% LUTs/MAC delta).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
