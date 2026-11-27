"""Reproducible FPGA gateware/bitstream builder and STA timing gate (Tasks 6.2 & 6.3)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from litex.soc.integration.builder import Builder
from migen.fhdl.bitcontainer import value_bits_sign
from migen.fhdl.structure import wrap

from targets.de10nano import SetunDE10NanoSoC
from targets.ecpix5 import SetunECPIX5SoC


def _write_cyclonev_yosys_shims(gw: Path, soc: SetunDE10NanoSoC) -> Path:
    pll_params: list[str] = []
    pll_ports: list[str] = []
    for k, v in sorted(soc.crg.pll.params.items()):
        if k.startswith("p_"):
            pll_params.append(f"parameter {k[2:]} = 0")
        elif k.startswith(("i_", "o_")):
            d = "input" if k.startswith("i_") else "output"
            w = max(1, int(value_bits_sign(wrap(v))[0]))
            pll_ports.append(f"{d} [{w - 1}:0] {k[2:]}")
    shim = gw / "cyclonev_yosys_bb.v"
    shim.write_text(
        "module DFF(input d, clk, clrn, prn, output reg q);\n"
        "  always @(posedge clk or negedge clrn or negedge prn)\n"
        "    if (!clrn) q <= 1'b0; else if (!prn) q <= 1'b1; else q <= d;\n"
        "endmodule\n"
        "module DFFE(input d, clk, clrn, prn, ena, output reg q);\n"
        "  always @(posedge clk or negedge clrn or negedge prn)\n"
        "    if (!clrn) q <= 1'b0; else if (!prn) q <= 1'b1; else if (ena) q <= d;\n"
        "endmodule\n"
        f"(* blackbox *) module ALTPLL #({', '.join(pll_params)}) "
        f"({', '.join(pll_ports)}); endmodule\n"
        "(* blackbox *) module cyclonev_hps_interface_fpga2sdram(\n"
        "  input clk, read, write, input [29:0] address, input [3:0] byteenable,\n"
        "  input [31:0] writedata, output [31:0] readdata, output readdatavalid, waitrequest\n"
        "); endmodule\n",
        "utf-8",
    )
    return shim


class TimingClosureError(RuntimeError):
    """Raised when strict STA timing fails to meet target frequency (WNS < 0 ns)."""


def _resolve_tool(candidates: tuple[str, ...]) -> str:
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(f"[fpga_build] missing required executable from {candidates}.")


def _extract_critical_path(pnr_log: str) -> list[str]:
    keep: list[str] = []
    in_crit = False
    for raw in pnr_log.splitlines():
        line = raw.strip()
        if "Critical path report for clock" in line:
            in_crit = True
        if in_crit:
            keep.append(line)
            if line.startswith("Info: Max frequency for clock"):
                in_crit = False
        elif line.startswith("Info: Max frequency for clock"):
            keep.append(line)
    return keep[-80:]


def compute_ecp5_utilization_report(
    pnr_data: dict[str, object],
    target_mhz: float,
    seed: int,
    nodsp: bool,
    bitstream_sha256: str,
    critical_path_lines: list[str],
    strict_timing: bool = False,
) -> dict[str, object]:
    util = pnr_data.get("utilization", {})
    assert isinstance(util, dict)

    def _used(keys: tuple[str, ...]) -> tuple[int, int]:
        for k in keys:
            if k in util and isinstance(util[k], dict):
                return int(util[k].get("used", 0)), int(util[k].get("available", 0))
        return 0, 0

    lut4, lut4_avail = _used(("TRELLIS_COMB", "LUT4"))
    ffs, ffs_avail = _used(("TRELLIS_FF",))
    bram, bram_avail = _used(("DP16KD",))
    dsps, dsps_avail = _used(("MULT18X18D",))
    if nodsp and dsps != 0:
        raise RuntimeError(f"[fpga_build] 0-DSP invariant violated: used {dsps} MULT18X18D.")

    fmax_map = pnr_data.get("fmax", {})
    assert isinstance(fmax_map, dict) and fmax_map, "[fpga_build] empty nextpnr fmax map."
    clocks: dict[str, dict[str, float]] = {}
    sys_fmax = 0.0
    for net in sorted(fmax_map):
        entry = fmax_map[net]
        ach = float(entry["achieved"])
        con = float(entry.get("constraint", target_mhz))
        wns = round((1000.0 / con) - (1000.0 / ach), 3)
        clocks[net] = {
            "achieved_mhz": round(ach, 2),
            "constraint_mhz": round(con, 2),
            "wns_ns": wns,
        }
        if ("sys_clk" in net and "sys2x" not in net) or abs(con - target_mhz) < 1.0:
            if sys_fmax == 0.0 or "sys_clk" in net:
                sys_fmax = ach
    if sys_fmax == 0.0:
        sys_fmax = min(float(v["achieved"]) for v in fmax_map.values())

    sys_wns = round((1000.0 / target_mhz) - (1000.0 / sys_fmax), 3)
    if strict_timing and sys_wns < 0.0:
        raise TimingClosureError(
            f"[fpga_build] sys clock {sys_fmax:.2f} MHz < {target_mhz:.2f} MHz "
            f"(WNS = {sys_wns:+.3f} ns)."
        )
    return {
        "board": "ecpix5",
        "device": "LFE5UM5G-85F-8BG554C",
        "seed": int(seed),
        "nodsp": bool(nodsp),
        "lut4": lut4,
        "lut4_available": lut4_avail,
        "ffs": ffs,
        "ffs_available": ffs_avail,
        "dp16kd": bram,
        "dp16kd_available": bram_avail,
        "dsps": dsps,
        "dsps_available": dsps_avail,
        "sys_target_mhz": round(target_mhz, 2),
        "sys_fmax_mhz": round(sys_fmax, 2),
        "sys_wns_ns": sys_wns,
        "clocks": clocks,
        "bitstream_sha256": bitstream_sha256,
        "critical_path": list(critical_path_lines),
    }


def compute_de10nano_utilization_report(
    stat_data: dict[str, object],
    target_mhz: float,
    nodsp: bool,
) -> dict[str, object]:
    design = stat_data.get("design", {})
    assert isinstance(design, dict)
    cells = design.get("num_cells_by_type", {})
    assert isinstance(cells, dict)
    aluts = sum(int(v) for k, v in cells.items() if "ALUT" in k)
    ff_names = ("MISTRAL_FF", "DFF", "DFFE")
    ffs = sum(int(v) for k, v in cells.items() if k in ff_names or k.endswith("_FF"))
    m10k = sum(int(v) for k, v in cells.items() if "M10K" in k)
    dsps = sum(int(v) for k, v in cells.items() if "DSP" in k or "MUL" in k)
    if nodsp and dsps != 0:
        raise RuntimeError(f"[fpga_build] Cyclone V 0-DSP invariant violated: {dsps} DSPs.")
    return {
        "board": "de10nano",
        "device": "5CSEBA6U23I7",
        "nodsp": bool(nodsp),
        "aluts": aluts,
        "ffs": ffs,
        "m10k": m10k,
        "dsps": dsps,
        "sys_target_mhz": round(target_mhz, 2),
    }


def build_board(
    board: str = "ecpix5",
    output_dir: Path | str | None = None,
    rows: int = 4,
    lanes: int = 64,
    group_size: int = 128,
    max_k: int = 2048,
    max_rows: int = 2048,
    sys_clk_freq: int = int(100e6),
    seed: int = 1,
    nodsp: bool = True,
    strict_timing: bool = False,
    with_etherbone: bool = False,
    archive: bool = False,
) -> dict[str, object]:
    out = Path(output_dir or f"build/fpga/{board}")
    out.mkdir(parents=True, exist_ok=True)
    yosys_bin = _resolve_tool(("yosys",))
    nodsp_flag = " -nodsp" if nodsp else ""
    target_mhz = sys_clk_freq / 1e6

    if board == "ecpix5":
        soc = SetunECPIX5SoC(
            sys_clk_freq=sys_clk_freq,
            accel_rows=rows,
            accel_lanes=lanes,
            accel_group_size=group_size,
            accel_max_k=max_k,
            accel_max_rows=max_rows,
            with_etherbone=with_etherbone,
        )
        Builder(
            soc,
            output_dir=str(out),
            csr_json=str(out / "csr.json"),
            compile_software=False,
            compile_gateware=False,
        ).build(run=False)
        gw = out / "gateware"
        v_file = next(p for p in sorted(gw.glob("*.v")) if p.name != "cyclonev_yosys_bb.v")
        top = v_file.stem
        subprocess.run(
            [
                yosys_bin,
                "-ql",
                "synth.log",
                "-p",
                f"read_verilog {v_file.name}; synth_ecp5 -top {top}{nodsp_flag} -json {top}.json",
            ],
            cwd=gw,
            check=True,
        )
        nextpnr_bin = _resolve_tool(("nextpnr-ecp5", "yowasp-nextpnr-ecp5"))
        pnr_cmd = [
            nextpnr_bin,
            "--um5g-85k",
            "--package",
            "CABGA554",
            "--speed",
            "8",
            "--seed",
            str(seed),
            "--freq",
            f"{target_mhz:.2f}",
            "--json",
            f"{top}.json",
            "--lpf",
            f"{top}.lpf",
            "--textcfg",
            f"{top}.config",
            "--report",
            "pnr_report.json",
        ]
        if not strict_timing:
            pnr_cmd.append("--timing-allow-fail")
        pnr_proc = subprocess.run(pnr_cmd, cwd=gw, capture_output=True, text=True)
        pnr_log = (pnr_proc.stdout or "") + "\n" + (pnr_proc.stderr or "")
        (gw / "pnr.log").write_text(pnr_log, "utf-8")
        if pnr_proc.returncode != 0:
            raise RuntimeError("\n".join(pnr_log.splitlines()[-25:]))

        ecppack_bin = _resolve_tool(("ecppack", "yowasp-ecppack"))
        bit_path = gw / f"{top}.bit"
        subprocess.run(
            [ecppack_bin, "--compress", f"{top}.config", "--bit", bit_path.name],
            cwd=gw,
            check=True,
        )
        sha = hashlib.sha256(bit_path.read_bytes()).hexdigest()
        pnr_data = json.loads((gw / "pnr_report.json").read_text("utf-8"))
        report = compute_ecp5_utilization_report(
            pnr_data=pnr_data,
            target_mhz=target_mhz,
            seed=seed,
            nodsp=nodsp,
            bitstream_sha256=sha,
            critical_path_lines=_extract_critical_path(pnr_log),
            strict_timing=strict_timing,
        )
    elif board == "de10nano":
        soc = SetunDE10NanoSoC(
            sys_clk_freq=sys_clk_freq,
            accel_rows=rows,
            accel_lanes=lanes,
            accel_group_size=group_size,
            accel_max_k=max_k,
            accel_max_rows=max_rows,
        )
        Builder(
            soc,
            output_dir=str(out),
            csr_json=str(out / "csr.json"),
            compile_software=False,
            compile_gateware=False,
        ).build(run=False)
        gw = out / "gateware"
        v_file = next(p for p in sorted(gw.glob("*.v")) if p.name != "cyclonev_yosys_bb.v")
        top = v_file.stem
        subprocess.run(
            [
                yosys_bin,
                "-ql",
                "synth.log",
                "-p",
                f"read_verilog {_write_cyclonev_yosys_shims(gw, soc).name} {v_file.name}; "
                f"synth_intel_alm -family cyclonev{nodsp_flag} -top {top}; "
                "tee -q -o stat.json stat -json",
            ],
            cwd=gw,
            check=True,
        )
        stat_data = json.loads((gw / "stat.json").read_text("utf-8"))
        report = compute_de10nano_utilization_report(
            stat_data=stat_data,
            target_mhz=target_mhz,
            nodsp=nodsp,
        )
    else:
        raise ValueError(f"[fpga_build] unsupported board {board!r}.")

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (out / "utilization.json").write_text(payload, "utf-8")
    if archive:
        Path(f"reports/fpga_{board}.json").write_text(payload, "utf-8")
        arch = Path(f"build/fpga/archive/{board}")
        arch.mkdir(parents=True, exist_ok=True)
        (arch / "utilization.json").write_text(payload, "utf-8")
        if board == "ecpix5":
            shutil.copy2(bit_path, arch / bit_path.name)
            shutil.copy2(gw / "pnr_report.json", arch / "pnr_report.json")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Setun Reproducible FPGA Build (Tasks 6.2 & 6.3)")
    p.add_argument("--board", choices=("ecpix5", "de10nano"), default="ecpix5")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--lanes", type=int, default=64)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--sys-clk-freq", type=int, default=int(100e6))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--use-dsp", action="store_true")
    p.add_argument("--strict-timing", action="store_true")
    p.add_argument("--with-etherbone", action="store_true")
    p.add_argument("--archive", action="store_true")
    args = p.parse_args()
    rep = build_board(
        board=args.board,
        output_dir=args.output_dir,
        rows=args.rows,
        lanes=args.lanes,
        group_size=args.group_size,
        sys_clk_freq=args.sys_clk_freq,
        seed=args.seed,
        nodsp=not args.use_dsp,
        strict_timing=args.strict_timing,
        with_etherbone=args.with_etherbone,
        archive=args.archive,
    )
    print(json.dumps({k: v for k, v in rep.items() if k != "critical_path"}, indent=2))


if __name__ == "__main__":
    main()
