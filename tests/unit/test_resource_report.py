"""Unit tests for Yosys multi-vendor resource characterization and 0-DSP default (Task 2.6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.resource_report import (
    SynthMetrics,
    VendorRowMetrics,
    _synthesize_single_row,
    check_baseline,
    render_markdown,
)


@pytest.mark.slow
def test_synthesize_single_row_nodsp_default_vs_optin_dsp() -> None:
    """Verify default -nodsp uses 0 DSPs while opt-in nodsp=False uses 1 DSP48E1/row."""
    luts_nodsp, ffs_nodsp, dsps_nodsp = _synthesize_single_row(
        lanes=16, group_size=64, act_bits=8, nodsp=True, vendor="xc7"
    )
    luts_dsp, ffs_dsp, dsps_dsp = _synthesize_single_row(
        lanes=16, group_size=64, act_bits=8, nodsp=False, vendor="xc7"
    )

    assert dsps_nodsp == 0, f"Expected 0 DSP48E1 in default -nodsp mode, got {dsps_nodsp}"
    assert dsps_dsp == 1, f"Expected 1 DSP48E1 in opt-in DSP mode, got {dsps_dsp}"
    assert luts_nodsp > luts_dsp
    assert ffs_nodsp >= ffs_dsp


@pytest.mark.slow
def test_synthesize_multivendor_ecp5_and_cyclonev() -> None:
    """Verify Lattice ECP5 and Intel Cyclone V synthesis succeed with 0 DSPs in default mode."""
    for vendor in ("ecp5", "cyclonev"):
        logic, ffs, dsps = _synthesize_single_row(
            lanes=16, group_size=64, act_bits=8, nodsp=True, vendor=vendor
        )
        assert logic > 0
        assert ffs > 0
        assert dsps == 0


def test_render_markdown_and_baseline_check(tmp_path: Path) -> None:
    """Verify markdown rendering and baseline regression gate."""
    sample = [
        SynthMetrics(
            rows=1,
            lanes=64,
            group_size=128,
            act_bits=8,
            macs_per_cycle=64,
            luts=3300,
            ffs=114,
            dsps=0,
            luts_per_mac=51.562,
            dsps_per_row=0.0,
            nodsp=True,
            vendor="xc7",
        )
    ]
    vrows = [
        VendorRowMetrics("xc7", "yosys xc7", "LUT6", 3300, 114, 0, 51.562),
        VendorRowMetrics("ecp5", "yosys ecp5", "LUT4", 4200, 114, 0, 65.625, 62.5),
        VendorRowMetrics("cyclonev", "yosys cyclonev", "ALM/LCELL", 2800, 114, 0, 43.75),
    ]
    md = render_markdown(sample, vrows, nodsp=True)
    assert "0-DSP Default / -nodsp" in md
    assert "ecp5" in md
    assert "cyclonev" in md

    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(
        json.dumps(
            {
                "synth_flow": "test",
                "configs": [
                    {
                        "rows": 1,
                        "lanes": 64,
                        "group_size": 128,
                        "act_bits": 8,
                        "luts_per_mac": 50.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    check_baseline(sample, baseline_file, max_regression=0.10)
    with pytest.raises(RuntimeError, match="Resource regression detected"):
        check_baseline(sample, baseline_file, max_regression=0.02)


@pytest.mark.slow
def test_ecp5_nextpnr_measured_fmax_and_extrapolated_labels() -> None:
    """Task C1.1b: Measure real nextpnr-ecp5 fmax_mhz and label R=4/R=8 [EXTRAPOLATED from R=1]."""
    import scripts.resource_report as rr

    assert hasattr(rr, "_measure_ecp5_fmax_mhz"), (
        "[test_resource_report] expected _measure_ecp5_fmax_mhz in scripts/resource_report.py"
    )
    fmax = rr._measure_ecp5_fmax_mhz(lanes=16, group_size=64, act_bits=8, nodsp=True)
    assert isinstance(fmax, float) and 10.0 < fmax < 500.0 and fmax != 100.0, (
        f"[test_resource_report] expected real measured nextpnr-ecp5 fmax_mhz != 100.0, got {fmax}"
    )

    sample = [
        SynthMetrics(
            rows=1,
            lanes=64,
            group_size=128,
            act_bits=8,
            macs_per_cycle=64,
            luts=3300,
            ffs=114,
            dsps=0,
            luts_per_mac=51.562,
            dsps_per_row=0.0,
            nodsp=True,
            vendor="xc7",
            basis="SYNTHESIZED (R=1)",
        ),
        SynthMetrics(
            rows=4,
            lanes=64,
            group_size=128,
            act_bits=8,
            macs_per_cycle=256,
            luts=13200,
            ffs=456,
            dsps=0,
            luts_per_mac=51.562,
            dsps_per_row=0.0,
            nodsp=True,
            vendor="xc7",
            basis="[EXTRAPOLATED from R=1]",
        ),
        SynthMetrics(
            rows=8,
            lanes=64,
            group_size=128,
            act_bits=8,
            macs_per_cycle=512,
            luts=26400,
            ffs=912,
            dsps=0,
            luts_per_mac=51.562,
            dsps_per_row=0.0,
            nodsp=True,
            vendor="xc7",
            basis="[EXTRAPOLATED from R=1]",
        ),
    ]
    vrows = [
        VendorRowMetrics("xc7", "yosys xc7", "LUT6", 3300, 114, 0, 51.562, None),
        VendorRowMetrics("ecp5", "yosys+nextpnr-ecp5", "LUT4", 4741, 114, 0, 74.078, fmax),
        VendorRowMetrics("cyclonev", "yosys cyclonev", "ALM/LCELL", 2677, 114, 0, 41.828, None),
    ]
    md = render_markdown(sample, vrows, nodsp=True)
    assert md.count("[EXTRAPOLATED from R=1]") >= 2, (
        "[test_resource_report] expected both R=4 and R=8 rows labeled '[EXTRAPOLATED from R=1]'"
    )
    assert f"{fmax:.2f}" in md, (
        f"[test_resource_report] expected measured fmax {fmax:.2f} MHz in rendered markdown"
    )
    assert "at 100 MHz" not in md, (
        "[test_resource_report] hardcoded 'at 100 MHz' must be removed from rendered markdown"
    )


def test_synthesize_row_and_tile_elaboration_and_vendor_cell_parsers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify row/tile Verilog elaboration, multi-vendor stat.json cell parsers, and guards."""
    import scripts.resource_report as rr

    with pytest.raises(ValueError, match="Unsupported synthesis vendor"):
        rr._synthesize_single_row(16, 64, 8, nodsp=True, vendor="unknown_fpga")
    with pytest.raises(ValueError, match="Unsupported synthesis vendor"):
        rr._synthesize_tile(2, 16, 64, 8, nodsp=True, vendor="unknown_fpga")

    def _mock_yosys_and_pnr(cmd: list[str], **kwargs: object) -> None:
        cwd = kwargs.get("cwd")
        if cwd is not None:
            out_dir = Path(str(cwd))
            for rep_name in ("pnr_report.json", "ecp5_pnr.json"):
                (out_dir / rep_name).write_text(
                    json.dumps({"fmax": {"sys_clk": {"achieved": 64.25}}}),
                    encoding="utf-8",
                )
            return

        prompt = cmd[-1]
        if "tee -q -o " not in prompt:
            return
        stat_file = Path(prompt.split("tee -q -o ")[1].split(" stat -json")[0])
        top = prompt.split("-top ")[1].split()[0]
        if "synth_xilinx" in prompt:
            cells = {"LUT6": 640, "FDRE": 64, "DSP48E1": 0}
        elif "synth_ecp5" in prompt:
            cells = {"LUT4": 512, "TRELLIS_FF": 64}
        else:
            cells = {"MISTRAL_ALM": 400, "MISTRAL_FF": 64}
        stat_file.write_text(
            json.dumps({"modules": {f"\\{top}": {"num_cells_by_type": cells}}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(rr.subprocess, "run", _mock_yosys_and_pnr)
    assert rr._synthesize_single_row(16, 64, 8, nodsp=True, vendor="xc7") == (640, 64, 0)
    assert rr._synthesize_single_row(16, 64, 8, nodsp=True, vendor="ecp5") == (512, 64, 0)
    assert rr._synthesize_single_row(16, 64, 8, nodsp=True, vendor="cyclonev") == (400, 64, 0)

    # Bound TernaryTile max_k to 64 during unit-test elaboration so Verilog emit runs in <0.1s
    orig_tile = rr.TernaryTile
    monkeypatch.setattr(
        rr,
        "TernaryTile",
        lambda **kw: orig_tile(**{**kw, "max_k": kw["group_size"]}),
    )
    assert rr._synthesize_tile(2, 16, 64, 8, nodsp=True, vendor="xc7") == (640, 64, 0)
    assert rr._synthesize_tile(2, 16, 64, 8, nodsp=True, vendor="ecp5") == (512, 64, 0)
    assert rr._synthesize_tile(2, 16, 64, 8, nodsp=True, vendor="cyclonev") == (400, 64, 0)

    assert rr._measure_ecp5_fmax_mhz(16, 64, 8, nodsp=True) == 64.25
    req_m = rr.run_requantize_characterization(in_bits=16, out_bits=8, nodsp=True)
    assert (req_m.xc7_lut6, req_m.xc7_ffs, req_m.xc7_dsps) == (640, 64, 0)
    assert (req_m.ecp5_lut4, req_m.ecp5_ffs, req_m.ecp5_dsps) == (512, 64, 0)
    assert req_m.ecp5_fmax_mhz == 64.25


def test_resource_report_pipelines_error_guards_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cover _resolve_nextpnr_ecp5, error guards, DSP-bound render, check_baseline, and main()."""
    import scripts.resource_report as rr

    # 1. _resolve_nextpnr_ecp5 venv_bin fallback & missing binary RuntimeError
    fake_venv = tmp_path / "venv_bin"
    fake_venv.mkdir()
    fake_py = fake_venv / "python3"
    fake_py.write_text("", encoding="utf-8")
    monkeypatch.setattr(rr.sys, "executable", str(fake_py))
    monkeypatch.setattr(rr.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="_resolve_nextpnr_ecp5"):
        rr._resolve_nextpnr_ecp5()

    fake_pnr = fake_venv / "nextpnr-ecp5"
    fake_pnr.write_text("", encoding="utf-8")
    assert rr._resolve_nextpnr_ecp5() == str(fake_pnr)

    # 2. Missing yosys guards across entry points
    with pytest.raises(RuntimeError, match="yosys not found"):
        rr._measure_ecp5_fmax_mhz(16, 64, 8)
    with pytest.raises(RuntimeError, match="yosys not found"):
        rr.run_sweep()
    with pytest.raises(RuntimeError, match="yosys not found"):
        rr.run_requantize_characterization()

    # Restore shutil.which for mocked pipeline tests
    monkeypatch.undo()

    # 3. _measure_ecp5_fmax_mhz empty fmax & non-positive achieved guards
    def _fake_pnr_writer(fmax_payload: dict[str, object]):
        def _fake_run(cmd: list[str], **kwargs: object) -> None:
            cwd = kwargs.get("cwd")
            if cwd is not None:
                (Path(str(cwd)) / "pnr_report.json").write_text(
                    json.dumps(fmax_payload), encoding="utf-8"
                )

        return _fake_run

    monkeypatch.setattr(rr.subprocess, "run", _fake_pnr_writer({"fmax": {}}))
    with pytest.raises(RuntimeError, match="empty 'fmax' section"):
        rr._measure_ecp5_fmax_mhz(16, 64, 8)

    monkeypatch.setattr(
        rr.subprocess, "run", _fake_pnr_writer({"fmax": {"sys_clk": {"achieved": 0.0}}})
    )
    with pytest.raises(RuntimeError, match="invalid achieved fmax_mhz"):
        rr._measure_ecp5_fmax_mhz(16, 64, 8)

    # 3b. run_requantize_characterization 0-DSP violation & non-positive fmax guards
    def _fake_req_run(dsps: int, achieved: float):
        def _run(cmd: list[str], **kwargs: object) -> None:
            cwd = kwargs.get("cwd")
            if cwd is not None:
                (Path(str(cwd)) / "ecp5_pnr.json").write_text(
                    json.dumps({"fmax": {"sys_clk": {"achieved": achieved}}}), encoding="utf-8"
                )
            else:
                prompt = cmd[-1]
                stat_file = Path(prompt.split("tee -q -o ")[1].split(" stat -json")[0])
                top = prompt.split("-top ")[1].split()[0]
                cells = {"LUT6": 10, "FDRE": 5, "DSP48E1": dsps, "LUT4": 10, "TRELLIS_FF": 5}
                stat_file.write_text(
                    json.dumps({"modules": {f"\\{top}": {"num_cells_by_type": cells}}}),
                    encoding="utf-8",
                )

        return _run

    monkeypatch.setattr(rr.subprocess, "run", _fake_req_run(dsps=1, achieved=50.0))
    with pytest.raises(RuntimeError, match="0-DSP invariant violated"):
        rr.run_requantize_characterization(in_bits=16, out_bits=8, nodsp=True)

    monkeypatch.setattr(rr.subprocess, "run", _fake_req_run(dsps=0, achieved=0.0))
    with pytest.raises(RuntimeError, match="invalid measured ecp5_fmax_mhz"):
        rr.run_requantize_characterization(in_bits=16, out_bits=8, nodsp=True)
    monkeypatch.undo()

    # 4. render_markdown with nodsp=False (DSP-limited board fit), requantize_stage, and bad fmax
    sample_dsp = [
        SynthMetrics(1, 64, 128, 8, 64, 2697, 66, 1, 42.141, 1.0, nodsp=False, vendor="xc7")
    ]
    vrows_dsp = [
        VendorRowMetrics("xc7", "yosys xc7", "LUT6", 100, 66, 50, 1.56, None),  # DSP-capped!
        VendorRowMetrics("ecp5", "yosys ecp5", "LUT4", 3500, 66, 1, 54.68, 75.5),
        VendorRowMetrics("cyclonev", "yosys cycv", "ALM/LCELL", 2200, 66, 1, 34.37, None),
    ]
    req_stage = rr.RequantizeStageMetrics(48, 8, False, 2100, 117, 1, 5000, 117, 1, 55.25)
    md_dsp = render_markdown(sample_dsp, vrows_dsp, nodsp=False, requantize_stage=req_stage)
    assert "Opt-In 1 DSP/row Mode" in md_dsp
    assert "Output Requantization Stage" in md_dsp
    assert "55.25 MHz" in md_dsp

    bad_vrows = [
        VendorRowMetrics("xc7", "yosys xc7", "LUT6", 3300, 114, 0, 51.562, None),
        VendorRowMetrics("ecp5", "yosys ecp5", "LUT4", 4741, 114, 0, 74.078, None),
        VendorRowMetrics("cyclonev", "yosys cycv", "ALM/LCELL", 2677, 114, 0, 41.828, None),
    ]
    with pytest.raises(ValueError, match="expected measured ecp5 fmax_mhz > 0"):
        render_markdown(sample_dsp, bad_vrows, nodsp=True)

    # 5. check_baseline missing file & untracked config skip
    with pytest.raises(FileNotFoundError, match="Baseline file"):
        check_baseline(sample_dsp, tmp_path / "nonexistent_baseline.json")

    base_p = tmp_path / "base.json"
    base_p.write_text(json.dumps({"configs": []}), encoding="utf-8")
    check_baseline(sample_dsp, base_p)  # untracked key skipped cleanly

    # 6. run_sweep, run_multivendor_reference, and main() CLI modes (mocked synthesis calls)
    monkeypatch.setattr(rr, "SWEEP_CONFIGS", [(1, 64, 128, 8), (4, 64, 128, 8)])
    monkeypatch.setattr(rr, "_synthesize_single_row", lambda *_a, **_kw: (3300, 114, 0))
    monkeypatch.setattr(rr, "_synthesize_tile", lambda *_a, **_kw: (13745, 699, 0))
    monkeypatch.setattr(rr, "_measure_ecp5_fmax_mhz", lambda *_a, **_kw: 62.08)
    monkeypatch.setattr(
        rr,
        "run_requantize_characterization",
        lambda **_kw: rr.RequantizeStageMetrics(48, 8, True, 2640, 117, 0, 6017, 117, 0, 49.08),
    )

    rep_dir = tmp_path / "reports"
    j_path = rep_dir / "resources.json"
    m_path = rep_dir / "resources.md"
    b_path = rep_dir / "baseline.json"
    monkeypatch.setattr(rr, "REPORTS_DIR", rep_dir)
    monkeypatch.setattr(rr, "DEFAULT_JSON", j_path)
    monkeypatch.setattr(rr, "DEFAULT_MD", m_path)
    monkeypatch.setattr(rr, "DEFAULT_BASELINE", b_path)

    assert rr.main(["--update-baseline", "--check-baseline"]) == 0
    assert j_path.exists() and m_path.exists() and b_path.exists()
    assert rr.main(["--check"]) == 0
    assert "passed" in capsys.readouterr().out

    # Drift detection in --check for resources.md and resources.json
    m_orig = m_path.read_text(encoding="utf-8")
    m_path.write_text(m_orig + "drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="deterministic report invariant violated"):
        rr.main(["--check"])
    m_path.write_text(m_orig, encoding="utf-8")

    j_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="deterministic report invariant violated"):
        rr.main(["--check"])
