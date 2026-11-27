"""Unit tests for Yosys/nextpnr ECP5 + Cyclone V resource & timing characterization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.resource_report as rr
from scripts.resource_report import (
    SynthMetrics,
    VendorRowMetrics,
    _synthesize_single_row,
    check_baseline,
    render_markdown,
)


@pytest.mark.slow
def test_synthesize_single_row_nodsp_default_vs_optin_dsp() -> None:
    """Verify default -nodsp uses 0 DSPs while opt-in nodsp=False uses 1 DSP/row on ECP5 & CycV."""
    for vendor in ("ecp5", "cyclonev"):
        logic_nodsp, ffs_nodsp, dsps_nodsp = _synthesize_single_row(
            lanes=16, group_size=64, act_bits=8, nodsp=True, vendor=vendor
        )
        logic_dsp, ffs_dsp, dsps_dsp = _synthesize_single_row(
            lanes=16, group_size=64, act_bits=8, nodsp=False, vendor=vendor
        )
        assert dsps_nodsp == 0, f"Expected 0 DSPs on {vendor} in -nodsp mode, got {dsps_nodsp}"
        assert dsps_dsp == 1, f"Expected 1 DSP on {vendor} in --use-dsp mode, got {dsps_dsp}"
        assert logic_nodsp > logic_dsp
        assert ffs_nodsp >= ffs_dsp


def test_no_xilinx_xc7_vendor_or_artix_boards_in_policy() -> None:
    """ADR-007 policy: only Lattice ECP5 and Intel Cyclone V are targeted (no AMD/Xilinx xc7)."""
    board_vendors = {b["vendor"] for b in rr.CANDIDATE_BOARDS}
    assert board_vendors == {"ecp5", "cyclonev"}, (
        f"[test_resource_report:ADR-007] expected ecp5 and cyclonev boards, got {board_vendors}"
    )
    board_names = " ".join(str(b["name"]) for b in rr.CANDIDATE_BOARDS)
    assert "Artix" not in board_names and "XC7" not in board_names
    assert "ECPIX-5" in board_names and "DE10-Nano" in board_names

    with pytest.raises(ValueError, match="Unsupported synthesis vendor"):
        _synthesize_single_row(16, 64, 8, nodsp=True, vendor="xc7")
    with pytest.raises(ValueError, match="Unsupported synthesis vendor"):
        rr._synthesize_tile(2, 16, 64, 8, nodsp=True, vendor="xc7")


def test_governing_fmax_uses_accel_min_and_marks_cyclonev_synth_only(tmp_path: Path) -> None:
    """Verify throughput uses governing min Fmax (never isolated row 62.08 MHz) and N/A for CycV."""
    sample = [
        SynthMetrics(
            rows=1,
            lanes=64,
            group_size=128,
            act_bits=8,
            macs_per_cycle=64,
            luts=4741,
            ffs=114,
            dsps=0,
            luts_per_mac=74.078,
            dsps_per_row=0.0,
            nodsp=True,
            vendor="ecp5",
            basis="SYNTHESIZED (TernaryRow)",
        ),
        SynthMetrics(
            rows=1,
            lanes=64,
            group_size=128,
            act_bits=8,
            macs_per_cycle=64,
            luts=2677,
            ffs=114,
            dsps=0,
            luts_per_mac=41.828,
            dsps_per_row=0.0,
            nodsp=True,
            vendor="cyclonev",
            basis="SYNTHESIZED (TernaryRow)",
        ),
    ]
    vrows = [
        VendorRowMetrics(
            "ecp5", "yosys synth_ecp5 -nodsp + nextpnr-ecp5", "LUT4", 4741, 114, 0, 74.078, 62.08
        ),
        VendorRowMetrics(
            "cyclonev",
            "yosys synth_intel_alm -family cyclonev -nodsp",
            "ALUT",
            2677,
            114,
            0,
            41.828,
            None,
        ),
    ]
    req_stage = rr.RequantizeStageMetrics(
        in_bits=48,
        out_bits=8,
        nodsp=True,
        ecp5_lut4=6017,
        ecp5_ffs=117,
        ecp5_dsps=0,
        ecp5_fmax_mhz=49.08,
        cyclonev_aluts=2100,
        cyclonev_ffs=117,
        cyclonev_dsps=0,
    )
    accel_soc = rr.AccelBramMetrics(
        rows=4,
        lanes=64,
        group_size=128,
        max_k=2048,
        max_rows=2048,
        nodsp=True,
        ecp5_lut4=40072,
        ecp5_ffs=2938,
        ecp5_dp16kd=19,
        ecp5_dsps=0,
        ecp5_fmax_mhz=41.50,
        cyclonev_aluts=18500,
        cyclonev_ffs=2910,
        cyclonev_m10k=24,
        cyclonev_dsps=0,
    )

    gov_fmax, gov_stage = rr.resolve_governing_ecp5_fmax(
        row_fmax_mhz=62.08,
        requantize_stage=req_stage,
        accel_soc=accel_soc,
    )
    assert gov_fmax == 41.50 and "TernaryAccel" in gov_stage

    # Also verify min selects Requantize if Requantize happens to be slower than TernaryAccel
    slower_req = rr.RequantizeStageMetrics(48, 8, True, 6017, 117, 0, 38.25, 2100, 117, 0)
    gov_req, stage_req = rr.resolve_governing_ecp5_fmax(62.08, slower_req, accel_soc)
    assert gov_req == 38.25 and "Requantize" in stage_req

    md = render_markdown(
        sample,
        vrows,
        nodsp=True,
        requantize_stage=req_stage,
        accel_soc=accel_soc,
    )
    # 1. Section 3 ceiling MUST use governing 41.50 MHz, NOT isolated TernaryRow 62.08 MHz
    assert "at 41.50 MHz" in md
    assert "at 62.08 MHz" not in md
    # At 41.50 MHz & 1.6 GB/s (ECPIX-5): round(1.6e9 / (0.25 * 41.50e6)) = 154 MACs/cyc (not 103)
    assert "| 154 |" in md
    assert "| 103 |" not in md
    # 2. Synth-only Cyclone V rows MUST show N/A (synth-only, pending Quartus STA)
    assert md.count("| N/A (synth-only, pending Quartus STA) |") == sum(
        1 for b in rr.CANDIDATE_BOARDS if b["vendor"] == "cyclonev"
    )
    # 3. Section 5 MUST report measured TernaryAccel ECP5 Fmax and Cyclone V M10K BRAMs
    assert "41.50 MHz" in md
    assert "M10K" in md and "24" in md
    assert "XC7" not in md and "LUT6" not in md and "DSP48" not in md

    # 4. Baseline check keyed by (vendor, rows, lanes, group_size, act_bits)
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(
        json.dumps(
            {
                "synth_flow": "test",
                "configs": [
                    {
                        "vendor": "ecp5",
                        "rows": 1,
                        "lanes": 64,
                        "group_size": 128,
                        "act_bits": 8,
                        "luts_per_mac": 72.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    check_baseline(sample, baseline_file, max_regression=0.10)
    with pytest.raises(RuntimeError, match="Resource regression detected"):
        check_baseline(sample, baseline_file, max_regression=0.01)


@pytest.mark.slow
def test_ecp5_nextpnr_measured_fmax_and_extrapolated_labels() -> None:
    """Measure real nextpnr-ecp5 row Fmax and verify [EXTRAPOLATED from R=1] labels."""
    fmax = rr._measure_ecp5_fmax_mhz(lanes=16, group_size=64, act_bits=8, nodsp=True)
    assert isinstance(fmax, float) and 10.0 < fmax < 500.0 and fmax != 100.0

    sample = [
        SynthMetrics(
            1, 64, 128, 8, 64, 4741, 114, 0, 74.078, 0.0, True, "ecp5", "SYNTHESIZED (R=1)"
        ),
        SynthMetrics(
            4, 64, 128, 8, 256, 18964, 456, 0, 74.078, 0.0, True, "ecp5", "[EXTRAPOLATED from R=1]"
        ),
        SynthMetrics(
            8, 64, 128, 8, 512, 37928, 912, 0, 74.078, 0.0, True, "ecp5", "[EXTRAPOLATED from R=1]"
        ),
    ]
    vrows = [
        VendorRowMetrics("ecp5", "yosys+nextpnr-ecp5", "LUT4", 4741, 114, 0, 74.078, fmax),
        VendorRowMetrics("cyclonev", "yosys cyclonev", "ALUT", 2677, 114, 0, 41.828, None),
    ]
    md = render_markdown(sample, vrows, nodsp=True)
    assert md.count("[EXTRAPOLATED from R=1]") >= 2
    assert f"{fmax:.2f}" in md
    assert "at 100 MHz" not in md
    assert "N/A (synth-only, pending Quartus STA)" in md


def test_synthesize_row_and_tile_elaboration_and_vendor_cell_parsers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify row/tile/accel Verilog elaboration, ECP5+CycloneV cell parsers, and PnR Fmax."""
    with pytest.raises(ValueError, match="Unsupported synthesis vendor"):
        rr._synthesize_single_row(16, 64, 8, nodsp=True, vendor="unknown_fpga")
    with pytest.raises(ValueError, match="Unsupported synthesis vendor"):
        rr._synthesize_tile(2, 16, 64, 8, nodsp=True, vendor="unknown_fpga")

    def _mock_yosys_and_pnr(cmd: list[str], **kwargs: object) -> None:
        cwd = kwargs.get("cwd")
        if cwd is not None:
            out_dir = Path(str(cwd))
            for rep_name in ("pnr_report.json", "ecp5_pnr.json", "accel_ecp5_pnr.json"):
                (out_dir / rep_name).write_text(
                    json.dumps({"fmax": {"sys_clk": {"achieved": 44.75}}}),
                    encoding="utf-8",
                )
            return

        prompt = cmd[-1]
        if "tee -q -o " not in prompt:
            return
        stat_file = Path(prompt.split("tee -q -o ")[1].split(" stat -json")[0])
        top = prompt.split("-top ")[1].split()[0]
        if "synth_ecp5" in prompt:
            cells = {"LUT4": 512, "TRELLIS_FF": 64, "DP16KD": 4}
        else:
            cells = {"MISTRAL_ALUT4": 400, "MISTRAL_FF": 64, "MISTRAL_M10K": 6}
        stat_file.write_text(
            json.dumps({"modules": {f"\\{top}": {"num_cells_by_type": cells}}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(rr.subprocess, "run", _mock_yosys_and_pnr)
    assert rr._synthesize_single_row(16, 64, 8, nodsp=True, vendor="ecp5") == (512, 64, 0)
    assert rr._synthesize_single_row(16, 64, 8, nodsp=True, vendor="cyclonev") == (400, 64, 0)

    orig_tile = rr.TernaryTile
    monkeypatch.setattr(
        rr,
        "TernaryTile",
        lambda **kw: orig_tile(**{**kw, "max_k": kw["group_size"]}),
    )
    assert rr._synthesize_tile(2, 16, 64, 8, nodsp=True, vendor="ecp5") == (512, 64, 0)
    assert rr._synthesize_tile(2, 16, 64, 8, nodsp=True, vendor="cyclonev") == (400, 64, 0)

    assert rr._measure_ecp5_fmax_mhz(16, 64, 8, nodsp=True) == 44.75
    req_m = rr.run_requantize_characterization(in_bits=16, out_bits=8, nodsp=True)
    assert (req_m.ecp5_lut4, req_m.ecp5_ffs, req_m.ecp5_dsps) == (512, 64, 0)
    assert (req_m.cyclonev_aluts, req_m.cyclonev_ffs, req_m.cyclonev_dsps) == (400, 64, 0)
    assert req_m.ecp5_fmax_mhz == 44.75

    acc_m = rr.run_accel_bram_characterization(
        rows=2, lanes=16, group_size=64, beat_bits=64, max_k=64, max_rows=64, nodsp=True
    )
    assert (acc_m.ecp5_lut4, acc_m.ecp5_dp16kd, acc_m.ecp5_dsps) == (512, 4, 0)
    assert (acc_m.cyclonev_aluts, acc_m.cyclonev_m10k, acc_m.cyclonev_dsps) == (400, 6, 0)
    assert acc_m.ecp5_fmax_mhz == 44.75


def test_resource_report_pipelines_error_guards_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cover _resolve_nextpnr_ecp5, error guards, DSP-bound render, check_baseline, and main()."""
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
    with pytest.raises(RuntimeError, match="yosys not in PATH"):
        rr.run_accel_bram_characterization()

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
                cells = {
                    "LUT4": 10,
                    "TRELLIS_FF": 5,
                    "MULT18X18D": dsps,
                    "MISTRAL_ALUT4": 10,
                    "MISTRAL_FF": 5,
                }
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

    # 3c. run_accel_bram_characterization 0-DSP, 0-BRAM, empty fmax, and <=0 fmax guards
    def _fake_accel_run(dsps: int, brams: int, fmax_payload: dict[str, object]):
        def _run(cmd: list[str], **kwargs: object) -> None:
            cwd = kwargs.get("cwd")
            if cwd is not None:
                (Path(str(cwd)) / "accel_ecp5_pnr.json").write_text(
                    json.dumps(fmax_payload), encoding="utf-8"
                )
                return
            prompt = cmd[-1]
            stat_file = Path(prompt.split("tee -q -o ")[1].split(" stat -json")[0])
            top = prompt.split("-top ")[1].split()[0]
            if "synth_ecp5" in prompt:
                cells = {"LUT4": 100, "TRELLIS_FF": 20, "DP16KD": brams, "MULT18X18D": dsps}
            else:
                cells = {"MISTRAL_ALUT4": 100, "MISTRAL_FF": 20, "MISTRAL_M10K": brams}
            stat_file.write_text(
                json.dumps({"modules": {f"\\{top}": {"num_cells_by_type": cells}}}),
                encoding="utf-8",
            )

        return _run

    ok_fmax = {"fmax": {"sys_clk": {"achieved": 42.0}}}
    monkeypatch.setattr(
        rr.subprocess, "run", _fake_accel_run(dsps=1, brams=4, fmax_payload=ok_fmax)
    )
    with pytest.raises(RuntimeError, match="expected 0 DSPs"):
        rr.run_accel_bram_characterization(
            rows=2, lanes=16, group_size=64, beat_bits=64, max_k=64, max_rows=64, nodsp=True
        )
    monkeypatch.setattr(
        rr.subprocess, "run", _fake_accel_run(dsps=0, brams=0, fmax_payload=ok_fmax)
    )
    with pytest.raises(RuntimeError, match="synchronous BRAM inference failed"):
        rr.run_accel_bram_characterization(
            rows=2, lanes=16, group_size=64, beat_bits=64, max_k=64, max_rows=64, nodsp=True
        )
    monkeypatch.setattr(
        rr.subprocess, "run", _fake_accel_run(dsps=0, brams=4, fmax_payload={"fmax": {}})
    )
    with pytest.raises(RuntimeError, match="empty 'fmax' section"):
        rr.run_accel_bram_characterization(
            rows=2, lanes=16, group_size=64, beat_bits=64, max_k=64, max_rows=64, nodsp=True
        )
    monkeypatch.setattr(
        rr.subprocess,
        "run",
        _fake_accel_run(dsps=0, brams=4, fmax_payload={"fmax": {"sys_clk": {"achieved": 0.0}}}),
    )
    with pytest.raises(RuntimeError, match="invalid measured ecp5_fmax_mhz"):
        rr.run_accel_bram_characterization(
            rows=2, lanes=16, group_size=64, beat_bits=64, max_k=64, max_rows=64, nodsp=True
        )
    monkeypatch.undo()

    # 4. render_markdown with nodsp=False (DSP-limited board fit), requantize_stage, and bad fmax
    sample_dsp = [
        SynthMetrics(1, 64, 128, 8, 64, 1900, 114, 1, 29.688, 1.0, nodsp=False, vendor="ecp5")
    ]
    vrows_dsp = [
        VendorRowMetrics("ecp5", "yosys ecp5", "LUT4", 100, 114, 50, 1.56, 75.5),  # DSP-capped!
        VendorRowMetrics("cyclonev", "yosys cycv", "ALUT", 1500, 114, 1, 23.438, None),
    ]
    req_stage = rr.RequantizeStageMetrics(48, 8, False, 5000, 117, 1, 55.25, 1800, 117, 1)
    md_dsp = render_markdown(sample_dsp, vrows_dsp, nodsp=False, requantize_stage=req_stage)
    assert "Opt-In 1 DSP/row Mode" in md_dsp
    assert "Output Requantization Stage" in md_dsp
    assert "55.25 MHz" in md_dsp

    bad_vrows = [
        VendorRowMetrics("ecp5", "yosys ecp5", "LUT4", 4741, 114, 0, 74.078, None),
        VendorRowMetrics("cyclonev", "yosys cycv", "ALUT", 2677, 114, 0, 41.828, None),
    ]
    with pytest.raises(ValueError, match="expected measured ecp5 fmax_mhz > 0"):
        render_markdown(sample_dsp, bad_vrows, nodsp=True)

    # 5. check_baseline missing file & untracked config skip
    with pytest.raises(FileNotFoundError, match="Baseline file"):
        check_baseline(sample_dsp, tmp_path / "nonexistent_baseline.json")

    base_p = tmp_path / "base.json"
    base_p.write_text(json.dumps({"configs": []}), encoding="utf-8")
    check_baseline(sample_dsp, base_p)

    # 6. run_sweep, run_multivendor_reference, and main() CLI modes (mocked synthesis calls)
    monkeypatch.setattr(rr, "SWEEP_CONFIGS", [(1, 64, 128, 8), (4, 64, 128, 8)])
    monkeypatch.setattr(rr, "_synthesize_single_row", lambda *_a, **_kw: (4741, 114, 0))
    monkeypatch.setattr(rr, "_synthesize_tile", lambda *_a, **_kw: (19200, 699, 0))
    monkeypatch.setattr(rr, "_measure_ecp5_fmax_mhz", lambda *_a, **_kw: 62.08)
    monkeypatch.setattr(
        rr,
        "run_requantize_characterization",
        lambda **_kw: rr.RequantizeStageMetrics(48, 8, True, 6017, 117, 0, 49.08, 2100, 117, 0),
    )
    monkeypatch.setattr(
        rr,
        "run_accel_bram_characterization",
        lambda **_kw: rr.AccelBramMetrics(
            4, 64, 128, 2048, 2048, True, 40072, 2938, 19, 0, 41.50, 18500, 2910, 24, 0
        ),
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
    saved_json = json.loads(j_path.read_text(encoding="utf-8"))
    assert saved_json["governing_ecp5_fmax_mhz"] == 41.50
    assert rr.main(["--check"]) == 0
    assert "41.50 MHz" in capsys.readouterr().out

    # Drift detection in --check for resources.md and resources.json
    m_orig = m_path.read_text(encoding="utf-8")
    m_path.write_text(m_orig + "drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="deterministic report invariant violated"):
        rr.main(["--check"])
    m_path.write_text(m_orig, encoding="utf-8")

    j_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="deterministic report invariant violated"):
        rr.main(["--check"])


@pytest.mark.slow
def test_ternary_accel_bram_and_fmax_characterization() -> None:
    """Verify TernaryAccel infers ECP5 DP16KD + CycloneV M10K BRAMs and real nextpnr-ecp5 Fmax."""
    bram_metrics = rr.run_accel_bram_characterization(
        rows=4, lanes=16, group_size=64, beat_bits=64, max_k=512, max_rows=512, nodsp=True
    )
    assert bram_metrics.ecp5_dsps == 0 and bram_metrics.cyclonev_dsps == 0
    assert bram_metrics.ecp5_dp16kd >= 2, f"Expected ECP5 DP16KD inference, got {bram_metrics}"
    assert bram_metrics.cyclonev_m10k >= 2, f"Expected CycloneV M10K inference, got {bram_metrics}"
    assert 10.0 < bram_metrics.ecp5_fmax_mhz < 500.0 and bram_metrics.ecp5_fmax_mhz != 100.0


def test_synth_cache_roundtrip_and_disable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify content-addressed synthesis cache hit/miss and SETUN_DISABLE_SYNTH_CACHE."""
    import scripts.resource_report as rr

    monkeypatch.setattr(rr, "SYNTH_CACHE_DIR", tmp_path / "synth_cache")
    monkeypatch.setenv("SETUN_FORCE_SYNTH_CACHE_TEST", "1")
    monkeypatch.delenv("SETUN_DISABLE_SYNTH_CACHE", raising=False)

    ckey = rr._synth_cache_key("unit:tag", "module top(); endmodule")
    assert rr._synth_cache_load(ckey) is None

    rr._synth_cache_store(ckey, [100, 50, 0])
    assert rr._synth_cache_load(ckey) == [100, 50, 0]

    monkeypatch.setenv("SETUN_DISABLE_SYNTH_CACHE", "1")
    assert rr._synth_cache_load(ckey) is None
    ckey2 = rr._synth_cache_key("unit:tag2", "module top2(); endmodule")
    rr._synth_cache_store(ckey2, [1, 2, 0])
    assert not ckey2.exists()


def test_board_fit_cyclonev_uses_aluts_and_rejects_unit_mismatch() -> None:
    """Verify Cyclone V boards use 83,820 ALUTs (25 rows) and guard unit mismatches."""
    cycv_boards = [b for b in rr.CANDIDATE_BOARDS if b["vendor"] == "cyclonev"]
    assert len(cycv_boards) >= 2
    for b in cycv_boards:
        assert b["logic_unit"] == "ALUT"
        assert b["logic_cap"] == 83_820

    vrows = [
        VendorRowMetrics("ecp5", "ecp5", "LUT4", 4741, 114, 0, 74.078, 62.08),
        VendorRowMetrics("cyclonev", "cyclonev", "ALUT", 2677, 114, 0, 41.828, None),
    ]
    md = render_markdown([], vrows, nodsp=True)
    assert "| 83,820 ALUT | 112 | 3.2 | 25 | 1,600 |" in md

    bad_vrows = [
        VendorRowMetrics("ecp5", "ecp5", "LUT4", 4741, 114, 0, 74.078, 62.08),
        VendorRowMetrics("cyclonev", "cyclonev", "ALM", 2677, 114, 0, 41.828, None),
    ]
    with pytest.raises(ValueError, match="logic_unit mismatch"):
        render_markdown([], bad_vrows, nodsp=True)
