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
