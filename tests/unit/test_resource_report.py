"""Unit tests for Yosys resource characterization and -nodsp mode (Task 2.6, ADR-004)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.resource_report import (
    SynthMetrics,
    _synthesize_single_row,
    check_baseline,
    render_markdown,
)


def test_synthesize_single_row_dsp_vs_nodsp() -> None:
    """Verify default synthesis uses 1 DSP48E1/row while -nodsp uses 0 DSP48E1/row."""
    luts_dsp, ffs_dsp, dsps_dsp = _synthesize_single_row(
        lanes=16, group_size=64, act_bits=8, nodsp=False
    )
    luts_nodsp, ffs_nodsp, dsps_nodsp = _synthesize_single_row(
        lanes=16, group_size=64, act_bits=8, nodsp=True
    )

    assert dsps_dsp == 1, f"Expected 1 DSP48E1 in default mode, got {dsps_dsp}"
    assert dsps_nodsp == 0, f"Expected 0 DSP48E1 in -nodsp mode, got {dsps_nodsp}"
    assert luts_nodsp > luts_dsp, "Expected -nodsp LUT multiplier to use more LUT6 than DSP mode"
    # DSP48E1 absorbs internal pipeline/accumulator registers (48 FFs), so -nodsp uses >= FFs
    assert ffs_nodsp >= ffs_dsp


def test_render_markdown_and_baseline_check(tmp_path: Path) -> None:
    """Verify markdown rendering and baseline regression gate."""
    sample = [
        SynthMetrics(
            rows=1,
            lanes=64,
            group_size=128,
            act_bits=8,
            macs_per_cycle=64,
            luts=2697,
            ffs=72,
            dsps=0,
            luts_per_mac=42.141,
            dsps_per_row=0.0,
            nodsp=True,
        )
    ]
    md = render_markdown(sample, nodsp=True)
    assert "-nodsp" in md
    assert "0 DSPs via Yosys -nodsp LUT multipliers" in md

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
                        "luts_per_mac": 40.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # 42.141 vs 40.0 is +5.35% (<= 10%), must pass
    check_baseline(sample, baseline_file, max_regression=0.10)

    # With a 2% threshold, must raise RuntimeError
    with pytest.raises(RuntimeError, match="Resource regression detected"):
        check_baseline(sample, baseline_file, max_regression=0.02)
