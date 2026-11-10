"""Unit tests for GroupAccumulateScale and TernaryRow (Task 2.2)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from migen.fhdl import verilog

from setun.accelerators.group_scale import GroupAccumulateScale, TernaryRow
from setun.ref.ops import group_accumulate, row_dot, ternary_dot
from tests.sim_helper import run_migen_sim


def _pack_trits_signed(trits: np.ndarray) -> int:
    code_map = {0: 0b00, 1: 0b01, -1: 0b10}
    val = 0
    for i, t in enumerate(trits):
        val |= code_map[int(t)] << (2 * i)
    return val


def _pack_acts_int8(acts: np.ndarray) -> int:
    val = 0
    for i, a in enumerate(acts):
        val |= (int(a) & 0xFF) << (8 * i)
    return val


@pytest.mark.parametrize(
    "group_size,lanes",
    [(64, 16), (64, 64), (128, 16), (128, 64), (128, 128), (256, 16), (256, 64), (256, 128)],
)
@pytest.mark.parametrize("k", [1, 7, 127, 128, 129, 2048, 6144])
def test_group_accumulate_scale_matrix_parity(k: int, group_size: int, lanes: int) -> None:
    """Verify bit-exact parity vs row_dot across K x G x L matrix (Task 2.2)."""
    rng = np.random.default_rng(k * 1000 + group_size * 10 + lanes)
    trits = rng.choice([-1, 0, 1], size=k).astype(np.int8)
    acts = rng.integers(-128, 128, size=k, dtype=np.int8)
    n_groups = (k + group_size - 1) // group_size
    scales = rng.integers(0, 65536, size=n_groups, dtype=np.uint16)

    expected = row_dot(trits, acts, scales, group_size=group_size, lanes=lanes, shift=4)

    # Pad ragged K to a multiple of group_size just as the feeder/unpacker does
    k_pad = n_groups * group_size
    w_pad = np.zeros(k_pad, dtype=np.int8)
    x_pad = np.zeros(k_pad, dtype=np.int8)
    w_pad[:k] = trits
    x_pad[:k] = acts

    beats_per_group = group_size // lanes
    total_beats = n_groups * beats_per_group
    beat_dots = [
        ternary_dot(w_pad[b * lanes : (b + 1) * lanes], x_pad[b * lanes : (b + 1) * lanes])
        for b in range(total_beats)
    ]

    dut = GroupAccumulateScale(group_size=group_size, lanes=lanes, act_bits=8, shift=4)

    def tb():
        for b_idx, d_val in enumerate(beat_dots):
            g_idx = b_idx // beats_per_group
            is_last = 1 if (b_idx == total_beats - 1) else 0
            yield dut.dot_in.eq(d_val)
            yield dut.scale_u16.eq(int(scales[g_idx]))
            yield dut.in_valid.eq(1)
            yield dut.row_last.eq(is_last)
            yield
        yield dut.in_valid.eq(0)
        yield dut.row_last.eq(0)
        yield
        assert (yield dut.out_valid) == 1
        assert (yield dut.result) == expected

    run_migen_sim(dut, tb)


def test_ternary_row_end_to_end_ragged_k() -> None:
    """Verify full TernaryRow (TernaryDot + GroupAccumulateScale) on ragged K=129."""
    k, group_size, lanes = 129, 128, 64
    rng = np.random.default_rng(1958)
    trits = rng.choice([-1, 0, 1], size=k).astype(np.int8)
    acts = rng.integers(-128, 128, size=k, dtype=np.int8)
    n_groups = (k + group_size - 1) // group_size
    scales = rng.integers(0, 65536, size=n_groups, dtype=np.uint16)

    expected = row_dot(trits, acts, scales, group_size=group_size, lanes=lanes, shift=0)

    k_pad = n_groups * group_size
    w_pad = np.zeros(k_pad, dtype=np.int8)
    x_pad = np.zeros(k_pad, dtype=np.int8)
    w_pad[:k] = trits
    x_pad[:k] = acts

    beats_per_group = group_size // lanes
    total_beats = n_groups * beats_per_group

    dut = TernaryRow(group_size=group_size, lanes=lanes, act_bits=8, shift=0)

    def tb():
        for b_idx in range(total_beats):
            g_idx = b_idx // beats_per_group
            w_slice = w_pad[b_idx * lanes : (b_idx + 1) * lanes]
            x_slice = x_pad[b_idx * lanes : (b_idx + 1) * lanes]
            yield dut.trits_packed.eq(_pack_trits_signed(w_slice))
            yield dut.acts_packed.eq(_pack_acts_int8(x_slice))
            yield dut.scale_u16.eq(int(scales[g_idx]))
            yield dut.in_valid.eq(1)
            yield dut.row_last.eq(1 if b_idx == total_beats - 1 else 0)
            yield
        yield dut.in_valid.eq(0)
        yield dut.row_last.eq(0)
        yield
        assert (yield dut.out_valid) == 1
        assert (yield dut.result) == expected
        # Ensure group_accumulate helper is also exercised
        assert group_accumulate(w_pad[:group_size], x_pad[:group_size], lanes=lanes) == sum(
            ternary_dot(w_pad[i * lanes : (i + 1) * lanes], x_pad[i * lanes : (i + 1) * lanes])
            for i in range(beats_per_group)
        )

    run_migen_sim(dut, tb)


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys not installed")
def test_ternary_row_yosys_exactly_one_multiplier() -> None:
    """Verify TernaryRow synthesizes with exactly 1 multiplier ($mul) per row."""
    dut = TernaryRow(group_size=128, lanes=64, act_bits=8)
    with tempfile.TemporaryDirectory() as tmpdir:
        v_path = Path(tmpdir) / "ternary_row.v"
        json_path = Path(tmpdir) / "stat.json"
        v_out = verilog.convert(dut, ios=dut.ios, name="ternary_row")
        v_out.write(str(v_path))

        cmd = [
            "yosys",
            "-q",
            "-p",
            f"read_verilog {v_path}; proc; opt; tee -q -o {json_path} stat -json",
        ]
        subprocess.run(cmd, check=True)
        stat_data = json.loads(json_path.read_text())
        cells = stat_data["modules"]["\\ternary_row"]["num_cells_by_type"]
        mul_count = cells.get("$mul", 0)
        assert mul_count == 1, f"Expected exactly 1 $mul per row, got {mul_count} ({cells})"
