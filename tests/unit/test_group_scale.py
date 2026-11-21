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


@pytest.mark.slow
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


@pytest.mark.parametrize(
    "k,group_size,lanes,shift",
    [
        (64, 64, 64, 3),  # 1 beat/group (single-beat row)
        (128, 64, 16, 4),  # 4 beats/group, 2 groups
        (129, 128, 64, 5),  # ragged K across 2 groups, 2 beats/group
        (256, 128, 16, 0),  # 8 beats/group, 2 groups, unshifted
    ],
)
def test_group_accumulate_scale_mid_row_in_valid_bubbles(
    k: int, group_size: int, lanes: int, shift: int
) -> None:
    """Task C1.1a: Mid-row 1/multi-cycle bubbles and back-to-back rows on GroupAccumulateScale."""
    rng = np.random.default_rng(195800 + k * 31 + group_size * 7 + lanes + shift)
    n_rows = 16
    n_groups = (k + group_size - 1) // group_size
    k_pad = n_groups * group_size
    beats_per_group = group_size // lanes
    total_beats = n_groups * beats_per_group

    dut = GroupAccumulateScale(group_size=group_size, lanes=lanes, act_bits=8, shift=shift)
    poison_dot = -(1 << (dut.dot_bits - 1))

    # Build cycle-accurate stimulus with forced 0-cycle, 1-cycle, and multi-cycle gaps
    # (both intra-row mid-group/cross-group and inter-row back-to-back).
    # Each entry: (in_valid, dot_in, scale_u16, row_last, completed_row_expected_or_none, row_id)
    schedule: list[tuple[int, int, int, int, int | None, int]] = []
    gap_pool = np.array([0, 0, 1, 1, 2, 3, 4], dtype=np.int64)

    for r_idx in range(n_rows):
        trits = rng.choice([-1, 0, 1], size=k).astype(np.int8)
        acts = rng.integers(-128, 128, size=k, dtype=np.int8)
        scales = rng.integers(0, 65536, size=n_groups, dtype=np.uint16)
        # Exercise corner rows: row 0 = -128 saturation, row 1 = zero scales
        if r_idx == 0:
            trits.fill(-1)
            acts.fill(-128)
            scales.fill(65535)
        elif r_idx == 1:
            scales.fill(0)

        expected_row = row_dot(trits, acts, scales, group_size=group_size, lanes=lanes, shift=shift)

        w_pad = np.zeros(k_pad, dtype=np.int8)
        x_pad = np.zeros(k_pad, dtype=np.int8)
        w_pad[:k] = trits
        x_pad[:k] = acts

        # Inter-row gap before row r_idx: force 0 on even rows > 0 (back-to-back) and >0 on odd
        inter_gap = 0 if (r_idx > 0 and r_idx % 2 == 0) else int(rng.integers(1, 5))
        for _ in range(inter_gap):
            schedule.append((0, poison_dot, 0xFFFF, 1, None, r_idx))

        for b_idx in range(total_beats):
            if b_idx > 0:
                # Intra-row mid-row bubble: guarantee 1-cycle and multi-cycle gaps occur
                if b_idx == 1 and r_idx % 3 == 0:
                    intra_gap = 1
                elif b_idx == total_beats - 1 and r_idx % 3 == 1:
                    intra_gap = 3
                else:
                    intra_gap = int(rng.choice(gap_pool))
                for _ in range(intra_gap):
                    schedule.append((0, poison_dot, 0xFFFF, 1, None, r_idx))

            g_idx = b_idx // beats_per_group
            d_val = ternary_dot(
                w_pad[b_idx * lanes : (b_idx + 1) * lanes],
                x_pad[b_idx * lanes : (b_idx + 1) * lanes],
            )
            is_last = 1 if (b_idx == total_beats - 1) else 0
            exp_after_edge = expected_row if is_last else None
            schedule.append((1, d_val, int(scales[g_idx]), is_last, exp_after_edge, r_idx))

    for _ in range(4):
        schedule.append((0, poison_dot, 0xFFFF, 1, None, n_rows))

    def tb():
        seen_rows = 0
        for cyc, (v_in, d_val, s_val, r_last, _exp_now, _r_id) in enumerate(schedule):
            if cyc >= 2:
                prev_exp = schedule[cyc - 2][4]
                prev_row = schedule[cyc - 2][5]
                exp_valid = 1 if prev_exp is not None else 0
                got_valid = yield dut.out_valid
                assert got_valid == exp_valid, (
                    f"[test_group_scale:GroupAccumulateScale] out_valid invariant violated at "
                    f"cycle {cyc} (row={prev_row}, K={k}, G={group_size}, L={lanes}): "
                    f"expected {exp_valid}, got {got_valid}"
                )
                if exp_valid == 1:
                    got_res = yield dut.result
                    assert got_res == prev_exp, (
                        f"[test_group_scale:GroupAccumulateScale] result parity violated at "
                        f"cycle {cyc} (row={prev_row}, K={k}, G={group_size}, L={lanes}, "
                        f"shift={shift}): expected {prev_exp}, got {got_res}"
                    )
                    seen_rows += 1

            yield dut.dot_in.eq(d_val)
            yield dut.scale_u16.eq(s_val)
            yield dut.in_valid.eq(v_in)
            yield dut.row_last.eq(r_last)
            yield

        assert seen_rows == n_rows, (
            f"[test_group_scale:GroupAccumulateScale] completed row count mismatch: "
            f"expected {n_rows}, got {seen_rows}"
        )

    run_migen_sim(dut, tb)


@pytest.mark.slow
def test_ternary_row_mid_row_bubbles_and_back_to_back() -> None:
    """Task C1.1a: Full TernaryRow bit-exact parity under mid-row bubbles and back-to-back rows."""
    k, group_size, lanes, shift = 129, 64, 16, 4
    rng = np.random.default_rng(20261120)
    n_rows = 12
    n_groups = (k + group_size - 1) // group_size
    k_pad = n_groups * group_size
    beats_per_group = group_size // lanes
    total_beats = n_groups * beats_per_group

    dut = TernaryRow(group_size=group_size, lanes=lanes, act_bits=8, shift=shift)
    poison_trits = _pack_trits_signed(np.full(lanes, -1, dtype=np.int8))
    poison_acts = _pack_acts_int8(np.full(lanes, -128, dtype=np.int8))

    schedule: list[tuple[int, int, int, int, int, int | None, int]] = []
    for r_idx in range(n_rows):
        trits = rng.choice([-1, 0, 1], size=k).astype(np.int8)
        acts = rng.integers(-128, 128, size=k, dtype=np.int8)
        scales = rng.integers(0, 65536, size=n_groups, dtype=np.uint16)
        if r_idx == 0:
            trits.fill(-1)
            acts.fill(-128)
            scales.fill(65535)

        expected_row = row_dot(trits, acts, scales, group_size=group_size, lanes=lanes, shift=shift)

        w_pad = np.zeros(k_pad, dtype=np.int8)
        x_pad = np.zeros(k_pad, dtype=np.int8)
        w_pad[:k] = trits
        x_pad[:k] = acts

        inter_gap = 0 if (r_idx > 0 and r_idx % 2 == 1) else int(rng.integers(1, 4))
        for _ in range(inter_gap):
            schedule.append((0, poison_trits, poison_acts, 0xFFFF, 1, None, r_idx))

        for b_idx in range(total_beats):
            if b_idx > 0:
                intra_gap = int(rng.choice([0, 1, 2, 4]))
                for _ in range(intra_gap):
                    schedule.append((0, poison_trits, poison_acts, 0xFFFF, 1, None, r_idx))

            g_idx = b_idx // beats_per_group
            w_slice = w_pad[b_idx * lanes : (b_idx + 1) * lanes]
            x_slice = x_pad[b_idx * lanes : (b_idx + 1) * lanes]
            is_last = 1 if (b_idx == total_beats - 1) else 0
            exp_after_edge = expected_row if is_last else None
            schedule.append(
                (
                    1,
                    _pack_trits_signed(w_slice),
                    _pack_acts_int8(x_slice),
                    int(scales[g_idx]),
                    is_last,
                    exp_after_edge,
                    r_idx,
                )
            )

    for _ in range(4):
        schedule.append((0, poison_trits, poison_acts, 0xFFFF, 1, None, n_rows))

    def tb():
        seen_rows = 0
        for cyc, (v_in, p_w, p_x, s_val, r_last, _exp, _r_id) in enumerate(schedule):
            if cyc >= 2:
                prev_exp = schedule[cyc - 2][5]
                prev_row = schedule[cyc - 2][6]
                exp_valid = 1 if prev_exp is not None else 0
                got_valid = yield dut.out_valid
                assert got_valid == exp_valid, (
                    f"[test_group_scale:TernaryRow] out_valid invariant violated at cycle {cyc} "
                    f"(row={prev_row}): expected {exp_valid}, got {got_valid}"
                )
                if exp_valid == 1:
                    got_res = yield dut.result
                    assert got_res == prev_exp, (
                        f"[test_group_scale:TernaryRow] result parity violated at cycle {cyc} "
                        f"(row={prev_row}): expected {prev_exp}, got {got_res}"
                    )
                    seen_rows += 1

            yield dut.trits_packed.eq(p_w)
            yield dut.acts_packed.eq(p_x)
            yield dut.scale_u16.eq(s_val)
            yield dut.in_valid.eq(v_in)
            yield dut.row_last.eq(r_last)
            yield

        assert seen_rows == n_rows, (
            f"[test_group_scale:TernaryRow] completed row count mismatch: "
            f"expected {n_rows}, got {seen_rows}"
        )

    run_migen_sim(dut, tb)


@pytest.mark.slow
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


def test_group_accumulate_scale_constructor_guards() -> None:
    """Verify GroupAccumulateScale rejects invalid geometry and negative shift."""
    with pytest.raises(ValueError, match="positive multiple of lanes"):
        GroupAccumulateScale(group_size=32, lanes=64)
    with pytest.raises(ValueError, match="positive multiple of lanes"):
        GroupAccumulateScale(group_size=64, lanes=24)
    with pytest.raises(ValueError, match="shift must be >= 0"):
        GroupAccumulateScale(group_size=64, lanes=16, shift=-1)
