"""High-speed vectorized cocotb + Verilator regression testbench (Task 2.8)."""

from __future__ import annotations

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


def _precompute_dot_batches(
    rng: np.random.Generator,
    n_cases: int,
    lanes: int,
) -> tuple[list[int], list[int], np.ndarray]:
    """Vectorized generation & bit-packing of n_cases x lanes ternary dot vectors."""
    trits = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n_cases, lanes))
    acts = rng.integers(-128, 128, size=(n_cases, lanes), dtype=np.int8)

    # Golden dot products in exact int64
    expected = np.sum(trits.astype(np.int64) * acts.astype(np.int64), axis=1)

    # Map {-1: 2 (0b10), 0: 0 (0b00), +1: 1 (0b01)} and pack 4 trits per byte (little-endian)
    codes = np.where(trits == 1, 1, np.where(trits == -1, 2, 0)).astype(np.uint8)
    c_grouped = codes.reshape(n_cases, lanes // 4, 4)
    trit_bytes = (
        c_grouped[:, :, 0]
        | (c_grouped[:, :, 1] << 2)
        | (c_grouped[:, :, 2] << 4)
        | (c_grouped[:, :, 3] << 6)
    )
    packed_trits = [int.from_bytes(row.tobytes(), "little") for row in trit_bytes]

    # Pack int8 activations directly from raw two's-complement bytes (little-endian)
    act_u8 = acts.view(np.uint8)
    packed_acts = [int.from_bytes(row.tobytes(), "little") for row in act_u8]

    return packed_trits, packed_acts, expected


@cocotb.test()
async def tb_regression_ternary_dot(dut) -> None:
    """Stream n_cases random vectors through TernaryDot in Verilator."""
    seed = int(os.environ.get("TEST_SEED", "1958"))
    lanes = int(os.environ.get("SETUN_LANES", "64"))
    n_cases = int(os.environ.get("SETUN_CASES", "250"))

    rng = np.random.default_rng(seed + lanes)
    packed_trits, packed_acts, expected = _precompute_dot_batches(rng, n_cases, lanes)

    dut.in_valid.value = 1
    for idx in range(n_cases):
        dut.trits_packed.value = packed_trits[idx]
        dut.acts_packed.value = packed_acts[idx]
        await Timer(1, units="ns")
        actual = dut.result.value.signed_integer
        if actual != int(expected[idx]):
            raise AssertionError(
                f"[TernaryDot REGRESSION] TEST_SEED={seed} lanes={lanes} case={idx}: "
                f"got {actual}, expected {int(expected[idx])}"
            )


@cocotb.test()
async def tb_regression_group_scale(dut) -> None:
    """Stream n_cases multi-beat rows through GroupAccumulateScale in Verilator."""
    seed = int(os.environ.get("TEST_SEED", "1958"))
    k = int(os.environ.get("SETUN_K", "128"))
    group_size = int(os.environ.get("SETUN_G", "128"))
    lanes = int(os.environ.get("SETUN_LANES", "64"))
    shift = int(os.environ.get("SETUN_SHIFT", "0"))
    n_cases = int(os.environ.get("SETUN_CASES", "120"))

    rng = np.random.default_rng(seed + k * 17 + group_size + lanes)
    n_groups = (k + group_size - 1) // group_size
    k_pad = n_groups * group_size
    beats_per_group = group_size // lanes
    total_beats = n_groups * beats_per_group

    # Vectorized pre-computation of all n_cases rows, beats, scales, and golden results
    trits = np.zeros((n_cases, k_pad), dtype=np.int64)
    acts = np.zeros((n_cases, k_pad), dtype=np.int64)
    trits[:, :k] = rng.choice([-1, 0, 1], size=(n_cases, k))
    acts[:, :k] = rng.integers(-128, 128, size=(n_cases, k))
    scales = rng.integers(0, 65536, size=(n_cases, n_groups), dtype=np.int64)

    # Shape into (n_cases, total_beats, lanes) to compute beat dot products
    beat_dots = np.sum(
        (trits * acts).reshape(n_cases, total_beats, lanes),
        axis=2,
    )
    # Sum beats into groups -> multiply by scale -> sum across groups -> round_shift_right
    group_sums = np.sum(beat_dots.reshape(n_cases, n_groups, beats_per_group), axis=2)
    row_accs = np.sum(group_sums * scales, axis=1)
    if shift > 0:
        expected_rows = (row_accs + (1 << (shift - 1))) >> shift
    else:
        expected_rows = row_accs

    clock = Clock(dut.sys_clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.sys_rst.value = 1
    dut.in_valid.value = 0
    dut.row_last.value = 0
    dut.dot_in.value = 0
    dut.scale_u16.value = 0
    await RisingEdge(dut.sys_clk)
    dut.sys_rst.value = 0
    await RisingEdge(dut.sys_clk)

    # Pipeline streaming: drive 1 row after another; out_valid asserts 1 cycle after row_last
    for idx in range(n_cases):
        row_beats = beat_dots[idx]
        row_scales = scales[idx]
        for b_idx in range(total_beats):
            g_idx = b_idx // beats_per_group
            dut.dot_in.value = int(row_beats[b_idx])
            dut.scale_u16.value = int(row_scales[g_idx])
            dut.in_valid.value = 1
            dut.row_last.value = 1 if b_idx == total_beats - 1 else 0
            await RisingEdge(dut.sys_clk)

        dut.in_valid.value = 0
        dut.row_last.value = 0
        await RisingEdge(dut.sys_clk)
        assert int(dut.out_valid.value) == 1
        actual = dut.result.value.signed_integer
        if actual != int(expected_rows[idx]):
            raise AssertionError(
                f"[GroupScale REGRESSION] TEST_SEED={seed} K={k} G={group_size} L={lanes} "
                f"case={idx}: got {actual}, expected {int(expected_rows[idx])}"
            )
