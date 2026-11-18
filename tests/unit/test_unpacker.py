"""Verilator + cocotb tests for WeightUnpacker and Decode5TritsByte (Task 2.3)."""

from __future__ import annotations

import os
import random

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from setun.accelerators.unpacker import (
    Decode5TritsByte,
    WeightUnpacker,
    decode_5trits_byte_ref,
)
from setun.ref.format import pack_ternary_tensor, unpack_trits_1d
from tests.sim_helper import run_cocotb_verilator


@cocotb.test()
async def tb_unpacker_bitexact_and_zero_bubble(dut) -> None:
    """Verify bit-exact unpacking vs format.py and 1 beat/cycle throughput with zero bubbles."""
    beat_bits = int(os.environ["SETUN_BEAT_BITS"])
    lanes = int(os.environ["SETUN_LANES"])
    group_size = int(os.environ["SETUN_GROUP_SIZE"])
    num_beats = int(os.environ["SETUN_NUM_BEATS"])
    random_bp = int(os.environ["SETUN_RANDOM_BP"]) == 1

    slices_per_beat = beat_bits // (2 * lanes)
    beats_per_group = group_size // lanes
    assert num_beats % (slices_per_beat * beats_per_group) == 0

    rng = np.random.default_rng(20260203)
    py_rng = random.Random(20260203)

    n_rows = 4
    n_cols = (num_beats // n_rows) * lanes
    weights = rng.choice([-1, 0, 1], size=(n_rows, n_cols)).astype(np.int8)
    n_groups = n_cols // group_size
    scales = rng.integers(1, 65535, size=(n_rows, n_groups), dtype=np.uint16)

    packed = pack_ternary_tensor(weights, scales, group_size=group_size, shift=0)
    flat_bytes = packed.packed_trits.reshape(-1)
    bytes_per_beat = beat_bits // 8
    total_words = len(flat_bytes) // bytes_per_beat
    words_per_row = total_words // n_rows

    beat_words: list[tuple[int, int]] = []
    for w_idx in range(total_words):
        chunk = flat_bytes[w_idx * bytes_per_beat : (w_idx + 1) * bytes_per_beat]
        val = int.from_bytes(chunk.tobytes(), byteorder="little")
        is_last = 1 if ((w_idx + 1) % words_per_row == 0) else 0
        beat_words.append((val, is_last))

    flat_scales = [int(s) for s in scales.reshape(-1)]

    expected: list[tuple[int, int, int, int]] = []
    out_bytes_per_beat = (2 * lanes) // 8
    for b_idx in range(num_beats):
        chunk = flat_bytes[b_idx * out_bytes_per_beat : (b_idx + 1) * out_bytes_per_beat]
        t_val = int.from_bytes(chunk.tobytes(), byteorder="little")
        s_val = flat_scales[b_idx // beats_per_group]
        g_last = 1 if ((b_idx + 1) % beats_per_group == 0) else 0
        r_last = 1 if ((b_idx + 1) % (n_cols // lanes) == 0) else 0
        expected.append((t_val, s_val, g_last, r_last))

    cocotb.start_soon(Clock(dut.sys_clk, 10, units="ns").start())
    dut.sys_rst.value = 1
    dut.trit_data.value = 0
    dut.trit_last.value = 0
    dut.trit_valid.value = 0
    dut.scale_data.value = 0
    dut.scale_valid.value = 0
    dut.out_ready.value = 0

    for _ in range(3):
        await RisingEdge(dut.sys_clk)
    dut.sys_rst.value = 0
    await RisingEdge(dut.sys_clk)

    w_ptr = 0
    s_ptr = 0
    out_idx = 0
    active_cycles = 0
    group_transition_bubbles = 0
    prev_was_group_last = False

    while out_idx < num_beats:
        if w_ptr < len(beat_words):
            dut.trit_data.value = beat_words[w_ptr][0]
            dut.trit_last.value = beat_words[w_ptr][1]
            dut.trit_valid.value = 1
        else:
            dut.trit_valid.value = 0

        if s_ptr < len(flat_scales):
            dut.scale_data.value = flat_scales[s_ptr]
            dut.scale_valid.value = 1
        else:
            dut.scale_valid.value = 0

        if not random_bp:
            dut.out_ready.value = 1
        else:
            dut.out_ready.value = 1 if py_rng.random() < 0.75 else 0

        await RisingEdge(dut.sys_clk)
        active_cycles += 1

        out_v = int(dut.out_valid.value)
        out_r = int(dut.out_ready.value)
        t_r = int(dut.trit_ready.value)
        s_r = int(dut.scale_ready.value)

        if out_r == 1 and out_v == 0 and prev_was_group_last:
            group_transition_bubbles += 1

        if out_v == 1 and out_r == 1:
            got_t = int(dut.trits_packed.value)
            got_s = int(dut.scale_u16.value)
            got_gl = int(dut.group_last.value)
            got_rl = int(dut.row_last.value)

            exp_t, exp_s, exp_gl, exp_rl = expected[out_idx]
            assert got_t == exp_t, f"Beat {out_idx}: trits {got_t:#x} != {exp_t:#x}"
            assert got_s == exp_s, f"Beat {out_idx}: scale {got_s} != {exp_s}"
            assert got_gl == exp_gl, f"Beat {out_idx}: group_last {got_gl} != {exp_gl}"
            assert got_rl == exp_rl, f"Beat {out_idx}: row_last {got_rl} != {exp_rl}"

            if out_idx < 32:
                got_bytes = np.frombuffer(
                    got_t.to_bytes(out_bytes_per_beat, byteorder="little"), dtype=np.uint8
                )
                unpacked = unpack_trits_1d(got_bytes)
                row_i = out_idx // (n_cols // lanes)
                col_start = (out_idx % (n_cols // lanes)) * lanes
                np.testing.assert_array_equal(
                    unpacked, weights[row_i, col_start : col_start + lanes]
                )

            prev_was_group_last = got_gl == 1
            out_idx += 1

        if int(dut.trit_valid.value) == 1 and t_r == 1:
            w_ptr += 1
        if int(dut.scale_valid.value) == 1 and s_r == 1:
            s_ptr += 1

    assert group_transition_bubbles == 0, (
        f"Detected {group_transition_bubbles} bubbles at group boundaries!"
    )
    if not random_bp:
        assert active_cycles == num_beats, (
            f"Expected exact 1.0 beat/cycle ({num_beats} cycles), took {active_cycles}"
        )


@cocotb.test()
async def tb_decode_5trits_byte_exhaustive(dut) -> None:
    """Exhaustively test all 256 byte values against decode_5trits_byte_ref."""
    for b in range(256):
        dut.byte_in.value = b
        await Timer(1, units="ns")
        got = int(dut.trits_out.value)
        exp = decode_5trits_byte_ref(b)
        assert got == exp, f"Byte {b}: got {got:#05x}, expected {exp:#05x}"


def _run_with_env(
    dut,
    ios,
    testcase: str,
    toplevel: str,
    env_vars: dict[str, str],
) -> None:
    saved = {k: os.environ.get(k) for k in env_vars}
    os.environ.update(env_vars)
    try:
        run_cocotb_verilator(
            dut=dut,
            ios=ios,
            test_module="test_unpacker",
            toplevel=toplevel,
            testcase=testcase,
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_unpacker_100k_beats_zero_bubble() -> None:
    """Sustains 1 beat/cycle for 100,000 beats with zero bubbles between groups."""
    dut = WeightUnpacker(beat_bits=256, lanes=64, group_size=128)
    _run_with_env(
        dut=dut,
        ios=dut.ios,
        testcase="tb_unpacker_bitexact_and_zero_bubble",
        toplevel="weight_unpacker_100k",
        env_vars={
            "SETUN_BEAT_BITS": "256",
            "SETUN_LANES": "64",
            "SETUN_GROUP_SIZE": "128",
            "SETUN_NUM_BEATS": "100000",
            "SETUN_RANDOM_BP": "0",
        },
    )


def test_unpacker_random_backpressure() -> None:
    """Bit-exact unpacking under randomized downstream back-pressure with zero group bubbles."""
    dut = WeightUnpacker(beat_bits=256, lanes=64, group_size=128)
    _run_with_env(
        dut=dut,
        ios=dut.ios,
        testcase="tb_unpacker_bitexact_and_zero_bubble",
        toplevel="weight_unpacker_bp",
        env_vars={
            "SETUN_BEAT_BITS": "256",
            "SETUN_LANES": "64",
            "SETUN_GROUP_SIZE": "128",
            "SETUN_NUM_BEATS": "4096",
            "SETUN_RANDOM_BP": "1",
        },
    )


def test_decode_5trits_byte_lut() -> None:
    """Verify optional 5-trits-per-byte LUT decoder across all 256 inputs."""
    dut = Decode5TritsByte()
    run_cocotb_verilator(
        dut=dut,
        ios=dut.ios,
        test_module="test_unpacker",
        toplevel="decode_5trits_byte",
        testcase="tb_decode_5trits_byte_exhaustive",
    )
