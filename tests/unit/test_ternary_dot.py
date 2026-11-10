"""Unit tests for TernaryDot Migen module against golden reference (Task 2.1)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from migen.fhdl import verilog

from setun.accelerators.ternary_dot import TernaryDot
from setun.ref.ops import lane_select_code, ternary_dot
from tests.sim_helper import run_migen_sim


def _pack_trits_codes(codes: list[int]) -> int:
    val = 0
    for i, c in enumerate(codes):
        val |= (c & 0x3) << (2 * i)
    return val


def _pack_trits_signed(trits: np.ndarray) -> int:
    code_map = {0: 0b00, 1: 0b01, -1: 0b10}
    return _pack_trits_codes([code_map[int(t)] for t in trits])


def _pack_acts(acts: np.ndarray, act_bits: int = 8) -> int:
    mask = (1 << act_bits) - 1
    val = 0
    for i, a in enumerate(acts):
        val |= (int(a) & mask) << (act_bits * i)
    return val


def test_ternary_dot_directed_cases_and_reserved_code() -> None:
    """Directed cases: all +1, all -1, all 0, alternating, x in {min, max}, and code 0b11."""
    lanes = 16
    dut = TernaryDot(lanes=lanes, act_bits=8, pipeline_depth=0)
    assert dut.latency == 0

    alt_trits = np.array([1 if i % 2 == 0 else -1 for i in range(lanes)], dtype=np.int8)
    alt_acts = np.array([-128 if i % 2 == 0 else 127 for i in range(lanes)], dtype=np.int8)

    directed = [
        (np.ones(lanes, dtype=np.int8), np.full(lanes, 127, dtype=np.int8)),
        (np.ones(lanes, dtype=np.int8), np.full(lanes, -128, dtype=np.int8)),
        (np.full(lanes, -1, dtype=np.int8), np.full(lanes, -128, dtype=np.int8)),
        (np.full(lanes, -1, dtype=np.int8), np.full(lanes, 127, dtype=np.int8)),
        (np.zeros(lanes, dtype=np.int8), np.full(lanes, -128, dtype=np.int8)),
        (alt_trits, alt_acts),
    ]

    def tb():
        for trits, acts in directed:
            yield dut.trits_packed.eq(_pack_trits_signed(trits))
            yield dut.acts_packed.eq(_pack_acts(acts, 8))
            yield dut.in_valid.eq(1)
            yield
            assert (yield dut.result) == ternary_dot(trits, acts)
            assert (yield dut.out_valid) == 1

        # Reserved code 0b11 must evaluate to 0
        codes = [0b11, 0b01, 0b10, 0b11, 0b00, 0b11, 0b01, 0b10] * 2
        acts_rng = np.array(
            [100, 50, -50, -128, 77, 127, -30, 20] * 2,
            dtype=np.int8,
        )
        yield dut.trits_packed.eq(_pack_trits_codes(codes))
        yield dut.acts_packed.eq(_pack_acts(acts_rng, 8))
        yield
        expected = sum(lane_select_code(c, int(x)) for c, x in zip(codes, acts_rng, strict=True))
        assert (yield dut.result) == expected

    run_migen_sim(dut, tb)


def test_ternary_dot_1000_random_vectors_and_latency() -> None:
    """Bit-exact vs ref/ops.py on >= 1000 random vectors and fixed pipeline latency check."""
    lanes = 16
    dut = TernaryDot(lanes=lanes, act_bits=8, pipeline_depth=1)
    assert dut.latency == 1
    rng = np.random.default_rng(1958)

    def tb():
        # Verify 1-cycle latency on in_valid -> out_valid
        yield dut.in_valid.eq(0)
        yield
        assert (yield dut.out_valid) == 0

        for _ in range(1000):
            trits = rng.choice([-1, 0, 1], size=lanes).astype(np.int8)
            acts = rng.integers(-128, 128, size=lanes, dtype=np.int8)
            expected = ternary_dot(trits, acts)

            yield dut.trits_packed.eq(_pack_trits_signed(trits))
            yield dut.acts_packed.eq(_pack_acts(acts, 8))
            yield dut.in_valid.eq(1)
            yield  # synchronous capture on clock edge
            yield  # combinational settle after edge
            assert (yield dut.result) == expected
            assert (yield dut.out_valid) == 1

    run_migen_sim(dut, tb)


@pytest.mark.parametrize("lanes,act_bits,depth", [(16, 8, 2), (64, 8, 1), (16, 16, 1)])
def test_ternary_dot_parameter_sweep(lanes: int, act_bits: int, depth: int) -> None:
    """Verify L in {16, 64}, A in {8, 16}, and pipeline_depth in {1, 2}."""
    dut = TernaryDot(lanes=lanes, act_bits=act_bits, pipeline_depth=depth)
    assert dut.latency == depth
    rng = np.random.default_rng(2026)
    a_min, a_max = -(1 << (act_bits - 1)), (1 << (act_bits - 1)) - 1

    def tb():
        for _ in range(25):
            trits = rng.choice([-1, 0, 1], size=lanes).astype(np.int64)
            acts = rng.integers(a_min, a_max + 1, size=lanes, dtype=np.int64)
            expected = int(np.sum(trits * acts))

            yield dut.in_valid.eq(0)
            yield
            yield dut.trits_packed.eq(_pack_trits_signed(trits))
            yield dut.acts_packed.eq(_pack_acts(acts, act_bits))
            yield dut.in_valid.eq(1)
            for _ in range(depth):
                yield
            yield
            assert (yield dut.out_valid) == 1
            assert (yield dut.result) == expected

    run_migen_sim(dut, tb)


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys not installed")
def test_ternary_dot_yosys_zero_dsp_no_multipliers() -> None:
    """Synthesize TernaryDot with Yosys and assert 0 DSPs and 0 multiplier cells."""
    dut = TernaryDot(lanes=64, act_bits=8, pipeline_depth=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        v_path = Path(tmpdir) / "ternary_dot.v"
        json_path = Path(tmpdir) / "stat.json"
        v_out = verilog.convert(dut, ios=dut.ios, name="ternary_dot")
        v_out.write(str(v_path))

        cmd = [
            "yosys",
            "-q",
            "-p",
            f"read_verilog {v_path}; proc; opt; tee -q -o {json_path} stat -json",
        ]
        subprocess.run(cmd, check=True)
        stat_data = json.loads(json_path.read_text())
        cells = stat_data["modules"]["\\ternary_dot"]["num_cells_by_type"]
        bad_cells = [k for k in cells if "mul" in k.lower() or "dsp" in k.lower()]
        assert bad_cells == [], f"Unexpected multiplier/DSP cells found: {bad_cells}"
