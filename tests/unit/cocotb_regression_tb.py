"""High-speed vectorized cocotb + Verilator regression testbench (Tasks 2.8 & 2.8b)."""

from __future__ import annotations

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


def _to_signed(val: int, bits: int) -> int:
    mask = (1 << bits) - 1
    v = int(val) & mask
    if v & (1 << (bits - 1)):
        v -= 1 << bits
    return v


def _precompute_dot_batches(
    rng: np.random.Generator,
    n_cases: int,
    lanes: int,
) -> tuple[list[int], list[int], np.ndarray]:
    """Vectorized generation & bit-packing of n_cases x lanes ternary dot vectors."""
    trits = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n_cases, lanes))
    acts = rng.integers(-128, 128, size=(n_cases, lanes), dtype=np.int8)

    expected = np.sum(trits.astype(np.int64) * acts.astype(np.int64), axis=1)

    codes = np.where(trits == 1, 1, np.where(trits == -1, 2, 0)).astype(np.uint8)
    c_grouped = codes.reshape(n_cases, lanes // 4, 4)
    trit_bytes = (
        c_grouped[:, :, 0]
        | (c_grouped[:, :, 1] << 2)
        | (c_grouped[:, :, 2] << 4)
        | (c_grouped[:, :, 3] << 6)
    )
    packed_trits = [int.from_bytes(row.tobytes(), "little") for row in trit_bytes]

    act_u8 = acts.view(np.uint8)
    packed_acts = [int.from_bytes(row.tobytes(), "little") for row in act_u8]

    return packed_trits, packed_acts, expected


def _precompute_tile_batch(
    rng: np.random.Generator,
    n_rows: int,
    k: int,
    rows: int,
    lanes: int,
    group_size: int,
    shift: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
    list[tuple[int, int, int]],
    np.ndarray,
]:
    """Vectorized generation & R-row lockstep bit-packing for TernaryTile."""
    n_groups = (k + group_size - 1) // group_size
    k_pad = n_groups * group_size
    beats_per_stripe = k_pad // lanes
    beats_per_group = group_size // lanes
    n_stripes = (n_rows + rows - 1) // rows
    n_pad = n_stripes * rows

    w = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n_rows, k))
    x = rng.integers(-128, 128, size=(k,), dtype=np.int8)
    s = rng.integers(0, 65536, size=(n_rows, n_groups), dtype=np.uint16)

    w_pad = np.zeros((n_pad, k_pad), dtype=np.int8)
    w_pad[:n_rows, :k] = w
    if n_pad > n_rows:
        # Non-zero padding rows verify hardware suppresses row_idx >= num_rows
        w_pad[n_rows:, :] = 1

    x_pad = np.zeros(k_pad, dtype=np.int8)
    x_pad[:k] = x

    s_pad = np.zeros((n_pad, n_groups), dtype="<u2")
    s_pad[:n_rows, :] = s
    if n_pad > n_rows:
        s_pad[n_rows:, :] = 0x7FFF

    # Vectorized golden GEMV matching setun.ref.ops.tile_gemv
    w_grp = w_pad[:n_rows].astype(np.int64).reshape(n_rows, n_groups, group_size)
    x_grp = x_pad.astype(np.int64).reshape(1, n_groups, group_size)
    group_sums = np.sum(w_grp * x_grp, axis=2)
    row_accs = np.sum(group_sums * s.astype(np.int64), axis=1)
    expected = (row_accs + (1 << (shift - 1))) >> shift if shift > 0 else row_accs

    # Pack L-lane activation slices (little-endian raw bytes)
    act_u8 = x_pad.view(np.uint8).reshape(beats_per_stripe, lanes)
    act_slices = [int.from_bytes(sl.tobytes(), "little") for sl in act_u8]

    # Vectorized R-row trit packing: transpose to (n_stripes, beats_per_stripe, rows, L//4, 4)
    codes = np.where(w_pad == 1, 1, np.where(w_pad == -1, 2, 0)).astype(np.uint8)
    c4 = codes.reshape(n_stripes, rows, beats_per_stripe, lanes // 4, 4).transpose(0, 2, 1, 3, 4)
    trit_bytes = (c4[..., 0] | (c4[..., 1] << 2) | (c4[..., 2] << 4) | (c4[..., 3] << 6)).astype(
        np.uint8
    )

    s_view = np.ascontiguousarray(s_pad.reshape(n_stripes, rows, n_groups), dtype="<u2")
    scale_words_by_group = [
        [int.from_bytes(s_view[st, :, g].tobytes(), "little") for g in range(n_groups)]
        for st in range(n_stripes)
    ]

    beats: list[tuple[int, int, int]] = []
    for st in range(n_stripes):
        st_scales = scale_words_by_group[st]
        for b in range(beats_per_stripe):
            t_word = int.from_bytes(trit_bytes[st, b].tobytes(), "little")
            s_word = st_scales[b // beats_per_group]
            last_f = 1 if (b == beats_per_stripe - 1) else 0
            beats.append((t_word, s_word, last_f))

    return w, x, s, act_slices, beats, expected


def _precompute_requantize_batch(
    rng: np.random.Generator,
    n_items: int,
    scale_u16: int,
    shift: int,
    symmetric: bool,
) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
    """Generate mixed 48-bit wide + near-boundary values and exact 65-bit int8 golden outputs."""
    min_i48 = -(1 << 47)
    max_i48 = (1 << 47) - 1
    mask_48 = (1 << 48) - 1

    half = n_items // 2
    wide_vals = rng.integers(min_i48, max_i48 + 1, size=half, dtype=np.int64)
    near_vals = rng.integers(-512, 513, size=n_items - half, dtype=np.int64)
    vals = np.concatenate([wide_vals, near_vals])
    if n_items >= 8:
        vals[:8] = np.array([0, 127, 128, -127, -128, -129, max_i48, min_i48], dtype=np.int64)
    rng.shuffle(vals)

    s_u16 = int(scale_u16)
    sh = int(shift)
    q_min = -127 if symmetric else -128
    q_max = 127

    u48_words: list[int] = []
    exp_q = np.empty(n_items, dtype=np.int8)
    exp_sat = np.empty(n_items, dtype=bool)

    for idx, item in enumerate(vals.tolist()):
        v = int(item)
        u48_words.append(v & mask_48)
        prod = v * s_u16
        shifted = prod if sh == 0 else (prod + (1 << (sh - 1))) >> sh
        if shifted < q_min:
            exp_q[idx] = q_min
            exp_sat[idx] = True
        elif shifted > q_max:
            exp_q[idx] = q_max
            exp_sat[idx] = True
        else:
            exp_q[idx] = shifted
            exp_sat[idx] = False

    return vals, u48_words, exp_q, exp_sat


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

    trits = np.zeros((n_cases, k_pad), dtype=np.int64)
    acts = np.zeros((n_cases, k_pad), dtype=np.int64)
    trits[:, :k] = rng.choice([-1, 0, 1], size=(n_cases, k))
    acts[:, :k] = rng.integers(-128, 128, size=(n_cases, k))
    scales = rng.integers(0, 65536, size=(n_cases, n_groups), dtype=np.int64)

    beat_dots = np.sum(
        (trits * acts).reshape(n_cases, total_beats, lanes),
        axis=2,
    )
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


@cocotb.test()
async def tb_regression_ternary_tile(dut) -> None:
    """Stream n_cases rows across ragged & aligned GEMVs through TernaryTile (Task 2.8b)."""
    seed = int(os.environ.get("TEST_SEED", "1958"))
    rows = int(os.environ.get("SETUN_ROWS", "4"))
    lanes = int(os.environ.get("SETUN_LANES", "16"))
    group_size = int(os.environ.get("SETUN_G", "64"))
    k = int(os.environ.get("SETUN_K", "65"))
    target_rows = int(os.environ.get("SETUN_CASES", "512"))

    rng = np.random.default_rng(seed + rows * 31 + lanes * 7 + group_size + k)

    cocotb.start_soon(Clock(dut.sys_clk, 10, units="ns").start())
    dut.sys_rst.value = 1
    dut.clear.value = 0
    dut.shift.value = 0
    dut.num_rows.value = 0
    dut.act_wr_en.value = 0
    dut.act_wr_addr.value = 0
    dut.act_wr_data.value = 0
    dut.sink_valid.value = 0
    dut.sink_last.value = 0
    dut.sink_trits.value = 0
    dut.sink_scales.value = 0
    dut.source_ready.value = 0
    await RisingEdge(dut.sys_clk)
    dut.sys_rst.value = 0
    await RisingEdge(dut.sys_clk)

    # Cycle ragged & aligned batch row counts and shifts across target_rows
    batch_schedule = [13, 64, 7, 127, 256, 511, 2048, 4093]
    shift_schedule = [0, 1, 4, 8, 12, 19, 24, 30]
    done_rows = 0
    batch_idx = 0
    lcg = (seed ^ 0x5A5A5A5A) & 0xFFFFFFFF

    while done_rows < target_rows:
        rem = target_rows - done_rows
        n_batch = min(rem, batch_schedule[batch_idx % len(batch_schedule)])
        shift = shift_schedule[batch_idx % len(shift_schedule)]
        use_stalls = (batch_idx % 2) == 0

        _w, _x, _s, act_slices, beats, expected = _precompute_tile_batch(
            rng,
            n_rows=n_batch,
            k=k,
            rows=rows,
            lanes=lanes,
            group_size=group_size,
            shift=shift,
        )

        dut.clear.value = 1
        dut.sink_valid.value = 0
        dut.source_ready.value = 0
        await RisingEdge(dut.sys_clk)
        dut.clear.value = 0
        dut.num_rows.value = n_batch
        dut.shift.value = shift
        await RisingEdge(dut.sys_clk)

        for addr_i, sl_val in enumerate(act_slices):
            dut.act_wr_addr.value = addr_i
            dut.act_wr_data.value = sl_val
            dut.act_wr_en.value = 1
            await RisingEdge(dut.sys_clk)
        dut.act_wr_en.value = 0
        await RisingEdge(dut.sys_clk)

        in_ptr = 0
        out_cnt = 0
        held_sink = False
        total_beats = len(beats)

        while out_cnt < n_batch:
            lcg = (1664525 * lcg + 1013904223) & 0xFFFFFFFF
            if in_ptr < total_beats:
                if held_sink or (not use_stalls) or ((lcg & 0x7) != 0):
                    t_w, s_w, last_f = beats[in_ptr]
                    dut.sink_trits.value = t_w
                    dut.sink_scales.value = s_w
                    dut.sink_last.value = last_f
                    dut.sink_valid.value = 1
                    held_sink = True
                else:
                    dut.sink_valid.value = 0
            else:
                dut.sink_valid.value = 0
                held_sink = False

            dut.source_ready.value = 1 if ((not use_stalls) or (((lcg >> 4) & 0x7) != 0)) else 0

            await RisingEdge(dut.sys_clk)

            if int(dut.sink_valid.value) == 1 and int(dut.sink_ready.value) == 1:
                in_ptr += 1
                held_sink = False

            if int(dut.source_valid.value) == 1 and int(dut.source_ready.value) == 1:
                got_idx = int(dut.source_row_idx.value)
                got_val = _to_signed(int(dut.source_result.value), 48)
                got_last = int(dut.source_last.value)
                exp_val = int(expected[out_cnt])
                exp_last = 1 if (out_cnt == n_batch - 1) else 0
                if (got_idx, got_val, got_last) != (out_cnt, exp_val, exp_last):
                    raise AssertionError(
                        f"[TernaryTile REGRESSION] TEST_SEED={seed} R={rows} L={lanes} "
                        f"G={group_size} K={k} batch={batch_idx} row={out_cnt}/{n_batch}: "
                        f"expected (idx={out_cnt}, val={exp_val}, last={exp_last}), "
                        f"got (idx={got_idx}, val={got_val}, last={got_last})"
                    )
                out_cnt += 1

        done_rows += n_batch
        batch_idx += 1


@cocotb.test()
async def tb_regression_requantize(dut) -> None:
    """Stream n_cases randomized 48-bit inputs through Requantize (Task 2.8b)."""
    seed = int(os.environ.get("TEST_SEED", "1958"))
    target_cases = int(os.environ.get("SETUN_CASES", "1000"))
    rng = np.random.default_rng(seed + 205)

    cocotb.start_soon(Clock(dut.sys_clk, 10, units="ns").start())
    dut.sys_rst.value = 1
    dut.clear.value = 0
    dut.scale_u16.value = 1
    dut.shift.value = 0
    dut.symmetric.value = 0
    dut.sink_valid.value = 0
    dut.sink_last.value = 0
    dut.sink_result.value = 0
    dut.sink_row_idx.value = 0
    dut.source_ready.value = 0
    await RisingEdge(dut.sys_clk)
    dut.sys_rst.value = 0
    await RisingEdge(dut.sys_clk)

    cfgs = [
        (1, 0, False),
        (1, 1, True),
        (317, 14, False),
        (65535, 15, True),
        (65535, 31, False),
        (65535, 47, True),
        (65535, 62, False),
        (0, 9, False),
    ]
    base_chunk = max(target_cases // len(cfgs), 1)
    done = 0
    lcg = (seed ^ 0xA5A5A5A5) & 0xFFFFFFFF

    for c_idx, (scale_u16, shift, sym) in enumerate(cfgs):
        n_chunk = (target_cases - done) if (c_idx == len(cfgs) - 1) else base_chunk
        if n_chunk <= 0:
            break
        use_stalls = (c_idx % 2) == 0
        vals, u48_words, exp_q, exp_sat = _precompute_requantize_batch(
            rng, n_items=n_chunk, scale_u16=scale_u16, shift=shift, symmetric=sym
        )

        dut.clear.value = 1
        dut.sink_valid.value = 0
        dut.source_ready.value = 0
        await RisingEdge(dut.sys_clk)
        dut.clear.value = 0
        dut.scale_u16.value = scale_u16
        dut.shift.value = shift
        dut.symmetric.value = 1 if sym else 0
        await RisingEdge(dut.sys_clk)

        in_ptr = 0
        out_cnt = 0
        held_sink = False

        while out_cnt < n_chunk:
            lcg = (1664525 * lcg + 1013904223) & 0xFFFFFFFF
            if in_ptr < n_chunk:
                if held_sink or (not use_stalls) or ((lcg & 0x7) != 0):
                    dut.sink_result.value = u48_words[in_ptr]
                    dut.sink_row_idx.value = in_ptr & 0xFFFF
                    dut.sink_last.value = 1 if (in_ptr == n_chunk - 1) else 0
                    dut.sink_valid.value = 1
                    held_sink = True
                else:
                    dut.sink_valid.value = 0
            else:
                dut.sink_valid.value = 0
                held_sink = False

            dut.source_ready.value = 1 if ((not use_stalls) or (((lcg >> 4) & 0x7) != 0)) else 0

            await RisingEdge(dut.sys_clk)

            if int(dut.sink_valid.value) == 1 and int(dut.sink_ready.value) == 1:
                in_ptr += 1
                held_sink = False

            if int(dut.source_valid.value) == 1 and int(dut.source_ready.value) == 1:
                got_idx = int(dut.source_row_idx.value)
                got_q = _to_signed(int(dut.source_result.value), 8)
                got_sat = int(dut.source_saturated.value)
                got_last = int(dut.source_last.value)
                exp_idx = out_cnt & 0xFFFF
                exp_q_i = int(exp_q[out_cnt])
                exp_sat_i = 1 if bool(exp_sat[out_cnt]) else 0
                exp_last = 1 if (out_cnt == n_chunk - 1) else 0
                if (got_idx, got_q, got_sat, got_last) != (
                    exp_idx,
                    exp_q_i,
                    exp_sat_i,
                    exp_last,
                ):
                    raise AssertionError(
                        f"[Requantize REGRESSION] TEST_SEED={seed} cfg=({scale_u16}, {shift}, "
                        f"{sym}) item={out_cnt} (val={int(vals[out_cnt])}): "
                        f"expected (idx={exp_idx}, q={exp_q_i}, sat={exp_sat_i}, last={exp_last}), "
                        f"got (idx={got_idx}, q={got_q}, sat={got_sat}, last={got_last})"
                    )
                out_cnt += 1

        done += n_chunk
