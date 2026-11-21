"""Multi-row TernaryTile GEMV engine with LiteX stream Endpoints (Task 2.4, ADR-015)."""

from __future__ import annotations

from litex.soc.interconnect import stream
from migen import Array, Case, ClockDomain, If, Memory, Module, Signal

from setun.accelerators.group_scale import TernaryRow


def tile_sink_description(rows: int, lanes: int) -> list[tuple[str, int]]:
    """Payload layout for TernaryTile lockstep R-row weight + group-scale input stream."""
    return [
        ("trits", rows * 2 * lanes),
        ("scales", rows * 16),
    ]


def tile_source_description(acc_bits: int = 48) -> list[tuple[str, int]]:
    """Payload layout for TernaryTile serialized 1-row/cycle output stream."""
    return [
        ("result", acc_bits),
        ("row_idx", 16),
    ]


class TernaryTile(Module):
    """R-row x L-lane streaming ternary GEMV tile (Task 2.4, ADR-015).

    Broadcasts shared int8 activations from an inferred LUTRAM/BRAM buffer across R
    parallel TernaryRow engines, latches stripe completions into an R-deep shadow buffer,
    and drains valid rows (< num_rows) at 1 row/cycle over a LiteX stream.Endpoint.
    """

    def __init__(
        self,
        rows: int = 4,
        lanes: int = 64,
        group_size: int = 128,
        act_bits: int = 8,
        acc_bits: int = 48,
        max_k: int = 2048,
    ) -> None:
        if rows < 1:
            raise ValueError(
                f"[setun/accelerators/tile.py:TernaryTile] expected rows >= 1, got {rows}."
            )
        if lanes < 1 or group_size < lanes or group_size % lanes != 0:
            raise ValueError(
                f"[setun/accelerators/tile.py:TernaryTile] group_size ({group_size}) must be a "
                f"positive multiple of lanes ({lanes})."
            )
        if max_k < group_size or max_k % lanes != 0:
            raise ValueError(
                f"[setun/accelerators/tile.py:TernaryTile] max_k ({max_k}) must be >= "
                f"group_size ({group_size}) and divisible by lanes ({lanes})."
            )

        self.rows = rows
        self.lanes = lanes
        self.group_size = group_size
        self.act_bits = act_bits
        self.acc_bits = acc_bits
        self.max_k = max_k
        self.row_trit_bits = 2 * lanes
        self.act_slice_bits = lanes * act_bits
        self.max_act_slices = max_k // lanes

        # Configuration & control signals
        self.clear = Signal(name="clear")
        self.shift = Signal(6, name="shift")
        self.num_rows = Signal(16, name="num_rows")

        # Shared activation buffer write port (1 L-lane slice per write)
        addr_max = max(self.max_act_slices, 2)
        self.act_wr_addr = Signal(max=addr_max, name="act_wr_addr")
        self.act_wr_data = Signal(self.act_slice_bits, name="act_wr_data")
        self.act_wr_en = Signal(name="act_wr_en")

        # LiteX stream.Endpoint interfaces (ADR-015)
        self.sink = stream.Endpoint(tile_sink_description(rows=rows, lanes=lanes))
        self.source = stream.Endpoint(tile_source_description(acc_bits=acc_bits))

        self.clock_domains.cd_sys = ClockDomain("sys")
        self.ios = {
            self.cd_sys.clk,
            self.cd_sys.rst,
            self.clear,
            self.shift,
            self.num_rows,
            self.act_wr_addr,
            self.act_wr_data,
            self.act_wr_en,
            self.sink.valid,
            self.sink.ready,
            self.sink.last,
            self.sink.trits,
            self.sink.scales,
            self.source.valid,
            self.source.ready,
            self.source.last,
            self.source.result,
            self.source.row_idx,
        }

        # Shared activation memory (async read -> distributed LUTRAM inference, 0 bubble)
        act_mem = Memory(width=self.act_slice_bits, depth=self.max_act_slices)
        act_wr_port = act_mem.get_port(write_capable=True)
        act_rd_port = act_mem.get_port(async_read=True)
        self.specials += act_mem, act_wr_port, act_rd_port

        act_beat_idx = Signal(max=addr_max, name="act_beat_idx")
        self.comb += [
            act_wr_port.adr.eq(self.act_wr_addr),
            act_wr_port.dat_w.eq(self.act_wr_data),
            act_wr_port.we.eq(self.act_wr_en),
            act_rd_port.adr.eq(act_beat_idx),
        ]

        # Instantiate R parallel unmodified TernaryRow engines (shift=0; shift applied post-mux)
        row_engines: list[TernaryRow] = []
        for r in range(rows):
            eng = TernaryRow(
                group_size=group_size,
                lanes=lanes,
                act_bits=act_bits,
                acc_bits=acc_bits,
                shift=0,
            )
            setattr(self.submodules, f"row_{r}", eng)
            row_engines.append(eng)

        # Input stripe tracking & 1-cycle TernaryRow commit tracking
        in_stripe_base = Signal(16, name="in_stripe_base")
        commit_pending = Signal(name="commit_pending")
        commit_base = Signal(16, name="commit_base")

        # R-deep shadow output buffer
        shadow_active = Signal(name="shadow_active")
        shadow_base = Signal(16, name="shadow_base")
        shadow_ptr = Signal(max=max(rows, 2), name="shadow_ptr")
        shadow_limit = Signal(max=rows + 1, name="shadow_limit")
        shadow_regs = Array([Signal((acc_bits, True), name=f"shadow_reg_{r}") for r in range(rows)])

        # Compute valid row count in the committing stripe (clamped to [0, R])
        rem_commit = Signal(17, name="rem_commit")
        commit_valid_cnt = Signal(max=rows + 1, name="commit_valid_cnt")
        self.comb += [
            If(
                self.num_rows > commit_base,
                rem_commit.eq(self.num_rows - commit_base),
            ).Else(
                rem_commit.eq(0),
            ),
            If(
                rem_commit >= rows,
                commit_valid_cnt.eq(rows),
            ).Else(
                commit_valid_cnt.eq(rem_commit),
            ),
        ]

        # Shadow buffer drain status & backpressure safety on sink.last (ADR-015 §3 & §4)
        source_fire = Signal(name="source_fire")
        draining_last_shadow = Signal(name="draining_last_shadow")
        can_accept_stripe_end = Signal(name="can_accept_stripe_end")
        accept_stripes = Signal(name="accept_stripes")
        sink_fire = Signal(name="sink_fire")

        self.comb += [
            source_fire.eq(self.source.valid & self.source.ready),
            draining_last_shadow.eq(
                shadow_active & source_fire & ((shadow_ptr + 1) >= shadow_limit)
            ),
            # Next cycle's shadow buffer is free iff no commit is landing this cycle AND
            # shadow is either already empty or draining its final valid row this cycle.
            can_accept_stripe_end.eq((~commit_pending) & ((~shadow_active) | draining_last_shadow)),
            accept_stripes.eq((self.num_rows != 0) & (in_stripe_base < self.num_rows)),
            If(
                ~accept_stripes,
                self.sink.ready.eq(0),
            )
            .Elif(
                self.sink.last,
                self.sink.ready.eq(can_accept_stripe_end),
            )
            .Else(
                self.sink.ready.eq(1),
            ),
            sink_fire.eq(self.sink.valid & self.sink.ready),
        ]

        # Broadcast shared activations & route per-row trits/scales to all R TernaryRow engines
        for r, eng in enumerate(row_engines):
            t_lo = r * self.row_trit_bits
            t_hi = (r + 1) * self.row_trit_bits
            s_lo = r * 16
            s_hi = (r + 1) * 16
            self.comb += [
                eng.trits_packed.eq(self.sink.trits[t_lo:t_hi]),
                eng.acts_packed.eq(act_rd_port.dat_r),
                eng.scale_u16.eq(self.sink.scales[s_lo:s_hi]),
                eng.in_valid.eq(sink_fire),
                eng.row_last.eq(self.sink.last),
            ]

        # Post-mux programmable round-half-up right shift (0..30) on serialized source stream
        raw_sel = Signal((acc_bits, True), name="raw_sel")
        shifted_sel = Signal((acc_bits, True), name="shifted_sel")
        cur_out_row = Signal(16, name="cur_out_row")

        self.comb += [
            raw_sel.eq(shadow_regs[shadow_ptr]),
            cur_out_row.eq(shadow_base + shadow_ptr),
        ]

        shift_cases: dict[int | str, object] = {0: shifted_sel.eq(raw_sel)}
        for s in range(1, 31):
            bias = 1 << (s - 1)
            shift_cases[s] = shifted_sel.eq((raw_sel + bias) >> s)
        shift_cases["default"] = shifted_sel.eq((raw_sel + (1 << 29)) >> 30)
        self.comb += Case(self.shift, shift_cases)

        self.comb += [
            self.source.valid.eq(shadow_active),
            self.source.result.eq(shifted_sel),
            self.source.row_idx.eq(cur_out_row),
            self.source.last.eq(
                shadow_active & (self.num_rows != 0) & ((cur_out_row + 1) == self.num_rows)
            ),
        ]

        # Sequential state updates
        row0_out_valid = row_engines[0].out_valid
        latch_shadow_stmts = [shadow_regs[r].eq(row_engines[r].result) for r in range(rows)]

        self.sync += [
            If(
                self.clear,
                act_beat_idx.eq(0),
                in_stripe_base.eq(0),
                commit_pending.eq(0),
                commit_base.eq(0),
                shadow_active.eq(0),
                shadow_base.eq(0),
                shadow_ptr.eq(0),
                shadow_limit.eq(0),
            ).Else(
                # Default: clear 1-cycle commit_pending unless a new stripe ends this cycle
                commit_pending.eq(0),
                If(
                    sink_fire,
                    If(
                        self.sink.last,
                        act_beat_idx.eq(0),
                        commit_pending.eq(1),
                        commit_base.eq(in_stripe_base),
                        in_stripe_base.eq(in_stripe_base + rows),
                    )
                    .Elif(
                        act_beat_idx == (self.max_act_slices - 1),
                        act_beat_idx.eq(0),
                    )
                    .Else(
                        act_beat_idx.eq(act_beat_idx + 1),
                    ),
                ),
                # Shadow buffer load (from TernaryRow completion) & 1-row/cycle drain
                If(
                    row0_out_valid & (commit_valid_cnt != 0),
                    *latch_shadow_stmts,
                    shadow_base.eq(commit_base),
                    shadow_ptr.eq(0),
                    shadow_limit.eq(commit_valid_cnt),
                    shadow_active.eq(1),
                ).Elif(
                    source_fire,
                    If(
                        (shadow_ptr + 1) >= shadow_limit,
                        shadow_active.eq(0),
                        shadow_ptr.eq(0),
                    ).Else(
                        shadow_ptr.eq(shadow_ptr + 1),
                    ),
                ),
            )
        ]
