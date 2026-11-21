"""2-stage pipelined output requantization and int8 saturation core (Task 2.5, NUMERICS.md §4)."""

from __future__ import annotations

from litex.soc.interconnect import stream
from migen import Case, ClockDomain, Constant, If, Module, Signal


def requantize_sink_description(in_bits: int = 48) -> list[tuple[str, int]]:
    """Payload layout matching TernaryTile.source for direct stream.Endpoint.connect()."""
    return [
        ("result", in_bits),
        ("row_idx", 16),
    ]


def requantize_source_description(out_bits: int = 8) -> list[tuple[str, int]]:
    """Payload layout for requantized int8 row output stream + saturation telemetry."""
    return [
        ("result", out_bits),
        ("row_idx", 16),
        ("saturated", 1),
    ]


class Requantize(Module):
    """Scale, round-half-up shift (65-bit signed safe), and saturate wide row results to int8.

    Implements a 2-stage interlocked valid/ready pipeline (II = 1):
      - Stage 1: signed `in_bits` * zero-extended unsigned `scale_u16` -> registered 65-bit product
      - Stage 2: 65-bit signed round-half-up shift (`shift == 0` bias bypass) + clamp to
        `[-128, +127]` (`symmetric = 0`) or `[-127, +127]` (`symmetric = 1`) -> registered output.
    """

    def __init__(
        self,
        in_bits: int = 48,
        out_bits: int = 8,
    ) -> None:
        if out_bits < 2 or in_bits < out_bits:
            raise ValueError(
                f"[setun/accelerators/requantize.py:Requantize] expected in_bits ({in_bits}) >= "
                f"out_bits ({out_bits}) >= 2."
            )

        self.in_bits = in_bits
        self.out_bits = out_bits
        # Minimum 65-bit signed width guarantees (2^47 - 1)*65535 + (1 << 61) never overflows
        self.sum_bits = max(in_bits + 17, 65)

        # Control & configuration inputs
        self.clear = Signal(name="clear")
        self.scale_u16 = Signal(16, name="scale_u16")
        self.shift = Signal(6, name="shift")
        self.symmetric = Signal(name="symmetric")

        # LiteX stream.Endpoint interfaces
        self.sink = stream.Endpoint(requantize_sink_description(in_bits=in_bits))
        self.source = stream.Endpoint(requantize_source_description(out_bits=out_bits))

        self.clock_domains.cd_sys = ClockDomain("sys")
        self.ios = {
            self.cd_sys.clk,
            self.cd_sys.rst,
            self.clear,
            self.scale_u16,
            self.shift,
            self.symmetric,
            self.sink.valid,
            self.sink.ready,
            self.sink.last,
            self.sink.result,
            self.sink.row_idx,
            self.source.valid,
            self.source.ready,
            self.source.last,
            self.source.result,
            self.source.row_idx,
            self.source.saturated,
        }

        # Stage 1 combinational signed x positive-unsigned multiply
        in_signed = Signal((in_bits, True), name="in_signed")
        scale_pos = Signal((17, True), name="scale_pos")
        prod_comb = Signal((self.sum_bits, True), name="prod_comb")

        self.comb += [
            in_signed.eq(self.sink.result),
            scale_pos.eq(self.scale_u16),
            prod_comb.eq(in_signed * scale_pos),
        ]

        # Stage 1 pipeline registers
        s1_valid = Signal(name="s1_valid")
        s1_prod = Signal((self.sum_bits, True), name="s1_prod")
        s1_row_idx = Signal(16, name="s1_row_idx")
        s1_last = Signal(name="s1_last")
        s1_shift = Signal(6, name="s1_shift")
        s1_sym = Signal(name="s1_sym")

        # Stage 2 output registers
        s2_valid = Signal(name="s2_valid")
        s2_result = Signal((out_bits, True), name="s2_result")
        s2_row_idx = Signal(16, name="s2_row_idx")
        s2_saturated = Signal(name="s2_saturated")
        s2_last = Signal(name="s2_last")

        # Interlocked II=1 valid/ready stall propagation
        s2_ready = Signal(name="s2_ready")
        s1_ready = Signal(name="s1_ready")
        self.comb += [
            s2_ready.eq((~s2_valid) | self.source.ready),
            s1_ready.eq((~s1_valid) | s2_ready),
            self.sink.ready.eq(s1_ready),
        ]

        # Stage 2 combinational 65-bit signed round-half-up shift (shift == 0 bypasses bias)
        shifted_val = Signal((self.sum_bits, True), name="shifted_val")
        shift_cases: dict[int | str, object] = {0: shifted_val.eq(s1_prod)}
        for s in range(1, 63):
            bias_c = Constant(1 << (s - 1), (self.sum_bits, True))
            shift_cases[s] = shifted_val.eq((s1_prod + bias_c) >> s)
        default_bias_c = Constant(1 << 61, (self.sum_bits, True))
        shift_cases["default"] = shifted_val.eq((s1_prod + default_bias_c) >> 62)
        self.comb += Case(s1_shift, shift_cases)

        # Stage 2 combinational signed int8 saturation & telemetry flag
        q_max = (1 << (out_bits - 1)) - 1
        q_min_asym = -(1 << (out_bits - 1))
        q_min_sym = -((1 << (out_bits - 1)) - 1)

        min_bound = Signal((self.sum_bits, True), name="min_bound")
        max_bound_c = Constant(q_max, (self.sum_bits, True))
        underflow = Signal(name="underflow")
        overflow = Signal(name="overflow")
        clamped_comb = Signal((out_bits, True), name="clamped_comb")
        sat_comb = Signal(name="sat_comb")

        self.comb += [
            If(
                s1_sym,
                min_bound.eq(Constant(q_min_sym, (self.sum_bits, True))),
            ).Else(
                min_bound.eq(Constant(q_min_asym, (self.sum_bits, True))),
            ),
            underflow.eq(shifted_val < min_bound),
            overflow.eq(shifted_val > max_bound_c),
            sat_comb.eq(underflow | overflow),
            If(
                underflow,
                clamped_comb.eq(min_bound[:out_bits]),
            )
            .Elif(
                overflow,
                clamped_comb.eq(q_max),
            )
            .Else(
                clamped_comb.eq(shifted_val[:out_bits]),
            ),
        ]

        # Drive source stream Endpoint from Stage 2 registers
        self.comb += [
            self.source.valid.eq(s2_valid),
            self.source.result.eq(s2_result),
            self.source.row_idx.eq(s2_row_idx),
            self.source.saturated.eq(s2_saturated),
            self.source.last.eq(s2_last),
        ]

        # Sequential 2-stage pipeline advancement
        self.sync += [
            If(
                self.clear,
                s1_valid.eq(0),
                s1_prod.eq(0),
                s1_row_idx.eq(0),
                s1_last.eq(0),
                s1_shift.eq(0),
                s1_sym.eq(0),
                s2_valid.eq(0),
                s2_result.eq(0),
                s2_row_idx.eq(0),
                s2_saturated.eq(0),
                s2_last.eq(0),
            ).Else(
                If(
                    s2_ready,
                    s2_valid.eq(s1_valid),
                    If(
                        s1_valid,
                        s2_result.eq(clamped_comb),
                        s2_row_idx.eq(s1_row_idx),
                        s2_saturated.eq(sat_comb),
                        s2_last.eq(s1_last),
                    ),
                ),
                If(
                    s1_ready,
                    s1_valid.eq(self.sink.valid),
                    If(
                        self.sink.valid,
                        s1_prod.eq(prod_comb),
                        s1_row_idx.eq(self.sink.row_idx),
                        s1_last.eq(self.sink.last),
                        s1_shift.eq(self.shift),
                        s1_sym.eq(self.symmetric),
                    ),
                ),
            )
        ]
