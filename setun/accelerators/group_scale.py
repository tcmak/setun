"""Group accumulate, scale multiply, and cross-group row accumulator (Task 2.2)."""

from __future__ import annotations

from migen import If, Module, Signal

from setun.accelerators.ternary_dot import TernaryDot
from setun.ref.ops import dot_bit_width, group_sum_bit_width


class GroupAccumulateScale(Module):
    """Accumulate G/L dot-product beats into a group sum, multiply by scale_u16,
    and accumulate across groups into a wide fixed-point row accumulator.

    Uses exactly 1 multiplier (group_sum * scale_u16) per row engine.
    """

    def __init__(
        self,
        group_size: int = 128,
        lanes: int = 64,
        act_bits: int = 8,
        acc_bits: int = 48,
        shift: int = 0,
    ) -> None:
        if group_size < lanes or group_size % lanes != 0:
            raise ValueError(
                f"group_size ({group_size}) must be a positive multiple of lanes ({lanes})."
            )
        if shift < 0:
            raise ValueError(f"shift must be >= 0, got {shift}.")

        self.group_size = group_size
        self.lanes = lanes
        self.act_bits = act_bits
        self.acc_bits = acc_bits
        self.shift = shift
        self.beats_per_group = group_size // lanes

        self.dot_bits = dot_bit_width(lanes, act_bits=act_bits)
        self.group_bits = group_sum_bit_width(group_size, act_bits=act_bits)
        self.prod_bits = self.group_bits + 17  # signed group_sum * unsigned 16-bit scale

        # Input ports (1 beat per valid cycle)
        self.dot_in = Signal((self.dot_bits, True), name="dot_in")
        self.scale_u16 = Signal(16, name="scale_u16")
        self.in_valid = Signal(name="in_valid")
        self.row_last = Signal(name="row_last")

        # Output ports (asserted 1 cycle after the beat where row_last=1)
        self.result = Signal((acc_bits, True), name="result")
        self.out_valid = Signal(name="out_valid")

        self.ios = {
            self.dot_in,
            self.scale_u16,
            self.in_valid,
            self.row_last,
            self.result,
            self.out_valid,
        }

        # Internal state
        beat_max = max(self.beats_per_group, 2)
        beat_idx = Signal(max=beat_max, name="beat_idx")
        group_acc = Signal((self.group_bits, True), name="group_acc")
        row_acc = Signal((acc_bits, True), name="row_acc")

        # Combinational current-beat group partial sum
        cur_group_sum = Signal((self.group_bits, True), name="cur_group_sum")
        is_last_beat = Signal(name="is_last_beat")

        if self.beats_per_group == 1:
            self.comb += [
                cur_group_sum.eq(self.dot_in),
                is_last_beat.eq(1),
            ]
        else:
            self.comb += [
                If(
                    beat_idx == 0,
                    cur_group_sum.eq(self.dot_in),
                ).Else(
                    cur_group_sum.eq(group_acc + self.dot_in),
                ),
                is_last_beat.eq((beat_idx == (self.beats_per_group - 1)) | self.row_last),
            ]

        # Single signed x unsigned multiplier per row
        scale_signed = Signal((17, True), name="scale_signed")
        scaled_group = Signal((self.prod_bits, True), name="scaled_group")
        next_row_acc = Signal((acc_bits, True), name="next_row_acc")

        self.comb += [
            scale_signed.eq(self.scale_u16),
            scaled_group.eq(cur_group_sum * scale_signed),
            next_row_acc.eq(row_acc + scaled_group),
        ]

        # Optional round-half-up right shift on final row output
        shifted_out = Signal((acc_bits, True), name="shifted_out")
        if shift == 0:
            self.comb += shifted_out.eq(next_row_acc)
        else:
            bias = 1 << (shift - 1)
            self.comb += shifted_out.eq((next_row_acc + bias) >> shift)

        # Sequential state updates
        self.sync += [
            self.out_valid.eq(0),
            If(
                self.in_valid,
                If(
                    is_last_beat,
                    beat_idx.eq(0),
                    group_acc.eq(0),
                    If(
                        self.row_last,
                        self.result.eq(shifted_out),
                        self.out_valid.eq(1),
                        row_acc.eq(0),
                    ).Else(
                        row_acc.eq(next_row_acc),
                    ),
                ).Else(
                    beat_idx.eq(beat_idx + 1),
                    group_acc.eq(cur_group_sum),
                ),
            ),
        ]


class TernaryRow(Module):
    """Single-row streaming ternary dot + group accumulate + scale unit (2.1 + 2.2)."""

    def __init__(
        self,
        group_size: int = 128,
        lanes: int = 64,
        act_bits: int = 8,
        acc_bits: int = 48,
        shift: int = 0,
    ) -> None:
        self.group_size = group_size
        self.lanes = lanes
        self.act_bits = act_bits

        self.submodules.dot = TernaryDot(lanes=lanes, act_bits=act_bits, pipeline_depth=0)
        self.submodules.acc = GroupAccumulateScale(
            group_size=group_size,
            lanes=lanes,
            act_bits=act_bits,
            acc_bits=acc_bits,
            shift=shift,
        )

        self.trits_packed = self.dot.trits_packed
        self.acts_packed = self.dot.acts_packed
        self.scale_u16 = self.acc.scale_u16
        self.in_valid = self.acc.in_valid
        self.row_last = self.acc.row_last
        self.result = self.acc.result
        self.out_valid = self.acc.out_valid

        self.comb += [
            self.dot.in_valid.eq(self.in_valid),
            self.acc.dot_in.eq(self.dot.result),
        ]

        self.ios = {
            self.trits_packed,
            self.acts_packed,
            self.scale_u16,
            self.in_valid,
            self.row_last,
            self.result,
            self.out_valid,
        }
