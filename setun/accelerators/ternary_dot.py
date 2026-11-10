"""L-lane combinational/pipelined ternary dot-product unit (Task 2.1)."""

from __future__ import annotations

from migen import If, Module, Signal

from setun.ref.ops import dot_bit_width


class TernaryDot(Module):
    """Compute sum_{i=0}^{L-1} w_i * x_i using 2-bit trit muxes and a balanced adder tree.

    Parameters
    ----------
    lanes : int
        Number of parallel lanes L (power of two, e.g. 16..128).
    act_bits : int
        Signed activation bit width A (8 or 16).
    pipeline_depth : int
        Number of synchronous pipeline stages (0 = combinational, 1 = output reg,
        2 = mid-tree reg + output reg). Fixed latency = `pipeline_depth` cycles.
    registered : bool | None
        Backwards-compatible alias: if True, sets pipeline_depth=1.
    """

    def __init__(
        self,
        lanes: int = 64,
        act_bits: int = 8,
        pipeline_depth: int = 0,
        registered: bool | None = None,
    ) -> None:
        if lanes < 1 or (lanes & (lanes - 1)) != 0:
            raise ValueError(f"lanes must be a power of 2 >= 1, got {lanes}.")
        if act_bits not in (8, 16):
            raise ValueError(f"act_bits must be 8 or 16, got {act_bits}.")
        if registered is not None:
            pipeline_depth = 1 if registered else 0
        if pipeline_depth not in (0, 1, 2):
            raise ValueError(f"pipeline_depth must be 0, 1, or 2, got {pipeline_depth}.")

        self.lanes = lanes
        self.act_bits = act_bits
        self.pipeline_depth = pipeline_depth
        self.latency = pipeline_depth
        self.out_bits = dot_bit_width(lanes, act_bits=act_bits)

        # Ports
        self.trits_packed = Signal(2 * lanes, name="trits_packed")
        self.acts_packed = Signal(act_bits * lanes, name="acts_packed")
        self.in_valid = Signal(name="in_valid")
        self.result = Signal((self.out_bits, True), name="result")
        self.out_valid = Signal(name="out_valid")

        self.ios = {
            self.trits_packed,
            self.acts_packed,
            self.in_valid,
            self.result,
            self.out_valid,
        }

        # Stage 0: Per-lane (act_bits + 1)-bit signed mux {00: 0, 01: +x, 10: -x, 11: 0}
        lane_w = act_bits + 1
        level_signals: list[Signal] = []
        for i in range(lanes):
            code = self.trits_packed[2 * i : 2 * i + 2]
            raw_act = self.acts_packed[act_bits * i : act_bits * (i + 1)]

            x_signed = Signal((act_bits, True), name=f"x_s_{i}")
            x_ext = Signal((lane_w, True), name=f"x_ext_{i}")
            lane_out = Signal((lane_w, True), name=f"lane_out_{i}")

            self.comb += [
                x_signed.eq(raw_act),
                x_ext.eq(x_signed),
                If(
                    code == 0b01,
                    lane_out.eq(x_ext),
                )
                .Elif(
                    code == 0b10,
                    lane_out.eq(-x_ext),
                )
                .Else(
                    lane_out.eq(0),
                ),
            ]
            level_signals.append(lane_out)

        # Balanced binary adder tree with +1 bit growth per level
        total_levels = lanes.bit_length() - 1
        mid_level = total_levels // 2
        valid_s1 = Signal(name="valid_s1")

        lvl = 0
        while len(level_signals) > 1:
            next_level: list[Signal] = []
            cur_width = len(level_signals[0]) + 1
            for j in range(0, len(level_signals), 2):
                s = Signal((cur_width, True), name=f"tree_l{lvl}_{j // 2}")
                self.comb += s.eq(level_signals[j] + level_signals[j + 1])
                if pipeline_depth == 2 and lvl == mid_level:
                    s_reg = Signal((cur_width, True), name=f"tree_l{lvl}_reg_{j // 2}")
                    self.sync += s_reg.eq(s)
                    next_level.append(s_reg)
                else:
                    next_level.append(s)
            level_signals = next_level
            lvl += 1

        if pipeline_depth == 2:
            self.sync += valid_s1.eq(self.in_valid)
        else:
            self.comb += valid_s1.eq(self.in_valid)

        tree_sum = level_signals[0]

        if pipeline_depth >= 1:
            self.sync += [
                self.result.eq(tree_sum),
                self.out_valid.eq(valid_s1),
            ]
        else:
            self.comb += [
                self.result.eq(tree_sum),
                self.out_valid.eq(valid_s1),
            ]
