"""Streaming packed-weight unpacker and optional 5-trits-per-byte LUT decoder (Task 2.3)."""

from __future__ import annotations

from migen import Array, Case, ClockDomain, If, Module, Signal

from setun.ref.format import CODE_NEG, CODE_POS, CODE_ZERO


class WeightUnpacker(Module):
    """Unpack wide packed trit beats (beat_bits) and aligned uint16 group scales
    into an L-lane trit stream + aligned scale_u16 stream at 1 beat/cycle with zero bubbles.

    Parameters
    ----------
    beat_bits : int
        Width in bits of the incoming packed trit bus (e.g., 128 or 256).
        Must be a positive multiple of (2 * lanes).
    lanes : int
        Number of parallel ternary lanes L emitted per output beat (default 64).
    group_size : int
        Number of weights G sharing one uint16 scale (default 128).
        Must be a positive multiple of lanes.
    """

    def __init__(
        self,
        beat_bits: int = 256,
        lanes: int = 64,
        group_size: int = 128,
    ) -> None:
        self.out_trit_bits = 2 * lanes
        if beat_bits < self.out_trit_bits or beat_bits % self.out_trit_bits != 0:
            raise ValueError(
                f"beat_bits ({beat_bits}) must be a positive multiple of "
                f"2*lanes ({self.out_trit_bits})."
            )
        if group_size < lanes or group_size % lanes != 0:
            raise ValueError(
                f"group_size ({group_size}) must be a positive multiple of lanes ({lanes})."
            )

        self.beat_bits = beat_bits
        self.lanes = lanes
        self.group_size = group_size
        self.slices_per_beat = beat_bits // self.out_trit_bits
        self.beats_per_group = group_size // lanes

        # Upstream packed trit stream (AXI-Stream / LiteX stream semantics)
        self.trit_data = Signal(beat_bits, name="trit_data")
        self.trit_last = Signal(name="trit_last")
        self.trit_valid = Signal(name="trit_valid")
        self.trit_ready = Signal(name="trit_ready")

        # Upstream aligned group scale stream (1 uint16 per G weights)
        self.scale_data = Signal(16, name="scale_data")
        self.scale_valid = Signal(name="scale_valid")
        self.scale_ready = Signal(name="scale_ready")

        # Downstream L-lane beat stream (connects directly to TernaryRow / TernaryTile)
        self.trits_packed = Signal(self.out_trit_bits, name="trits_packed")
        self.scale_u16 = Signal(16, name="scale_u16")
        self.row_last = Signal(name="row_last")
        self.group_last = Signal(name="group_last")
        self.out_valid = Signal(name="out_valid")
        self.out_ready = Signal(name="out_ready")

        self.clock_domains.cd_sys = ClockDomain("sys")
        self.ios = {
            self.cd_sys.clk,
            self.cd_sys.rst,
            self.trit_data,
            self.trit_last,
            self.trit_valid,
            self.trit_ready,
            self.scale_data,
            self.scale_valid,
            self.scale_ready,
            self.trits_packed,
            self.scale_u16,
            self.row_last,
            self.group_last,
            self.out_valid,
            self.out_ready,
        }

        # Sub-word slice index within the incoming beat_bits word
        slice_max = max(self.slices_per_beat, 2)
        slice_idx = Signal(max=slice_max, name="slice_idx")

        # Beat counter within the current scale group (0 .. beats_per_group - 1)
        grp_max = max(self.beats_per_group, 2)
        grp_beat_idx = Signal(max=grp_max, name="grp_beat_idx")

        # Combinational slice selection from incoming beat_bits word
        if self.slices_per_beat == 1:
            self.comb += self.trits_packed.eq(self.trit_data)
        else:
            slices = Array(
                [
                    self.trit_data[i * self.out_trit_bits : (i + 1) * self.out_trit_bits]
                    for i in range(self.slices_per_beat)
                ]
            )
            self.comb += self.trits_packed.eq(slices[slice_idx])

        is_last_slice = Signal(name="is_last_slice")
        is_last_grp_beat = Signal(name="is_last_grp_beat")
        fire = Signal(name="fire")

        if self.slices_per_beat == 1:
            self.comb += is_last_slice.eq(1)
        else:
            self.comb += is_last_slice.eq(slice_idx == (self.slices_per_beat - 1))

        if self.beats_per_group == 1:
            self.comb += is_last_grp_beat.eq(1)
        else:
            self.comb += is_last_grp_beat.eq(
                (grp_beat_idx == (self.beats_per_group - 1)) | (self.trit_last & is_last_slice)
            )

        # Zero-bubble join: output is valid iff both trit word and group scale are valid
        self.comb += [
            self.scale_u16.eq(self.scale_data),
            self.out_valid.eq(self.trit_valid & self.scale_valid),
            self.group_last.eq(is_last_grp_beat),
            self.row_last.eq(self.trit_last & is_last_slice),
            fire.eq(self.out_valid & self.out_ready),
            # Consume upstream trit word only on its final slice
            self.trit_ready.eq(self.out_ready & self.scale_valid & is_last_slice),
            # Consume upstream scale word only on the final beat of the group
            self.scale_ready.eq(self.out_ready & self.trit_valid & is_last_grp_beat),
        ]

        # Sequential slice & group beat counters
        if self.slices_per_beat > 1:
            self.sync += [
                If(
                    fire,
                    If(
                        is_last_slice,
                        slice_idx.eq(0),
                    ).Else(
                        slice_idx.eq(slice_idx + 1),
                    ),
                )
            ]

        if self.beats_per_group > 1:
            self.sync += [
                If(
                    fire,
                    If(
                        is_last_grp_beat,
                        grp_beat_idx.eq(0),
                    ).Else(
                        grp_beat_idx.eq(grp_beat_idx + 1),
                    ),
                )
            ]


def decode_5trits_byte_ref(byte_val: int) -> int:
    """Reference base-3 decoder: unpack 1 byte (0..242) into 5 x 2-bit LSB-first trit codes.

    Values 243..255 (out of 3^5 range) map to 0 (all zero trits).
    """
    if not 0 <= byte_val <= 255:
        raise ValueError(f"byte_val must be in 0..255, got {byte_val}")
    if byte_val >= 243:
        return 0

    rem = byte_val
    packed_10b = 0
    for i in range(5):
        digit = rem % 3  # 0 -> -1, 1 -> 0, 2 -> +1 (standard I2_S / TQ1_0 base-3 mapping)
        rem //= 3
        if digit == 0:
            code = CODE_NEG
        elif digit == 1:
            code = CODE_ZERO
        else:
            code = CODE_POS
        packed_10b |= code << (2 * i)
    return packed_10b


class Decode5TritsByte(Module):
    """Combinational 256-entry LUT decoding 1 packed base-3 byte into 5 x 2-bit trit codes."""

    def __init__(self) -> None:
        self.byte_in = Signal(8, name="byte_in")
        self.trits_out = Signal(10, name="trits_out")
        self.ios = {self.byte_in, self.trits_out}

        cases = {b: self.trits_out.eq(decode_5trits_byte_ref(b)) for b in range(256)}
        self.comb += Case(self.byte_in, cases)
