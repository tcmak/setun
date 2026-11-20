# Setun Loader & In-Memory Weight Representation

> **Single Wire-Format Source of Truth:** Setun does **not** define its own binary container format. All serialized `.ttape` files produced or consumed by `setun` conform strictly to the canonical **`TTAPE` v1.0 (Draft) Specification** in [`../third_party/TernaryTape/spec/ttape-v1-draft.md`](../third_party/TernaryTape/spec/ttape-v1-draft.md) (vendored from `TernaryTape.jl` at `spec-v0.2-draft`), verified against the shared golden corpus in [`../third_party/TernaryTape/fixtures/`](../third_party/TernaryTape/fixtures/).

This document specifies **only** Setun-specific loader constraints and how `setun/ref/format.py` maps validated `.ttape` sections into `PackedTernaryTensor` in-memory hardware buffers (`ADR-003`, `ADR-004`, `ADR-011`, `ADR-014`).

---

## 1. Supported `TTAPE` Profiles (`ADR-014`)

Setun's loader (`setun.ref.format.validate_ttape` and `load_ttape_layers`) defaults to `reader_profile="grouped"`, supporting the two integer-datapath profiles and failing closed on host-FP32 extensions unless explicitly requested:

1. **`TTAPE-core` (`flags` bit 2 = `0`):**
   * Contains base sections `0x0001` (`weights`), `0x0002` (`scales`), and `0x0003` (`arch_descriptor`) (plus optional `0x0004` `manifest` / `0x0005` `link_table`).
   * Each `ternary_dense` layer carries a single signed `s16` power-of-two exponent `scale_exp` in `0x0002` (`[-30, +30]`).
   * **Setun Shift-Mode Mapping:** Loaded into `PackedTernaryTensor` with `group_size = 64` ($\lceil K / 64 \rceil$ groups per row):
     * When `scale_exp <= 0`: `scales = uint16(1)` across all groups and `shift = -scale_exp` (`0 DSPs` shift-only execution).
     * When `0 < scale_exp <= 15`: `scales = uint16(1 << scale_exp)` across all groups and `shift = 0`.
     * Exponents `16 <= scale_exp <= 30` exceed `uint16` group-scale mantissa range and raise a Goose-style `TtapeValidationError` on load into `PackedTernaryTensor`.
2. **`TTAPE-grouped` (`flags` bit 2 = `1`, tag `0x0006` `group_scales_u16`):**
   * Carries explicit `group_size` $G \in \{64, 128, 256\}$, shared right-shift `shift` $\in [0, 30]$, `n_groups` $= \lceil K / G \rceil$, and row-major `uint16` scale mantissas of shape `(N, n_groups)` in section `0x0006` (`scales_s16 == -shift` in `0x0002`).
   * Effective real group scale: $\text{scale}_{\text{real}}(n, g) = \text{scale\_u16}(n, g) \cdot 2^{-\text{shift}}$.
3. **`TTAPE-extended` (`flags` bit 2 = `1`, tag `0x0007` `fp32_aux`):**
   * Rejected fail-closed with `TTAPE_ERR_UNSUPPORTED_PROFILE` under Setun's default `reader_profile="grouped"` gate (since `0x0007` `fp32_aux` and `layer_kind in {6, 7}` are host-consumed FP32 tensors, never hardware-visible). Pass `reader_profile="extended"` to `validate_ttape` / `load_ttape_layers` when inspecting hybrid M0/Lux tapes on the host.

---

## 2. Ragged-$K$ Wire-to-Memory Padding (`PackedTernaryTensor`)

In `TTAPE` section `0x0001` (`weights`), each `ternary_dense` row of logical width $K = \text{dim\_in}$ is padded to a **byte boundary** ($\text{row\_bytes} = \lfloor (K + 3) / 4 \rfloor$, 4 LSB-first 2-bit trits per byte, trailing lanes `0b00`).

In Setun's streaming hardware (`WeightUnpacker` → `TernaryDot` → `GroupAccumulateScale`), rows are streamed in full groups of $G \in \{64, 128, 256\}$ trits ($G / 4$ bytes per group, $N_{\text{groups}} = \lceil K / G \rceil$, $K_{\text{pad}} = N_{\text{groups}} \cdot G$):

* **Aligned Dimensions ($K \bmod G = 0$, e.g., all `M0` projections):** $\text{row\_bytes} = N_{\text{groups}} \cdot (G / 4)$. Wire bytes in `0x0001` and `PackedTernaryTensor.packed_trits` (`uint8[N, n_groups, G // 4]`) are **1:1 identical**.
* **Ragged Dimensions ($K \bmod G \ne 0$, e.g., `accept/grouped_with_u16_scales.ttape` with $N=1, K=6, G=64$):**
  * **On wire (`0x0001`):** `row_bytes = (6 + 3) // 4 = 2` bytes (`49 0a`), with lanes `6..7` verified to be `0b00` (`TTAPE_ERR_NONZERO_PADDING` otherwise).
  * **In memory (`PackedTernaryTensor`):** `deserialize_packed_weights` / `load_ttape_layers` unpacks the $K$ logical trits, zero-pads trailing elements $k \in [K, K_{\text{pad}})$ with `0` (`0b00`), and repacks into contiguous shape `(N, n_groups, G // 4)` (`16` bytes `49 0a 00 ... 00` for $G=64$). Because $0 \cdot x_k = 0$, zero-padded group tails contribute zero to `GroupAccumulateScale` without dynamic lane-masking hardware.
  * **On serialize (`serialize_packed_weights`):** Unpacks `PackedTernaryTensor` (verifying any in-memory tail trits $\ge K$ are zero), packs each logical row of $K$ trits to `row_bytes = (K + 3) // 4` bytes in `0x0001`, and emits a canonical single-layer `TTAPE-grouped` `.ttape` artifact.

---

## 3. Normative `0b11` Rejection vs. Silicon Defense-in-Depth (`ADR-003`)

* **Software / Loader Contract:** Code `0b11` (`0x3`) is strictly reserved. Both `validate_ttape` (`TTAPE_ERR_RESERVED_TERNARY_CODE`) and `unpack_trits_1d` (`ValueError`) reject any active lane containing `0b11` before weights can enter simulation or DMA buffers.
* **RTL Defense-in-Depth:** Inside `setun/accelerators/ternary_dot.py`, the combinational 2-bit lane mux maps `0b11 -> 0` solely as a deterministic safe hardware fallback against uninitialized SRAM/FIFO beats prior to `valid` qualification.
