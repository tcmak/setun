# Setun `TernaryAccel` CSR Register Contract (Phase 3 · Tasks 3.1b & 3.2)

Status: Accepted (`2026-11-21`, Task `3.1b`) · Source of truth: `setun/soc/ternary_accel.py` · Generated artifacts: `reports/csr.json`, `reports/csr.h`

---

## 1. Overview & `AutoCSR` Offset Stability

`TernaryAccel` wraps the multi-row `TernaryTile` (`ADR-015`) and 2-stage `Requantize` (`docs/NUMERICS.md` §4) cores behind a LiteX `AutoCSR` register bank (`CSR_TERNARY_ACCEL_BASE = 0x00001800` in `SetunSimSoC`).

- **Default SoC geometry:** `rows = 4`, `lanes = 64`, `group_size = 128`, `beat_bits = 256`, `max_k = 2048`, `max_rows = 2048`.
- **Offset stability invariant:** Registers `#0`–`#9` (`ctrl` through `out_row_data`, `0x1800`–`0x1824`) preserve their exact Task `3.1a` addresses and bit offsets. All Phase-3 multi-row, DMA pointer, requantization, capability, and performance registers (`#10`–`#21`, `0x1828`–`0x1854`) are appended strictly after `out_row_data`.
- **Synchronous BRAM scratchpads:** Both the quantized `int8` activation buffer (`max_k // 4` 32-bit words) and the row output buffer (`max_rows` 32-bit words), plus the `(R - 1)` row-to-stripe staging banks, use synchronous read ports (`async_read=False`) so Yosys infers dedicated block RAM (`RAMB18E1`/`RAMB36E1` on Xilinx XC7; `DP16KD` on Lattice ECP5) rather than LUT/FF arrays.

---

## 2. Register Map (`CSR_TERNARY_ACCEL_BASE + offset`)

| # | Offset | Name | Dir | Fields (`[msb:lsb]`) & Semantics |
|---:|---:|---|---|---|
| 0 | `0x00` | `ctrl` | RW | `[0] enable` (reset `1`): allow stream handshakes<br>`[1] clear` (pulse): reset counters, pointers, and sticky flags<br>`[2] start` (pulse): atomically clear sticky status/errors, validate config, load staged activations from sync BRAM into `TernaryTile`, and arm GEMV<br>`[3] req_enable` (reset `0`): `0` = store shifted `int32` (`result_lo`/`result_hi`), `1` = pass `TernaryTile.source` through `Requantize` (`int8` sign-extended to `int32`)<br>`[4] req_sym` (reset `0`): `Requantize` clamp (`0` = `[-128, +127]`, `1` = `[-127, +127]`) |
| 1 | `0x04` | `cfg` | RW | `[5:0] shift` (reset `0`): tile round-half-up right shift in `[0, 30]`<br>`[23:8] num_rows` (reset `1`): target output rows `N` (`1..max_rows`) |
| 2 | `0x08` | `info` | RO | `[7:0] lanes`, `[19:8] group_size`, `[31:20] beat_bits` (legacy `3.1a` packed geometry) |
| 3 | `0x0C` | `status` | RO | `[0] busy`: `1` while run active (`loading_acts` or `rows_completed < num_rows`) and `error == 0`<br>`[1] done`: `1` when `error == 0`, `num_rows != 0`, and `rows_completed >= num_rows`<br>`[2] trit_ready`: live `unpacker.trit_ready`<br>`[3] scale_ready`: live `unpacker.scale_ready`<br>`[4] error`: sticky `1` if last `ctrl.start` failed parameter/alignment checks<br>`[5] req_saturated`: sticky `1` if `Requantize` clipped any row during current run<br>`[11:8] err_code`: sticky 4-bit error bitmask (see §3)<br>`[31:16] rows_completed`: completed output rows since last `clear`/`start` |
| 4 | `0x10` | `act_word_idx` | RW | `[15:0]`: 32-bit word write pointer into activation scratchpad BRAM (`0 .. max_k/4 - 1`) |
| 5 | `0x14` | `act_word_data` | RW | `[31:0]`: write 4 packed `int8` activations (little-endian) to BRAM and auto-increment pointer |
| 6 | `0x18` | `result_lo` | RO | `[31:0]`: lower 32 bits of most recently committed row output |
| 7 | `0x1C` | `result_hi` | RO | `[31:0]`: sign-extended upper bits `[47:32]` of most recently committed row output |
| 8 | `0x20` | `out_row_idx` | RW | `[15:0]`: row selector driving synchronous read port of output scratchpad BRAM |
| 9 | `0x24` | `out_row_data` | RO | `[31:0]`: signed `int32` output word from synchronous BRAM at `out_row_idx` |
| 10 | `0x28` | `tile_info` | RO | `[7:0] rows` (`R`), `[15:8] lanes` (`L`), `[31:16] hw_group_size` (`G`) |
| 11 | `0x2C` | `buf_caps` | RO | `[15:0] max_k` (up to `65535` for M1 FFN), `[31:16] max_rows` (up to `65535`) |
| 12 | `0x30` | `dim_k` | RW | `[15:0] k_cols` (`K`, reset `group_size`), `[31:16] group_size` (`G`, reset `group_size`) |
| 13 | `0x34` | `weight_addr` | RW | `[31:0]`: 4-byte-aligned byte pointer to `.ttape` `packed_trits` in RAM (`0` = CSR/stream push) |
| 14 | `0x38` | `scale_addr` | RW | `[31:0]`: 4-byte-aligned byte pointer to `.ttape` `scales_u16` in RAM (`0` = CSR/stream push) |
| 15 | `0x3C` | `act_addr` | RW | `[31:0]`: 4-byte-aligned byte pointer to `int8` activations in RAM (`0` = use staged BRAM) |
| 16 | `0x40` | `out_addr` | RW | `[31:0]`: 4-byte-aligned byte pointer for DMA output writeback (`0` = keep in output BRAM) |
| 17 | `0x44` | `act_scale` | RW | `[31:0]`: host per-token activation dequantization scale metadata (IEEE-754 float32 bits) |
| 18 | `0x48` | `req_cfg` | RW | `[15:0] scale_u16` (reset `1`), `[21:16] shift` (reset `0`, range `0..62`) for `Requantize` |
| 19 | `0x4C` | `perf_cycles` | RO | `[31:0]`: elapsed `sys_clk` cycles while `status.busy == 1` in current run |
| 20 | `0x50` | `perf_bus_beats` | RO | `[31:0]`: accepted weight stream / DMA bus beats in current run |
| 21 | `0x54` | `perf_sat_rows` | RO | `[31:0]`: number of rows clipped by `Requantize` (`saturated == 1`) in current run |

---

## 3. Parameter Validation, Error Codes & Sticky Recovery (`ctrl.start`)

Pulsing `ctrl.start = 1` **first** clears prior `rows_completed`, `error`, `err_code`, `req_saturated`, and `perf_*` counters, and **then** evaluates all four hardware legality checks combinationally on the same cycle:

| Bit | Mask | Constant | Trigger Condition |
|---:|---:|---|---|
| `err_code[0]` | `0x1` | `ERR_K_INVALID` | `dim_k.k_cols == 0` or `dim_k.k_cols > buf_caps.max_k` |
| `err_code[1]` | `0x2` | `ERR_N_INVALID` | `cfg.num_rows == 0` or `cfg.num_rows > buf_caps.max_rows` |
| `err_code[2]` | `0x4` | `ERR_G_MISMATCH` | `dim_k.group_size != tile_info.hw_group_size` |
| `err_code[3]` | `0x8` | `ERR_ADDR_ALIGN` | Any of `weight_addr`, `scale_addr`, `act_addr`, `out_addr` has `addr[1:0] != 0` (`0x0` is aligned/valid) |

- **Non-hanging guarantee:** Whenever `err_code != 0`, hardware sets `status.error = 1`, `status.busy = 0`, and `status.done = 0` immediately without entering the activation-load or DMA FSM, and gates off `trit_ready`/`scale_ready`.
- **Sticky recovery without SoC reset:** Writing corrected CSRs and pulsing `ctrl.start` (or `ctrl.clear`) clears `error` and `err_code` atomically and launches the valid GEMV cleanly.

---

## 4. Launch & Synchronous BRAM Readback Sequence

1. Stage quantized `int8` activations via `act_word_idx = 0` followed by `ceil(K / G) * (G / 4)` writes to `act_word_data` (or specify `act_addr` in Task `3.2`).
2. Write `cfg` (`shift`, `num_rows = N`), `dim_k` (`k_cols = K`, `group_size = G`), optional `req_cfg`, and pointer CSRs (`0` in stream-push mode).
3. Write `ctrl` with `enable = 1, start = 1` (+ optional `req_enable`, `req_sym`).
4. Check `status.error`; if `0`, feed `.ttape` row-major weights/scales (or let Task `3.2` DMA stream them) and poll `status` until `status.done == 1`.
5. For each row `r in 0 .. N - 1`, write `out_row_idx = r` and read `out_row_data` (the Wishbone CSR turnaround provides `>= 3` `sys_clk` cycles, exceeding the 1-cycle synchronous BRAM read latency; in bare Migen cycle-level tests without Wishbone, wait 2 `sys_clk` cycles after updating `out_row_idx.storage` before reading `out_row_data.status`).
