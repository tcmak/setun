# Next Up

- **Current Season:** Season 2 (Streaming `TernaryTile`, M0 golden model, and first token in SimSoC — Gate 3)
- **Completed Seasons:** Season 1 (`0.1`, `0.2a`, `0.3`, `0.5a/b`, `0.6a`, `1.3a`, `1.4a`, `2.1`, `2.2`, `2.6`, `2.8`, `S1.end` ADRs)
- **Completed in Season 2:**
  - **Block A (`1.5`, `1.6` — COMPLETE):** `setun/ref/m0.py`, `scripts/make_toy.py`, frozen `TTAPE` `spec-v1.0` submodule (`third_party/TernaryTape`), and end-to-end `Lux.Chain` grouped `.ttape` golden verification.
  - **Block B (`2.3`, `2.4`, `2.5`, `2.8b` — COMPLETE):** `WeightUnpacker`, multi-row `TernaryTile` (`ADR-015`), 2-stage `II=1` `Requantize` (`NUMERICS.md` §4), PR/nightly (`>= 10^5`) Verilator regression sweeps, and multi-vendor synthesis reports.
  - **Block C (`0.4`, `3.1a`, `3.1b`):** `AccelStub`, `docs/CSR.md`, and multi-row `TernaryAccel` (`R=4, L=64, G=128, max_k=2048, max_rows=2048`) wrapping `TernaryTile` + `Requantize` with synchronous BRAM inference (`9 RAMB18/36` on XC7, `17 DP16KD` on ECP5, `0 DSPs`), sticky `req_saturated` telemetry, non-hanging error recovery, and deterministic `reports/csr.{json,h}`.
- **Next Tasks (in `PLAN.md` order):**
  1. **`3.2` — SimSoC SDRAM/RAM Weight DMA:** Burst Wishbone DMA reader (`bus_bursting=True`, `CTI_BURST_INCREMENTING`) feeding `TernaryAccel` across `N, K in {64, 256, 2048}` at `>= 80%` steady-state bus utilization, plus `--ram-init` preload support.
  2. **`3.3a` — `AccelDriver` Sim Backend:** Host Python driver (`setun/runtime/`) supporting cocotb/sim & Etherbone backends.
  3. **`3.4`, `3.5`, `3.6` — Hybrid M0 Decode, `setun diff`, & Gate 3 CI:** Full 14-projection M0 greedy token generation bit-exact vs. `setun/ref/m0.py`.
