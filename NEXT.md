# Next Up

- **Current Season:** Season 2 (Streaming `TernaryTile`, M0 golden model, and first token in SimSoC — Gate 3)
- **Completed Seasons:** Season 1 (`0.1`, `0.2a`, `0.3`, `0.5a/b`, `0.6a`, `1.3a`, `1.4a`, `2.1`, `2.2`, `2.6`, `2.8`, `S1.end` ADRs)
- **Completed in Season 2:**
  - **Block A (`1.5`, `1.6` — COMPLETE):** `setun/ref/m0.py`, `scripts/make_toy.py`, frozen `TTAPE` `spec-v1.0` submodule (`third_party/TernaryTape`), and end-to-end `Lux.Chain` grouped `.ttape` golden verification.
  - **Block B (`2.3`, `2.4`, `2.5`, `2.8b` — COMPLETE):** `WeightUnpacker`, multi-row `TernaryTile` (`ADR-015`), 2-stage `II=1` `Requantize` (`NUMERICS.md` §4), PR/nightly (`>= 10^5`) Verilator regression sweeps, and multi-vendor synthesis reports.
  - **Block C (`0.4`, `3.1a`, `3.1b`, `3.2`, `3.2b` — COMPLETE):** `AccelStub`, `docs/CSR.md`, multi-row `TernaryAccel` with synchronous BRAM inference, 32-bit Wishbone burst DMA (`86%–97%+` bus utilization, `--ram-init`), and zero-copy `4 x int8` coalesced output DMA writer (`sel = (1 << (N % 4)) - 1` byte-masked tail flush).
- **Remaining Season 2 Tasks (Block D — Hybrid Decode & Gate 3):**
  1. **`3.3a` — `AccelDriver` Python Sim Backend (`setun/runtime/`):** Host driver API (`write_csr`, `read_csr`, `dma_write`, `dma_read`, `launch`, `wait`, `perf`) selected by `--backend ref|sim|board`.
  2. **`3.4` — Hybrid M0 Decode Loop & `setun run` CLI (`setun/cli.py`):** `setun run <model.ttape> --backend ref|sim --tokens N` running all 14 M0 linear projections on `SetunSimSoC` `TernaryAccel` (`16` tokens bit-exact vs. `--backend ref`), logging CSR `cycles/tok`, bus utilization %, and `MB/s` to stderr.
  3. **`3.5` — `setun diff` First-Divergence Debugger (`setun/cli.py`):** Localize injected 1-bit GEMV faults to exact op and row.
  4. **`3.6` — System-Sim CI (`make sim-m0`, Gate 3):** Run `setun run m0_toy.ttape --backend sim --tokens 16` bit-exact vs. ref with randomized `reset_less` register init and SDRAM wait-state jitter.
