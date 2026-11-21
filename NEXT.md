# Next Up

- **Current Season:** Season 2 (Streaming `TernaryTile`, M0 golden model, and first token in SimSoC — Gate 3)
- **Completed Seasons:** Season 1 (`0.1`, `0.2a`, `0.3`, `0.5a/b`, `0.6a`, `1.3a`, `1.4a`, `2.1`, `2.2`, `2.6`, `2.8`, `S1.end` ADRs)
- **Completed in Season 2:**
  - **Block A (`1.5`, `1.6`):** `setun/ref/m0.py`, `scripts/make_toy.py`, frozen `TTAPE` `spec-v1.0` submodule (`third_party/TernaryTape`), and end-to-end `Lux.Chain` grouped `.ttape` golden verification.
  - **Block B (`2.3`):** `setun/accelerators/unpacker.py`, mid-row `in_valid` bubble resilience across `TernaryDot`/`GroupAccumulateScale`/`TernaryRow`, and real `nextpnr-ecp5` `fmax_mhz` extraction in `scripts/resource_report.py`.
  - **Block C (`0.4`, `3.1a`):** `AccelStub` (`targets/sim.py`) and single-row `TernaryAccel` LiteX wrapper (`setun/soc/ternary_accel.py`) with `.ttape` composition tests (`tests/unit/test_soc_pipeline.py`) and generated `reports/csr.{json,h}`.
- **Next Tasks (in `PLAN.md` order):**
  1. **`2.4` — Multi-row `TernaryTile`:** Reference model + red unit tests for `R` parallel rows with shared activation broadcast and `valid`/`ready` backpressure; once RTL lands (maintainer-owned or explicitly delegated), synthesize real `R=4` and `R=8` figures to replace `[EXTRAPOLATED from R=1]` in `reports/resources.{json,md}`.
  2. **`2.5` & `2.8b` — Output Requantization & Tile Regression:** `int8` saturation/requantization stage and nightly Verilator sweep coverage for `TernaryTile`.
  3. **`3.1b` & `3.2` — Multi-row SoC & SDRAM DMA:** Write `docs/CSR.md`, wire `TernaryTile` into `TernaryAccel`, and add SimSoC SDRAM weight DMA burst reader + activation/output scratchpad BRAMs.
