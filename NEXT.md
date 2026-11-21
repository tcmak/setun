# Next Up

- **Current Season:** Season 2 (Streaming `TernaryTile`, M0 golden model, and first token in SimSoC — Gate 3)
- **Completed Seasons:** Season 1 (`0.1`, `0.2a`, `0.3`, `0.5a/b`, `0.6a`, `1.3a`, `1.4a`, `2.1`, `2.2`, `2.6`, `2.8`, `S1.end` ADRs)
- **Completed in Season 2:**
  - **Block A (`1.5`, `1.6`):** `setun/ref/m0.py`, `scripts/make_toy.py`, frozen `TTAPE` `spec-v1.0` submodule (`third_party/TernaryTape`), and end-to-end `Lux.Chain` grouped `.ttape` golden verification.
  - **Block B (`2.3`, `2.4`):** `setun/accelerators/unpacker.py`, multi-row `TernaryTile` (`setun/accelerators/tile.py`, `ADR-015`, verified on `2048x2048` M1 GEMV at >= 99.9% utilization + synthesized `R=4`/`R=8` XC7 figures), mid-row `in_valid` bubble resilience, and real `nextpnr-ecp5` `fmax_mhz` extraction in `scripts/resource_report.py`.
  - **Block C (`0.4`, `3.1a`):** `AccelStub` (`targets/sim.py`) and single-row `TernaryAccel` LiteX wrapper (`setun/soc/ternary_accel.py`) with `.ttape` composition tests (`tests/unit/test_soc_pipeline.py`) and generated `reports/csr.{json,h}`.
- **Next Tasks (in `PLAN.md` order):**
  1. **`2.5` — Output Requantization & `int8` Saturation:** Reference op + red-first corner tests (saturation at both ends, round-half-up ties, zero) and hardware stage.
  2. **`2.8b` — Tile Nightly Regression:** Extend randomized nightly Verilator sweep (`tests/unit/test_regression_sweep.py`) to `TernaryTile`.
  3. **`3.1b` & `3.2` — Multi-row SoC & SDRAM DMA:** Write `docs/CSR.md`, upgrade `TernaryAccel` to wrap multi-row `TernaryTile`, and add SimSoC SDRAM weight DMA burst reader + activation/output scratchpad BRAMs.
