# Next Up

- **Current Season:** Season 2 (Streaming `TernaryTile`, M0 golden model, and first token in SimSoC — Gate 3)
- **Completed Seasons:** Season 1 (`0.1`, `0.2a`, `0.3`, `0.5a/b`, `0.6a`, `1.3a`, `1.4a`, `2.1`, `2.2`, `2.6`, `2.8`, `S1.end` ADRs)
- **Completed in Season 2:**
  - **Block A (`1.5`, `1.6`):** `setun/ref/m0.py`, `scripts/make_toy.py`, frozen `TTAPE` `spec-v1.0` submodule (`third_party/TernaryTape`), and end-to-end `Lux.Chain` grouped `.ttape` golden verification.
  - **Block B (`2.3`, `2.4`, `2.5`):** `WeightUnpacker` (`unpacker.py`), multi-row `TernaryTile` (`tile.py`, `ADR-015`), and 2-stage `II=1` `Requantize` (`requantize.py`, `docs/NUMERICS.md` §4, 0 DSPs, measured `49.08 MHz` on ECP5-85F), with real `R=4`/`R=8` XC7 synthesis in `reports/resources.{json,md}`.
  - **Block C (`0.4`, `3.1a`):** `AccelStub` (`targets/sim.py`) and single-row `TernaryAccel` LiteX wrapper (`setun/soc/ternary_accel.py`) with `.ttape` composition tests (`tests/unit/test_soc_pipeline.py`) and generated `reports/csr.{json,h}`.
- **Next Tasks (in `PLAN.md` order):**
  1. **`2.8b` — Tile Nightly Regression:** Extend randomized nightly Verilator sweep (`tests/unit/test_regression_sweep.py`) to `TernaryTile` (and `Requantize`).
  2. **`3.1b` — Multi-row `TernaryAccel` & `docs/CSR.md`:** Document CSR register contract (`docs/CSR.md`) and upgrade `TernaryAccel` to wrap multi-row `TernaryTile`.
  3. **`3.2` — SimSoC SDRAM Weight DMA:** Burst reader from SimSoC SDRAM + activation/output scratchpad BRAMs.
