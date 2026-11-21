# Next Up

- **Current Season:** Season 2 (Streaming `TernaryTile`, M0 golden model, and first token in SimSoC — Gate 3)
- **Completed Seasons:** Season 1 (`0.1`, `0.2a`, `0.3`, `0.5a/b`, `0.6a`, `1.3a`, `1.4a`, `2.1`, `2.2`, `2.6`, `2.8`, `S1.end` ADRs)
- **Completed in Season 2:**
  - **Block A (`1.5`, `1.6`):** `setun/ref/m0.py`, `scripts/make_toy.py`, frozen `TTAPE` `spec-v1.0` submodule (`third_party/TernaryTape`), and end-to-end `Lux.Chain` grouped `.ttape` golden verification.
  - **Block B (`2.3`, `2.4`, `2.5`, `2.8b` — COMPLETE):** `WeightUnpacker`, multi-row `TernaryTile` (`ADR-015`), 2-stage `II=1` `Requantize` (`NUMERICS.md` §4), PR/nightly (`>= 10^5`) Verilator regression sweeps across all datapath blocks, and synthesized `R=4`/`R=8` XC7 + ECP5 `nextpnr` figures in `reports/resources.{json,md}`.
  - **Block C (`0.4`, `3.1a`):** `AccelStub` (`targets/sim.py`) and single-row `TernaryAccel` LiteX wrapper (`setun/soc/ternary_accel.py`) with `.ttape` composition tests (`tests/unit/test_soc_pipeline.py`) and generated `reports/csr.{json,h}`.
- **Next Tasks (in `PLAN.md` order):**
  1. **`3.1b` — Multi-row `TernaryAccel` & `docs/CSR.md`:** Document CSR register contract (`docs/CSR.md`), upgrade `TernaryAccel` (`setun/soc/ternary_accel.py`) to wrap multi-row `TernaryTile` (+ optional `Requantize`), enforce illegal-config error flags (`K=0`, misaligned addr, `N > max_rows`) without hanging, and update `reports/csr.{json,h}`.
  2. **`3.2` — SimSoC SDRAM Weight DMA:** Burst reader from SimSoC SDRAM + activation/output scratchpad BRAMs (`N, K in {64, 256, 2048}`, >= 80% steady-state bus utilization).
  3. **`3.3a` & `3.4` — `AccelDriver` Sim Backend & Hybrid M0 Decode:** Python runtime driver (`setun/runtime/`) and full 14-projection M0 decode loop in SimSoC.
