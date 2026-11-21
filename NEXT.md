# Next Up

- **Current Season:** Season 2 (Streaming `TernaryTile`, M0 golden model, and first token in SimSoC — Gate 3)
- **Completed Seasons:** Season 1 (`0.1`, `0.2a`, `0.3`, `0.5a/b`, `0.6a`, `1.3a`, `1.4a`, `2.1`, `2.2`, `2.6`, `2.8`, `S1.end` ADRs)
- **Completed in Season 2:** `0.4` (`AccelStub` CSR peripheral in `targets/sim.py`), `1.5` & `1.6` (`setun/ref/m0.py` + `scripts/make_toy.py`), `2.3` (`setun/accelerators/unpacker.py`), `A1` (toolchain pin `migen==0.10.0`, LF enforcement, host/compiler stub pruning, CI workflow), `B2` & `B2a` (vendored `TernaryTape.jl` via HTTPS submodule, `ADR-014`, migrated `setun/ref/format.py` to `"TTAP"`), `C1.3` (verified end-to-end `Lux.Chain` grouped `.ttape` forward inference in `setun/ref/m0.py` across 21 golden fixtures and pinned frozen `TTAPE` `spec-v1.0`), `C1.1` (mid-row `in_valid` bubble resilience tests for `TernaryDot`/`GroupAccumulateScale`/`TernaryRow`, real `nextpnr-ecp5` `fmax_mhz` extraction + `[EXTRAPOLATED from R=1]` labels in `reports/resources.{json,md}`, and `TernaryAccel` LiteX SoC wrapper + `.ttape` composition tests + deterministic `reports/csr.{json,h}`)
- **Next Tasks (Season 2 — Block B & Block C):**
  1. `2.4` & `2.5`: Multi-row `TernaryTile` (`R` parallel rows, shared activation broadcast, `valid`/`ready` backpressure) and output requantization (`int8` saturation)
  2. `3.1` & `3.2`: Extend `TernaryAccel` with SDRAM weight DMA burst reader (`docs/CSR.md`) and multi-row tile integration
