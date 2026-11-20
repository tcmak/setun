# Next Up

- **Current Season:** Season 2 (Streaming `TernaryTile`, M0 golden model, and first token in SimSoC — Gate 3)
- **Completed Seasons:** Season 1 (`0.1`, `0.2a`, `0.3`, `0.5a/b`, `0.6a`, `1.3a`, `1.4a`, `2.1`, `2.2`, `2.6`, `2.8`, `S1.end` ADRs)
- **Completed in Season 2:** `0.4` (`AccelStub` CSR peripheral in `targets/sim.py`), `1.5` & `1.6` (`setun/ref/m0.py` + `scripts/make_toy.py`), `2.3` (`setun/accelerators/unpacker.py`), `A1` (toolchain pin `migen==0.10.0`, LF enforcement, host/compiler stub pruning, CI workflow), `B2` & `B2a` (vendored `TernaryTape.jl` via HTTPS submodule at `spec-v0.2-draft`, `ADR-014`, migrated `setun/ref/format.py` to `"TTAP"` with 20-fixture corpus validation)
- **Next Tasks (Season 2 — Block B & Block C / Phase C1.1):**
  1. `C1.1`: Mid-row `in_valid` bubble resilience tests, `nextpnr-ecp5` real `fmax_mhz` extraction, and LiteX SoC CSR export (`reports/csr.json`, `reports/csr.h`)
  2. `2.4` & `2.5`: Multi-row `TernaryTile` (`R` parallel rows, shared activation broadcast, `valid`/`ready` backpressure) and output requantization (`int8` saturation)
