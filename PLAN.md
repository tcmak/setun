# Setun — Project Plan (Seasons & Gates)

Draft 0.3 · 2026-10-17 (renamed to Setun) · companion to [README.md](README.md) · full task catalogue: [docs/phases.md](docs/phases.md) · ADRs: [docs/decisions.md](docs/decisions.md)

---

## Season 1 (`S1`): Foundations & Single-Row Ternary Engine (COMPLETE)

**Goal:** Stand up the reproducible verification/synthesis harness (**Gate 0**) and characterize a bit-exact single-row ternary dot + group scale engine (**Gate 2 / G1**).

- [X] **Block A — Repo Skeleton, Format & Numerics:** `0.1`, `0.2a` (`toolchain.lock` + `scripts/doctor.py`), `0.5a` (`tests/sim_helper.py`), `0.6a` (`make ci`), `1.3a` (`setun/ref/format.py` + `docs/FORMAT.md`), `1.4a` (`setun/ref/ops.py` + `docs/NUMERICS.md`).
- [X] **Block B — LiteX SimSoC & First Cocotb Testbench:** `0.3` (`targets/sim.py` hello world), `0.5b` (`SyncFIFO` cocotb + Verilator testbench).
- [X] **Block C — `TernaryDot` RTL & Verification:** `2.1` (`setun/accelerators/ternary_dot.py`, 0 DSPs, balanced adder tree, 1000+ vectors + edge cases).
- [X] **Block D — `GroupAccumulateScale` & `TernaryRow`:** `2.2` (`setun/accelerators/group_scale.py`, 58-case `K × G × L` matrix, ragged `K`, 1 DSP48E1 per row).
- [X] **Block E — Resource Sweep, Nightly Regression & `S1.end` Gate:** `2.6` (`scripts/resource_report.py`, `reports/resources.md`, `reports/baseline.json`), `2.8` (`tests/unit/test_regression_sweep.py`, $10^5$ Verilator vectors/config), `S1.end` ADR sign-off (`ADR-004`, `ADR-009`).

---

## Season 2 (`S2`): Streaming `TernaryTile`, M0 Golden Model & First Token in SimSoC

**Goal:** Build the multi-row `TernaryTile` accelerator, wrap it as a LiteX CSR + DMA peripheral in `SimSoC`, and generate 16 bit-exact tokens from the M0 tiny ternary transformer (**Gate 3**).

- [X] **Block A — M0 Reference & Toy Weights:**
  - `1.5` Tiny ternary transformer M0 in pure NumPy (`setun/ref/m0.py`)
  - `1.6` Deterministic M0 toy weight generator (`scripts/make_toy_model.py`, < 1 MB)
- [ ] **Block B — Tile Datapath RTL:**
  - `2.3` Weight unpacker (packed beat → `L` trits + aligned group scale stream)
  - `2.4` Multi-row `TernaryTile` (`R` parallel rows, shared activation broadcast, `valid`/`ready` backpressure)
  - `2.5` Output requantization & `int8` saturation
  - `2.8` Extend randomized nightly regression to `TernaryTile`
- [ ] **Block C — LiteX SoC Integration & DMA:**
  - `0.4` Minimal CSR peripheral stub in `SimSoC`
  - `3.1` `TernaryAccel` LiteX wrapper & CSR register map (`docs/CSR.md`)
  - `3.2` Weight DMA reader from SimSoC SDRAM + activation/output scratchpad BRAMs
- [ ] **Block D — Hybrid Decode & Gate 3:**
  - `3.3` `AccelDriver` Python simulation backend
  - `3.4` Hybrid M0 decode loop (linear layers on SimSoC `TernaryAccel`, non-linear ops on host)
  - `3.5` `setun diff` layer-by-layer mismatch debugger
  - `3.6` System-sim CI (`make sim-m0`, 16 tokens bit-exact vs. M0 golden)

---

## Season 3 (`S3`): First FPGA Board Bring-Up & M0 on Silicon

**Goal:** Synthesize, place-and-route, and run M0 bit-exact on a real DDR-equipped FPGA board at ≥ 100 MHz (**Gate 6**).

- [ ] `6.1` Candidate board selection (`ADR-007`)
- [ ] `6.2` Board target & `make fpga-build`
- [ ] `6.3` Fit & timing closure (≥ 100 MHz, zero negative slack)
- [ ] `6.4` Weight loader to board DDR (Etherbone / SD)
- [ ] `3.3` `AccelDriver` board backend
- [ ] `6.5` Run M0 on real hardware (16 tokens bit-exact vs. golden)
- [ ] `6.6` First real tok/s, MB/s, and wall-plug power measurements

---

## Season 4+: Parallel Tracks (`Track A` – `Track E`)

- **Track A — Real Dense Model (`M1`) on DDR Board (Gate 1 & Gate 7 / v0.1 Demo):** `1.1`, `1.2`, `1.3` (GGUF readers), `1.4` (RMSNorm/RoPE/SwiGLU/Softmax), `1.5` (M1 golden), `1.7`, `1.8`, `7.1`–`7.6`.
- **Track B — Full Decoder on Chip with Command Sequencer (Gate 4):** `4.1`–`4.8`, `4.10`, `4.11`.
- **Track C — Compiler MVP `setun compile | run | inspect` (Gate 5):** `5.1`–`5.8`.
- **Track D — Runtime Library, OpenAI Server & `setun bench`:** `8.1`–`8.3`.
- **Track E — Prefill & GEMM Batching:** `4.9`, `8.4`.
