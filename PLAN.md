# Setun — Project Plan (Seasons & Gates)

Draft 0.4 · 2026-11-21 · companion to [README.md](README.md) · full task catalogue: [docs/phases.md](docs/phases.md) · ADRs: [docs/decisions.md](docs/decisions.md)

---

## Season 1 (`S1`): Foundations & Single-Row Ternary Engine (COMPLETE)

**Goal:** Stand up the reproducible verification/synthesis harness (**Gate 0**) and characterize a bit-exact single-row ternary dot + group scale engine (**Gate 2 / G1**).

- [x] **Block A — Repo Skeleton, Format & Numerics:** `0.1`, `0.2a` (`toolchain.lock` + `scripts/doctor.py`), `0.5a` (`tests/sim_helper.py`), `0.6a` (`make ci`), `1.3a` (`setun/ref/format.py` + `docs/FORMAT.md`), `1.4a` (`setun/ref/ops.py` + `docs/NUMERICS.md`).
- [x] **Block B — LiteX SimSoC & First Cocotb Testbench:** `0.3` (`targets/sim.py` hello world), `0.5b` (`SyncFIFO` cocotb + Verilator testbench).
- [x] **Block C — `TernaryDot` RTL & Verification:** `2.1` (`setun/accelerators/ternary_dot.py`, 0 DSPs default, balanced adder tree, 1000+ vectors + mid-row `in_valid` bubble resilience).
- [x] **Block D — `GroupAccumulateScale` & `TernaryRow`:** `2.2` (`setun/accelerators/group_scale.py`, 58-case `K × G × L` matrix, ragged `K`, mid-row bubble resilience).
- [x] **Block E — Resource Sweep, Nightly Regression & `S1.end` Gate:** `2.6` (`scripts/resource_report.py` with measured `nextpnr-ecp5` `fmax_mhz`, `reports/resources.md`, `reports/baseline.json`), `2.8` (`tests/unit/test_regression_sweep.py`, $10^5$ Verilator vectors/config), `S1.end` ADR sign-off (`ADR-004`, `ADR-009`, `ADR-014`).

---

## Season 2 (`S2`): Streaming `TernaryTile`, M0 Golden Model & First Token in SimSoC

**Goal:** Build the multi-row `TernaryTile` accelerator, wrap it as a LiteX CSR + DMA peripheral in `SimSoC`, and generate 16 bit-exact tokens from the M0 tiny ternary transformer (**Gate 3**).

### Block A — M0 Reference, `TTAPE` v1.0 & Toy Weights (COMPLETE)
- [x] `1.5` Tiny ternary transformer M0 in pure NumPy (`setun/ref/m0.py`, verified against frozen `TTAPE` `spec-v1.0` and exported `Lux.Chain` grouped fixtures)
- [x] `1.6` Deterministic M0 toy weight generator (`scripts/make_toy.py`, < 1 MB `.ttape`)

### Block B — Tile Datapath RTL *(Note: `setun/accelerators/` is maintainer-owned per `AGENTS.md` unless explicitly delegated)*
- [x] `2.3` Weight unpacker (`setun/accelerators/unpacker.py`: packed beat → `L` trits + aligned group scale stream)
- [x] `2.4` Multi-row `TernaryTile` (`R` parallel rows, shared activation broadcast, `valid`/`ready` backpressure; replace `[EXTRAPOLATED from R=1]` in `reports/resources.{json,md}` with synthesized `R=4` / `R=8` figures)
- [x] `2.5` Output requantization & `int8` saturation
- [x] `2.8b` Extend randomized nightly Verilator regression (`tests/unit/test_regression_sweep.py`) to `TernaryTile`

### Block C — LiteX SoC Integration & DMA
- [x] `0.4` Minimal CSR peripheral stub in `SimSoC` (`setun/accelerators/stub.py`, `targets/sim.py`)
- [x] `3.1a` Single-row `TernaryAccel` LiteX wrapper, `.ttape` composition test & deterministic CSR export (`setun/soc/ternary_accel.py`, `reports/csr.json`, `reports/csr.h`)
- [x] `3.1b` Document CSR register contract (`docs/CSR.md`) and upgrade `TernaryAccel` to wrap multi-row `TernaryTile`
- [x] `3.2` Weight DMA burst reader from SimSoC SDRAM + activation/output scratchpad BRAMs
- [x] `3.2b` Packed `int8` output DMA writer (`4 × int8` rows per 32-bit beat when `req_enable = 1`, byte-masked tail flush for `N % 4 != 0`, matching `act_addr` layout)
  - **Do:** Audit and upgrade the `TernaryAccel` output write path (`setun/soc/ternary_accel.py` and `docs/CSR.md`) when `ctrl.req_enable == 1`: coalesce 4 consecutive `int8` row results into one 32-bit little-endian word (`row 4m+k` at bits `[8k+7 : 8k]`, matching the `act_addr` input activation layout) written to `out_addr + 4*m`. On `last == 1` when `N % 4 != 0`, flush the partial word using Wishbone byte-enable strobes (`sel = (1 << (N % 4)) - 1`) so trailing bytes after `out_addr + N - 1` are never clobbered. Preserve unquantized wide-word writes when `ctrl.req_enable == 0`.
  - **Accept:**
    - [x] Output memory layout for `req_enable == 1` matches `int8[N]` contiguous bytes (`act_addr`-compatible zero-copy layout) for both aligned (`N ∈ {4, 16, 64}`) and ragged (`N ∈ {1, 2, 3, 5, 7, 13}`) row counts
    - [x] Trailing sentinel bytes in SDRAM immediately at `out_addr + N` remain untouched when `N % 4 != 0`
    - [x] `perf_bus_beats` (or bus write counter) confirms exactly `⌈N / 4⌉` output write beats for `req_enable == 1` (4× reduction vs. 1 beat/row)
    - [x] `req_enable == 0` wide accumulator readback remains bit-exact and backwards-compatible

### Block D — Hybrid Decode & Gate 3
- [x] `3.3a` `AccelDriver` Python simulation backend (`setun/runtime/`)
- [x] `3.4` Hybrid M0 decode loop & `setun run` CLI (`setun run <model.ttape> --backend ref|sim --tokens N` in `setun/cli.py`: linear layers on SimSoC `TernaryAccel`, non-linear ops on host, printing token stream + CSR `cycles/tok` and `MB/s` summary)
- [x] `3.5` `setun diff` layer-by-layer mismatch debugger (`setun/cli.py`)
- [x] `3.6` System-sim CI (`make sim-m0`: `setun run` `--backend sim` vs. `--backend ref` 16 tokens bit-exact, including randomized `reset_less` register init and SDRAM wait-state jitter)
- [x] `3.7` End-User Quickstart & Simulation User Guide (`README.md` / `docs/USER_GUIDE.md`: 5-minute walkthrough covering install, `setun run models/m0_toy.ttape --backend sim --tokens 16`, and `setun diff`)

---

## Season 3 (`S3`): First FPGA Board Bring-Up & M0 on Silicon

**Goal:** Synthesize, place-and-route, and run M0 bit-exact on a real DDR-equipped FPGA board at ≥ 100 MHz (**Gate 6**).

- [x] `6.1` Candidate board selection (`ADR-007`, evaluate Lattice ECP5-85F / LambdaConcept ECPIX-5 512MB via open yosys+nextpnr flow vs. Altera Cyclone V / Terasic DE10-Nano 1GB via Quartus Lite + ARM HPS AXI bridge)
- [ ] `6.2` Board target & `make fpga-build`
- [ ] `6.3` Fit & timing closure (≥ 100 MHz measured `nextpnr` / vendor STA, zero negative slack)
- [ ] `6.4` `.ttape` weight loader to board DDR (Etherbone / SD)
- [ ] `3.3b` `AccelDriver` board backend (`setun/runtime/`, wired into `setun run --backend board`)
- [ ] `6.5` Run M0 on real hardware (`setun run m0.ttape --backend board --tokens 16` bit-exact vs. `--backend ref` + 256-token continuous DDR burst stability soak)
- [ ] `6.6` First real tok/s, MB/s, and wall-plug power measurements
- [ ] `6.8` Public hardware showcase & traction release (`docs/results.md` + `README.md` showcase with measured `tok/s`, `MB/s`, LUT/BRAM/0-DSP utilization, and board bring-up photo on ECPIX-5 or DE10-Nano running bit-exact; share with LiteX Discord/mailing list, `r/FPGA`, and Hacker News)

---

## Season 4+: Hardware-Focused Parallel Tracks (`Track A` – `Track E`)

*(Boundary invariant: `setun` consumes strictly `.ttape` containers from `TernaryTape.jl` and never implements GGUF parsers, host language compilers, or HTTP/OpenAI model servers.)*

- **Track A — Real Dense Model (`M1`) on DDR Board (Gate 1 & Gate 7 / v0.1 Demo):** Add standalone converter in `TernaryTape.jl` (`hf_bitnet_to_ttape.py` packing native 1.58-bit checkpoints into `TTAPE v1.0`) and bump `third_party/TernaryTape`; `1.4` (`RMSNorm`/`SubLN`, `RoPE`, `SwiGLU` + `ReLU²`, dynamic `int8` absmax, and `Softmax` in `setun/ref/`), `1.5b` (`M1` golden oracle + perplexity gate), `1.7`, `1.8`, `7.1`–`7.6` (extending `setun run` with `--prompt` + `--tokenizer` for live terminal text generation).
- **Track B — Full Decoder on Chip with Command Sequencer (Gate 4):** `4.1`–`4.8`, `4.10`, `4.11`.
- **Track C — Hardware Command-Schedule Emitter `setun pack-schedule | inspect` (Gate 5):** Lower `.ttape` layer graphs into static DMA/command-sequencer descriptor tables (`5.1`–`5.8`; strictly hardware descriptor packing, not a general host compiler).
- **Track D — Minimal Board Driver & Hardware CLI (`setun run | bench`):** Thin UART/DMA/Etherbone driver in `setun/runtime/`, `--schedule` fast-path execution in `setun run`, and cycle/throughput/power measurement harness (`8.1`, `8.3`; zero HTTP/serving stack).
- **Track E — Prefill & GEMM Batching:** `4.9`, `8.4`.
