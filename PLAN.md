PLAN
Draft 0.3 · 2026-10-17 (renamed to Setun) · companion to README.md · full task catalogue: docs/phases.md
# Phase reference (task catalogue)

The draft 0.2 task list, kept as the catalogue that [PLAN.md](../PLAN.md) pulls from. PLAN.md decides what to do and when (seasons, sessions, owners); this file holds the full Do/Accept text for each task ID. Each task heading notes where it is scheduled: S1–S3 (seasons), Track A–E (Season 4 and later), deferred, or north star.

- Sizes are rough relative effort (full-time terms): **S** ≤ 1 day · **M** 2–4 days · **L** 1–2 weeks · **XL** > 2 weeks. Use them to compare tasks, not to schedule.
- Numbers tagged **(initial)** are guesses, replaced by measured values at the task that says so.
- The gates below are the draft 0.2 phase gates. Season exits in PLAN.md take precedence.
- Model rungs M0–M4 and the numerics profiles are defined in the [README](../README.md).

## Mapping from the first draft

| First draft | Now |
|---|---|
| Phase 0 scaffold and sim | Phases 0 and 1 |
| Phase 1 TernaryLinear | Phases 2 and 3 |
| Phase 2 "LiteX CUDA" compiler | Phases 4 and 5 |
| Phase 3 Rise Pro SoC | Phases 6 and 7 |
| Phase 4 HBM on Agilex 7 | Phase 9 (stretch); MoE in Phase 10 |
| Phase 5 serving | Phase 8 (moved ahead of HBM so a real demo lands first) |

## Overview

The order matters more than the dates. Each phase ends at a gate, which is the milestone.

| Phase | Name | Gate |
|---|---|---|
| 0 | Foundations | Fresh clone → green sim |
| 1 | Numerics, formats, golden model | Golden model meets thresholds on M1 |
| 2 | Ternary compute primitives | **G1: architecture decision** |
| 3 | First token from hardware (sim) | M0 tokens == golden |
| 4 | Whole decoder on chip (sim) | M0 from command stream, CPU idle |
| 5 | Compiler MVP | Compiler reproduces hand-written M0 program |
| 6 | Small-board bring-up | M0 bit-exact on silicon |
| 7 | Real dense model on a DDR board | **v0.1 demo (M1)** |
| 8 | Serving, benchmarks, release | Release v0.1 |
| 9 | HBM platform (stretch) | Go/no-go after spike |
| 10 | MoE and hybrid models (stretch) | Toy hybrid bit-exact |

---|---|---|---|
| 0 | Foundations | ~1 wk | Fresh clone → green sim |
| 1 | Numerics, formats, golden model | ~3 wk | Golden model meets thresholds on M1 |
| 2 | Ternary compute primitives | ~3 wk | **G1: architecture decision** |
| 3 | First token from hardware (sim) | ~3 wk | M0 tokens == golden |
| 4 | Whole decoder on chip (sim) | ~5–6 wk | M0 from command stream, CPU idle |
| 5 | Compiler MVP | ~4–6 wk | Compiler reproduces hand-written M0 program |
| 6 | Small-board bring-up | ~3–4 wk | M0 bit-exact on silicon |
| 7 | Real dense model on a DDR board | ~3–5 wk | **v0.1 demo (M1)** |
| 8 | Serving, benchmarks, release | ~3–4 wk | Release v0.1 |
| 9 | HBM platform (stretch) | open | Go/no-go after spike |
| 10 | MoE and hybrid models (stretch) | open | Toy hybrid bit-exact |

---

## Phase 0: Foundations

**Goal:** a reproducible environment where "LiteX sim + custom peripheral + cocotb test" runs in CI.

### 0.1 Repo skeleton [S] · *S1*
- **Do:** layout per README; `pyproject.toml` (package `setun`, console script `setun`); LICENSE; CONTRIBUTING; `AGENTS.md` (rules from README); ruff + formatter + pre-commit; pytest config; `Makefile` with `doctor`, `lint`, `test`, `sim-smoke`, `ci`.
- **Accept:**
  - [X] `pip install -e .[dev]` works on a clean checkout
  - [X] `make lint test` pass
  - [X] `setun --version` prints version and git hash

### 0.2 Pinned, reproducible toolchain [M] · *S1 (lock, doctor) · deferred (container)*
- **Do:** pin Python, LiteX and its libraries (migen, litedram, liteeth, litepcie) by git commit in `toolchain.lock`; pin Verilator, cocotb, Yosys, RISC-V GCC (if firmware is used) and optionally SymbiYosys; Dockerfile (or devcontainer/Nix) that builds the environment; `make doctor`.
- **Accept:**
  - [ ] `docker build` from scratch, then `make doctor` passes
  - [X] `make doctor` fails with a clear message when any version differs from the lock
  - [X] versions are printed at the top of every CI log

### 0.3 LiteX SimSoC hello-world [S] · *S1*
- **Do:** `targets/sim.py` extends LiteX's `SimSoC` (see `litex.tools.litex_sim`; check `--help` for current flags); headless run; log capture.
- **Accept:**
  - [X] `make sim-smoke` builds and runs the sim, exits 0 once the BIOS banner appears in the log
  - [X] wall-clock ≤ 3 min on CI **(initial)**
  - [ ] on timeout: non-zero exit and the log is kept as an artifact

### 0.4 CSR peripheral stub in SimSoC [M] · *S2*
- **Do:** `setun/accelerators/stub.py` with ID, VERSION, SCRATCH and COUNTER CSRs; enabled by `--with-accel-stub`; export `csr.json`.
- **Accept:**
  - [ ] a test reads ID/VERSION through the real bus in sim (firmware or Etherbone/host path)
  - [ ] SCRATCH read-back verified with ≥ 100 random 32-bit patterns
  - [ ] addresses come from `csr.json`; none are hard-coded

### 0.5 Unit-test harness for Migen modules [M] · *S1*
- **Do:** helper that converts a Migen module to Verilog and runs it under cocotb + Verilator via pytest; Migen's own simulator for tiny combinational tests; seed logging (`TEST_SEED`); FST waveform on failure.
- **Accept:**
  - [X] example tests (adder, FIFO) pass in ≤ 30 s
  - [X] a deliberately failing test prints its seed and leaves an FST file
  - [X] the same command reproduces the failure locally and in CI

### 0.6 `make ci` and pipeline wiring [M] · *S1*
- **Do:** `make ci` = lint + unit + sim-smoke; CI config of any runner calls only `make ci` and uploads artifacts; `make nightly`; Verilator build cache.
- **Accept:**
  - [X] `make ci` ≤ 10 min on a clean machine and in CI **(initial)**
  - [ ] a PR with a failing test is red
  - [ ] logs, waveforms and reports are downloadable artifacts

### 0.7 Optional: formal harness [M] · *deferred*
- **Do:** SymbiYosys flow for a FIFO and a valid/ready stability property.
- **Accept:**
  - [ ] `make formal` proves the properties
  - [ ] it fails when a bug is injected

### 0.8 Optional: Linux on SimSoC, nightly [M] · *deferred*
- **Do:** nightly job that boots Linux in sim (the first draft's Phase 0 criterion).
- **Accept:**
  - [ ] login prompt reached. Not a gate; needed only if on-SoC userspace becomes a goal.

**Gate 0:** 0.1–0.6 green on `main`; fresh clone → green sim ≤ 30 min, documented.

---

## Phase 1: Numerics, formats, golden model

**Goal:** define exactly what the hardware must compute, as executable code and tests, before any RTL.

### 1.1 Freeze reference models [S] · *Track A*
- **Do:** `docs/models.md` for M0/M1 (and later rungs): layers, d_model, heads, KV heads, head_dim, FFN dim, vocab, context, norm type, activation, RoPE params, tied embeddings, file sizes, SHA256, licence, source URL. `scripts/fetch_models.py` with hash verification. Record ADR-006.
- **Accept:**
  - [ ] the table is generated from GGUF/config metadata, not typed by hand
  - [ ] the fetch script downloads and verifies M1
  - [ ] ADR-006 states the chosen M1 and why

### 1.2 Reference runtime baseline [M] · *Track A*
- **Do:** build the reference runtime for M1 (PrismML llama.cpp fork, `prism` branch, for `Q2_0`; `bitnet.cpp` for the BitNet profile). Record tokens/s, RAM and wall power if a meter is available.
- **Accept:**
  - [ ] `docs/baselines.md` has exact commit hashes, commands and outputs
  - [ ] `tests/vectors/ref_tokens.json` holds deterministic greedy output for 5 fixed prompts × 64 tokens (first end-to-end truth)

### 1.3 Weight format spec and pack/unpack [M] · *S1 (internal format) · Track A (GGUF readers)*
- **Do:** `docs/FORMAT.md`. Internal format: 2-bit trits {00: 0, 01: +1, 10: −1, 11: reserved}; group size G as a parameter (64/128/256); FP16 per-group scales; alignment; header. Readers for GGUF `Q2_0` (PrismML) and `TQ2_0`/`TQ1_0` (stock llama.cpp). Optional 5-trits-per-byte codec (243 of 256 codes) for density. Record ADR-003.
- **Accept:**
  - [X] pack → unpack is lossless on 10⁶ random tensors, including odd shapes and K not a multiple of G
  - [ ] for one real M1 tensor, GGUF → internal → dequantized floats equals the reference runtime's dequantization
  - [X] `FORMAT.md` contains a byte-exact worked example

### 1.4 Numerics spec [L] · *S1 (dot, group scale) · Track A (rest)*
- **Do:** `docs/NUMERICS.md`: profile definitions; activation quantization options (int8 per-token absmax; block-scaled int8/int16 with block B; FP16 passthrough); accumulator widths; group-scale application; RMSNorm and QK-norm (rsqrt method); RoPE table precision; softmax (exp approximation, normalization); SiLU/SwiGLU and ReLU²; residual-stream precision; head precision; sampling (argmax first).
  - Group partial-sum width: `⌈log2(G · A_max + 1)⌉ + 1` bits (G = 128, int8 → 16 bits). Cross-group accumulation uses wide fixed-point.
- **Accept:**
  - [X] every op has a definition, a bit-width table, a reference implementation in `setun/ref/ops.py`, and unit tests
  - [X] tests cover zero vector (scale 0), max-abs ties, saturation, −128, padding groups

### 1.5 Golden integer model [L] · *S2 (M0) · Track A (M1)*
- **Do:** integer-exact decoder in NumPy: embeddings, norms, QK-norm, RoPE, GQA attention with KV cache, FFN, head, greedy sampling; per-op trace hooks.
- **Accept:**
  - [ ] M0: final hidden state matches an FP32 reference with cosine ≥ 0.999 **(initial)**
  - [ ] M1: perplexity on a fixed 20k-token sample within +2% of the reference runtime **(initial)**
  - [ ] M1: top-1 next-token agreement ≥ 95% over 5k positions **(initial)**
  - [ ] any miss is explained and bounded in `NUMERICS.md`

### 1.6 Toy model generator [M] · *S2*
- **Do:** `scripts/make_toy.py`: seeded random model with M1's *structure* (GQA, norm type, FFN type, tied embeddings) at tiny dims (e.g. d_model 64, 2 layers, 4 heads / 2 KV heads, vocab 256); writes the internal format plus float source weights. Sizes `tiny` (CI) and `small` (nightly).
- **Accept:**
  - [ ] same seed → same SHA256
  - [ ] golden model runs on it; logits are not degenerate (entropy within a documented range)
  - [ ] note in `models.md`: random weights have no activation outliers, so error budgets are judged on M1, not M0

### 1.7 Golden vector exporter [M] · *Track A*
- **Do:** per-op test vectors (inputs, outputs, hex) for several shapes into `tests/vectors/`, with manifest and SHA256.
- **Accept:**
  - [ ] `make vectors` regenerates byte-identical files
  - [ ] CI verifies the manifest
  - [ ] committed size < 20 MB **(initial)**; larger sets are generated on demand

### 1.8 Activation-precision study [M] · *Track A*
- **Do:** with the golden model on M1, compare int8 per-token, block-scaled int8 (B ∈ {32, 128}), int16 and FP16 activations. Record ADR-002.
- **Accept:**
  - [ ] results table in `NUMERICS.md` (perplexity delta, top-1 agreement, accumulator width, estimated LUTs per MAC)
  - [ ] a recommendation and ADR-002

**Gate 1:** golden model meets thresholds on M1; vectors regenerate deterministically; ADR-002, ADR-003 and ADR-006 decided.

---

## Phase 2: Ternary compute primitives

**Goal:** the smallest, bit-exact ternary datapath, with measured cost.

### 2.1 `TernaryDot`: one group, L lanes [M] · *S1*
- **Do:** parameters L (16–128), activation width A (8/16), pipeline depth. Per lane select {0, +x, −x}; adder tree; optional negation via bit-invert plus a correction count. Output width per spec.
- **Accept:**
  - [X] bit-exact vs `ref/ops.py` on ≥ 1000 random vectors
  - [X] directed cases pass: all +1, all −1, all 0, alternating, x ∈ {min, max}
  - [X] synthesized netlist has 0 DSP and no multipliers
  - [X] latency is fixed, documented and asserted

### 2.2 Group accumulate and scale [M] · *S1*
- **Do:** accumulate G/L beats into a group partial sum; multiply by the group scale (fixed-point or FP16 → fixed); accumulate across groups in wide fixed-point; end-of-row flag; handle K not a multiple of G.
- **Accept:**
  - [X] bit-exact vs golden for K ∈ {1, 7, 127, 128, 129, 2048, 6144} × G ∈ {64, 128, 256} × L ∈ {16, 64, 128}
  - [X] multiplier count per row documented (DSP use is allowed in this stage only)

### 2.3 Weight unpacker [S–M] · *S2*
- **Do:** packed beat (e.g. 256-bit) → L trits and aligned scale stream; separate optional 5-trits-per-byte decoder (LUT).
- **Accept:**
  - [ ] bit-exact vs the 1.3 unpacker
  - [ ] sustains 1 beat/cycle for 10⁵ beats under random back-pressure, with no bubble between groups

### 2.4 `TernaryTile`: R rows × L lanes GEMV engine [L] · *S2*
- **Do:** activation buffer (BRAM/regs) broadcast to R row engines; weight and scale streams in; output stream with row index; parameters R, L, G; LiteX `stream.Endpoint` interfaces. Decode is GEMV (batch 1): activations stay put, weights stream.
- **Accept:**
  - [ ] GEMV (N × K) bit-exact for shapes from M0 and M1 (for example N = 2048, K = 2048)
  - [ ] N not a multiple of R works
  - [ ] steady-state utilization ≥ 95% of R·L MACs/cycle when fed at line rate **(initial)**, measured and reported

### 2.5 Requantize / output stage [M] · *S2*
- **Do:** convert row results to the next op's input format (scale, round, saturate) per spec.
- **Accept:**
  - [ ] bit-exact on corner cases: saturation at both ends, rounding ties, zero

### 2.6 Resource and Fmax characterization [M] · *S1*
- **Do:** scripted Yosys runs (generic and `synth_xilinx` for LUT6 counts); sweep R, L, G, A; write `reports/resources.md` and `.json`; CI compares against `reports/baseline.json`.
- **Accept:**
  - [ ] `make resource-report` reproduces the tables
  - [ ] CI fails if LUTs-per-MAC regresses > 10% **(initial)**
  - [ ] the report states how many MACs/cycle fit the small-board candidates at ≥ 100 MHz
  - [ ] the report states the ops/cycle needed to match the memory beat rate (rule of thumb: extra MACs beyond the weight-stream rate are wasted)

### 2.7 Optional: table-lookup variant [L] · *deferred*
- **Do:** TeLLMe-style table-lookup ternary matmul (grouped activations, precomputed partial sums) behind the same interface as 2.4.
- **Accept:**
  - [ ] same bit-exact tests
  - [ ] resource report compares LUTs/MAC and Fmax vs 2.4 at equal throughput
  - [ ] ADR-004 records the choice

### 2.8 Randomized regression suite [S] · *S1 · S2*
- **Do:** seeds, small PR configs and large nightly configs.
- **Accept:**
  - [ ] nightly runs ≥ 10⁵ random cases per config with zero mismatches
  - [ ] a failing seed is printed and replayable with one command

### 2.9 Optional: formal properties for stream handshakes [M] · *deferred*
- **Accept:**
  - [ ] valid/ready stability and no-data-loss proven for the tile's input and output streams

**Gate 2 (G1, architecture decision):** 2.6 shows tile throughput can match the memory beat rate within the small board's resources; ADR-001 (overlay) and ADR-004 (micro-architecture) decided. If not, revisit before Phase 3.

---

## Phase 3: First token from hardware, in sim

**Goal:** greedy generation on M0 where the only on-chip op is ternary GEMV and the host does everything else with golden ops. After this the system always works end to end.

### 3.1 `TernaryAccel` wrapper with CSRs [M] · *S2*
- **Do:** CSRs for control/status, N, K, G, input/output/weight/scale pointers, activation scale, error flags. Completion by polling (IRQ optional).
- **Accept:**
  - [ ] `csr.json` is generated
  - [ ] one GEMV launched by register writes matches golden
  - [ ] illegal parameters (K = 0, misaligned address, N too large) set error bits and never hang (tested)

### 3.2 Weight streaming from SimSoC memory [L] · *S2*
- **Do:** SDRAM/RAM model in the sim; burst reader (LiteDRAM DMA or Wishbone/AXI) feeding the tile; preload weights with the sim's memory init-file option (check `litex_sim --help`); document burst and alignment rules.
- **Accept:**
  - [ ] GEMV with memory-resident weights is bit-exact for N, K ∈ {64, 256, 2048}
  - [ ] steady-state bus utilization ≥ 80% **(initial)**, reported

### 3.3 `AccelDriver` host abstraction [M] · *S2 · S3 (board backend)*
- **Do:** Python API (`write_csr`, `dma_write`, `dma_read`, `launch`, `wait`, `perf`) with backends: cocotb-direct, Etherbone via `litex_server`/`RemoteClient` (sim has an Etherbone option), later board.
- **Accept:**
  - [ ] the same test script passes on both sim backends
  - [ ] the backend is chosen by one flag

### 3.4 Hybrid decode loop [M] · *S2*
- **Do:** `examples/hybrid_decode.py`: golden ops in Python, GEMVs through the driver, KV cache in host memory.
- **Accept:**
  - [ ] M0 generates 32 tokens identical to the golden model
  - [ ] cycles per GEMV are logged
  - [ ] runs in ≤ 10 min nightly **(initial)**

### 3.5 First-divergence tool [M] · *S2*
- **Do:** `setun diff traceA traceB` reports the first op/tensor/index mismatch with context; trace format has op IDs, shapes and hashes.
- **Accept:**
  - [ ] an injected 1-bit fault in the GEMV (test hook) is localized to the exact op and row by an automated test

### 3.6 System-sim job in CI [S] · *S2*
- **Accept:**
  - [ ] 3.4 runs nightly (and on PRs if ≤ 10 min)
  - [ ] trace and cycle log are saved as artifacts

**Gate 3 ("first token"):** M0 tokens from hardware GEMV == golden; cycles/token recorded; CI green.

---

## Phase 4: Whole decoder on chip, in sim

**Goal:** the decode loop runs from a command stream with the CPU idle.

### 4.1 Command format v1 [M] · *Track B*
- **Do:** `docs/COMMANDS.md`: opcodes, operand encodings (addresses, shapes, scales, flags), fences, loops, and a reserved **indirect-address mode** (needed later for MoE routing and dynamic KV length). Python encoder/decoder. Record ADR-005 (control plane: host-driven, soft CPU, or HPS).
- **Accept:**
  - [ ] round-trip tests for every opcode
  - [ ] reserved fields are rejected
  - [ ] each command has semantics and a cycle model in the spec

### 4.2 Sequencer [L] · *Track B*
- **Do:** fetches descriptors by DMA, dispatches to units, enforces dependencies with fences, reports errors and timeouts.
- **Accept:**
  - [ ] runs a hand-written program of ≥ 100 commands in sim
  - [ ] illegal opcodes and timeouts set an error status without hanging
  - [ ] the executed-command trace is readable
  - [ ] optional: fence logic proven deadlock-free with SymbiYosys

### 4.3 Activation quantizer [M] · *Track B*
- **Do:** absmax → scale → quantize, two passes, per `NUMERICS.md`; zero vector handled.
- **Accept:**
  - [ ] bit-exact on golden vectors for sizes {64, 2048, 6144}

### 4.4 Norm units: RMSNorm and QK-norm [M] · *Track B*
- **Do:** RMSNorm (sum of squares, rsqrt per `NUMERICS.md`, per-channel weight) and per-head QK-norm; epsilon and widths per spec.
- **Accept:**
  - [ ] bit-exact vs golden, or measured max error ≤ spec
  - [ ] rsqrt method documented

### 4.5 RoPE [M] · *Track B*
- **Do:** table generator script; position input; head_dim and partial-rotary parameters.
- **Accept:**
  - [ ] bit-exact for positions {0, 1, 17, 1023, max}
  - [ ] table memory size reported

### 4.6 Attention engine (decode) [XL] · *Track B*
- **Do:** QKᵀ, scaling, softmax, ·V; GQA head mapping; causal masking; KV append and read by DMA; context-length parameter.
- **Accept:**
  - [ ] bit-exact (or documented error bound) vs golden for context ∈ {1, 2, 17, 64, 1024}
  - [ ] softmax approximation error ≤ spec
  - [ ] KV bytes/cycle reported

### 4.7 FFN activation, elementwise ops, residual add [M] · *Track B*
- **Do:** SiLU approximation and elementwise multiply (SwiGLU) or ReLU² per model; residual add with saturation/width per spec.
- **Accept:**
  - [ ] bit-exact, or error ≤ spec, on golden vectors

### 4.8 Embedding, head, sampler [L] · *Track B*
- **Do:** embedding row gather and dequantization; head GEMV (format per M1; the vocabulary head is often the single largest matrix); argmax. Top-k/temperature stay on the host for now.
- **Accept:**
  - [ ] M0 token IDs match golden
  - [ ] head bytes/token reported

### 4.9 Prefill baseline [L] · *Track E*
- **Do:** decide token-by-token vs GEMM mode (activation matrix through the tile) and implement at least the functional baseline.
- **Accept:**
  - [ ] a 64-token prompt leaves state (KV cache, last hidden state) identical to golden
  - [ ] prefill throughput measured; if GEMM mode is deferred to 8.4, the TTFT cost is documented in the README

### 4.10 Full decoder from a command stream [L] · *Track B*
- **Do:** hand-written program for M0.
- **Accept:**
  - [ ] M0 generates 64 tokens identical to golden
  - [ ] bus trace shows the CPU only starts and polls
  - [ ] per-op cycle breakdown report

### 4.11 Performance counters [M] · *Track B*
- **Do:** cycles per op class, stall cycles, DMA bytes, exposed through CSRs.
- **Accept:**
  - [ ] counters within 5% of an independent sim measurement **(initial)**

**Gate 4:** M0 end-to-end on chip in sim; command format v1 frozen and recorded.

---

## Phase 5: Compiler MVP

**Goal:** compiler output replaces the hand-written programs; compile time stays in seconds.

### 5.1 IR [M] · *Track C*
- **Do:** ops, tensor types (shape, dtype, layout, quantization params), graph container, text dump, JSON serialization, verifier.
- **Accept:**
  - [ ] IR round-trip tests
  - [ ] the verifier rejects ≥ 10 malformed graphs (negative tests)

### 5.2 Frontends [L] · *Track C*
- **Do:** (a) internal toy format; (b) GGUF `Q2_0` and `TQ2_0`/`TQ1_0`; (c) safetensors + `config.json` for checkpoints that ship ternary weights and scales. Architecture adapters: Qwen3-class dense first; hybrid/MoE later.
- **Accept:**
  - [ ] M0 and M1 import with shapes matching `docs/models.md`
  - [ ] an unsupported op or format fails with an actionable message

### 5.3 IR interpreter [M] · *Track C*
- **Do:** executes the IR with golden ops.
- **Accept:**
  - [ ] equals the golden model bit-exactly on M0

### 5.4 Passes [XL] · *Track C*
- **Do:** legalize (map model ops to overlay ops); fold scales; fuse (norm + quant, matmul + requant, SwiGLU); weight layout and packing (tile-major); tiling and scheduling for the (R, L, G) in `hwconfig.json`; static memory planner (activation arena, KV cache, weights, alignment); command emission.
- **Accept:**
  - [ ] unit tests per pass, with IR dumps before and after
  - [ ] memory-plan checker finds no overlaps
  - [ ] schedule checker finds no dependency violations

### 5.5 Backend and artifacts [M] · *Track C*
- **Do:** emit a command stream, a weight image and a manifest (hashes, shapes, required hwconfig, ISA version). File names and extensions are decided when Phase 5 starts.
- **Accept:**
  - [ ] compiled M0 run in sim gives tokens == golden (CI)
  - [ ] the manifest check rejects a mismatched tile config or ISA version

### 5.6 Hardware-config contract [M] · *Track C*
- **Do:** each SoC build emits `hwconfig.json` (R, L, G, SRAM sizes, bus width, ISA version); the compiler validates fit.
- **Accept:**
  - [ ] a model that does not fit (for example KV cache too large) fails with a message naming the limiting resource

### 5.7 CLI and Python API [M] · *Track C*
- **Do:** `setun compile | run | inspect | diff | bench`.
- **Accept:**
  - [ ] `setun compile models/toy --target sim -o build/toy` works
  - [ ] `setun inspect` prints the per-layer schedule and predicted cycles
  - [ ] M1 compile ≤ 60 s **(initial)**

### 5.8 Performance model [M] · *Track C*
- **Do:** predicted cycles per token from the schedule and a bandwidth model.
- **Accept:**
  - [ ] within ±15% of measured sim cycles on M0 **(initial)**

### 5.9 Optional: design-space exploration [L] · *deferred*
- **Do:** given board resources and a model, suggest (R, L, buffer sizes).
- **Accept:**
  - [ ] on the small board it reproduces or beats the hand-picked configuration from Phase 6

### 5.10 Stretch: kernel DSL v0 [XL] · *north star*
- **Do:** `@setun.kernel` for a restricted Python subset lowered to a *command-stream program* (software side). RTL generation from Python is out of scope until after Gate 5.
- **Accept:**
  - [ ] a custom elementwise op is compiled and runs in sim

**Gate 5:** compiling M0 and running it in sim gives tokens == golden; the compiler reproduces (or improves on) the hand-written program from 4.10; M1 compiles, and at least one layer runs in sim with outputs equal to golden.

---

## Phase 6: Small-board bring-up, toy model

**Goal:** the toy model runs on real silicon with external DDR.

### 6.1 Board selection [S] · *S3*
- **Do:** pick a board with a LiteX target, ≥ 1 GB DDR3/DDR4 (for M1 later), Ethernet or SD, and enough LUTs for the tile from 2.6. The first draft mentioned "EmbedFire Rise Pro / small Artix-7"; neither is verified. Record ADR-007.
- **Accept:**
  - [ ] BIOS boots; LiteDRAM memtest passes; console works (UART and Ethernet if present)
  - [ ] ADR-007 records board, memory, toolchain and cost

### 6.2 Reproducible FPGA build [M] · *S3*
- **Do:** `make fpga-build BOARD=…` (vendor flow, or open flow if LiteX supports it for the board).
- **Accept:**
  - [ ] same inputs → same utilization report
  - [ ] bitstream and reports are archived

### 6.3 Fit and timing [L] · *S3*
- **Do:** choose (R, L, G) and pipeline depth.
- **Accept:**
  - [ ] worst negative slack ≥ 0 at ≥ 100 MHz **(initial)**
  - [ ] utilization report and critical-path notes committed

### 6.4 Weight loader [M] · *S3*
- **Do:** load weights over Etherbone/UDP or from SD. Not UART: 115200 baud ≈ 11 KB/s.
- **Accept:**
  - [ ] M0 weights loaded and CRC-verified
  - [ ] load throughput reported

### 6.5 Run M0 on hardware [M] · *S3*
- **Accept:**
  - [ ] 64 tokens identical to golden
  - [ ] 100 consecutive runs without mismatch

### 6.6 Measurements [M] · *S3*
- **Do:** tokens/s, bandwidth utilization, power (inline meter), compared with the Phase 1 baselines.
- **Accept:**
  - [ ] `docs/results.md` with method and raw data

### 6.7 Hardware-in-the-loop CI [L] · *deferred*
- **Do:** board attached to a self-hosted runner; nightly flash-and-test; automatic power-cycle on a hang.
- **Accept:**
  - [ ] nightly HIL job green for 7 consecutive days **(initial)**

**Gate 6:** toy model bit-exact on silicon.

---

## Phase 7: Real dense model on a DDR board

**Goal:** M1 on hardware. This is the v0.1 demo. In PLAN.md it runs in Track A in hybrid mode (GEMV on chip, host does the rest), so it does not have to wait for Phases 4–5.

### 7.1 Memory budget [S] · *Track A*
- **Accept:**
  - [ ] table of weights + KV cache (for a chosen context length) + buffers ≤ board memory

### 7.2 Compile and verify M1 layer by layer [L] · *Track A*
- **Do:** run layer 0 first, then all layers, comparing traces with `setun diff`.
- **Accept:**
  - [ ] per-layer outputs equal golden for all layers on hardware

### 7.3 Memory-system tuning [L] · *Track A*
- **Do:** burst size, bank/row locality, double-buffered activations, DMA/compute overlap.
- **Accept:**
  - [ ] ≥ 70% of theoretical DDR bandwidth sustained during tile streaming **(initial)**, reported

### 7.4 Text in / text out demo [M] · *Track A*
- **Do:** host tokenizer, sampling, streamed output.
- **Accept:**
  - [ ] the 5 fixed prompts produce the same first 64 greedy tokens as the golden model

### 7.5 Stability [M] · *Track A*
- **Accept:**
  - [ ] 1-hour soak with zero errors or hangs
  - [ ] hardware perplexity on a fixed sample equals the golden model's (bit-exact design → identical)

### 7.6 Performance vs bound [S] · *Track A*
- **Do:** compare measured tokens/s with `usable_bandwidth ÷ bytes_read_per_token` (for M1 about 0.44 GB/token plus KV-cache reads; 9 GB/s usable, which needs a 64-bit DDR3/DDR4 interface, gives a ceiling near 20 tokens/s, while a 16- or 32-bit DDR3 board at ~1–4.5 GB/s gives ~2–10 tokens/s).
- **Accept:**
  - [ ] written analysis of the gap, with the top three causes

**Gate 7 (v0.1 demo):** M1 generates text on hardware, matching golden; results published in `docs/results.md`.

---

## Phase 8: Serving, benchmarks, release

### 8.1 Runtime library [M] · *Track D*
- **Do:** sessions, KV reset/continue, token streaming, sampler (temperature, top-k, top-p) on the host.
- **Accept:**
  - [ ] multi-turn session reproduces the golden model's output for the same token history

### 8.2 OpenAI-compatible server [M] · *Track D*
- **Do:** `/v1/models`, `/v1/chat/completions` (SSE streaming), usage fields.
- **Accept:**
  - [ ] the official `openai` Python client works
  - [ ] smoke tests with Open WebUI, Continue and Cline pass
  - [ ] Cursor compatibility is verified separately; it may require a publicly reachable endpoint, so it is not a gate

### 8.3 Benchmark harness [M] · *Track D*
- **Do:** `setun bench`: decode tok/s, prefill tok/s, time to first token, joules/token, perplexity vs the reference runtime.
- **Accept:**
  - [ ] reproducible with one command on sim and on the board
  - [ ] every result includes a config hash (hwconfig + model + commit)

### 8.4 Prefill performance [L] · *Track E*
- **Do:** GEMM mode or batched tile, if deferred in 4.9.
- **Accept:**
  - [ ] time to first token for a 512-token prompt ≥ 4× faster than token-by-token **(initial)**

### 8.5 Docs and packaging [M] · *deferred*
- **Do:** tutorial, architecture doc, ADR index, Docker image, wheels, licence headers, release notes.
- **Accept:**
  - [ ] a new user reproduces the sim demo from the README in ≤ 30 min on a clean machine (timed)

### 8.6 Release v0.1 [S] · *deferred*
- **Accept:**
  - [ ] tagged release with artifacts and a results page

**Gate 8:** v0.1 tagged; a new user reproduces the sim demo from the README on a clean machine.

---

## Phase 9: HBM platform (stretch)

**Goal:** prove the design scales with memory bandwidth.

### 9.0 Feasibility spike and board decision [M] · *north star*
- **Do:** compare (a) an Alveo U280-class card (LiteX has a `USPHBM2` wrapper and an `xilinx_alveo_u280` target with `--with-hbm`; HBM2 capacity is about 8 GB, to be verified; Vivado) with (b) the Agilex 7 M-series kit DK-DEV-AGM039EA (HBM2e, quad-core Cortex-A53 HPS, 32 GB class, roughly US$10–17k, Quartus Pro; LiteX has Agilex PLL and HPS support but no HBM2e wrapper that I found; Altera's reference design for this kit instantiates the HBM2e IP with the HPS using it as memory). Record ADR-008.
- **Accept:**
  - [ ] a go/no-go note with licence, cost, access and a list of blockers
  - [ ] ADR-008

### 9.1 Platform bring-up [L–XL] · *north star*
- **Do:** LiteX target and HBM wrapper (AXI) for the chosen board; host link (LitePCIe on Xilinx, or HPS/Ethernet).
- **Accept:**
  - [ ] memtest across all pseudo-channels
  - [ ] sustained read bandwidth ≥ 70% of theoretical in a streaming test **(initial)**

### 9.2 Multi-tile / multi-channel scaling [XL] · *north star*
- **Do:** bind tiles to channels; spread weights across channels; extend the memory planner.
- **Accept:**
  - [ ] bandwidth scales ≥ 0.8× linearly with channels in the streaming test **(initial)**
  - [ ] M1 runs bit-exact on the board

### 9.3 Timing closure [L–XL] · *north star*
- **Do:** floorplanning per channel, pipelining.
- **Accept:**
  - [ ] worst negative slack ≥ 0 at the target Fmax (initial 250–350 MHz, device-dependent)

### 9.4 Larger dense models [L] · *north star*
- **Do:** M2 (Bonsai 4B, then 8B).
- **Accept:**
  - [ ] first 64 tokens bit-exact vs golden
  - [ ] tokens/s reported against the bandwidth bound

---

## Phase 10: MoE and hybrid architectures (stretch)

### 10.1 MoE routing and expert dispatch [XL] · *north star*
- **Do:** top-k over many experts; indirect descriptors (reserved in 4.1); expert prefetch from DDR/HBM; shared expert.
- **Accept:**
  - [ ] a toy MoE is bit-exact vs golden
  - [ ] expert-fetch bytes per token reported

### 10.2 Gated DeltaNet ops [XL] · *north star*
- **Do:** short causal conv, SiLU/softplus, L2 norm, delta-rule state update, gated output norm. First on the CPU/vector unit (offload ladder), then hardened.
- **Accept:**
  - [ ] a toy hybrid model is bit-exact vs golden
  - [ ] one real layer of M3 matches the reference runtime within tolerance

### 10.3 M3 / M4 runs [XL] · *north star*
- **Do:** Bonsai 27B, then Qwen3.6-35B-A3B if a ternary checkpoint of acceptable quality exists.
- **Accept:**
  - [ ] first 32 tokens agree with the reference runtime, or the divergence is explained by documented numerics
  - [ ] tokens/s against the bandwidth bound (active-weights-per-token for MoE)

### 10.4 Long-context memory plan [M] · *north star*
- **Do:** KV cache only for full-attention layers; fixed-size state for DeltaNet layers.
- **Accept:**
  - [ ] memory table for 32k and 128k context, checked by the compiler's fit test

---

## Decisions, risks, open questions

Maintained in [PLAN.md](../PLAN.md).
