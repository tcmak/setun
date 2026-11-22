# Phase reference (task catalogue)

The task list catalogue that [PLAN.md](../PLAN.md) pulls from. PLAN.md decides what to do and when (seasons, sessions, owners); this file holds the full Do/Accept text for each task ID. Each task heading notes where it is scheduled: S1–S3 (seasons), Track A–E (Season 4 and later), deferred, or north star.

- Sizes are rough relative effort (full-time terms): **S** ≤ 1 day · **M** 2–4 days · **L** 1–2 weeks · **XL** > 2 weeks. Use them to compare tasks, not to schedule.
- Numbers tagged **(initial)** are guesses, replaced by measured values at the task that says so.
- The gates below are the draft phase gates. Season exits in PLAN.md take precedence.
- Model rungs M0–M4 and the numerics profiles are defined in the [README](../README.md).
- **Boundary Invariants (Draft 0.4):** `setun` consumes strictly `.ttape` containers from `TernaryTape.jl` and provides local terminal hardware CLIs (`setun run | diff | pack-schedule | inspect | bench`); it never implements GGUF parsers, general host compilers, or HTTP/OpenAI model servers.

## Mapping from the first draft

| First draft | Now |
|---|---|
| Phase 0 scaffold and sim | Phases 0 and 1 |
| Phase 1 TernaryLinear | Phases 2 and 3 |
| Phase 2 "LiteX CUDA" compiler | Phases 4 and 5 (Hardware Schedule Packer) |
| Phase 3 Small FPGA SoC | Phases 6 and 7 |
| Phase 4 HBM on Agilex 7 | Phase 9 (stretch); MoE in Phase 10 |
| Phase 5 serving / CLI | Phase 8 (Runtime CLI & benchmarks; HTTP serving cut) |

## Overview

The order matters more than the dates. Each phase ends at a gate, which is the milestone.

| Phase | Name | Gate |
|---|---|---|
| 0 | Foundations | Fresh clone → green sim |
| 1 | Numerics, formats, golden model | Golden model meets thresholds on M1 |
| 2 | Ternary compute primitives | **G1: architecture decision** |
| 3 | First token from hardware (sim) | M0 tokens == golden (`setun run`) |
| 4 | Whole decoder on chip (sim) | M0 from command stream, CPU idle |
| 5 | Command-Schedule Packer MVP | Packer reproduces hand-written M0 program |
| 6 | Small-board bring-up | M0 bit-exact on silicon |
| 7 | Real dense model on a DDR board | **v0.1 demo (M1)** |
| 8 | Runtime CLI, benchmarks, release | Release v0.1 |
| 9 | High-bandwidth platform (stretch) | Go/no-go after spike |
| 10 | MoE and hybrid models (stretch) | Toy hybrid bit-exact |

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
  - [X] a test reads ID/VERSION through the real bus in sim (firmware or Etherbone/host path)
  - [X] SCRATCH read-back verified with ≥ 100 random 32-bit patterns
  - [X] addresses come from `csr.json`; none are hard-coded

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
- **Do:** nightly job that boots Linux in sim.
- **Accept:**
  - [ ] login prompt reached. Not a gate; needed only if on-SoC userspace becomes a goal.

**Gate 0:** 0.1–0.6 green on `main`; fresh clone → green sim ≤ 30 min, documented.

---

## Phase 1: Numerics, formats, golden model

**Goal:** define exactly what the hardware must compute, as executable code and tests, before any RTL.

### 1.1 Freeze reference models [S] · *Track A*
- **Do:** `docs/models.md` for M0/M1 (and later rungs): layers, d_model, heads, KV heads, head_dim, FFN dim, vocab, context, norm type, activation, RoPE params, tied embeddings, file sizes, SHA256, licence, source URL. Record ADR-006.
- **Accept:**
  - [ ] the table is generated from Hugging Face / TTAPE metadata, not typed by hand
  - [ ] the fetch script downloads and verifies M1 native 1.58-bit checkpoint assets
  - [ ] ADR-006 states the chosen M1 checkpoint (`microsoft/bitnet-b1.58-2B-4T` or open equivalent) and why

### 1.2 Reference runtime baseline [M] · *Track A*
- **Do:** build reference runtime baseline for M1 (using native BitNet / HF reference models such as `TriLM`, `Falcon-E`, or `bitnet-b1.58-2B-4T`). Record tokens/s, RAM and wall power if a meter is available.
- **Accept:**
  - [ ] `docs/baselines.md` has exact commit hashes, commands and outputs
  - [ ] `tests/vectors/ref_tokens.json` holds deterministic greedy output for 5 fixed prompts × 64 tokens (first end-to-end truth)

### 1.3 Weight format spec and TTAPE container loader [M] · *S1 (internal format) · Track A (TTAPE integration)*
- **Do:** `docs/FORMAT.md`. Internal format and container alignment for `TTAPE v1.0` (`third_party/TernaryTape`). 2-bit trits {00: 0, 01: +1, 10: −1, 11: reserved}; group size G parameter (64/128/256); `uint16` group scales + exponent shift; alignment; header; FNV-1a checksum verification. Record ADR-003.
- **Accept:**
  - [X] pack → unpack is lossless on 10⁶ random tensors, including odd shapes and K not a multiple of G
  - [X] `FORMAT.md` contains a byte-exact worked example
  - [X] `TTAPE v1.0` binary container loader verified; passes exhaustive single-byte corruption and metadata mutation rejection

### 1.4 Numerics spec [L] · *S1 (dot, group scale) · Track A (rest)*
- **Do:** `docs/NUMERICS.md`: profile definitions; activation quantization options (int8 per-token absmax; dynamic per-token int8 absmax); accumulator widths; group-scale application; RMSNorm and SubLN (rsqrt method); RoPE table precision; softmax (exp approximation, normalization); SwiGLU and ReLU²; residual-stream precision; head precision; sampling (argmax first).
  - Group partial-sum width: `⌈log2(G · A_max + 1)⌉ + 1` bits (G = 128, int8 → 16 bits). Cross-group accumulation uses wide fixed-point.
- **Accept:**
  - [X] every op has a definition, a bit-width table, a reference implementation in `setun/ref/ops.py`, and unit tests
  - [X] tests cover zero vector (scale 0), max-abs ties, saturation, −128, padding groups

### 1.5 Golden integer model [L] · *S2 (M0) · Track A (M1)*
- **Do:** integer-exact decoder in NumPy (`setun/ref/`): embeddings, RMSNorm / SubLN, QK-norm, RoPE, GQA attention with KV cache, FFN (SwiGLU / ReLU²), head, greedy sampling; per-op trace hooks.
- **Accept:**
  - [X] M0: final hidden state matches an FP32 reference with cosine ≥ 0.999
  - [ ] M1: perplexity on a fixed 20k-token sample within +2% of the reference runtime **(initial)**
  - [ ] M1: top-1 next-token agreement ≥ 95% over 5k positions **(initial)**
  - [X] any miss is explained and bounded in `NUMERICS.md`

### 1.6 Toy model generator [M] · *S2*
- **Do:** `scripts/make_toy.py`: seeded random model with M1's *structure* (GQA, norm type, FFN type, tied embeddings) at tiny dims (e.g. d_model 64, 2 layers, 4 heads / 2 KV heads, vocab 256); writes the `.ttape` format plus float source weights. Sizes `tiny` (CI) and `small` (nightly).
- **Accept:**
  - [X] same seed → same SHA256
  - [X] golden model runs on it; logits are not degenerate (entropy within a documented range)
  - [X] note in `models.md`: random weights have no activation outliers, so error budgets are judged on M1, not M0

### 1.7 Golden vector exporter [M] · *Track A*
- **Do:** per-op test vectors (inputs, outputs, hex) for several shapes into `tests/vectors/`, with manifest and SHA256.
- **Accept:**
  - [ ] `make vectors` regenerates byte-identical files
  - [ ] CI verifies the manifest
  - [ ] committed size < 20 MB **(initial)**; larger sets are generated on demand

### 1.8 Activation-precision study [M] · *Track A*
- **Do:** with the golden model on M1, compare int8 per-token, dynamic int8, and FP16 passthrough activations. Record ADR-002.
- **Accept:**
  - [ ] results table in `NUMERICS.md` (perplexity delta, top-1 agreement, accumulator width, estimated LUTs per MAC)
  - [ ] a recommendation and ADR-002

**Gate 1:** golden model meets thresholds on M1; vectors regenerate deterministically; ADR-002, ADR-003 and ADR-006 decided.

---

## Phase 2: Ternary compute primitives

**Goal:** the smallest, bit-exact ternary datapath, with measured cost (0 DSPs default).

### 2.1 `TernaryDot`: one group, L lanes [M] · *S1*
- **Do:** parameters L (16–128), activation width A (8/16), pipeline depth. Per lane select {0, +x, −x}; adder tree; optional negation via bit-invert plus a correction count. Output width per spec.
- **Accept:**
  - [X] bit-exact vs `ref/ops.py` on ≥ 1000 random vectors
  - [X] directed cases pass: all +1, all −1, all 0, alternating, x ∈ {min, max}
  - [X] synthesized netlist has 0 DSP and no multipliers
  - [X] latency is fixed, documented and asserted

### 2.2 Group accumulate and scale [M] · *S1*
- **Do:** accumulate G/L beats into a group partial sum; multiply by the group scale (`uint16` + exponent shift → fixed); accumulate across groups in wide fixed-point; end-of-row flag; handle K not a multiple of G.
- **Accept:**
  - [X] bit-exact vs golden for K ∈ {1, 7, 127, 128, 129, 2048, 6144} × G ∈ {64, 128, 256} × L ∈ {16, 64, 128}
  - [X] zero DSP multipliers verified by Yosys synth check

### 2.3 Weight unpacker [S–M] · *S2*
- **Do:** packed beat (e.g. 256-bit) → L trits and aligned scale stream; separate optional 5-trits-per-byte decoder (LUT).
- **Accept:**
  - [X] bit-exact vs the 1.3 unpacker
  - [X] sustains 1 beat/cycle for 10⁵ beats under random back-pressure, with no bubble between groups

### 2.4 `TernaryTile`: R rows × L lanes GEMV engine [L] · *S2*
- **Do:** activation buffer (BRAM/regs) broadcast to R row engines; weight and scale streams in; output stream with row index; parameters R, L, G; LiteX `stream.Endpoint` interfaces. Decode is GEMV (batch 1): activations stay put, weights stream.
- **Accept:**
  - [X] GEMV (N × K) bit-exact for shapes from M0 and M1 (for example N = 2048, K = 2048)
  - [X] N not a multiple of R works
  - [X] steady-state utilization ≥ 95% of R·L MACs/cycle when fed at line rate **(initial)**, measured and reported

### 2.5 Requantize / output stage [M] · *S2*
- **Do:** convert row results to the next op's input format (scale, round, saturate) per spec.
- **Accept:**
  - [X] bit-exact on corner cases: saturation at both ends, rounding ties, zero

### 2.6 Resource and Fmax characterization [M] · *S1*
- **Do:** scripted Yosys runs (`synth_ecp5` + `nextpnr-ecp5` and `synth_xilinx` for LUT counts and fmax); sweep R, L, G, A; write `reports/resources.md` and `.json`; CI compares against `reports/baseline.json`.
- **Accept:**
  - [X] `make resource-report` reproduces the tables
  - [X] CI fails if LUTs-per-MAC regresses > 10% **(initial)**
  - [X] the report states how many MACs/cycle fit candidate dev boards (ECPIX-5 / DE10-Nano) at ≥ 100 MHz
  - [X] the report states the ops/cycle needed to match the memory beat rate

### 2.7 Optional: table-lookup variant [L] · *deferred*
- **Do:** TeLLMe-style table-lookup ternary matmul (grouped activations, precomputed partial sums) behind the same interface as 2.4.
- **Accept:**
  - [ ] same bit-exact tests
  - [ ] resource report compares LUTs/MAC and Fmax vs 2.4 at equal throughput
  - [ ] ADR-004 records the choice

### 2.8 Randomized regression suite [S] · *S1 · S2*
- **Do:** seeds, small PR configs and large nightly configs.
- **Accept:**
  - [X] nightly runs ≥ 10⁵ random cases per config with zero mismatches
  - [X] a failing seed is printed and replayable with one command

### 2.9 Optional: formal properties for stream handshakes [M] · *deferred*
- **Accept:**
  - [ ] valid/ready stability and no-data-loss proven for the tile's input and output streams

**Gate 2 (G1, architecture decision):** 2.6 shows tile throughput can match the memory beat rate within small-board resources; ADR-001 (overlay) and ADR-004 (micro-architecture) decided.

---

## Phase 3: First token from hardware, in sim

**Goal:** greedy generation on M0 via `setun run <model.ttape> --backend sim` where hardware does GEMV and host handles non-linear ops.

### 3.1 `TernaryAccel` wrapper with CSRs [M] · *S2*
- **Do:** CSRs for control/status, N, K, G, input/output/weight/scale pointers, activation scale, error flags. Completion by polling (IRQ optional).
- **Accept:**
  - [X] `csr.json` is generated
  - [X] one GEMV launched by register writes matches golden
  - [X] illegal parameters (K = 0, misaligned address, N too large) set error bits and never hang (tested)

### 3.2 Weight streaming from SimSoC memory [L] · *S2*
- **Do:** SDRAM/RAM model in sim; burst reader (LiteDRAM DMA or Wishbone/AXI) feeding tile; preload weights with sim memory init file option; document burst and alignment rules.
- **Accept:**
  - [X] GEMV with memory-resident weights is bit-exact for N, K ∈ {64, 256, 2048}
  - [X] steady-state bus utilization ≥ 80% **(initial)**, reported

### 3.2b Packed `int8` output DMA writer [S] · *S2*
- **Do:** Coalesce 4 consecutive `int8` row outputs into one 32-bit LE word (`out_addr + 4*m`) when `ctrl.req_enable == 1`, flushing ragged tails (`N % 4 != 0`) with Wishbone byte-enable strobes `sel = (1 << (N % 4)) - 1`; preserve wide `int32[N]` DMA writes when `ctrl.req_enable == 0`.
- **Accept:**
  - [X] contiguous `int8[N]` (`act_addr`-compatible zero-copy layout) verified for aligned (`N ∈ {4, 16, 64}`) and ragged (`N ∈ {1, 2, 3, 5, 7, 13}`) row counts
  - [X] trailing sentinel bytes at `out_addr + N` untouched when `N % 4 != 0`
  - [X] `perf_bus_beats` confirms `⌈N / 4⌉` output write beats (`req_enable == 1`) and wide `int32[N]` (`req_enable == 0`, including `N > 512`)

### 3.3 `AccelDriver` host abstraction [M] · *S2 · S3 (board backend)*
- **Do:** Python API (`write_csr`, `dma_write`, `dma_read`, `launch`, `wait`, `perf`) with backends: cocotb-direct, Etherbone via `litex_server`/`RemoteClient` (`sim`), later `board`.
- **Accept:**
  - [ ] the same test script passes on both sim backends
  - [ ] the backend is chosen by `--backend ref|sim|board` flag

### 3.4 Hybrid M0 decode loop & `setun run` CLI [M] · *S2*
- **Do:** `setun run <model.ttape> --backend ref|sim --tokens N` in `setun/cli.py`: linear GEMV layers executed on `SimSoC` via `AccelDriver`, non-linear ops on host (`setun/ref/`), streaming token IDs + printing CSR `cycles/tok`, bus utilization %, and effective `MB/s` telemetry summary on stderr.
- **Accept:**
  - [ ] `setun run models/m0_toy.ttape --backend sim --tokens 16` produces 16 tokens bit-exact vs `--backend ref`
  - [ ] CSR cycle counters and throughput metrics logged to stderr
  - [ ] runs in ≤ 10 min in CI

### 3.5 First-divergence debugger (`setun diff`) [M] · *S2*
- **Do:** `setun diff traceA traceB` reports the first op/tensor/index mismatch with context; trace format has op IDs, shapes and hashes.
- **Accept:**
  - [ ] an injected 1-bit fault in GEMV (test hook) is localized to the exact op and row by an automated test

### 3.6 System-sim job in CI (`make sim-m0`) [S] · *S2*
- **Do:** Wire `make sim-m0` to run `setun run m0_toy.ttape --backend sim --tokens 16` bit-exact vs ref, incorporating randomized `reset_less` register initialization and SDRAM wait-state jitter.
- **Accept:**
  - [ ] `make sim-m0` runs green nightly in CI
  - [ ] trace and cycle logs saved as downloadable build artifacts

**Gate 3 ("first token"):** `setun run m0_toy.ttape --backend sim` tokens == ref (16 tokens bit-exact); cycles/token & MB/s telemetry logged; CI green.

---

## Phase 4: Whole decoder on chip, in sim

**Goal:** the decode loop runs from a command stream with the CPU idle.

### 4.1 Command format v1 [M] · *Track B*
- **Do:** `docs/COMMANDS.md`: opcodes, operand encodings (addresses, shapes, scales, flags), fences, loops, and a reserved **indirect-address mode** (needed later for MoE routing and dynamic KV length). Python encoder/decoder. Record ADR-005 (control plane architecture decision).
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
- **Do:** absmax → scale → quantize, dynamic per-token, per `NUMERICS.md`; zero vector handled.
- **Accept:**
  - [ ] bit-exact on golden vectors for sizes {64, 2048, 6144}

### 4.4 Norm units: RMSNorm and SubLN [M] · *Track B*
- **Do:** RMSNorm / SubLN (sum of squares, rsqrt per `NUMERICS.md`, per-channel weight) and per-head QK-norm; epsilon and widths per spec.
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
- **Do:** SwiGLU / ReLU² per model spec; residual add with saturation/width per spec.
- **Accept:**
  - [ ] bit-exact, or error ≤ spec, on golden vectors

### 4.8 Embedding, head, sampler [L] · *Track B*
- **Do:** embedding row gather and dequantization; head GEMV; argmax. Top-k/temperature stay on host.
- **Accept:**
  - [ ] M0 token IDs match golden
  - [ ] head bytes/token reported

### 4.9 Prefill baseline [L] · *Track E*
- **Do:** implement prefill baseline (streaming activation matrix through the tile) so prompt evaluation updates state bit-exactly.
- **Accept:**
  - [ ] a 64-token prompt leaves state (KV cache, last hidden state) identical to golden
  - [ ] prefill throughput measured; if GEMM mode is deferred to 8.4, TTFT cost is documented in README

### 4.10 Full decoder from a command stream [L] · *Track B*
- **Do:** hand-written program for M0.
- **Accept:**
  - [ ] M0 generates 64 tokens identical to golden
  - [ ] bus trace shows CPU only starts and polls
  - [ ] per-op cycle breakdown report

### 4.11 Performance counters [M] · *Track B*
- **Do:** cycles per op class, stall cycles, DMA bytes, exposed through CSRs.
- **Accept:**
  - [ ] counters within 5% of an independent sim measurement **(initial)**

**Gate 4:** M0 end-to-end on chip in sim; command format v1 frozen and recorded.

---

## Phase 5: Hardware Command-Schedule Packer MVP

**Goal:** `setun pack-schedule` lowers `.ttape` layer graphs into hardware command streams (`setun pack-schedule | inspect`).

### 5.1 IR [M] · *Track C*
- **Do:** ops, tensor types (shape, dtype, layout, quantization params), graph container, text dump, JSON serialization, verifier.
- **Accept:**
  - [ ] IR round-trip tests
  - [ ] verifier rejects ≥ 10 malformed graphs (negative tests)

### 5.2 Container Frontend (`.ttape`) [L] · *Track C*
- **Do:** `.ttape` container frontend loading frozen `TTAPE v1.0` model manifests (`M0` toy and `M1` dense profiles exported via `TernaryTape.jl`), validating tensor shapes and architecture descriptors against `docs/models.md`.
- **Accept:**
  - [ ] M0 and M1 `.ttape` containers import with shapes matching `docs/models.md`
  - [ ] unsupported ops or malformed `.ttape` headers fail with actionable error messages

### 5.3 IR interpreter [M] · *Track C*
- **Do:** executes graph IR with golden ops in `setun/ref/`.
- **Accept:**
  - [ ] equals golden model bit-exactly on M0

### 5.4 Lowering & Scheduling Passes [XL] · *Track C*
- **Do:** legalize (map model ops to overlay ops); fold scales; fuse (norm + quant, matmul + requant); weight layout and packing (tile-major); tiling and scheduling for (R, L, G) in `hwconfig.json`; static memory planner (activation arena, KV cache, weights, alignment); command descriptor stream emission.
- **Accept:**
  - [ ] unit tests per pass, with IR dumps before and after
  - [ ] memory-plan checker finds no overlaps
  - [ ] schedule checker finds no dependency violations

### 5.5 Schedule Emitter & Manifest [M] · *Track C*
- **Do:** emit binary command stream, weight image pointers, and schedule manifest (hashes, shapes, required `hwconfig`, ISA version).
- **Accept:**
  - [ ] packed M0 schedule run in sim gives tokens == golden
  - [ ] manifest check rejects mismatched tile config or ISA version

### 5.6 Hardware-config contract [M] · *Track C*
- **Do:** each SoC build emits `hwconfig.json` (R, L, G, SRAM sizes, bus width, ISA version); schedule packer validates fit.
- **Accept:**
  - [ ] a model that does not fit (for example KV cache too large) fails with a message naming limiting resource

### 5.7 CLI Hardware Surface (`setun pack-schedule | inspect`) [M] · *Track C*
- **Do:** `setun pack-schedule | run | inspect | diff | bench` CLI.
- **Accept:**
  - [ ] `setun pack-schedule models/m0_toy.ttape --target sim -o build/m0.sched` works
  - [ ] `setun inspect build/m0.sched` prints per-layer command schedule and predicted cycles
  - [ ] M1 schedule packing completes in ≤ 60 s **(initial)**

### 5.8 Performance model [M] · *Track C*
- **Do:** predicted cycles per token from schedule and bandwidth model.
- **Accept:**
  - [ ] within ±15% of measured sim cycles on M0 **(initial)**

### 5.9 Optional: design-space exploration [L] · *deferred*
- **Do:** given board resources and a model, suggest optimal (R, L, SRAM buffer sizes).
- **Accept:**
  - [ ] reproduces or beats hand-picked configuration from Phase 6

### 5.10 Stretch: kernel DSL [PARKED / OUT OF SCOPE]
- **Note:** General host language compilers and RTL-generating DSLs are strictly out-of-scope for `setun` per Draft 0.4 invariants.

**Gate 5:** packing M0 schedule and running in sim gives tokens == golden; packer output reproduces hand-written program from 4.10; M1 packs and layer 0 runs in sim equal to golden.

---

## Phase 6: Small-board bring-up, toy model

**Goal:** the toy model runs on real silicon with external DDR over open/vendor toolchains.

### 6.1 Board selection [S] · *S3*
- **Do:** pick a non-AMD, non-Chinese target board with open/vendor toolchain support, ≥ 512 MB DDR3 (ECPIX-5) or 1 GB DDR3 (DE10-Nano), and enough LUTs for tile from 2.6. Record ADR-007.
  - *Candidate A:* Lattice ECP5-85F (LambdaConcept ECPIX-5, 512MB DDR3, open `yosys`+`nextpnr` flow).
  - *Candidate B:* Altera Cyclone V (Terasic DE10-Nano, 1GB DDR3 + ARM HPS AXI bridge, Quartus Lite).
- **Accept:**
  - [ ] BIOS boots; LiteDRAM memtest passes; console works over UART / Ethernet
  - [ ] ADR-007 records board, memory, toolchain and cost

### 6.2 Reproducible FPGA build [M] · *S3*
- **Do:** `make fpga-build BOARD=…` (open `yosys`+`nextpnr-ecp5` or Quartus Lite).
- **Accept:**
  - [ ] same inputs → same utilization report
  - [ ] bitstream and reports archived

### 6.3 Fit and timing [L] · *S3*
- **Do:** choose (R, L, G) and pipeline depth.
- **Accept:**
  - [ ] worst negative slack ≥ 0 at ≥ 100 MHz measured **(initial)**
  - [ ] utilization report and critical-path notes committed

### 6.4 Weight loader [M] · *S3*
- **Do:** stream `.ttape` weight files over Etherbone/UDP or SD card with header + payload FNV-1a verification.
- **Accept:**
  - [ ] M0 `.ttape` weights loaded and FNV-1a checksum verified
  - [ ] load throughput reported

### 6.5 Run M0 on hardware (`setun run --backend board`) [M] · *S3*
- **Do:** Execute `setun run m0_toy.ttape --backend board --tokens 16` bit-exact vs `--backend ref`, plus continuous DDR burst stability soak.
- **Accept:**
  - [ ] 16 tokens bit-exact vs golden reference
  - [ ] 256-token continuous DDR burst thermal and read-window stability soak with zero errors

### 6.6 Measurements [M] · *S3*
- **Do:** tokens/s, bandwidth utilization, power (inline meter), compared with Phase 1 baselines.
- **Accept:**
  - [ ] `docs/results.md` updated with method and raw data

### 6.7 Hardware-in-the-loop CI [L] · *deferred*
- **Do:** board attached to self-hosted runner; nightly flash-and-test.
- **Accept:**
  - [ ] nightly HIL job green for 7 consecutive days **(initial)**

**Gate 6:** toy model bit-exact on silicon via `setun run --backend board`.

---

## Phase 7: Real dense model on a DDR board

**Goal:** M1 on hardware. This is the v0.1 demo. In PLAN.md it runs in Track A in hybrid mode (GEMV on chip, host does the rest), so it does not have to wait for Phases 4–5.

### 7.1 Memory budget [S] · *Track A*
- **Accept:**
  - [ ] table of weights + KV cache (for chosen context length) + buffers ≤ board memory (512MB ECPIX-5 / 1GB DE10-Nano)

### 7.2 Compile and verify M1 layer by layer [L] · *Track A*
- **Do:** run layer 0 first, then all layers, comparing traces with `setun diff`.
- **Accept:**
  - [ ] per-layer outputs equal golden for all layers on hardware

### 7.3 Memory-system tuning [L] · *Track A*
- **Do:** burst size, bank/row locality, double-buffered activations, DMA/compute overlap.
- **Accept:**
  - [ ] ≥ 70% of theoretical DDR bandwidth sustained during tile streaming **(initial)**, reported

### 7.4 Terminal Text Generation (`setun run`) [M] · *Track A*
- **Do:** `setun run m1_dense.ttape --backend board --prompt "Once upon a time" --tokenizer tokenizer.json` streaming decoded text tokens to stdout and telemetry footer to stderr.
- **Accept:**
  - [ ] 5 fixed prompts produce first 64 greedy tokens bit-exact vs golden model

### 7.5 Stability [M] · *Track A*
- **Accept:**
  - [ ] 1-hour soak with zero errors or hangs
  - [ ] hardware perplexity on a fixed sample equals golden model

### 7.6 Performance vs bound [S] · *Track A*
- **Do:** compare measured tokens/s with `usable_bandwidth ÷ bytes_read_per_token`.
- **Accept:**
  - [ ] written analysis of gap with top three causes

**Gate 7 (v0.1 demo):** M1 generates text on hardware via `setun run`, matching golden; results published in `docs/results.md`.

---

## Phase 8: Runtime CLI, benchmarks, release

### 8.1 Runtime library (`AccelDriver`) [M] · *Track D*
- **Do:** sessions, KV reset/continue, token streaming, argmax sampler in `setun/runtime/`.
- **Accept:**
  - [ ] multi-turn session reproduces golden model output for same token history

### 8.2 [REMOVED — Out of Scope per Boundary Invariant]
- **Note:** HTTP/OpenAI model servers (`/v1/chat/completions`) are intentionally excluded from `setun` per Draft 0.4 invariants. `setun` provides strictly the hardware driver library (8.1) and terminal CLIs (`setun run | bench`).

### 8.3 Benchmark harness (`setun bench`) [M] · *Track D*
- **Do:** `setun bench`: decode tok/s, prefill tok/s, time to first token, joules/token, perplexity vs reference runtime.
- **Accept:**
  - [ ] reproducible with one command on sim and on board
  - [ ] every result includes config hash (hwconfig + model + commit)

### 8.4 Prefill performance [L] · *Track E*
- **Do:** GEMM mode or batched tile for prefill.
- **Accept:**
  - [ ] time to first token for 512-token prompt ≥ 4× faster than token-by-token **(initial)**

### 8.5 Docs and packaging [M] · *deferred*
- **Do:** tutorial, architecture doc, ADR index, wheels, licence headers, release notes.
- **Accept:**
  - [ ] new user reproduces sim demo from README in ≤ 30 min on clean machine

### 8.6 Release v0.1 [S] · *deferred*
- **Accept:**
  - [ ] tagged release with artifacts and results page

**Gate 8:** v0.1 tagged; new user reproduces sim demo from README on clean machine.

---

## Phase 9: High-bandwidth platform (stretch)

**Goal:** prove the design scales with high memory bandwidth on non-AMD platforms.

### 9.0 Feasibility spike and board decision [M] · *north star*
- **Do:** evaluate non-AMD HBM platform targets (e.g. Altera Agilex 7 M-series kit DK-DEV-AGM039EA with HBM2e, quad-core ARM Cortex-A53 HPS, 32 GB class via Quartus Pro). Record ADR-008.
- **Accept:**
  - [ ] go/no-go note with licence, cost, access and list of blockers
  - [ ] ADR-008

### 9.1 Platform bring-up [L–XL] · *north star*
- **Do:** LiteX target and HBM wrapper (AXI) for chosen board; host link (PCIe or HPS/Ethernet).
- **Accept:**
  - [ ] memtest across all pseudo-channels
  - [ ] sustained read bandwidth ≥ 70% of theoretical in streaming test **(initial)**

### 9.2 Multi-tile / multi-channel scaling [XL] · *north star*
- **Do:** bind tiles to channels; spread weights across channels; extend memory planner.
- **Accept:**
  - [ ] bandwidth scales ≥ 0.8× linearly with channels in streaming test **(initial)**
  - [ ] M1 runs bit-exact on board

### 9.3 Timing closure [L–XL] · *north star*
- **Do:** floorplanning per channel, pipelining.
- **Accept:**
  - [ ] worst negative slack ≥ 0 at target Fmax (250–350 MHz)

### 9.4 Larger dense model rungs (M2) [L] · *north star*
- **Do:** M2 (3B–8B dense ternary `.ttape` checkpoints).
- **Accept:**
  - [ ] first 64 tokens bit-exact vs golden
  - [ ] tokens/s reported against bandwidth bound

---

## Phase 10: MoE and hybrid architectures (stretch)

### 10.1 MoE routing and expert dispatch [XL] · *north star*
- **Do:** top-k over many experts; indirect descriptors (reserved in 4.1); expert prefetch from DDR/HBM; shared expert.
- **Accept:**
  - [ ] toy MoE `.ttape` is bit-exact vs golden
  - [ ] expert-fetch bytes per token reported

### 10.2 Gated DeltaNet ops [XL] · *north star*
- **Do:** short causal conv, SiLU/softplus, L2 norm, delta-rule state update, gated output norm.
- **Accept:**
  - [ ] toy hybrid model `.ttape` is bit-exact vs golden
  - [ ] one real layer matches reference runtime within tolerance

### 10.3 MoE / Hybrid model rungs (M3 / M4) [XL] · *north star*
- **Do:** M3/M4 (large ternary MoE and hybrid `.ttape` models).
- **Accept:**
  - [ ] first 32 tokens agree with reference runtime
  - [ ] tokens/s against bandwidth bound (active-weights-per-token for MoE)

### 10.4 Long-context memory plan [M] · *north star*
- **Do:** KV cache for full-attention layers; fixed-size state for DeltaNet layers.
- **Accept:**
  - [ ] memory table for 32k and 128k context, checked by schedule packer fit test

---

## Decisions, risks, open questions

Maintained in [PLAN.md](../PLAN.md).

