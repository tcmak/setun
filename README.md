# Setun

**A LiteX-native ternary LLM hardware accelerator overlay, golden reference model, and toolchain.**

*Informal pitch: "CUDA for LiteX FPGAs". Named after the Setun (Сетунь), the 1958 balanced-ternary computer built at Moscow State University.*

## Status & Roadmap

**Current State:**
- **Season 1 — COMPLETE:** 0-DSP `TernaryDot` + `GroupAccumulateScale` RTL verified across $6 \times 10^5$ Verilator vectors; Yosys + `nextpnr-ecp5` resource/Fmax baseline locked.
- **Season 2 — Blocks A–C COMPLETE:** `M0` golden model & `TTAPE v1.0` loader (`1.5`–`1.6`), `WeightUnpacker` + multi-row `TernaryTile` + `Requantize` (`2.3`–`2.5`, `2.8b`), and LiteX `TernaryAccel` SoC wrapper with `CTI_BURST_INCREMENTING` RAM DMA & zero-copy `4 × int8` output writeback at 86%–97%+ bus utilization (`0.4`, `3.1`–`3.2b`).
- **Up Next (Season 2 Block D — Gate 3):** `AccelDriver` (`3.3a`), `setun run` hybrid `M0` decode (`3.4`), `setun diff` mismatch debugger (`3.5`), and `make sim-m0` system-sim CI (`3.6`).

* **Facts Checked:** `2026-11-22`

* **Planning & Governance:**

  * [`PLAN.md`](PLAN.md) — Season roadmap, session blocks, and milestone gates.

  * [`NEXT.md`](NEXT.md) — Immediate active engineering tasks.

  * [`docs/phases.md`](docs/phases.md) — Full task catalogue and acceptance criteria checkboxes.

  * [`docs/decisions.md`](docs/decisions.md) — Architecture Decision Records (`ADR-001` through `ADR-015`).

  * [`docs/open-questions.md`](docs/open-questions.md) — Cross-repo blocker log (`OQ-001` resolved in Task B2).

## Why Ternary LLMs on LiteX FPGAs?

Single-batch autoregressive LLM decoding ($\text{batch size} = 1$) is fundamentally **memory-bandwidth bound**, not compute bound: every generated token streams the entire model weight matrix from external memory to perform a matrix-vector multiply ($\mathbf{y} = \mathbf{W}\mathbf{x}$).

By constraining linear projection weights to balanced ternary values $w \in \{-1, 0, +1\}$ (BitNet b1.58 / Grouped `Q2_0`):

1. **8x Memory Traffic Reduction vs. FP16:** Weights pack into **2 bits/trit** (4 weights per byte, LSB-first) plus compact per-group scales, drastically lowering the external DDR/SRAM bandwidth required per token.

2. **Multiplier-Free Ternary MAC Lanes (`0 DSPs` in `TernaryDot`):** Multiplying a signed `int8` activation $x \in [-128, +127]$ by $w \in \{-1, 0, +1\}$ requires **zero hardware multipliers**—only a 2-to-1 multiplexer/negator per lane feeding a balanced binary adder tree in standard FPGA LUTs.

3. **Abundant Compute Headroom on Low-Cost FPGAs:** In default zero-DSP mode, a canonical 64-lane, 128-group `TernaryRow` synthesizes to **3,300 LUT6, 0 DSPs** (`51.56 LUTs/MAC`) on Xilinx 7-Series fabric (or **2,697 LUT6, 1 DSP48E1** with `--use-dsp` opt-in), and places-and-routes at **62.08 MHz** out-of-context on Lattice `ECP5-85F` (`nextpnr-ecp5`, unpipelined single-cycle row accumulation; pipelined $\ge 100\text{ MHz}$ closure in Season 3 Task `6.3`). Even a small Artix-7 (`XC7A35T`) fits 5 parallel rows (**320 MACs/cycle**, `[EXTRAPOLATED from R=1]`), exceeding the external DDR3 weight-streaming ceiling.

## Architecture & How It Works

### 1. Hardware Datapath (`setun/accelerators/` · Migen HDL)

* **Dataflow Pipeline:**
  `Packed 2-bit Trits` + `uint16 Scales` $\longrightarrow$ **`WeightUnpacker` (`valid`/`ready`)** $\longrightarrow$ **`TernaryDot` (**$L$ **lanes, 0 DSPs, `valid`-only stall-free)** $\longrightarrow$ **`GroupAccumulateScale` (**$G$**, `valid`-only stall-free)** $\longrightarrow$ **Wide Row Accumulator (`round-half-up >> shift`)**

* **Streaming Handshake Contracts (`valid`/`ready` vs. `valid`-Only Stall-Free):**

  * **`WeightUnpacker` (`setun/accelerators/unpacker.py`):** Implements full **`valid`/`ready`** stream handshaking on both upstream (`trit_valid`/`trit_ready`, `scale_valid`/`scale_ready`) and downstream (`out_valid`/`out_ready`) interfaces, sustaining 1 beat/cycle with zero group-transition bubbles under downstream backpressure.

  * **`TernaryDot` & `GroupAccumulateScale` / `TernaryRow` (`ternary_dot.py`, `group_scale.py`):** Implement a **`valid`-only stall-free** sink/pipeline contract (`in_valid` / `out_valid`, plus `row_last` end-of-row framing on `GroupAccumulateScale` and `TernaryRow`—no `in_ready` or `out_ready` ports). Both cores accept up to 1 beat/cycle with zero internal stalls and are verified bit-exact under arbitrary 1-cycle and multi-cycle mid-row `in_valid=0` bubbles as well as zero-gap back-to-back rows.

* **`TernaryDot` (`setun/accelerators/ternary_dot.py`):** Parametric $L$-lane ($L \in \{16, 32, 64, 128\}$) combinational/pipelined (`pipeline_depth` $\in \{0, 1, 2\}$) ternary dot-product engine with `valid`-only stall-free streaming (`in_valid` $\to$ `out_valid`). Uses **0 DSP slices** across all configurations. Exact bit-growth sizing ($\lceil \log_2(L \cdot 128 + 1) \rceil + 1$ bits; 15-bit signed output for $L=64$) prevents overflow even at the corner case $x = -128, w = -1 \implies +128$.

* **`GroupAccumulateScale` & `TernaryRow` (`setun/accelerators/group_scale.py`):** Accumulates $G/L$ beats ($G \in \{64, 128, 256\}$) into a 16-bit signed group partial sum $P_g$, multiplies by the group's unsigned 16-bit scale (`scale_u16`), accumulates across groups along a matrix row, and applies a hardware-friendly round-half-up arithmetic right shift (`(acc + (1 << (shift - 1))) >> shift`).

  * **DSP Usage Modes (`ADR-004`):** By default, synthesis runs in **zero-DSP mode** (`0 DSPs` across the entire row, mapping the once-per-group $16\text{-bit} \times 16\text{-bit}$ scale multiply to pure LUT fabric: `3,300 LUT6, 114 FFs, 0 DSP48E1` at $L=64, G=128$). Passing `--use-dsp` to `scripts/resource_report.py` opts into **1 `DSP48E1` slice per row** (`2,697 LUT6, 66 FFs, 1 DSP48E1`).

* **LiteX SoC Wrapper & CSR Peripherals (`setun/soc/ternary_accel.py`, `setun/accelerators/stub.py` & `targets/sim.py`):** `SetunSimSoC` integrates both `AccelStub` (`ID`, `VERSION`, `SCRATCH`, free-running cycle `COUNTER`) and `TernaryAccel` (wiring `WeightUnpacker` $\to$ `TernaryDot` $\to$ `GroupAccumulateScale` with CSR activation staging, programmable round-half-up `shift`, completion status, and row result readback) over a 32-bit Wishbone bus, deterministically exporting [`reports/csr.json`](reports/csr.json) and [`reports/csr.h`](reports/csr.h).

### 2. Numerics, Weight Format & Golden Model (`setun/ref/`)

* **Bit-Exact Reference Ops (`setun/ref/ops.py`):** Executable Python/NumPy specification for `lane_select_code`, `ternary_dot`, `group_accumulate`, `apply_scale`, and `row_dot` (supporting ragged $K$ not divisible by $G$).

* **Weight Packing & Binary Codec (`setun/ref/format.py` · `docs/FORMAT.md`):**

  * Packs 2-bit trits LSB-first (`0b00: 0`, `0b01: +1`, `0b10: -1`, `0b11: reserved`) with per-group `uint16` scales ($G \in \{64, 128, 256\}$) and a shared right-shift exponent (`shift`).

  * **Normative `0b11` Contract vs. Hardware Defense-in-Depth (`ADR-003`):** Rejecting the reserved code `0b11` (and any non-zero tail padding) on load is the strict software/loader contract (`unpack_trits_1d` raises `ValueError`). The hardware `TernaryDot` lane aliases `0b11 -> 0` strictly as silicon defense-in-depth against uninitialized bus/FIFO beats, never as a valid weight encoding.

  * **Canonical `TTAPE` v1 Container (`ADR-014`):** Setun consumes the canonical `.ttape` (`magic = "TTAP"`, `0x50415454` LE) specification and golden corpus from `third_party/TernaryTape` (`spec-v1.0`), supporting both **`TTAPE-core`** (`DSP=0` power-of-two shift scales) and **`TTAPE-grouped`** (`0x0006` `group_scales_u16`) while rejecting `TTAPE-extended` (`0x0007` `fp32_aux`) fail-closed in hardware loader mode.

* **`M0` Golden Transformer Decoder (`setun/ref/m0.py` & `scripts/make_toy.py`):**

  * Complete 2-layer autoregressive decoder with `M1`'s modern architecture: Grouped-Query Attention (GQA) with KV cache, pre-attention & pre-FFN `RMSNorm`, per-head `QK-norm`, Rotary Position Embeddings (`RoPE`), `SwiGLU` FFN, and tied token embeddings / LM head.

  * Supports both hardware-exact `"integer"` mode (per-token symmetric `int8` absmax activation quantization + integer `row_dot`) and `"fp32"` reference mode, achieving $> 0.99998$ **hidden-state cosine similarity** and deterministic SHA256 toy weight generation (`tiny` and `small` presets).

## Repository Layout

* **`setun/accelerators/`** — Migen RTL modules (`WeightUnpacker`, `TernaryDot`, `GroupAccumulateScale`, `TernaryRow`, `TernaryTile`, `Requantize`, `AccelStub`).

* **`setun/ref/`** — Golden Python/NumPy reference ops (`ops.py`), pack/unpack binary codec (`format.py`), and `M0` decoder (`m0.py`).

* **`setun/soc/`** — LiteX `AutoCSR` wrapper (`TernaryAccel`) and deterministic `csr.json` / `csr.h` exporter (`export_soc_csr_artifacts`).

* **`targets/sim.py`** — LiteX `SetunSimSoC` target with Wishbone interconnect and deterministic CSR exporter (`--export-csr-reports`).

* **`scripts/`** — Toolchain verifier (`doctor.py`), deterministic `M0` weight generator (`make_toy.py`), and Yosys + `nextpnr-ecp5` resource/timing characterization (`resource_report.py`).

* **`tests/unit/`** — `pytest`, Migen, and Verilator `cocotb` testbenches, SoC `.ttape` composition tests, and randomized regression sweeps.

* **`docs/`** — Architecture specs (`FORMAT.md`, `NUMERICS.md`, `models.md`, `decisions.md`, `open-questions.md`, `phases.md`, `log.md`).

* **`reports/`** — Deterministically generated Yosys/`nextpnr-ecp5` resource tables (`resources.md`, `resources.json`), `baseline.json`, and LiteX SoC register maps (`csr.json`, `csr.h`).

## Quickstart & Usage

### 1. Environment Setup (WSL2 Ubuntu / Linux)

Setun uses a pinned Python virtual environment (`.venv`)—including `migen==0.10.0` and `yowasp-nextpnr-ecp5==0.11.1.0.post826` (`nextpnr-ecp5` `0.11.1` + `prjtrellis` ECP5 chipdb)—alongside system `verilator` (`5.020`) and `yosys` (`0.33`). Verify your toolchain health with:

* `source .venv/bin/activate`

* `make doctor`

### 2. Run the Test Suite & CI Gates

* **Format and lint (`ruff check` + `ruff format`):** `make format && make lint`

* **Run fast CI suite (unit tests, `M0` golden model, LiteX CSR sim, Verilator RTL tests):** `make ci`

* **Run LiteX `SetunSimSoC` smoke build & CSR map generation:** `make sim-smoke`

* **Run full 600,000+ vector Verilator hardware regression + Yosys resource gate:** `make nightly`

### 3. Generate Deterministic `M0` Toy Models

* **Tiny preset (CI,** $d_{\text{model}}=64, G=64$**):** `python3 -m scripts.make_toy --size tiny`

* **Small preset (Nightly,** $d_{\text{model}}=128, G=128$**):** `python3 -m scripts.make_toy --size small`

### 4. Run Yosys FPGA Resource Characterization

* **Default zero-DSP mode (`0 DSP48E1` everywhere) + check** $\le 10\%$ **LUT/MAC baseline gate:** `make resource-check`

* **Opt-in 1-DSP/row mode (`1 DSP48E1` per row for group scaling via `--use-dsp`):** `python3 -m scripts.resource_report --use-dsp`

## Engineering & Contribution Discipline

1. **Golden Reference First:** No RTL module is written before its exact integer math and bit-width equations are defined in `docs/NUMERICS.md` and implemented with unit tests in `setun/ref/`.

2. **Bit-Exact Cosimulation:** Every Migen hardware module under `setun/accelerators/` must pass Verilator + `cocotb` equivalence tests (including randomized backpressure, `-128` saturation corners, zero scales, and ragged dimensions) against `setun/ref/`.

3. **Zero Unverified Claims:** Resource numbers (`LUT6`, `FF`, `DSP48E1`) and numerical accuracy bounds (`cosine`, `entropy`, `perplexity`) are generated by reproducible scripts in `scripts/` and enforced in CI.

4. **Formatting & Linting:** All commits must pass `make format && make lint` (`ruff` targeting Python 3.10+ compatibility, 100-column line limit) and `make ci` before merge.

