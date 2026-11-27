Architecture Decision Records (ADRs)

ADR-001: Execution Model
Status: Provisional (Decide at start of Track B)
Decision: Fixed NPU overlay + binary command stream.

ADR-002: Activation Precision and Granularity (Provisional)
Status: Provisional for Seasons 1–3 (Revisited in Track A, Task 1.8)
Decision: int8 signed activations ([-128, 127]) with a per-token scale factor.
Context & Trade-offs: Keeps lane datapath width at A = 8 bits (and A + 1 = 9 bits after negation) so a group partial sum of size G = 128 fits cleanly inside a 16-bit signed integer (⌈log2(128 · 128 + 1)⌉ + 1 = 16 bits). Block-scaled int8/int16 and FP16 passthrough are deferred until real-model perplexity evaluations in Track A.

ADR-003: Internal Weight Format, Bit-Packing Endianness, and TTAPE v1 Alignment
Status: Accepted (Container migrated to canonical TTAPE v1 in Task B2; see ADR-014)
Decision:
Encoding: 2-bit trits {00: 0, 01: +1, 10: -1, 11: reserved} with parametric group size G ∈ {64, 128, 256}.
Packing Endianness (LSB-First): Trits are packed little-endian within each byte and across wide bus beats (trit[0] in bits[1:0], trit[1] in bits[3:2], ..., trit[i] in bits[2*i + 1 : 2*i]). This matches TTAPE v1 §3.1 bit-for-bit.
Reserved Code (11) Handling (Contract vs. Defense-in-Depth): Loader rejection is the normative contract (setun/ref/format.py rejects any tape containing 0b11 with TTAPE_ERR_RESERVED_TERNARY_CODE or non-zero tail padding with TTAPE_ERR_NONZERO_PADDING before weights become hardware-visible). The hardware TernaryDot combinational mux aliases 0b11 -> 0 strictly as silicon defense-in-depth (deterministic safe fallback against uninitialized BRAM/FIFO beats), never as an accepted encoding.
Container Framing: Interim STNW framing has been retired in favor of canonical TTAPE v1 (magic "TTAP", 0x50415454 LE) owned by TernaryTape.jl (third_party/TernaryTape/spec/ttape-v1.0.md); see ADR-014.

ADR-004: Ternary Matmul Micro-Architecture (0-DSP Default, Opt-In Row DSP)
Status: Accepted at S1.end (2026-11-14); Amended 2026-11-24 (0-DSP default, non-Xilinx ECP5 + Cyclone V baseline)
Decision: Multiplier-free add/sub tree in TernaryDot (0 DSPs per ternary MAC lane) paired with GroupAccumulateScale defaulting to 0 DSPs (Yosys -nodsp LUT multiplier for TTAPE-grouped u16 scales, or shift-only mode for TTAPE-core power-of-two tapes) and supporting opt-in 1 hard 18x18 DSP per row (--use-dsp) when hard DSP slices are abundant.
Evidence: On canonical L=64, G=128, A=8 TernaryRow, Yosys synth_ecp5 measures 4,741 LUT4 / 0 MULT18X18D (74.08 LUT4/MAC) in default -nodsp mode vs. 3,942 LUT4 / 1 MULT18X18D in --use-dsp mode, and synth_intel_alm -family cyclonev measures 2,677 ALUT / 0 MUL18X18 (41.83 ALUT/MAC) vs. 1,921 ALUT / 1 MUL18X18. On Lattice ECP5-85F (83,640 LUT4), 80% pure LUT fabric fits 14 rows (896 MACs/cycle, [EXTRAPOLATED from R=1]), exceeding the 305 MACs/cycle external DDR3 weight-streaming ceiling (1.6 GB/s at 20.99 MHz governing TernaryAccel Fmax).

ADR-005: Control Plane
Status: Provisional (Decide in Track B)
Decision: Host-driven in Seasons 1–3; command sequencer in Track B.

ADR-006: First Real Model and Reference Runtime
Status: Provisional (Decide at start of Track A)
Decision: Ternary Bonsai 1.7B + PrismML llama.cpp fork (fallback: BitNet b1.58 2B4T).

ADR-007: FPGA Board Selection & Non-Xilinx Dual-Target Strategy (Task 6.1)
Status: Accepted (2026-11-27)
Decision: Start with two desk-sized DDR3 targets via `make fpga-build BOARD=ecpix5|de10nano`:
1. Primary Open-STA & Bitstream Target (`BOARD=ecpix5`): LambdaConcept ECPIX-5 (France; Lattice ECP5-5G `LFE5UM5G-85F-8BG554C` speed -8, 83,640 LUT4, 208 DP16KD, 156 MULT18X18D; 512 MB 16-bit DDR3 via LiteDRAM `ECP5DDRPHY` + RGMII GbE + FT2232H USB-C; 100% open `yosys` + `yowasp-nextpnr-ecp5` + `yowasp-ecppack` flow in `.venv`/CI; ~EUR 329 / ~GBP 280 ex-VAT, currently backordered direct).
2. Primary Stocked UK Silicon Target (`BOARD=de10nano`): Terasic DE10-Nano `P0496` (Taiwan; Intel Cyclone V SE `5CSEBA6U23I7`, 110K LE / 41,910 ALM, 553 M10K, 112 DSPs; 1 GB 32-bit DDR3-800 + GbE via hardened Cortex-A9 HPS AXI bridge with decoupled fabric PLL; CI utilization via `yosys synth_intel_alm` + bitstream/STA via free Quartus Prime Lite; ~GBP 245 in UK stock). Radiona ULX3S-85F (~GBP 207) is deferred due to 32-64 MB SDR SDRAM and onboard ESP32.

ADR-008: HBM Board Selection
Status: Not planned (North star)

ADR-009: Hardware Description Language (Migen)
Status: Accepted at S1.end (2026-11-14)
Decision: Write RTL cores under setun/accelerators/ in Migen for Native LiteX/LiteDRAM/CSR integration, verified via Migen simulation and cocotb + Verilator.

ADR-010: License and Project Name
Status: Accepted (2026-10-17)
Decision: BSD-2-Clause license; package and CLI named setun.

ADR-011: Hardware Group-Scale Representation
Status: Accepted for Season 1 (Task 1.4a / 2.2b)
Decision: Pre-converted 16-bit unsigned fixed-point mantissa (uint16) per group, paired with a shared per-row (or per-tensor) right-shift exponent (shift).

ADR-012: Host Toolchain & Environment Pinning (Task 0.2a; Amended in Task C1.1b)
Status: Accepted for Season 1; Amended 2026-11-20
Decision: Native Ubuntu/Debian (WSL2) system packages via apt (verilator, yosys) combined with a local Python virtual environment (.venv, including migen==0.10.0 and yowasp-nextpnr-ecp5==0.11.1.0.post826 providing deterministic nextpnr-ecp5 0.11.1 + prjtrellis ECP5 chipdbs without requiring root) pinned via toolchain.lock and verified by make doctor. Containerization (Dockerfile) is deferred to Task 0.2b.

ADR-013: Cross-Portfolio Mythos-Loop Alignment & Season 3 HIL Parity Runner
Status: Accepted (2026-11-18)
Decision: Align Setun's synthesis, real-model evaluation, and board bring-up gates with the cross-repo VeriObjC / TernaryTape / veriStream-mythos loop:
1. Task 2.6 characterizes Lattice ECP5 (yosys synth_ecp5 + nextpnr-ecp5) and Intel Cyclone V (yosys synth_intel_alm) with 0-DSP (-nodsp) as the default baseline so ADR-007 board selection uses measured non-Xilinx multi-vendor data without extrapolating ECP5 timing onto synth-only Cyclone V rows.
2. Season 4 Track A enforces a real-checkpoint PTQ quality gate (perplexity / probe accuracy on <= 1B weights) prior to the M1 demo, splitting hardware vs. QAT research tracks if generic PTQ collapses.
3. Season 3 board bring-up (Etherbone / litex_server) doubles as the portfolio Hardware-in-the-Loop (HIL) parity runner against shared TTAPE golden fixtures.

ADR-014: Canonical TTAPE v1 Container Adoption (TTAPE-core & TTAPE-grouped)
Status: Accepted (2026-11-20, Task B2; promoted to frozen spec-v1.0 in Task C1.3)
Decision:
1. Single Spec & Fixture Source of Truth: Setun vendors TernaryTape.jl (via HTTPS submodule at third_party/TernaryTape, pinned at spec-v1.0) and removes duplicated wire-format tables from docs/FORMAT.md.
2. Supported Profiles: Setun's default loader gate (reader_profile="grouped") supports both TTAPE-core (flags bit 2 = 0, per-layer s16 power-of-two shift scales in 0x0002 executed in 0-DSP shift mode) and TTAPE-grouped (flags bit 2 = 1, section 0x0006 group_scales_u16 carrying explicit group_size G in {64, 128, 256}, shared uint8 shift in [0, 30], and row-major uint16 scale mantissas).
3. Fail-Closed Extended Gate: Tapes carrying host-only section 0x0007 (fp32_aux) or host layer kinds {6, 7} (TTAPE-extended) are rejected fail-closed with TTAPE_ERR_UNSUPPORTED_PROFILE under Setun's hardware-targeted "grouped" profile, and accepted only when host callers explicitly request reader_profile="extended".

ADR-015: TernaryTile Multi-Row Stream Contract, Ragged Tail Framing, and Backpressure Safety
Status: Accepted (2026-11-21, Task 2.4)
Decision:
1. Lockstep R-Row Sink Endpoint: TernaryTile(rows=R, lanes=L, group_size=G) exposes a LiteX stream.Endpoint sink carrying lockstep packed trits (R * 2 * L bits, row 0 in LSBs) and per-row group scales (R * 16 bits, row 0 in LSBs), with sink.last marking the final K-slice beat of an R-row stripe. Shared int8 activations (up to max_k >= 2048) live in an inferred BRAM/LUTRAM Memory(width=L * act_bits, depth=max_k // L) and broadcast to all R TernaryRow engines simultaneously.
2. Serialized 1-Row/Cycle Source Endpoint via R-Deep Shadow Buffer: Completed row accumulators latch into an R-deep shadow buffer and drain at 1 row/cycle over a LiteX stream.Endpoint source (payload: signed acc_bits result, 16-bit row_idx; packet framing: source.last). Programmable round-half-up shift (0..30) is applied once after the R-to-1 shadow-buffer mux rather than duplicated across all R rows, keeping tile wrapper overhead within the <= 10% LUTs/MAC baseline gate.
3. Full Backpressure Safety (ceil(K/L) < R and source.ready == 0): Non-final stripe beats (sink.last == 0) stream at full line rate while the shadow buffer drains the previous stripe, achieving >= 95% of R * L MACs/cycle whenever ceil(K/L) >= R. On sink.last == 1, sink.ready stalls cleanly whenever a prior row_last commit is in flight (1-cycle TernaryRow latency) or the shadow buffer still holds undrained valid rows that do not finish draining on the current cycle.
4. Ragged Tail & source.last Framing (N % R != 0 and num_rows == 0 guard): Padded tail lanes (row_idx >= num_rows) are skipped automatically without asserting source.valid, source.last == 1 asserts strictly on the final valid row (row_idx == num_rows - 1), and num_rows == 0 is guarded so row_idx == num_rows - 1 never underflows 0xFFFF.
