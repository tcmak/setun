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
Status: Accepted at S1.end (2026-11-14); Amended 2026-11-18 (0-DSP default, opt-in 1 DSP/row)
Decision: Multiplier-free add/sub tree in TernaryDot (0 DSPs per ternary MAC lane) paired with GroupAccumulateScale defaulting to 0 DSPs (Yosys -nodsp LUT multiplier for VTPE-Grouped u16 scales, or shift-only mode for VTPE-Core power-of-two tapes) and supporting opt-in 1 DSP48E1 per row (--use-dsp) when hard DSP slices are abundant.
Evidence: Yosys synth_xilinx (xc7) on a canonical L=64, G=128, A=8 TernaryRow measures 3,300 LUT6 / 0 DSP48E1 (51.56 LUTs/MAC) in default -nodsp mode and 2,697 LUT6 / 1 DSP48E1 (42.14 LUTs/MAC) in --use-dsp mode. Even on small Artix-7 (XC7A35T), pure LUT fabric fits 5 rows (320 MACs/cycle, 0 DSPs), still exceeding the 64 MACs/cycle external memory bandwidth ceiling at 100 MHz.

ADR-005: Control Plane
Status: Provisional (Decide in Track B)
Decision: Host-driven in Seasons 1–3; command sequencer in Track B.

ADR-006: First Real Model and Reference Runtime
Status: Provisional (Decide at start of Track A)
Decision: Ternary Bonsai 1.7B + PrismML llama.cpp fork (fallback: BitNet b1.58 2B4T).

ADR-007: FPGA Board Selection
Status: Open (Decide at S3.0)

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

ADR-012: Host Toolchain & Environment Pinning (Task 0.2a)
Status: Accepted for Season 1
Decision: Native Ubuntu/Debian (WSL2) system packages via apt (verilator, yosys, gtkwave) combined with a local Python virtual environment (.venv) pinned via toolchain.lock and verified by make doctor. Containerization (Dockerfile) is deferred to Task 0.2b.

ADR-013: Cross-Portfolio Mythos-Loop Alignment & Season 3 HIL Parity Runner
Status: Accepted (2026-11-18)
Decision: Align Setun's synthesis, real-model evaluation, and board bring-up gates with the cross-repo VeriObjC / TernaryTape / veriStream-mythos loop:
1. Task 2.6 characterizes XC7, Lattice ECP5, and Intel Cyclone V with 0-DSP (-nodsp) as the default baseline so ADR-007 board selection uses measured multi-vendor data.
2. Season 4 Track A enforces a real-checkpoint PTQ quality gate (perplexity / probe accuracy on <= 1B weights) prior to the M1 demo, splitting hardware vs. QAT research tracks if generic PTQ collapses.
3. Season 3 board bring-up (Etherbone / litex_server) doubles as the portfolio Hardware-in-the-Loop (HIL) parity runner against shared TTAPE golden fixtures.

ADR-014: Canonical TTAPE v1 Container Adoption (TTAPE-core & TTAPE-grouped)
Status: Accepted (2026-11-20, Task B2; promoted to frozen spec-v1.0 in Task C1.3)
Decision:
1. Single Spec & Fixture Source of Truth: Setun vendors TernaryTape.jl (via HTTPS submodule at third_party/TernaryTape, pinned at spec-v1.0) and removes duplicated wire-format tables from docs/FORMAT.md.
2. Supported Profiles: Setun's default loader gate (reader_profile="grouped") supports both TTAPE-core (flags bit 2 = 0, per-layer s16 power-of-two shift scales in 0x0002 executed in 0-DSP shift mode) and TTAPE-grouped (flags bit 2 = 1, section 0x0006 group_scales_u16 carrying explicit group_size G in {64, 128, 256}, shared uint8 shift in [0, 30], and row-major uint16 scale mantissas).
3. Fail-Closed Extended Gate: Tapes carrying host-only section 0x0007 (fp32_aux) or host layer kinds {6, 7} (TTAPE-extended) are rejected fail-closed with TTAPE_ERR_UNSUPPORTED_PROFILE under Setun's hardware-targeted "grouped" profile, and accepted only when host callers explicitly request reader_profile="extended".
