Architecture Decision Records (ADRs)

ADR-001: Execution Model
Status: Provisional (Decide at start of Track B)
Decision: Fixed NPU overlay + binary command stream.

ADR-002: Activation Precision and Granularity (Provisional)
Status: Provisional for Seasons 1–3 (Revisited in Track A, Task 1.8)
Decision: int8 signed activations ([-128, 127]) with a per-token scale factor.
Context & Trade-offs: Keeps lane datapath width at A = 8 bits (and A + 1 = 9 bits after negation) so a group partial sum of size G = 128 fits cleanly inside a 16-bit signed integer (⌈log2(128 · 128 + 1)⌉ + 1 = 16 bits). Block-scaled int8/int16 and FP16 passthrough are deferred until real-model perplexity evaluations in Track A.

ADR-003: Internal Weight Format, Bit-Packing Endianness, and VTPE v1 Reconciliation Plan
Status: Accepted for bit-packing; Container migration to VTPE v1 BLOCKED on VeriObjC Task 5.6
Decision:
Encoding: 2-bit trits {00: 0, 01: +1, 10: -1, 11: reserved} with parametric group size G ∈ {64, 128, 256}.
Packing Endianness (LSB-First): Trits are packed little-endian within each byte and across wide bus beats (trit[0] in bits[1:0], trit[1] in bits[3:2], ..., trit[i] in bits[2*i + 1 : 2*i]). This matches VeriObjC VTPE v1 §3 bit-for-bit.
Reserved Code (11) Handling: The software unpacker (setun/ref/) rejects 11 (and non-zero tail padding) with a ValueError. The hardware TernaryDot lane aliases 11 -> 0 in the combinational datapath.
VTPE v1 Alignment (Spec §11 & §12.5): Setun currently uses its interim STNW/STM0 framing. Once VeriObjC Task 5.6 reconciles the authoritative rt/ header layout, Setun will migrate its container framing to VTPE v1 as a dual-profile consumer: supporting both VTPE-Core (flags bit 2 = 0, s16 power-of-two shift scales, DSP=0) and VTPE-Grouped (flags bit 2 = 1, tag 0x0006 group_scales_u16 + tag 0x0007 host-only fp32_aux). Until that loader rewrite and conformance test land, STNW/STM0 and VTPE coexist and are not claimed unified.

ADR-004: Ternary Matmul Micro-Architecture
Status: Accepted at S1.end (2026-11-14); Updated 2026-11-16 for VTPE Profile Boundary
Decision: Multiplier-free add/sub tree in TernaryDot (0 DSPs per ternary MAC lane) paired with GroupAccumulateScale supporting both 1 DSP48E1 per row (default for VTPE-Grouped u16 scales) and 0 DSPs (via Yosys -nodsp LUT multiply, or shift-only mode for VTPE-Core power-of-two tapes).
Evidence: Yosys synth_xilinx (xc7) confirms 2,697 LUT6 (42.14 LUTs/MAC) and 1 DSP48E1 for a canonical L=64, G=128, A=8 TernaryRow. Even small Artix-7 (XC7A35T) fabric fits 6 rows (384 MACs/cycle), exceeding the 64 MACs/cycle external memory bandwidth ceiling at 100 MHz. The DSP=0 claim applies strictly to TernaryDot MAC lanes and VTPE-Core shift-mode execution; VTPE-Grouped execution requires 1 multiply per row per group (1 DSP48E1 or -nodsp LUTs).

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
