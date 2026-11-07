Architecture Decision Records (ADRs)
ADR-001: Execution Model
Status: Provisional (Decide at start of Track B)
Decision: Fixed NPU overlay + binary command stream.
ADR-002: Activation Precision and Granularity (Provisional)
Status: Provisional for Seasons 1–3 (Revisited in Track A, Task 1.8)
Decision: int8 signed activations ([-128, 127]) with a per-token scale factor.
Context & Trade-offs: Keeps lane datapath width at A = 8 bits (and A + 1 = 9 bits after negation) so a group partial sum of size G = 128 fits cleanly inside a 16-bit signed integer (⌈log2(128 · 128 + 1)⌉ + 1 = 16 bits). Block-scaled int8/int16 and FP16 passthrough are deferred until real-model perplexity evaluations in Track A.
ADR-003: Internal Weight Format, Bit-Packing Endianness, and Reserved Code
Status: Accepted for internal format (M1 group size G open until Track A)
Decision:
Encoding: 2-bit trits {00: 0, 01: +1, 10: -1, 11: reserved} with parametric group size G ∈ {64, 128, 256}.
Packing Endianness (LSB-First): Trits are packed little-endian within each byte and across wide bus beats (trit[0] in bits[1:0], trit[1] in bits[3:2], ..., trit[i] in bits[2*i + 1 : 2*i]).
Reserved Code (11) Handling: The software unpacker (setun/ref/) rejects 11 with a ValueError to catch corrupt weight images early. The hardware TernaryDot lane aliases 11 -> 0 in the combinational datapath.
ADR-004: Ternary Matmul Micro-Architecture
Status: Provisional (Decide at S1.end)
Decision: Multiplier-free add/sub tree (0 DSP in TernaryDot).
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
Status: Provisional for Season 1 (Reviewed at S1.end)
Decision: Write RTL cores under setun/accelerators/ in Migen.
ADR-010: License and Project Name
Status: Accepted (2026-10-17)
Decision: BSD-2-Clause license; package and CLI named setun.
ADR-011: Hardware Group-Scale Representation
Status: Accepted for Season 1 (Task 1.4a / 2.2b)
Decision: Pre-converted 16-bit unsigned fixed-point mantissa (uint16) per group, paired with a shared per-row (or per-tensor) right-shift exponent (shift).
ADR-012: Host Toolchain & Environment Pinning (Task 0.2a)
Status: Accepted for Season 1
Decision: Native Ubuntu/Debian (WSL2) system packages via apt (verilator, yosys, gtkwave) combined with a local Python virtual environment (.venv) pinned via toolchain.lock and verified by make doctor. Containerization (Dockerfile) is deferred to Task 0.2b.
