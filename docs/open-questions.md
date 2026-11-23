# Open Questions & Cross-Repo Blockers

## OQ-001: Reconciliation of `setun` Weight Framing with Canonical `TTAPE` v1.0
- **Status:** **RESOLVED** (2026-11-22, Tasks B1, B2 & C1.3 — `ADR-014`, frozen `spec-v1.0`).
- **Resolution:** `TernaryTape.jl` (`third_party/TernaryTape`, pinned at frozen tag `spec-v1.0`, commit `3c73f3cc85f0d7a9bf1ccd00b7dc8a100f379920`) establishes the single frozen `TTAPE` v1.0 specification (`spec/ttape-v1.0.md`, `magic = "TTAP"`, `0x50415454` LE) with distinct section tags `0x0006` (`group_scales_u16`, including explicit `group_size` $G \in \{64, 128, 256\}$ and `uint8` `shift`) and `0x0007` (`fp32_aux`). `setun/ref/format.py` and `setun/ref/m0.py` serialize, validate, and execute canonical `.ttape` artifacts across all 21 shared golden fixtures (`5` accept + `16` reject), enforced in `toolchain.lock` and `scripts/doctor.py`.
