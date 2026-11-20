# Open Questions & Cross-Repo Blockers

## OQ-001: Reconciliation of `setun` Weight Framing with Canonical `TTAPE` v1
- **Status:** **RESOLVED** (2026-11-20, Tasks B1 & B2 — `ADR-014`).
- **Resolution:** `TernaryTape.jl` (`third_party/TernaryTape`, pinned at `spec-v0.1-draft`) establishes the single `TTAPE` v1 draft specification (`magic = "TTAP"`, `0x50415454` LE) with distinct section tags `0x0006` (`group_scales_u16`, including explicit `group_size` $G \in \{64, 128, 256\}$ and `uint8` `shift`) and `0x0007` (`fp32_aux`). `setun/ref/format.py` now serializes and validates canonical `.ttape` artifacts across all 20 shared golden fixtures.
