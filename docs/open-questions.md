# Open Questions & Cross-Repo Blockers

## OQ-001: Reconciliation of `setun` Weight Framing with VeriObjC `VTPE v1` (Spec §12.2 & §12.5)
- **Status:** **BLOCKED** on `VeriObjC` TDD Plan Task 5.6 (`rt/` header field order, size, and tag numbering reconciliation).
- **Owner:** `setun` coding agent (blocked on `VeriObjC` coding agent).
- **Context:** `setun` currently uses `STNW` (per-tensor header) and `STM0` (JSON-indexed M0 bundle) in `setun/ref/format.py` and `setun/ref/m0.py`. To avoid maintaining two incompatible ternary tape formats across the portfolio, `setun` will migrate to `VTPE v1` as a dual-profile consumer (`VTPE-Core` with `flags` bit 2 = 0 for shift-only `DSP=0` tapes, and `VTPE-Grouped` with `flags` bit 2 = 1 for `0x0006 group_scales_u16` and `0x0007 fp32_aux` hybrid decode tensors).
- **Guardrails until unblocked:**
  1. Preparation (concept mapping in `docs/FORMAT.md` §6 and dual-consumer design sketch) is complete.
  2. **Spec gap to close during Task 5.6 reconciliation (`0x0006` shared `shift`):** `setun`'s grouped scale is `u16_mantissa * 2^-shift` (one shared `uint8` right-shift exponent per tensor alongside `G ∈ {64, 128, 256}`), whereas the `VTPE v1` proposal's `0x0006` (`group_scales_u16`) description currently specifies `G` and the per-group `u16` mantissas without naming the field for the shared `shift` exponent. Task 5.6 must define the exact byte offset of `shift` in the `0x0006` section header.
  3. **Normative `0b11` contract:** Loader rejection of `0b11` (`VTPE` §3) is the contract; `setun` RTL aliasing `0b11 -> 0` is strictly defense-in-depth against uninitialized hardware beats.
  4. Do **not** rewrite `setun/ref/format.py` until `VeriObjC` Task 5.6 publishes the reconciled `rt/` byte layout.
  5. Do **not** claim `STNW`/`STM0` and `VTPE` are unified until `setun` passes the `VTPE-Core` and `VTPE-Grouped` conformance and exhaustive byte-mutation rejection tests.
