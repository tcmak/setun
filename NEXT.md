# Next Up

- **Current Season:** Season 3 (First FPGA Board Bring-Up & M0 on Silicon — Gate 6)
- **Completed Seasons:** Season 1 (`COMPLETE`) and Season 2 (`COMPLETE — Gate 3 Passed`: Blocks A–D `1.5`–`3.7`, 16 tokens / 561 ops bit-exact under `rand_reset` + Wishbone wait-state jitter).
- **Active Season 3 Tasks (`PLAN.md`):**
  1. **`6.1`:** Candidate DDR FPGA board selection (`ECPIX-5` ECP5-85F vs. `DE10-Nano` Cyclone V).
  2. **`6.2`:** Board target & `make fpga-build` (`TernaryAccel` + LiteDRAM).
  3. **`6.3`:** Fit & >= 100 MHz timing closure with zero negative slack.
  4. **`6.4` & `3.3b`:** `.ttape` DDR weight loader and `--backend board` host transport.
  5. **`6.5` & `6.6`:** 16-token silicon parity, 256-token DDR soak, and real `tok/s`, `MB/s`, and power.
