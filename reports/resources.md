# Setun Ternary Datapath Resource & Throughput Report (0-DSP Default / -nodsp) (Task 2.6 / C1.1b)

Primary XC7 sweep via Yosys (`synth_xilinx -family xc7 -noiopad -nodsp`), paired with real `nextpnr-ecp5` place-and-route timing on Lattice ECP5-85F.

## 1. Xilinx 7-Series Configuration Sweep (`R × L × G × A`)

| R (rows) | L (lanes) | G | A (bits) | MACs/cycle | LUT6 | FFs | DSP48E1 | LUTs/MAC | DSPs/row | Basis |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 16 | 64 | 8 | 16 | 1552 | 114 | 0 | 97.000 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 16 | 128 | 8 | 16 | 1528 | 116 | 0 | 95.500 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 64 | 128 | 8 | 64 | 3300 | 114 | 0 | 51.562 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 64 | 256 | 8 | 64 | 3487 | 116 | 0 | 54.484 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 128 | 128 | 8 | 128 | 5656 | 97 | 0 | 44.188 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 64 | 128 | 16 | 64 | 5838 | 122 | 0 | 91.219 | 0.00 | SYNTHESIZED (TernaryRow) |
| 4 | 64 | 128 | 8 | 256 | 13745 | 699 | 0 | 53.691 | 0.00 | SYNTHESIZED (TernaryTile R=4) |
| 8 | 64 | 128 | 8 | 512 | 27316 | 1342 | 0 | 53.352 | 0.00 | SYNTHESIZED (TernaryTile R=8) |

## 2. Multi-Vendor Reference Row (`L=64, G=128, A=8`) for `ADR-007`

| Vendor Target | Synthesis / PnR Flow | Logic Unit | Logic Cells | FFs | DSPs | Cells/MAC | Measured Fmax |
|---|---|---|---:|---:|---:|---:|---:|
| `xc7` | `yosys synth_xilinx -family xc7 -nodsp` | LUT6 | 3300 | 114 | 0 | 51.562 | synth-only |
| `ecp5` | `yosys synth_ecp5 -nodsp + nextpnr-ecp5 (--85k -6)` | LUT4 | 4741 | 114 | 0 | 74.078 | 62.08 MHz |
| `cyclonev` | `yosys synth_intel_alm -family cyclonev -nodsp` | ALM/LCELL | 2677 | 114 | 0 | 41.828 | synth-only |

## 3. Candidate Board Fit & Memory Bandwidth Ceiling (at 62.08 MHz Measured ECP5 Fmax)

Reference XC7 row (`L=64, G=128, A=8`): **3300 LUT6**, **0 DSP48E1** (51.56 LUTs/MAC). Measured out-of-context `nextpnr-ecp5` (`LFE5U-85F-6BG381C`) clock frequency: **62.08 MHz**.

At 2 bits/trit (0.25 B/weight), a memory interface supplying $B_{mem}$ GB/s feeds at most:

$$\text{Max useful MACs/cycle at }62.08\text{ MHz} = \frac{B_{mem} \times 10^9}{0.25 \times 62.08 \times 10^6} \approx 64.43 \times B_{mem}$$

| Candidate Board | Logic Budget | DSP Budget | Usable BW (GB/s) | Max Rows (80% Logic, `[EXTRAPOLATED from R=1]`) | Fit MACs/cycle | BW-Matched MACs/cycle |
|---|---:|---:|---:|---:|---:|---:|
| Artix-7 XC7A35T (Arty A7-35) | 20,800 LUT6 | 90 | 1.6 | 5 | 320 | 103 |
| Artix-7 XC7A100T (Arty A7-100) | 63,400 LUT6 | 240 | 3.2 | 15 | 960 | 206 |
| Lattice ECP5-85F (ULX3S) | 83,640 LUT4 | 156 | 1.6 | 14 | 896 | 103 |
| Intel Cyclone V 5CSEMA6 (Arrow DECA) | 41,910 ALM | 112 | 3.2 | 12 | 768 | 206 |

### Key Architectural Takeaways (Gate 2 / G1 & `ADR-004`)
- **Zero DSPs in `TernaryDot`:** All ternary MAC lanes are multiplier-free add/sub trees.
- **Group Scale DSP Mode:** 0 DSPs everywhere (default -nodsp LUT multipliers).
- **Measured Single-Cycle Row Timing (`nextpnr-ecp5`):** Out-of-context place-and-route on `LFE5U-85F-6BG381C` measures **62.08 MHz** across the unpipelined `group_acc + dot_in -> scale_u16 multiply -> row_acc` critical path in `GroupAccumulateScale` (pipelining for >= 100 MHz board closure scheduled in Season 3 Task `6.3`). Multi-row (`R=4`, `R=8`) XC7 figures are directly synthesized from `TernaryTile` (`Task 2.4`, including shared activation LUTRAM/BRAM and serialized output mux); board capacity counts remain `[EXTRAPOLATED from R=1]` per-vendor row budgets.
- **Memory-bound regime:** Even on the smallest Artix-7 (`XC7A35T`) in 0-DSP mode, compute fabric fits more MACs/cycle than external DDR3/SPI weight bandwidth can feed at 62.08 MHz—confirming that memory burst efficiency (`Phase 3.2`) governs throughput.
