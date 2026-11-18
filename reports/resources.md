# Setun Ternary Datapath Resource & Throughput Report (0-DSP Default / -nodsp) (Task 2.6)

Primary XC7 sweep via Yosys (`synth_xilinx -family xc7 -noiopad -nodsp`).

## 1. Xilinx 7-Series Configuration Sweep (`R × L × G × A`)

| R (rows) | L (lanes) | G | A (bits) | MACs/cycle | LUT6 | FFs | DSP48E1 | LUTs/MAC | DSPs/row |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 64 | 8 | 16 | 1552 | 114 | 0 | 97.000 | 0.00 |
| 1 | 16 | 128 | 8 | 16 | 1528 | 116 | 0 | 95.500 | 0.00 |
| 1 | 64 | 128 | 8 | 64 | 3300 | 114 | 0 | 51.562 | 0.00 |
| 1 | 64 | 256 | 8 | 64 | 3487 | 116 | 0 | 54.484 | 0.00 |
| 1 | 128 | 128 | 8 | 128 | 5656 | 97 | 0 | 44.188 | 0.00 |
| 1 | 64 | 128 | 16 | 64 | 5838 | 122 | 0 | 91.219 | 0.00 |
| 4 | 64 | 128 | 8 | 256 | 13200 | 456 | 0 | 51.562 | 0.00 |
| 8 | 64 | 128 | 8 | 512 | 26400 | 912 | 0 | 51.562 | 0.00 |

## 2. Multi-Vendor Reference Row (`L=64, G=128, A=8`) for `ADR-007`

| Vendor Target | Synthesis Flow | Logic Unit | Logic Cells | FFs | DSPs | Cells/MAC |
|---|---|---|---:|---:|---:|---:|
| `xc7` | `yosys synth_xilinx -family xc7 -nodsp` | LUT6 | 3300 | 114 | 0 | 51.562 |
| `ecp5` | `yosys synth_ecp5 -nodsp` | LUT4 | 4741 | 114 | 0 | 74.078 |
| `cyclonev` | `yosys synth_intel_alm -family cyclonev -nodsp` | ALM/LCELL | 2677 | 114 | 0 | 41.828 |

## 3. Candidate Board Fit & Memory Bandwidth Ceiling (at 100 MHz)

Reference XC7 row (`L=64, G=128, A=8`): **3300 LUT6**, **0 DSP48E1** (51.56 LUTs/MAC).

At 2 bits/trit (0.25 B/weight), a memory interface supplying $B_{mem}$ GB/s feeds at most:

$$\text{Max useful MACs/cycle at } 100\text{ MHz} = \frac{B_{mem} \times 10^9}{0.25 \times 100 \times 10^6} = 40 \times B_{mem}$$

| Candidate Board | Logic Budget | DSP Budget | Usable BW (GB/s) | Max Rows (80% Logic) | Fit MACs/cycle | BW-Matched MACs/cycle |
|---|---:|---:|---:|---:|---:|---:|
| Artix-7 XC7A35T (Arty A7-35) | 20,800 LUT6 | 90 | 1.6 | 5 | 320 | 64 |
| Artix-7 XC7A100T (Arty A7-100) | 63,400 LUT6 | 240 | 3.2 | 15 | 960 | 128 |
| Lattice ECP5-85F (ULX3S) | 83,640 LUT4 | 156 | 1.6 | 14 | 896 | 64 |
| Intel Cyclone V 5CSEMA6 (Arrow DECA) | 41,910 ALM | 112 | 3.2 | 12 | 768 | 128 |

### Key Architectural Takeaways (Gate 2 / G1 & `ADR-004`)
- **Zero DSPs in `TernaryDot`:** All ternary MAC lanes are multiplier-free add/sub trees.
- **Group Scale DSP Mode:** 0 DSPs everywhere (default -nodsp LUT multipliers).
- **Memory-bound regime:** Even on the smallest Artix-7 (`XC7A35T`) in 0-DSP mode, compute fabric fits more MACs/cycle than external DDR3/SPI weight bandwidth can feed at 100 MHz—confirming that memory burst efficiency (`Phase 3.2`) governs throughput.
