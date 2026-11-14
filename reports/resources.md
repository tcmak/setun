# Setun Ternary Datapath Resource & Throughput Report (Task 2.6)

Synthesized via Yosys (`synth_xilinx -family xc7 -noiopad`).

## 1. Configuration Sweep (`R × L × G × A`)

| R (rows) | L (lanes) | G | A (bits) | MACs/cycle | LUT6 | FFs | DSP48E1 | LUTs/MAC | DSPs/row |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 64 | 8 | 16 | 645 | 66 | 1 | 40.312 | 1.00 |
| 1 | 16 | 128 | 8 | 16 | 649 | 68 | 1 | 40.562 | 1.00 |
| 1 | 64 | 128 | 8 | 64 | 2697 | 66 | 1 | 42.141 | 1.00 |
| 1 | 64 | 256 | 8 | 64 | 2704 | 68 | 1 | 42.250 | 1.00 |
| 1 | 128 | 128 | 8 | 128 | 5010 | 49 | 1 | 39.141 | 1.00 |
| 1 | 64 | 128 | 16 | 64 | 4873 | 74 | 1 | 76.141 | 1.00 |
| 4 | 64 | 128 | 8 | 256 | 10788 | 264 | 4 | 42.141 | 1.00 |
| 8 | 64 | 128 | 8 | 512 | 21576 | 528 | 8 | 42.141 | 1.00 |

## 2. Candidate Board Fit & Memory Bandwidth Ceiling (at 100 MHz)

Reference row (`L=64, G=128, A=8`): **2697 LUT6**, **1 DSP48E1** (42.14 LUTs/MAC).

At 2 bits/trit (0.25 B/weight), a memory interface supplying $B_{mem}$ GB/s feeds at most:

$$\text{Max useful MACs/cycle at } 100\text{ MHz} = \frac{B_{mem} \times 10^9}{0.25 \times 100 \times 10^6} = 40 \times B_{mem}$$

| Candidate Board | LUT6 Budget | DSP Budget | Usable BW (GB/s) | Max Rows (80% LUT budget) | Fit MACs/cycle | BW-Matched MACs/cycle |
|---|---:|---:|---:|---:|---:|---:|
| Artix-7 XC7A35T (Arty A7-35) | 20,800 | 90 | 1.6 | 6 | 384 | 64 |
| Artix-7 XC7A100T (Arty A7-100) | 63,400 | 240 | 3.2 | 18 | 1,152 | 128 |
| Zynq-7020 XC7Z020 (PYNQ-Z2) | 53,200 | 220 | 4.2 | 15 | 960 | 168 |
| Zynq US+ XCK26 (Kria KV260) | 117,120 | 1248 | 9.6 | 34 | 2,176 | 384 |

### Key Architectural Takeaways (Gate 2 / G1)
- **Zero DSPs in `TernaryDot`:** All multipliers live strictly in the per-group scale stage (`1 DSP48E1` per row).
- **Memory-bound regime:** Even on the smallest Artix-7 (`XC7A35T`), compute fabric fits more MACs/cycle than external DDR3/SPI weight bandwidth can feed at 100 MHz—confirming that memory burst efficiency (`Phase 3.2`) governs token throughput.
