# Setun Ternary Datapath Resource & Throughput Report (0-DSP Default / -nodsp) (ADR-007)

Primary non-Xilinx multi-vendor sweeps via Yosys (`synth_ecp5 -nodsp` + `nextpnr-ecp5 --85k -6` on Lattice ECP5-85F, and `synth_intel_alm -family cyclonev -nodsp` on Intel Cyclone V).

## 1a. Lattice ECP5 Configuration Sweep (`R × L × G × A`, `synth_ecp5`)

| R (rows) | L (lanes) | G | A (bits) | MACs/cycle | LUT4 | FFs | MULT18X18D | LUT4/MAC | DSPs/row | Basis |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 16 | 64 | 8 | 16 | 2548 | 114 | 0 | 159.250 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 16 | 128 | 8 | 16 | 2953 | 116 | 0 | 184.562 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 64 | 128 | 8 | 64 | 4741 | 114 | 0 | 74.078 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 64 | 256 | 8 | 64 | 5049 | 116 | 0 | 78.891 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 128 | 128 | 8 | 128 | 8488 | 97 | 0 | 66.312 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 64 | 128 | 16 | 64 | 8597 | 122 | 0 | 134.328 | 0.00 | SYNTHESIZED (TernaryRow) |
| 4 | 64 | 128 | 8 | 256 | 25130 | 699 | 0 | 98.164 | 0.00 | SYNTHESIZED (TernaryTile R=4) |
| 8 | 64 | 128 | 8 | 512 | 48887 | 1342 | 0 | 95.482 | 0.00 | SYNTHESIZED (TernaryTile R=8) |

## 1b. Intel Cyclone V Configuration Sweep (`R × L × G × A`, `synth_intel_alm`)

| R (rows) | L (lanes) | G | A (bits) | MACs/cycle | ALUT | FFs | MUL18X18 | ALUT/MAC | DSPs/row | Basis |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 16 | 64 | 8 | 16 | 1283 | 114 | 0 | 80.188 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 16 | 128 | 8 | 16 | 1321 | 116 | 0 | 82.562 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 64 | 128 | 8 | 64 | 2677 | 114 | 0 | 41.828 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 64 | 256 | 8 | 64 | 2923 | 116 | 0 | 45.672 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 128 | 128 | 8 | 128 | 4758 | 97 | 0 | 37.172 | 0.00 | SYNTHESIZED (TernaryRow) |
| 1 | 64 | 128 | 16 | 64 | 4806 | 122 | 0 | 75.094 | 0.00 | SYNTHESIZED (TernaryRow) |
| 4 | 64 | 128 | 8 | 256 | 12999 | 705 | 0 | 50.777 | 0.00 | SYNTHESIZED (TernaryTile R=4) |
| 8 | 64 | 128 | 8 | 512 | 23590 | 1351 | 0 | 46.074 | 0.00 | SYNTHESIZED (TernaryTile R=8) |

## 2. Multi-Vendor Reference Row (`L=64, G=128, A=8`) for `ADR-007`

| Vendor Target | Synthesis / PnR Flow | Logic Unit | Logic Cells | FFs | DSPs | Cells/MAC | Measured Fmax |
|---|---|---|---:|---:|---:|---:|---:|
| `ecp5` | `yosys synth_ecp5 -nodsp + nextpnr-ecp5 (--85k -6)` | LUT4 | 4741 | 114 | 0 | 74.078 | 62.08 MHz |
| `cyclonev` | `yosys synth_intel_alm -family cyclonev -nodsp` | ALUT | 2677 | 114 | 0 | 41.828 | synth-only |

## 3. Candidate Board Fit & Memory Bandwidth Ceiling (at 20.99 MHz Governing ECP5 Fmax — `TernaryAccel full top`)

Reference ECP5 row (`L=64, G=128, A=8`): **4741 LUT4**, **0 MULT18X18D** (74.08 LUT4/MAC). System throughput on Lattice ECP5-85F (`LFE5U-85F-6BG381C`) is strictly governed by **20.99 MHz** (`TernaryAccel full top`, `min` across measured pipeline stages: isolated `TernaryRow` = 62.08 MHz, `Requantize` = 49.08 MHz, full `TernaryAccel` = 20.99 MHz). ECP5 `nextpnr-ecp5` frequencies are **never** extrapolated to `synth-only` Intel Cyclone V targets.

At 2 bits/trit (0.25 B/weight), an ECP5 memory interface supplying $B_{mem}$ GB/s feeds at most:

$$\text{Max useful MACs/cycle at }20.99\text{ MHz} = \frac{B_{mem} \times 10^9}{0.25 \times 20.99 \times 10^6} \approx 190.57 \times B_{mem}$$

| Candidate Board | Logic Budget | DSP Budget | Usable BW (GB/s) | Max Rows (80% Logic, `[EXTRAPOLATED from R=1]`) | Fit MACs/cycle | BW-Matched MACs/cycle |
|---|---:|---:|---:|---:|---:|---:|
| LambdaConcept ECPIX-5 (Lattice ECP5-85F, DDR3) | 83,640 LUT4 | 156 | 1.6 | 14 | 896 | 305 |
| Radiona ULX3S (Lattice ECP5-85F, SDRAM) | 83,640 LUT4 | 156 | 0.3 | 14 | 896 | 57 |
| Terasic DE10-Nano (Intel Cyclone V 5CSEBA6, HPS DDR3) | 83,820 ALUT | 112 | 3.2 | 25 | 1,600 | N/A (synth-only, pending Quartus STA) |
| Arrow DECA (Intel Cyclone V 5CSEMA6, DDR3) | 83,820 ALUT | 112 | 1.6 | 25 | 1,600 | N/A (synth-only, pending Quartus STA) |

### Key Architectural Takeaways (Gate 2 / G1, `ADR-004`, & `ADR-007`)
- **Zero DSPs in `TernaryDot`:** All ternary MAC lanes are multiplier-free add/sub trees.
- **Group Scale DSP Mode:** 0 DSPs everywhere (default -nodsp LUT multipliers).
- **Governing End-to-End ECP5 Timing (`nextpnr-ecp5`):** System throughput is governed by **20.99 MHz** (`TernaryAccel full top`; isolated `TernaryRow` = 62.08 MHz, `Requantize` = 49.08 MHz, full `TernaryAccel` = 20.99 MHz), never by isolated `TernaryRow` alone when downstream `Requantize` or full `TernaryAccel` measures lower (pipelining for >= 100 MHz board closure scheduled in Season 3 Task `6.3`). Multi-row (`R=4`, `R=8`) figures in Section 1 are directly synthesized from `TernaryTile`, while Section 3 board row ceilings remain `[EXTRAPOLATED from R=1]`.
- **No Cross-Family Timing Extrapolation:** Intel Cyclone V (`DE10-Nano` / `DECA`) rows are synthesized via `synth_intel_alm` without open-source STA, so bandwidth-matched MAC targets remain `N/A (synth-only, pending Quartus STA)` until Quartus TimeQuest closure in Season 3.

## 4. Output Requantization Stage (`Requantize` 48-bit -> `int8`, Task 2.5)

Shared 1-row/cycle 2-stage pipelined requantizer (`II = 1`, 1 instance per `TernaryTile` behind the serialized output mux):

| Width (`in -> out`) | ECP5 LUT4 | ECP5 FFs | ECP5 MULT18X18D | Measured ECP5 Fmax | Cyclone V ALUT | Cyclone V FFs | Cyclone V MUL18X18 | Cyclone V Fmax |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `48b -> 8b` | 6017 | 117 | 0 | 49.08 MHz | 7147 | 117 | 0 | synth-only |

## 5. Multi-Row SoC, Sync BRAM Inference & Full-Top Timing (`TernaryAccel`)

Full `TernaryAccel(R=4, L=64, G=128, max_k=2048, max_rows=2048)` (`WeightUnpacker` + `(R-1)` sync-BRAM stripe banks + `TernaryTile` + `Requantize` + Wishbone burst DMA + sync scratchpad BRAMs):

| Geometry | ECP5 LUT4 | ECP5 FFs | ECP5 BRAM (`DP16KD`) | ECP5 MULT18X18D | Measured ECP5 Fmax | Cyclone V ALUT | Cyclone V FFs | Cyclone V BRAM (`M10K`) | Cyclone V MUL18X18 | Cyclone V Fmax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `R=4, L=64, K=2048` | 40072 | 2938 | 19 | 0 | 20.99 MHz | 26183 | 2616 | 31 | 0 | synth-only |
