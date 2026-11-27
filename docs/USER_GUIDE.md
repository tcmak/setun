# Setun End-User Quickstart & Simulation Guide

This 5-minute walkthrough covers environment verification, running the 2-layer `M0` ternary transformer bit-exact on the LiteX/Verilator `SetunSimSoC` (`--backend sim` vs. `--backend ref`), reading hardware CSR telemetry, and localizing layer mismatches with `setun diff`.

## 1. Setup & Toolchain Check

Activate the pinned Python virtual environment and verify system HDL tools (`verilator`, `yosys`, `nextpnr-ecp5`, and pinned LiteX commits):

```bash
source .venv/bin/activate
pip install -e ".[dev]"
make doctor
```

Ensure the deterministic `M0` toy container (`models/m0_toy.ttape`, `d_model=64, G=64`) exists:

```bash
python3 -m scripts.make_toy --size tiny --out models/m0_toy.ttape
```

## 2. Run 16-Token Greedy Decode (`ref` vs. `sim`)

Run the golden integer reference backend first and record a per-op `STRC` v1 binary trace:

```bash
setun run models/m0_toy.ttape --backend ref --tokens 16 --prompt 1 \
  --trace build/quickstart/ref.strc
```

Now run the exact same `.ttape` container on `SetunSimSoC` in Verilator (`--backend sim`). Linear projections (`q/k/v/o_proj`, `gate/up/down_proj`) execute on the 4-row `TernaryAccel` Wishbone DMA peripheral over `UARTBone`/`Etherbone`, while host NumPy runs `RMSNorm`, `RoPE`, GQA attention, and `SwiGLU`:

```bash
setun run models/m0_toy.ttape --backend sim --tokens 16 --prompt 1 \
  --trace build/quickstart/sim.strc
```

* First run compiles and caches `Vsim` under `sim_build/.litex_sim_cache/` (~60–90 s); subsequent runs reuse the cached binary in a few seconds.
* Generated token IDs stream space-separated to `stdout`; run summary and hardware CSR counters print to `stderr`.
* Pass `--rand-reset-seed 1958 --ram-jitter-seed 1958` (or run `make sim-m0`) to enable Verilator randomized `reset_less` init and 0–3 cycle LFSR Wishbone `main_ram` wait-state jitter.

## 3. Interpreting `setun perf` CSR Telemetry

Example `stderr` output from `--backend sim`:

```text
[setun run] backend=sim model=m0-tiny prompt=[1] tokens=16 forward_steps=17 gemv_calls=238
[setun perf] total_cycles=206568 cycles/tok=12151.1 bus_beats=109888 bus_util=53.20% bytes_moved=439552 MB/s=2.1279 (clk=1.00 MHz)
```

* **`forward_steps` / `gemv_calls`:** `1` prompt token + `16` generated steps = `17` forward passes x `14` ternary projections (`7` per layer x `2` layers) = `238` hardware GEMV launches.
* **`total_cycles` & `cycles/tok`:** Accumulated directly from the `perf_cycles` hardware CSR across all `238` `TernaryAccel` runs (`total_cycles / forward_steps`).
* **`bus_beats` & `bus_util`:** Total 32-bit Wishbone DMA read/write beats (`perf_bus_beats`) divided by `total_cycles`.
* **`bytes_moved` & `MB/s`:** Packed 2-bit trits + `uint16` group scales + `int8` activations + `int32` outputs transferred over DMA (`SimSoC` defaults to `1.00 MHz`; pass `--clk-mhz 100` to project throughput at a target FPGA clock).

## 4. Debugging Layer Mismatches with `setun diff`

Compare the two `STRC` traces (`561` traced ops across `17` steps):

```bash
setun diff build/quickstart/ref.strc build/quickstart/sim.strc
```

* **Exit code `0` (`MATCH`):** `[setun diff] MATCH: 561 traced ops verified identical.`
* **Exit code `1` (`MISMATCH`):** Reports the exact first diverging `step<S>.<op>` ID, tensor shape/dtype, FNV-1a-64 hashes, multi-dimensional `index=(row,)` and flat index, expected vs. actual values, `max_abs_diff`, and surrounding `--context` ops (pass `--atol` / `--rtol` for tolerance sweeps).
