# AGENTS.md — `setun` Working Protocol & Standing Engineering Rules

`setun` is the hardware and numerical authority for ternary LLM inference on FPGA.

## 1. Portfolio Ownership & Human/Agent Code Boundaries

* **`setun` OWNS:** 2-bit trit lane semantics, rounding/saturation rules (`docs/NUMERICS.md`), Migen RTL (`setun/accelerators/`), LiteX SoC integration & generated register maps (`csr.json`, `csr.h`), Yosys/`nextpnr` synthesis & timing reports (`reports/`), and the single Python golden reference oracle (`setun/ref/`).
* **`setun` DOES NOT OWN:** The `TTAPE` binary container specification (`TernaryTape.jl/spec/`), golden `.ttape` fixture generation (`TernaryTape.jl/fixtures/`), or host language compilers/runtimes (`veriObjC`). Never add host compiler or model-serving frameworks here.
* **Maintainer-Owned RTL Boundary:** RTL under `setun/accelerators/` is written by the maintainer through Season 2. By default, agents write tests, reference code (`setun/ref/`), SoC/CSR glue, scripts, and docs, and review RTL or diagnose bugs when asked—**do not modify or commit files under `setun/accelerators/` unless explicitly instructed in the prompt.**

---

## 2. Session Workflow & Version Control

1. **Orient first:** Read `README.md`, `PLAN.md`, and `NEXT.md` at the start of every session. Work on **one task at a time**, in `PLAN.md` order, unless told otherwise.
2. **Stop on open decisions:** If a task hits an unresolved architectural choice (see `PLAN.md` / `docs/decisions.md` / `docs/open-questions.md`), **stop and ask**—never silently invent or drop a spec behavior.
3. **Atomic commits:** Commit after each satisfied acceptance criterion using the format `<task-id>: <what>`. Enforce LF line endings (`* text=auto eol=lf`).
4. **Close the loop:** End every working session by pruning/updating `NEXT.md` and appending a dated summary line to `docs/log.md`, updating `README.md` or `docs/decisions.md` in the same PR whenever flags or architecture change.

---

## 3. Standing Engineering Rules (Apply to Every Task)

1. **Red first, never weaken tests.**
   * No reference op, loader change, script, or RTL fix lands without its failing test written and observed red first.
   * **Never weaken a test, tolerance, synthesis threshold, or golden vector to get green.** If a test or vector looks wrong, stop and document the discrepancy in `docs/open-questions.md`.

2. **The RTL & Reference Audit Trifecta:**
   * **Bit-exact oracle parity:** Hardware simulation must match `setun/ref/` bit-for-bit across corner cases (including round-half-up and saturation—never floor-shift silently).
   * **Bubble & backpressure resilience:** Every streaming module testbench must exercise randomized mid-row `in_valid` deassertion (1-cycle and multi-cycle bubbles) and backpressure, never just contiguous burst inputs.
   * **Synthesized zero-DSP & real timing:** Default synthesis must use `0` DSPs (unless `--use-dsp` is explicitly tested) and extract measured `fmax_mhz` from real place-and-route (`nextpnr`), never hard-coded clock constants.

3. **Single-source toolchain lock in the same commit.**
   Pin all versions—no floating dependencies or unverified git HEAD installs. `pyproject.toml` (`migen==0.10.0`) and `toolchain.lock` are the sole sources of truth, actively enforced by `scripts/doctor.py` (`make doctor`). Any dependency change must update `scripts/doctor.py` in the same diff.

4. **Goose-style diagnostics.**
   Errors in `setun/ref/format.py`, `scripts/doctor.py`, and testbench assertions must name the module/function, the violated invariant, expected vs. actual values, and the byte offset or simulation cycle. Never emit a bare `AssertionError` or generic `ValueError`.

5. **Composition + positive/negative corpus tests.**
   * Loader and format tests must validate both `accept/` and `reject/` cases from the canonical `TTAPE` fixtures.
   * Pipeline components (`unpacker` → `ternary_dot` → `group_scale` → CSR/SoC) require end-to-end composition tests against `setun/ref/m0.py`, not just isolated unit tests.

6. **Measure resources, claim nothing.**
   Regenerate `reports/resources.json` and `reports/resources.md` deterministically (`scripts/resource_report.py --check`). Any multi-row figure (`R=4`, `R=8`) not directly synthesized and routed MUST be labeled `[EXTRAPOLATED from R=1]`.

---

## 4. Mandatory Pre-Completion Gate

A task is done **only** when `make ci` stays green and within its fast time budget (keep heavy sweeps like the 600k Verilator run in nightly targets), passing:
1. `python scripts/doctor.py`
2. `ruff check . && ruff format --check .`
3. `pytest`
4. `python scripts/resource_report.py --check` (when synthesis/reporting scripts or RTL change)
