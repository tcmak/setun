"""System-sim Gate 3 harness (`make sim-m0`): 16-token sim vs ref with jitter & rand_reset."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

from setun.cli import main as cli_main


def _invoke_cli(argv: list[str]) -> tuple[int, str, str]:
    out_b, err_b = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out_b), contextlib.redirect_stderr(err_b):
        rc = cli_main(argv)
    return rc, out_b.getvalue(), err_b.getvalue()


def run_sim_m0(
    model_path: Path,
    *,
    tokens: int = 16,
    prompt: str = "1",
    seed: int = 1958,
    out_dir: Path = Path("build/sim_m0"),
    sim_backend: str = "sim",
) -> dict[str, object]:
    """Run M0 on `ref` vs `sim` (rand_reset + SDRAM wait jitter) and save trace/cycle logs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_t, sim_t = out_dir / "ref.strc", out_dir / "sim.strc"

    _, ref_out, ref_err = _invoke_cli(
        [
            "run",
            str(model_path),
            "--backend",
            "ref",
            "--tokens",
            str(tokens),
            "--prompt",
            prompt,
            "--trace",
            str(ref_t),
        ]
    )
    (out_dir / "ref_perf.log").write_text(ref_err, encoding="utf-8")
    ref_tokens = [int(t) for t in ref_out.strip().split()]

    _, sim_out, sim_err = _invoke_cli(
        [
            "run",
            str(model_path),
            "--backend",
            sim_backend,
            "--tokens",
            str(tokens),
            "--prompt",
            prompt,
            "--rand-reset-seed",
            str(seed),
            "--ram-jitter-seed",
            str(seed),
            "--trace",
            str(sim_t),
        ]
    )
    (out_dir / "sim_perf.log").write_text(sim_err, encoding="utf-8")
    sim_tokens = [int(t) for t in sim_out.strip().split()]

    rc_diff, diff_out, _ = _invoke_cli(["diff", str(ref_t), str(sim_t)])
    (out_dir / "diff_report.txt").write_text(diff_out, encoding="utf-8")

    matched = (ref_tokens == sim_tokens) and (rc_diff == 0)
    summary: dict[str, object] = {
        "model": str(model_path),
        "seed": seed,
        "tokens": tokens,
        "prompt": prompt,
        "ref_tokens": ref_tokens,
        "sim_tokens": sim_tokens,
        "diff_exit_code": rc_diff,
        "matched": matched,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not matched:
        raise RuntimeError(
            f"[sim-m0] MISMATCH (seed={seed}): {ref_tokens} vs {sim_tokens}\n{diff_out}"
        )

    print(f"[sim-m0] PASS: {len(sim_tokens)} tokens bit-exact (seed={seed}, artifacts={out_dir})")
    return summary


def main(argv: list[str] | None = None) -> int:
    def_seed = int(os.environ.get("TEST_SEED", os.environ.get("RANDOM_SEED", "1958")))
    p = argparse.ArgumentParser(description="Gate 3 M0 system-sim check (`make sim-m0`).")
    p.add_argument("--model", type=Path, default=Path("models/m0_toy.ttape"))
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--prompt", type=str, default="1")
    p.add_argument("--seed", type=int, default=def_seed)
    p.add_argument("--out-dir", type=Path, default=Path("build/sim_m0"))
    a = p.parse_args(argv)
    run_sim_m0(a.model, tokens=a.tokens, prompt=a.prompt, seed=a.seed, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
