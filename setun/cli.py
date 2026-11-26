"""Command-line interface for Setun."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from setun import __version__
from setun.ref.m0 import load_m0_model
from setun.runtime.driver import add_backend_argument
from setun.runtime.m0_runner import run_m0_greedy
from setun.runtime.trace import (
    TraceRecorder,
    diff_traces,
    format_diff_report,
    load_trace,
    save_trace,
)


def get_git_hash() -> str:
    """Return the short git commit hash of the repo, or 'unknown'."""
    repo_root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def get_version_string() -> str:
    return f"setun {__version__} ({get_git_hash()})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setun",
        description="Setun: LiteX-native ternary LLM toolchain",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version_string(),
    )
    subparsers = parser.add_subparsers(dest="command")

    run_p = subparsers.add_parser(
        "run",
        help="Run hybrid M0 greedy decode across ref, sim, or board backends (Task 3.4)",
    )
    run_p.add_argument(
        "model",
        type=Path,
        help="Path to M0 .ttape (TTAPE-extended v1.0) or .stm0 model file",
    )
    add_backend_argument(run_p, default="ref")
    run_p.add_argument(
        "--tokens",
        type=int,
        default=16,
        help="Number of new tokens to generate greedily (default: 16)",
    )
    run_p.add_argument(
        "--prompt",
        type=str,
        default="1",
        help="Comma-separated prompt token IDs (default: '1')",
    )
    run_p.add_argument(
        "--clk-mhz",
        type=float,
        default=None,
        help="Optional clock frequency override in MHz for MB/s calculation",
    )
    run_p.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Optional output path for per-op STRC binary trace (for `setun diff`)",
    )
    run_p.add_argument("--rand-reset-seed", type=int, default=None)
    run_p.add_argument("--ram-jitter-seed", type=int, default=0)

    diff_p = subparsers.add_parser(
        "diff",
        help="Compare two STRC execution traces and localize first divergence (Task 3.5)",
    )
    diff_p.add_argument("trace_a", type=Path, help="Reference STRC trace file")
    diff_p.add_argument("trace_b", type=Path, help="Candidate STRC trace file")
    diff_p.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance (default: 0.0 bit-exact)",
    )
    diff_p.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance (default: 0.0 bit-exact)",
    )
    diff_p.add_argument(
        "--context",
        type=int,
        default=3,
        help="Number of surrounding ops to display on mismatch (default: 3)",
    )
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    if args.tokens < 1:
        raise ValueError(f"[setun run] --tokens must be >= 1, got {args.tokens}.")

    prompt_parts = [p.strip() for p in args.prompt.split(",") if p.strip()]
    if not prompt_parts:
        raise ValueError("[setun run] --prompt must contain at least one integer token ID.")
    prompt_ids = [int(p) for p in prompt_parts]

    if args.clk_mhz is not None and args.clk_mhz <= 0.0:
        raise ValueError(f"[setun run] --clk-mhz must be > 0, got {args.clk_mhz}.")

    clk_hz = (args.clk_mhz * 1e6) if args.clk_mhz is not None else None
    model = load_m0_model(args.model)

    recorder = (
        TraceRecorder(
            metadata={
                "backend": args.backend,
                "model": model.config.name,
                "prompt": prompt_ids,
                "tokens": args.tokens,
            }
        )
        if args.trace is not None
        else None
    )

    emitted = 0

    def _emit_token(tok: int) -> None:
        nonlocal emitted
        prefix = " " if emitted > 0 else ""
        sys.stdout.write(f"{prefix}{tok}")
        sys.stdout.flush()
        emitted += 1

    _, _, _, summary = run_m0_greedy(
        model,
        backend=args.backend,
        prompt_tokens=prompt_ids,
        max_new_tokens=args.tokens,
        clk_freq_hz=clk_hz,
        rand_reset_seed=args.rand_reset_seed,
        ram_wait_jitter_seed=args.ram_jitter_seed,
        on_token=_emit_token,
        trace_hook=recorder,
    )
    print(flush=True)

    if args.trace is not None and recorder is not None:
        save_trace(args.trace, recorder.entries, metadata=recorder.metadata)

    clk_mhz = summary.backend_clk_hz / 1e6
    print(
        f"[setun run] backend={args.backend} model={model.config.name} "
        f"prompt={list(summary.prompt_tokens)} tokens={len(summary.generated_tokens)} "
        f"forward_steps={summary.forward_steps} gemv_calls={summary.gemv_calls}",
        file=sys.stderr,
    )
    print(
        f"[setun perf] total_cycles={summary.total_cycles} "
        f"cycles/tok={summary.cycles_per_token:.1f} "
        f"bus_beats={summary.total_bus_beats} "
        f"bus_util={summary.bus_utilization * 100.0:.2f}% "
        f"bytes_moved={summary.total_bytes_moved} "
        f"MB/s={summary.mb_per_sec:.4f} (clk={clk_mhz:.2f} MHz)",
        file=sys.stderr,
    )
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    if args.atol < 0.0:
        raise ValueError(f"[setun diff] --atol must be >= 0, got {args.atol}.")
    if args.rtol < 0.0:
        raise ValueError(f"[setun diff] --rtol must be >= 0, got {args.rtol}.")
    if args.context < 0:
        raise ValueError(f"[setun diff] --context must be >= 0, got {args.context}.")

    entries_a, _ = load_trace(args.trace_a)
    entries_b, _ = load_trace(args.trace_b)
    res = diff_traces(
        entries_a,
        entries_b,
        atol=args.atol,
        rtol=args.rtol,
        context=args.context,
    )
    print(format_diff_report(res))
    return 0 if res.matched else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "diff":
        return _cmd_diff(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
