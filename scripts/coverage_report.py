"""Zero-dependency Python line coverage reporter across setun, targets, and scripts."""

from __future__ import annotations

import argparse
import ast
import shlex
import sys
import types
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("setun", "targets", "scripts")

FAST_PYTEST_ARGS = [
    "-q",
    "tests/unit/test_version.py",
    "tests/unit/test_doctor.py",
    "tests/unit/test_format.py",
    "tests/unit/test_ops_ref.py",
    "tests/unit/test_m0_golden.py",
    "tests/unit/test_coverage_report.py",
    "tests/unit/test_tile.py",
    "tests/unit/test_requantize.py",
    "-k",
    "not m1_2048 and not m0_shapes and not short_k_eq_l and not rtl_corners and not end_to_end",
]


def discover_repo_python_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Return sorted resolved Paths for all .py files in setun/, targets/, and scripts/."""
    found: list[Path] = []
    for sub in SOURCE_DIRS:
        base = repo_root / sub
        if base.exists():
            found.extend(sorted(p.resolve() for p in base.rglob("*.py")))
    return found


def _excluded_ast_lines(tree: ast.AST) -> set[int]:
    """Collect line numbers inside `if __name__ == '__main__':`, `TYPE_CHECKING`, and docstrings."""
    excluded: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                doc_node = node.body[0]
                end_ln = doc_node.end_lineno or doc_node.lineno
                excluded.update(range(doc_node.lineno, end_ln + 1))

        if isinstance(node, ast.If):
            test = node.test
            is_main = (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            )
            is_tc = isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
            if is_main or is_tc:
                end_ln = node.end_lineno or node.lineno
                excluded.update(range(node.lineno, end_ln + 1))

    return excluded


def _walk_code_lines(code_obj: types.CodeType, out: set[int]) -> None:
    for _start, _end, lineno in code_obj.co_lines():
        if lineno is not None and lineno > 0:
            out.add(lineno)
    for const in code_obj.co_consts:
        if isinstance(const, types.CodeType):
            _walk_code_lines(const, out)


def extract_executable_lines(path: Path) -> set[int]:
    """Return exact bytecode-backed executable statement line numbers for `path`."""
    resolved = path.resolve()
    source = resolved.read_text(encoding="utf-8")
    if not source.strip():
        return set()

    tree = ast.parse(source, filename=str(resolved))
    excluded = _excluded_ast_lines(tree)
    code_obj = compile(source, str(resolved), "exec")
    lines: set[int] = set()
    _walk_code_lines(code_obj, lines)
    return lines - excluded


def format_line_ranges(lines: Iterable[int]) -> str:
    """Format a set of line numbers into compact ranges (e.g. '1-3, 7, 9-10')."""
    nums = sorted(set(lines))
    if not nums:
        return "-"
    ranges: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = n
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(ranges)


def run_with_line_tracer(
    target_files: Iterable[Path],
    workload: Callable[[], int],
) -> tuple[int, dict[Path, set[int]]]:
    """Execute `workload()` under a scoped `sys.settrace` hook and record hit line numbers."""
    hits: dict[Path, set[int]] = {p.resolve(): set() for p in target_files}
    by_str: dict[str, set[int]] = {str(p): s for p, s in hits.items()}

    prev_trace = sys.gettrace()

    def _tracer(frame: types.FrameType, event: str, arg: object):
        if prev_trace is not None:
            prev_trace(frame, event, arg)
        bucket = by_str.get(frame.f_code.co_filename)
        if bucket is None:
            return _tracer if prev_trace is not None else None
        if event == "line":
            bucket.add(frame.f_lineno)
        return _tracer

    sys.settrace(_tracer)
    try:
        rc = workload()
    finally:
        sys.settrace(prev_trace)
        f: types.FrameType | None = sys._getframe()
        while f is not None:
            f.f_trace = prev_trace
            f = f.f_back

    return rc, hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zero-dependency Python line coverage report for Setun",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast pure-Python unit tests (~4s, skipping Verilator/Yosys subprocesses)",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Display uncovered line number ranges per file",
    )
    parser.add_argument(
        "--pytest-args",
        type=str,
        default="",
        help="Custom pytest CLI arguments string (overrides --fast / default tests/unit)",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=0.0,
        help="Fail if total Python line coverage percentage is below this threshold",
    )
    args = parser.parse_args(argv)

    if args.pytest_args.strip():
        pytest_args = shlex.split(args.pytest_args)
    elif args.fast:
        pytest_args = list(FAST_PYTEST_ARGS)
    else:
        pytest_args = ["-q", "tests/unit"]

    files = discover_repo_python_files(REPO_ROOT)
    rc, hits = run_with_line_tracer(files, lambda: int(pytest.main(pytest_args)))
    if rc != 0:
        raise RuntimeError(
            f"[scripts/coverage_report.py:main] pytest exited with non-zero code {rc} "
            f"for args={pytest_args!r}."
        )

    mode_label = "FAST Python Suite" if args.fast else "Full Unit Suite"
    print(f"\n=== Setun Python Line Coverage ({mode_label}) ===")
    hdr = f"  {'File':<36} {'Stmts':>6} {'Hit':>6} {'Miss':>6} {'Cover':>7}"
    if args.show_missing:
        hdr += "  Missing"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    total_stmts = 0
    total_hits = 0
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        exec_lines = extract_executable_lines(path)
        if not exec_lines:
            continue
        hit_lines = hits[path] & exec_lines
        missed_lines = exec_lines - hit_lines
        n_stmts = len(exec_lines)
        n_hit = len(hit_lines)
        n_miss = len(missed_lines)
        pct = (100.0 * n_hit / n_stmts) if n_stmts else 100.0

        total_stmts += n_stmts
        total_hits += n_hit

        row = f"  {rel:<36} {n_stmts:>6} {n_hit:>6} {n_miss:>6} {pct:>6.1f}%"
        if args.show_missing:
            miss_str = format_line_ranges(missed_lines)
            if len(miss_str) > 44:
                miss_str = miss_str[:41] + "..."
            row += f"  {miss_str}"
        print(row)

    total_pct = (100.0 * total_hits / total_stmts) if total_stmts else 100.0
    print("  " + "-" * (len(hdr) - 2))
    print(
        f"  {'TOTAL':<36} {total_stmts:>6} {total_hits:>6} "
        f"{total_stmts - total_hits:>6} {total_pct:>6.1f}%"
    )

    if total_pct < args.min_pct:
        raise RuntimeError(
            f"[scripts/coverage_report.py:main] coverage threshold violated: "
            f"expected >= {args.min_pct:.1f}%, got {total_pct:.1f}% "
            f"({total_hits}/{total_stmts} lines)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
