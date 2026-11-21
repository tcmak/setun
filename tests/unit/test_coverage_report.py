"""Red-first unit tests for Python coverage reporter (scripts/coverage_report.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.coverage_report import (
    discover_repo_python_files,
    extract_executable_lines,
    format_line_ranges,
    main,
    run_with_line_tracer,
)


def test_format_line_ranges_compact() -> None:
    """Verify uncovered line numbers collapse into compact human-readable ranges."""
    assert format_line_ranges(set()) == "-"
    assert format_line_ranges({5}) == "5"
    assert format_line_ranges({1, 2, 3, 7, 9, 10, 15}) == "1-3, 7, 9-10, 15"


def test_extract_executable_lines_ignores_docstrings_and_main_guard(tmp_path: Path) -> None:
    """Verify bytecode line extraction skips comments, docstrings, and __main__ guards."""
    sample = tmp_path / "sample_mod.py"
    sample.write_text(
        '"""Module docstring.\n\nMulti-line text.\n"""\n'
        "\n"
        "# Pure comment line\n"
        "def clamp_pos(x: int) -> int:\n"
        '    """Function docstring."""\n'
        "    if x < 0:\n"
        "        raise ValueError('negative')\n"
        "    return x * 2\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    print(clamp_pos(3))\n",
        encoding="utf-8",
    )

    exec_lines = extract_executable_lines(sample)
    assert 1 not in exec_lines  # docstring excluded
    assert 6 not in exec_lines  # comment excluded
    assert 13 not in exec_lines  # __main__ guard excluded
    assert 14 not in exec_lines  # __main__ body excluded
    assert {7, 9, 10, 11}.issubset(exec_lines)


def test_line_tracer_records_hit_and_unhit_negative_branches(tmp_path: Path) -> None:
    """Verify sys.settrace hook records hit lines and flags unhit negative branches."""
    mod_path = tmp_path / "mini_target.py"
    mod_path.write_text(
        "def check_val(v: int) -> int:\n"
        "    if v < 0:\n"
        "        raise ValueError('bad')\n"
        "    return v + 1\n",
        encoding="utf-8",
    )

    def _workload() -> int:
        ns: dict[str, object] = {}
        code = compile(mod_path.read_text(encoding="utf-8"), str(mod_path.resolve()), "exec")
        exec(code, ns)
        fn = ns["check_val"]
        assert callable(fn)
        assert fn(4) == 5
        return 0

    rc, hits = run_with_line_tracer([mod_path], _workload)
    assert rc == 0
    exec_lines = extract_executable_lines(mod_path)
    hit_lines = hits[mod_path.resolve()] & exec_lines
    missed = exec_lines - hit_lines
    # Line 3 (`raise ValueError('bad')`) was never executed!
    assert missed == {3}


def test_discover_repo_python_files_covers_all_packages() -> None:
    """Verify auto-discovery includes setun/{ref,accelerators,soc,cli}, targets, and scripts."""
    repo_root = Path(__file__).resolve().parents[2]
    discovered = {p.relative_to(repo_root).as_posix() for p in discover_repo_python_files()}
    expected_subset = {
        "setun/cli.py",
        "setun/ref/ops.py",
        "setun/ref/format.py",
        "setun/ref/m0.py",
        "setun/accelerators/stub.py",
        "setun/accelerators/ternary_dot.py",
        "setun/accelerators/group_scale.py",
        "setun/accelerators/unpacker.py",
        "setun/accelerators/tile.py",
        "setun/accelerators/requantize.py",
        "setun/soc/ternary_accel.py",
        "targets/sim.py",
        "scripts/doctor.py",
        "scripts/make_toy.py",
        "scripts/resource_report.py",
    }
    missing = expected_subset - discovered
    assert not missing, f"Missing repo files from coverage discovery: {sorted(missing)}"


def test_cli_negative_min_pct_and_bad_args() -> None:
    """Verify Goose-style failure when --min-pct threshold is not met."""
    with pytest.raises(RuntimeError, match="coverage_report"):
        main(["--pytest-args", "tests/unit/test_version.py", "--min-pct", "99.9"])
