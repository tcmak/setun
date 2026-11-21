"""Red-first unit tests for Python coverage reporter (scripts/coverage_report.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.coverage_report import (
    FAST_PYTEST_ARGS,
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


def test_extract_executable_lines_empty_and_type_checking(tmp_path: Path) -> None:
    """Verify empty files return set() and TYPE_CHECKING blocks are excluded."""
    empty_file = tmp_path / "empty_mod.py"
    empty_file.write_text("   \n\n  ", encoding="utf-8")
    assert extract_executable_lines(empty_file) == set()

    tc_file = tmp_path / "tc_mod.py"
    tc_file.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import decimal\n"
        "def f(x: int) -> int:\n"
        "    return x + 1\n",
        encoding="utf-8",
    )
    exec_lines = extract_executable_lines(tc_file)
    assert 2 not in exec_lines
    assert 3 not in exec_lines
    assert {1, 4, 5}.issubset(exec_lines)


def test_nested_tracer_preserves_outer_sys_settrace(tmp_path: Path) -> None:
    """Verify run_with_line_tracer forwards events to and restores an existing trace hook."""
    import sys
    import types

    mod_path = tmp_path / "nested_target.py"
    mod_path.write_text("def inc(v: int) -> int:\n    return v + 1\n", encoding="utf-8")

    outer_events: list[str] = []

    def _outer_tracer(frame: types.FrameType, event: str, _arg: object):
        outer_events.append(event)
        return _outer_tracer

    def _workload() -> int:
        ns: dict[str, object] = {}
        code = compile(mod_path.read_text(encoding="utf-8"), str(mod_path.resolve()), "exec")
        exec(code, ns)
        fn = ns["inc"]
        assert callable(fn)
        assert fn(10) == 11
        return 0

    prev = sys.gettrace()
    sys.settrace(_outer_tracer)
    try:
        rc, hits = run_with_line_tracer([mod_path], _workload)
        assert sys.gettrace() is _outer_tracer
    finally:
        sys.settrace(prev)

    assert rc == 0
    assert {1, 2}.issubset(hits[mod_path.resolve()])
    assert len(outer_events) > 0


def test_cli_show_missing_success_and_nonzero_pytest_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify main() happy path with --show-missing truncation and pytest failure error."""
    rc = main(
        [
            "--pytest-args",
            "tests/unit/test_version.py -k test_version_string_contains_version_and_hash",
            "--show-missing",
            "--min-pct",
            "0.1",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Missing" in out
    assert "..." in out
    assert "TOTAL" in out

    with pytest.raises(RuntimeError, match="pytest exited with non-zero code"):
        main(["--pytest-args", "tests/unit/nonexistent_file_1958.py"])


def test_fast_pytest_args_uses_not_slow_marker_across_unit_suite() -> None:
    """Verify --fast uses pytest '-m not slow' across all of tests/unit (no file allowlist)."""
    assert FAST_PYTEST_ARGS == ["-q", "-m", "not slow", "tests/unit"], (
        f"[test_coverage_report:FAST_PYTEST_ARGS] expected ['-q', '-m', 'not slow', 'tests/unit'], "
        f"got {FAST_PYTEST_ARGS!r}"
    )
    pyproject_text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "markers" in pyproject_text and "slow:" in pyproject_text, (
        "[test_coverage_report:pyproject.toml] expected registered 'slow' pytest marker"
    )
