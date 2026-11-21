from __future__ import annotations

import subprocess
import sys

import pytest

import setun.cli as cli_mod
from setun import __version__
from setun.cli import build_parser, get_git_hash, get_version_string, main


def test_version_string_contains_version_and_hash() -> None:
    ver = get_version_string()
    assert __version__ in ver
    assert "(" in ver and ")" in ver


def test_get_git_hash_fallback_when_git_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_git_hash falls back cleanly to 'unknown' when git is unavailable."""

    def _raise_git_err(*_args: object, **_kwargs: object) -> str:
        raise subprocess.CalledProcessError(128, ["git", "rev-parse"])

    monkeypatch.setattr(cli_mod.subprocess, "check_output", _raise_git_err)
    assert get_git_hash() == "unknown"
    assert get_version_string() == f"setun {__version__} (unknown)"


def test_cli_main_help_and_version_in_process(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify build_parser() and main() in-process for empty args and --version."""
    parser = build_parser()
    assert parser.prog == "setun"

    rc = main([])
    assert rc == 0
    help_out = capsys.readouterr().out
    assert "Setun: LiteX-native ternary LLM toolchain" in help_out

    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    ver_out = capsys.readouterr().out.strip()
    assert ver_out.startswith(f"setun {__version__} (")


def test_cli_version_flag() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "setun.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    out = res.stdout.strip()
    assert out.startswith(f"setun {__version__} (")
