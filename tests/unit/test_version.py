from __future__ import annotations

import subprocess
import sys

from setun import __version__
from setun.cli import get_version_string


def test_version_string_contains_version_and_hash() -> None:
    ver = get_version_string()
    assert __version__ in ver
    assert "(" in ver and ")" in ver


def test_cli_version_flag() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "setun.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    out = res.stdout.strip()
    assert out.startswith(f"setun {__version__} (")
