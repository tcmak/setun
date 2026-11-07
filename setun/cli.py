"""Command-line interface for Setun."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from setun import __version__


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
