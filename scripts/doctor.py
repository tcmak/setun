"""Verify local toolchain versions and git commits against toolchain.lock."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

PINNED_MIGEN_VERSION = "0.10.0"


def parse_lockfile(path: Path) -> dict[str, str]:
    locks: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        locks[k.strip()] = v.strip()
    return locks


def get_cmd_output(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None


def probe_python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def probe_verilator() -> str:
    out = get_cmd_output(["verilator", "--version"])
    if not out:
        return "MISSING"
    m = re.search(r"Verilator\s+([0-9]+\.[0-9]+)", out)
    return m.group(1) if m else out.splitlines()[0]


def probe_yosys() -> str:
    out = get_cmd_output(["yosys", "-V"])
    if not out:
        return "MISSING"
    m = re.search(r"Yosys\s+([0-9]+\.[0-9]+(?:\+[0-9]+)?)", out)
    return m.group(1) if m else out.splitlines()[0]


def resolve_nextpnr_ecp5_bin() -> str | None:
    venv_bin = Path(sys.executable).resolve().parent
    for name in ("nextpnr-ecp5", "yowasp-nextpnr-ecp5"):
        found = shutil.which(name)
        if found:
            return found
        candidate = venv_bin / name
        if candidate.exists():
            return str(candidate)
    return None


def probe_nextpnr_ecp5() -> str:
    exe = resolve_nextpnr_ecp5_bin()
    if not exe:
        return "MISSING"
    out = get_cmd_output([exe, "--version"])
    if not out:
        return "MISSING"
    m = re.search(r"Version\s+(?:nextpnr-)?([0-9]+\.[0-9]+(?:\.[0-9]+)?)", out)
    return m.group(1) if m else out.splitlines()[-1]


def probe_pkg_version(pkg: str) -> str:
    try:
        ver = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"

    git_state = probe_git_pkg(pkg)
    if git_state not in ("INSTALLED_NON_GIT", "MISSING"):
        return f"git@{git_state} (unpinned git install of {pkg}=={ver})"
    return ver


REPO_ROOT = Path(__file__).resolve().parent.parent


def probe_git_pkg(pkg: str) -> str:
    # 1. Check PEP 610 direct_url.json first (pip install git+https://...)
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name and name.lower().replace("-", "_") == pkg.lower().replace("-", "_"):
            direct_url = dist.read_text("direct_url.json")
            if direct_url:
                m = re.search(r'"commit_id"\s*:\s*"([0-9a-f]{40})"', direct_url)
                if m:
                    return m.group(1)
                if '"vcs"' in direct_url or "git" in direct_url:
                    return "UNPINNED_GIT_VCS"

    # 2. Check editable git checkout only if its top-level is not REPO_ROOT
    spec = importlib.util.find_spec(pkg)
    if spec is None or spec.origin is None:
        return "MISSING"
    pkg_dir = Path(spec.origin).resolve().parent
    toplevel = get_cmd_output(["git", "-C", str(pkg_dir), "rev-parse", "--show-toplevel"])
    if toplevel and Path(toplevel).resolve() != REPO_ROOT:
        commit = get_cmd_output(["git", "-C", str(pkg_dir), "rev-parse", "HEAD"])
        if commit:
            return commit

    return "INSTALLED_NON_GIT"


def probe_submodule(rel_path: str) -> str:
    sub_dir = (REPO_ROOT / rel_path).resolve()
    if not sub_dir.exists():
        return "MISSING"
    commit = get_cmd_output(["git", "-C", str(sub_dir), "rev-parse", "HEAD"])
    return commit if commit else "MISSING"


def probe_all(keys: list[str]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for k in keys:
        if k == "python":
            actual[k] = probe_python()
        elif k == "verilator":
            actual[k] = probe_verilator()
        elif k == "yosys":
            actual[k] = probe_yosys()
        elif k == "nextpnr-ecp5":
            actual[k] = probe_nextpnr_ecp5()
        elif k.startswith("git:"):
            pkg = k.split(":", 1)[1]
            actual[k] = probe_git_pkg(pkg)
        elif k.startswith("submodule:"):
            rel_path = k.split(":", 1)[1]
            actual[k] = probe_submodule(rel_path)
        else:
            actual[k] = probe_pkg_version(k)
    return actual


def write_lockfile(path: Path, actual: dict[str, str]) -> None:
    lines = [
        "# Pinned environment specification for Setun (ADR-012: WSL2 Ubuntu apt + venv).",
        "# Run `make doctor` to verify against this lockfile.",
        "",
    ]
    for k, v in actual.items():
        lines.append(f"{k} = {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Setun toolchain lockfile.")
    parser.add_argument("--lockfile", type=Path, default=Path("toolchain.lock"))
    parser.add_argument("--banner", action="store_true", help="Print CI version banner")
    parser.add_argument("--write", action="store_true", help="Update lockfile from active env")
    args = parser.parse_args(argv)

    if not args.lockfile.exists():
        print(
            f"ERROR [scripts/doctor.py:main]: Lockfile invariant violated at offset 0: "
            f"expected existing file at '{args.lockfile}', found MISSING.",
            file=sys.stderr,
        )
        return 1

    expected = parse_lockfile(args.lockfile)
    probe_keys = list(expected.keys())
    if "migen" not in probe_keys:
        probe_keys.append("migen")
    actual = probe_all(probe_keys)

    if args.write:
        write_lockfile(args.lockfile, actual)
        print(f"Updated {args.lockfile} with current environment versions.")
        return 0

    if args.banner:
        print("=== Setun Toolchain Versions ===")
        for k, v in actual.items():
            print(f"  {k:16s} : {v}")
        print("================================")

    mismatches: list[str] = []
    if expected.get("migen") != PINNED_MIGEN_VERSION:
        mismatches.append(
            f"  - [scripts/doctor.py:check_migen_pin] invariant 'migen == {PINNED_MIGEN_VERSION}' "
            f"violated in {args.lockfile} (line/key 'migen'): "
            f"expected '{PINNED_MIGEN_VERSION}', found '{expected.get('migen', 'MISSING')}'"
        )

    act_migen = actual.get("migen", "MISSING")
    if act_migen != PINNED_MIGEN_VERSION:
        mismatches.append(
            f"  - [scripts/doctor.py:check_migen_version] invariant "
            f"'migen == {PINNED_MIGEN_VERSION}' violated in active Python environment: "
            f"expected '{PINNED_MIGEN_VERSION}', found '{act_migen}'"
        )

    for k, exp_val in expected.items():
        if k == "migen":
            continue
        act_val = actual.get(k, "MISSING")
        if k.startswith(("git:", "submodule:")) and act_val != "MISSING":
            if not act_val.startswith(exp_val) and not exp_val.startswith(act_val):
                mismatches.append(
                    f"  - [scripts/doctor.py:main] key '{k}' invariant violated: "
                    f"expected '{exp_val}', found '{act_val}'"
                )
        elif act_val != exp_val:
            mismatches.append(
                f"  - [scripts/doctor.py:main] key '{k}' invariant violated: "
                f"expected '{exp_val}', found '{act_val}'"
            )

    if mismatches:
        print(
            f"ERROR [scripts/doctor.py:main]: Toolchain check failed against {args.lockfile}:",
            file=sys.stderr,
        )
        for msg in mismatches:
            print(msg, file=sys.stderr)
        print(
            "\nFix the environment or run `make doctor-lock` if intentionally updating pins.",
            file=sys.stderr,
        )
        return 1

    if not args.banner:
        print("doctor: all toolchain versions match toolchain.lock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
