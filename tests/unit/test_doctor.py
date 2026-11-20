"""Unit tests for scripts/doctor.py toolchain and migen==0.10.0 enforcement (Task A1.1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import doctor


def _write_test_lockfile(tmp_path: Path, migen_line: str = "migen = 0.10.0") -> Path:
    lockfile = tmp_path / "toolchain.lock"
    lockfile.write_text(
        "\n".join(
            [
                "python = 3.12",
                "verilator = 5.020",
                "yosys = 0.33",
                "cocotb = 1.8.1",
                migen_line,
                "git:litex = 9187839f2d275352ef6f0d7ab14431e2a66b8f81",
                "git:litedram = f2d22975d67229676e10308ae6e71b819b6afe42",
                "git:liteeth = 9712c3899c4b9d162aca5aef403889ab2c8c4542",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return lockfile


def test_doctor_passes_on_valid_migen_0_10_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lockfile = _write_test_lockfile(tmp_path)
    monkeypatch.setattr(doctor, "probe_python", lambda: "3.12")
    monkeypatch.setattr(doctor, "probe_verilator", lambda: "5.020")
    monkeypatch.setattr(doctor, "probe_yosys", lambda: "0.33")
    monkeypatch.setattr(
        doctor,
        "probe_pkg_version",
        lambda pkg: "0.10.0" if pkg == "migen" else ("1.8.1" if pkg == "cocotb" else "MISSING"),
    )
    git_map = {
        "litex": "9187839f2d275352ef6f0d7ab14431e2a66b8f81",
        "litedram": "f2d22975d67229676e10308ae6e71b819b6afe42",
        "liteeth": "9712c3899c4b9d162aca5aef403889ab2c8c4542",
    }
    monkeypatch.setattr(
        doctor,
        "probe_git_pkg",
        lambda pkg: "INSTALLED_NON_GIT" if pkg == "migen" else git_map.get(pkg, "MISSING"),
    )

    rc = doctor.main(["--lockfile", str(lockfile)])
    assert rc == 0


def test_doctor_rejects_migen_0_9_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lockfile = _write_test_lockfile(tmp_path)
    monkeypatch.setattr(doctor, "probe_python", lambda: "3.12")
    monkeypatch.setattr(doctor, "probe_verilator", lambda: "5.020")
    monkeypatch.setattr(doctor, "probe_yosys", lambda: "0.33")
    monkeypatch.setattr(
        doctor,
        "probe_pkg_version",
        lambda pkg: "0.9.2" if pkg == "migen" else "1.8.1",
    )
    litex_sha = "9187839f2d275352ef6f0d7ab14431e2a66b8f81"
    monkeypatch.setattr(
        doctor,
        "probe_git_pkg",
        lambda pkg: "INSTALLED_NON_GIT" if pkg == "migen" else litex_sha,
    )

    rc = doctor.main(["--lockfile", str(lockfile)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "migen" in captured.err
    assert "0.10.0" in captured.err
    assert "0.9.2" in captured.err


def test_doctor_rejects_unpinned_git_migen_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Even if metadata version reports 0.10.0, a git install of migen must be rejected."""
    lockfile = _write_test_lockfile(tmp_path)
    monkeypatch.setattr(doctor, "probe_python", lambda: "3.12")
    monkeypatch.setattr(doctor, "probe_verilator", lambda: "5.020")
    monkeypatch.setattr(doctor, "probe_yosys", lambda: "0.33")

    fake_dist = MagicMock()
    fake_dist.metadata = {"Name": "migen"}
    fake_dist.read_text.return_value = (
        '{"url": "https://github.com/m-labs/migen.git", '
        '"vcs_info": {"vcs": "git", '
        '"commit_id": "d62d73602fa5ff6f7fb0f3f7d0f1b6bf0fa4885a"}}'
    )
    monkeypatch.setattr(doctor.importlib.metadata, "distributions", lambda: [fake_dist])
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "version",
        lambda pkg: "0.10.0" if pkg == "migen" else "1.8.1",
    )

    rc = doctor.main(["--lockfile", str(lockfile)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "migen" in captured.err
    assert "0.10.0" in captured.err


def test_repo_toolchain_lock_requires_migen_0_10_0() -> None:
    """The real repo toolchain.lock must pin migen == 0.10.0 (not git:migen = INSTALLED_NON_GIT)."""
    repo_lock = Path(__file__).resolve().parent.parent.parent / "toolchain.lock"
    locks = doctor.parse_lockfile(repo_lock)
    assert locks.get("migen") == "0.10.0", (
        f"toolchain.lock invariant violated: expected migen = '0.10.0', got {locks.get('migen')!r}"
    )
    assert "git:migen" not in locks, "toolchain.lock must not use floating git:migen entry"
