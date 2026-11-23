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


def test_doctor_enforces_nextpnr_ecp5_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Task C1.1b: doctor.py and toolchain.lock must pin and enforce nextpnr-ecp5."""
    repo_lock = Path(__file__).resolve().parent.parent.parent / "toolchain.lock"
    repo_locks = doctor.parse_lockfile(repo_lock)
    assert "nextpnr-ecp5" in repo_locks, (
        "[test_doctor:toolchain.lock] missing required 'nextpnr-ecp5' pin in toolchain.lock"
    )
    assert hasattr(doctor, "probe_nextpnr_ecp5"), (
        "[test_doctor:doctor] missing probe_nextpnr_ecp5() in scripts/doctor.py"
    )

    lockfile = _write_test_lockfile(tmp_path)
    lockfile.write_text(
        lockfile.read_text(encoding="utf-8") + "nextpnr-ecp5 = 0.11.1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(doctor, "probe_python", lambda: "3.12")
    monkeypatch.setattr(doctor, "probe_verilator", lambda: "5.020")
    monkeypatch.setattr(doctor, "probe_yosys", lambda: "0.33")
    monkeypatch.setattr(
        doctor,
        "probe_pkg_version",
        lambda pkg: "0.10.0" if pkg == "migen" else ("1.8.1" if pkg == "cocotb" else "MISSING"),
    )
    litex_map = {
        "litex": "9187839f2d275352ef6f0d7ab14431e2a66b8f81",
        "litedram": "f2d22975d67229676e10308ae6e71b819b6afe42",
        "liteeth": "9712c3899c4b9d162aca5aef403889ab2c8c4542",
    }
    monkeypatch.setattr(
        doctor,
        "probe_git_pkg",
        lambda pkg: "INSTALLED_NON_GIT" if pkg == "migen" else litex_map.get(pkg, "MISSING"),
    )

    monkeypatch.setattr(doctor, "probe_nextpnr_ecp5", lambda: "0.11.1")
    assert doctor.main(["--lockfile", str(lockfile)]) == 0

    monkeypatch.setattr(doctor, "probe_nextpnr_ecp5", lambda: "MISSING")
    rc = doctor.main(["--lockfile", str(lockfile)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "nextpnr-ecp5" in captured.err
    assert "MISSING" in captured.err


def test_doctor_live_environment_and_banner(capsys: pytest.CaptureFixture[str]) -> None:
    """Run real unmocked probes against the repo toolchain.lock with --banner."""
    repo_lock = Path(__file__).resolve().parents[2] / "toolchain.lock"
    rc = doctor.main(["--lockfile", str(repo_lock), "--banner"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== Setun Toolchain Versions ===" in out
    assert "migen" in out


def test_doctor_probe_fallbacks_and_cli_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cover probe missing/fallback branches, --write, missing lockfile, and git mismatch."""
    # 1. get_cmd_output failure -> None
    assert doctor.get_cmd_output(["definitely_nonexistent_cmd_1958"]) is None

    # 2. probe_verilator & probe_yosys missing vs non-standard banner fallback
    monkeypatch.setattr(doctor, "get_cmd_output", lambda _cmd: None)
    assert doctor.probe_verilator() == "MISSING"
    assert doctor.probe_yosys() == "MISSING"

    monkeypatch.setattr(doctor, "get_cmd_output", lambda _cmd: "CustomBanner v99\nextra")
    assert doctor.probe_verilator() == "CustomBanner v99"
    assert doctor.probe_yosys() == "CustomBanner v99"

    # 3. resolve_nextpnr_ecp5_bin venv fallback & missing
    fake_venv = tmp_path / "fake_bin"
    fake_venv.mkdir()
    fake_py = fake_venv / "python3"
    fake_py.write_text("", encoding="utf-8")
    monkeypatch.setattr(doctor.sys, "executable", str(fake_py))
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: None)
    assert doctor.resolve_nextpnr_ecp5_bin() is None
    assert doctor.probe_nextpnr_ecp5() == "MISSING"

    fake_pnr = fake_venv / "nextpnr-ecp5"
    fake_pnr.write_text("", encoding="utf-8")
    assert doctor.resolve_nextpnr_ecp5_bin() == str(fake_pnr)
    monkeypatch.setattr(doctor, "get_cmd_output", lambda _cmd: None)
    assert doctor.probe_nextpnr_ecp5() == "MISSING"
    monkeypatch.setattr(doctor, "get_cmd_output", lambda _cmd: "header\ncustom-pnr-build")
    assert doctor.probe_nextpnr_ecp5() == "custom-pnr-build"

    # 4. probe_pkg_version PackageNotFoundError & probe_git_pkg UNPINNED_GIT_VCS / editable git
    assert doctor.probe_pkg_version("nonexistent-setun-pkg-1958") == "MISSING"
    assert doctor.probe_git_pkg("nonexistent_setun_mod_1958") == "MISSING"

    unpinned_dist = MagicMock()
    unpinned_dist.metadata = {"Name": "custom_vcs_pkg"}
    unpinned_dist.read_text.return_value = (
        '{"url": "https://example.com/repo.git", "vcs_info": {"vcs": "git"}}'
    )
    monkeypatch.setattr(doctor.importlib.metadata, "distributions", lambda: [unpinned_dist])
    assert doctor.probe_git_pkg("custom_vcs_pkg") == "UNPINNED_GIT_VCS"

    monkeypatch.setattr(doctor.importlib.metadata, "distributions", lambda: [])
    fake_spec = MagicMock()
    fake_spec.origin = str(tmp_path / "ext_pkg" / "__init__.py")
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda _pkg: fake_spec)
    monkeypatch.setattr(
        doctor,
        "get_cmd_output",
        lambda cmd: str(tmp_path / "ext_pkg") if "--show-toplevel" in cmd else "deadbeef" * 5,
    )
    assert doctor.probe_git_pkg("ext_pkg") == "deadbeef" * 5

    # 5. Missing lockfile error
    rc_missing = doctor.main(["--lockfile", str(tmp_path / "missing.lock")])
    assert rc_missing == 1
    assert "Lockfile invariant violated" in capsys.readouterr().err

    # 6. Lockfile missing 'migen' key + git: commit mismatch + --write round-trip
    no_migen_lock = tmp_path / "no_migen.lock"
    no_migen_lock.write_text("# comment\npython = 3.12\ngit:litex = 11111111\n", encoding="utf-8")
    monkeypatch.setattr(
        doctor,
        "probe_all",
        lambda _keys: {"python": "3.12", "git:litex": "22222222", "migen": "0.10.0"},
    )
    rc_bad = doctor.main(["--lockfile", str(no_migen_lock)])
    err_bad = capsys.readouterr().err
    assert rc_bad == 1
    assert "check_migen_pin" in err_bad
    assert "git:litex" in err_bad

    write_target = tmp_path / "written.lock"
    write_target.write_text("python = 3.12\nmigen = 0.10.0\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "probe_all", lambda _keys: {"python": "3.12", "migen": "0.10.0"})
    assert doctor.main(["--lockfile", str(write_target), "--write"]) == 0
    parsed = doctor.parse_lockfile(write_target)
    assert parsed == {"python": "3.12", "migen": "0.10.0"}


def test_doctor_enforces_ternarytape_spec_v1_0_submodule_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """OQ-001: toolchain.lock and doctor.py must pin third_party/TernaryTape at spec-v1.0."""
    expected_sha = "3c73f3cc85f0d7a9bf1ccd00b7dc8a100f379920"
    sub_key = "submodule:third_party/TernaryTape"

    repo_lock = Path(__file__).resolve().parents[2] / "toolchain.lock"
    repo_locks = doctor.parse_lockfile(repo_lock)
    assert repo_locks.get(sub_key) == expected_sha, (
        f"[test_doctor:toolchain.lock] expected {sub_key} = {expected_sha!r} (spec-v1.0), "
        f"got {repo_locks.get(sub_key)!r}"
    )
    assert hasattr(doctor, "probe_submodule"), (
        "[test_doctor:doctor] missing probe_submodule() in scripts/doctor.py"
    )
    assert doctor.probe_submodule("third_party/TernaryTape") == expected_sha
    assert doctor.probe_submodule("third_party/nonexistent_submodule_1958") == "MISSING"

    lockfile = _write_test_lockfile(tmp_path)
    lockfile.write_text(
        lockfile.read_text(encoding="utf-8") + f"{sub_key} = {expected_sha}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(doctor, "probe_python", lambda: "3.12")
    monkeypatch.setattr(doctor, "probe_verilator", lambda: "5.020")
    monkeypatch.setattr(doctor, "probe_yosys", lambda: "0.33")
    monkeypatch.setattr(
        doctor,
        "probe_pkg_version",
        lambda pkg: "0.10.0" if pkg == "migen" else ("1.8.1" if pkg == "cocotb" else "MISSING"),
    )
    litex_map = {
        "litex": "9187839f2d275352ef6f0d7ab14431e2a66b8f81",
        "litedram": "f2d22975d67229676e10308ae6e71b819b6afe42",
        "liteeth": "9712c3899c4b9d162aca5aef403889ab2c8c4542",
    }
    monkeypatch.setattr(
        doctor,
        "probe_git_pkg",
        lambda pkg: "INSTALLED_NON_GIT" if pkg == "migen" else litex_map.get(pkg, "MISSING"),
    )

    monkeypatch.setattr(doctor, "probe_submodule", lambda _rel: expected_sha)
    assert doctor.main(["--lockfile", str(lockfile)]) == 0

    # Drifted draft SHA (spec-v0.1-draft) must fail doctor check
    monkeypatch.setattr(
        doctor, "probe_submodule", lambda _rel: "1c30a54130ca071dd7f051abb18ea9f5be4567c4"
    )
    rc = doctor.main(["--lockfile", str(lockfile)])
    err = capsys.readouterr().err
    assert rc == 1
    assert sub_key in err and expected_sha in err
