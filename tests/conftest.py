"""Pytest configuration hooks (RAM-aware worker scaling for pytest-xdist)."""

from __future__ import annotations

import os
from pathlib import Path

_GIB_PER_EDA_WORKER = 2.5


def _detect_total_ram_gib() -> float | None:
    """Return total system RAM in GiB on Linux/POSIX, or None if unavailable."""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                return kib / (1024.0 * 1024.0)
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return (pages * page_size) / (1024.0**3)
        except (ValueError, OSError):
            pass
    return None


def pytest_xdist_auto_num_workers(config) -> int:  # noqa: ARG001
    """Bound `-n auto` by both CPU cores and ~2.5 GiB/worker for Verilator/Yosys/nextpnr."""
    env_jobs = os.environ.get("SETUN_JOBS")
    if env_jobs is not None:
        return max(1, int(env_jobs))

    cpus = os.cpu_count() or 1
    ram_gib = _detect_total_ram_gib()
    if ram_gib is None:
        return cpus

    ram_cap = max(1, int(ram_gib // _GIB_PER_EDA_WORKER))
    return max(1, min(cpus, ram_cap))
