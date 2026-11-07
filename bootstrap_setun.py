#!/usr/bin/env python3
"""Materialize the entire Setun Season 1 repository with raw files."""
from pathlib import Path

FILES: dict[str, str] = {
    ".gitignore": """\
__pycache__/
*.py[cod]
.venv/
build/
dist/
*.egg-info/
.pytest_cache/
.ruff_cache/
sim_build/
*.vcd
*.fst
""",
    ".pre-commit-config.yaml": """\
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.10
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
""",
    "LICENSE": """\
BSD 2-Clause License

Copyright (c) 2026, Setun Contributors

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
""",
    "pyproject.toml": """\
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "setun"
version = "0.1.0.dev0"
description = "A LiteX-native ternary LLM toolchain: NPU overlay and compiler"
readme = "README.md"
license = { text = "BSD-2-Clause" }
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "ruff>=0.4",
    "pre-commit>=3.5",
    "cocotb>=1.8",
    "cocotb-test>=0.2.4",
]

[project.scripts]
setun = "setun.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["setun*", "targets*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
""",
    "toolchain.lock": """\
# Pinned environment specification for Setun (ADR-012: WSL2 Ubuntu apt + venv).
# Run `make doctor` to verify against this lockfile.

python = 3.12
verilator = 5.020
yosys = 0.33
cocotb = 1.9.2

# LiteX ecosystem git commit pins (installed as editable git packages)
git:migen = d62d73602fa5ff6f7fb0f3f7d0f1b6bf0fa4885a
git:litex = e3ebce1f33eb42cb0a7fd098a3eb4b0cf8eb48b4
git:litedram = 53a7f5d241b720b18eb015ca0d56c4cf0bc2ab9d
git:liteeth = 80fa158268295248330b7cd86a622a5a05593938
""",
    "Makefile": """\
.PHONY: doctor doctor-lock lint format test sim-smoke ci nightly

PYTHON ?= python3

doctor:
\t@$(PYTHON) scripts/doctor.py --lockfile toolchain.lock

doctor-lock:
\t@$(PYTHON) scripts/doctor.py --lockfile toolchain.lock --write

lint:
\truff check setun targets tests scripts
\truff format --check setun targets tests scripts

format:
\truff check --fix setun targets tests scripts
\truff format setun targets tests scripts

test:
\tpytest tests/unit

sim-smoke:
\t@if [ -f targets/sim.py ]; then \\
\t\t$(PYTHON) -m targets.sim; \\
\telse \\
\t\techo "[sim-smoke] targets/sim.py not yet implemented (scheduled for Task 0.3)"; \\
\tfi

ci:
\t@$(PYTHON) scripts/doctor.py --lockfile toolchain.lock --banner
\t$(MAKE) lint
\t$(MAKE) test
\t$(MAKE) sim-smoke

nightly:
\t@$(PYTHON) scripts/doctor.py --lockfile toolchain.lock --banner
\tpytest tests/
""",
    "NEXT.md": """\
# Next Up

- **Current Season:** Season 1 (Learn RTL, build a ternary row engine)
- **Completed Agent Tasks:** `0.1`, `0.2a`, `0.5a`, `0.6a`, `1.3a`, and drafted `1.4a`
- **Immediate Action (Maintainer):**
  1. Review the `1.4a` draft (`docs/NUMERICS.md` and `setun/ref/ops.py`) and check the hand-worked example in Section 3.
  2. Work through Block A (`F.1`-`F.6` on HDLBits) and Block B (`F.7`, `0.3`, `0.5b`, `F.8a`, `F.8b`).
- **Waiting Agent Tasks (Blocked on Maintainer RTL in Block D/E):**
  - `2.1e`: Yosys synthesis report parser (`scripts/synth.py`)
  - `2.6a`: Resource sweep script
  - `2.8a`: Randomized nightly suite for `TernaryRow`
""",
    "setun/__init__.py": '"""Setun: A LiteX-native ternary LLM toolchain."""\n\n__version__ = "0.1.0.dev0"\n',
    "setun/accelerators/__init__.py": '"""Migen/LiteX hardware accelerator cores for Setun."""\n',
    "setun/compiler/__init__.py": '"""Setun compiler: frontends, IR, passes, and backends."""\n',
    "setun/ref/__init__.py": '"""Golden integer reference models and format codecs for Setun."""\n',
    "setun/runtime/__init__.py": '"""Host runtime drivers (cocotb, Etherbone, board) for Setun."""\n',
    "setun/serve/__init__.py": '"""OpenAI-compatible server for Setun."""\n',
    "setun/soc/__init__.py": '"""LiteX SoC integration, CSRs, and DMA wrappers for Setun."""\n',
    "targets/__init__.py": '"""LiteX SoC targets (sim, DDR board, HBM board) for Setun."""\n',
    "tests/__init__.py": '"""Test suite for Setun."""\n',
}

for rel_path, content in FILES.items():
    p = Path(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print(f"Wrote {rel_path}")
