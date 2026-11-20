# Contributing to Setun

## WSL2 Ubuntu Toolchain Setup (Task 0.2a / ADR-012)

1. **System packages (via `apt`):**

```sh
sudo apt update
sudo apt install -y build-essential git python3 python3-venv python3-pip \
    verilator yosys gtkwave libevent-dev libjson-c-dev
```

2. **Python virtual environment & LiteX git pins:**

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Install migen==0.10.0 (via pyproject.toml) and pinned LiteX ecosystem commits from toolchain.lock:
pip install \
    "git+https://github.com/enjoy-digital/litex.git@9187839f2d275352ef6f0d7ab14431e2a66b8f81" \
    "git+https://github.com/enjoy-digital/litedram.git@f2d22975d67229676e10308ae6e71b819b6afe42" \
    "git+https://github.com/enjoy-digital/liteeth.git@9712c3899c4b9d162aca5aef403889ab2c8c4542"
```

See [`toolchain.lock`](toolchain.lock) and [`pyproject.toml`](pyproject.toml) (`migen==0.10.0`) as the single source of truth for all pinned versions and commits.

3. **Verify environment:**

```sh
# If initializing on your WSL2 distro for the first time, snapshot your apt versions:
make doctor-lock
# Verify lockfile and run checks:
make doctor
make lint test
```
