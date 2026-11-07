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

# Install pinned LiteX ecosystem packages (matches toolchain.lock)
pip install \
    "git+https://github.com/m-labs/migen.git@d62d73602fa5ff6f7fb0f3f7d0f1b6bf0fa4885a" \
    "git+https://github.com/enjoy-digital/litex.git@e3ebce1f33eb42cb0a7fd098a3eb4b0cf8eb48b4" \
    "git+https://github.com/enjoy-digital/litedram.git@53a7f5d241b720b18eb015ca0d56c4cf0bc2ab9d" \
    "git+https://github.com/enjoy-digital/liteeth.git@80fa158268295248330b7cd86a622a5a05593938"
```

3. **Verify environment:**

```sh
# If initializing on your WSL2 distro for the first time, snapshot your apt versions:
make doctor-lock
# Verify lockfile and run checks:
make doctor
make lint test
```
