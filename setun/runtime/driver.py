"""AccelDriver Python hardware & simulation host abstraction (Task 3.3a).

Supports:
  - CocotbWishboneBackend (`cocotb`-direct master over SetunSimSoC.host_wb)
  - EtherboneBackend / LitexSimSession (`--backend sim` via litex_server + RemoteClient)
  - RefBusBackend (`--backend ref` golden behavioral bus backend for fast unit tests)
  - `--backend board` placeholder raising NotImplementedError until Season 3 Task 3.3b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from setun.ref.format import PackedTernaryTensor
from setun.ref.m0 import ternary_matvec_integer
from setun.ref.ops import requantize_int8
from setun.soc import DEFAULT_CSR_JSON

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VALID_BACKENDS: tuple[str, ...] = ("ref", "sim", "board")

ERR_FLAGS: tuple[tuple[int, str], ...] = (
    (0x1, "ERR_K_INVALID"),
    (0x2, "ERR_N_INVALID"),
    (0x4, "ERR_G_MISMATCH"),
    (0x8, "ERR_ADDR_ALIGN"),
)

_ETHERBONE_MAX_BURST_WORDS = 64  # <= 256 B per packet: fits serial2tcp databuf[2048] & UARTBone


class AccelHardwareError(RuntimeError):
    """Raised when TernaryAccel sets status.error == 1 on launch/wait."""


@dataclass(frozen=True)
class PerfMetrics:
    """Cycle-accurate CSR telemetry read after a TernaryAccel GEMV run."""

    cycles: int
    bus_beats: int
    sat_rows: int
    req_saturated: bool
    bytes_moved: int
    bus_utilization: float
    mb_per_sec: float


@dataclass(frozen=True)
class CSRRegSpec:
    name: str
    addr: int
    access: str  # "rw" or "ro"


class CSRLayout:
    """Register & memory map loaded from LiteX csr.json (`reports/csr.json`)."""

    def __init__(self, csr_json_path: Path = DEFAULT_CSR_JSON) -> None:
        raw = json.loads(csr_json_path.read_text(encoding="utf-8"))
        self.registers: dict[str, CSRRegSpec] = {}
        self.short_aliases: dict[str, str] = {}
        for full_name, meta in raw["csr_registers"].items():
            spec = CSRRegSpec(
                name=full_name,
                addr=int(meta["addr"]),
                access=str(meta.get("type", "rw")),
            )
            self.registers[full_name] = spec
            if full_name.startswith("ternary_accel_"):
                short = full_name[len("ternary_accel_") :]
                self.short_aliases[short] = full_name

        mems = raw.get("memories", {})
        main_ram = mems.get("main_ram", {"base": 0x4000_0000, "size": 0x0040_0000})
        self.main_ram_base: int = int(main_ram["base"])
        self.main_ram_size: int = int(main_ram["size"])
        consts = raw.get("constants", {})
        self.clk_freq_hz: float = float(consts.get("config_clock_frequency") or 1_000_000)

    def resolve(self, name: str, caller: str) -> CSRRegSpec:
        canonical = self.short_aliases.get(name, name)
        spec = self.registers.get(canonical)
        if spec is None:
            raise KeyError(
                f"[setun.runtime.driver:{caller}] unknown CSR register {name!r} "
                f"(expected a register in csr.json such as 'ctrl', 'status', or 'ctrl_scratch')."
            )
        return spec


class AsyncBusBackend(Protocol):
    """Minimal 32-bit byte-addressed bus transport shared across cocotb, Etherbone, and ref."""

    async def read32(self, addr: int) -> int: ...

    async def write32(self, addr: int, value: int) -> None: ...

    async def read_words(self, addr: int, count: int) -> list[int]: ...

    async def write_words(self, addr: int, words: Sequence[int]) -> None: ...


class CocotbWishboneBackend:
    """Cocotb-direct 32-bit Wishbone master driving SetunSimSoC.host_wb."""

    def __init__(self, dut, clk_name: str = "sys_clk", prefix: str = "host_wb") -> None:
        self._dut = dut
        self._clk = getattr(dut, clk_name)
        self._adr = getattr(dut, f"{prefix}_adr")
        self._dat_w = getattr(dut, f"{prefix}_dat_w")
        self._dat_r = getattr(dut, f"{prefix}_dat_r")
        self._sel = getattr(dut, f"{prefix}_sel")
        self._cyc = getattr(dut, f"{prefix}_cyc")
        self._stb = getattr(dut, f"{prefix}_stb")
        self._we = getattr(dut, f"{prefix}_we")
        self._cti = getattr(dut, f"{prefix}_cti")
        self._bte = getattr(dut, f"{prefix}_bte")
        self._ack = getattr(dut, f"{prefix}_ack")

    async def read32(self, addr: int) -> int:
        from cocotb.triggers import ReadOnly, RisingEdge

        self._adr.value = (addr >> 2) & 0x3FFF_FFFF
        self._dat_w.value = 0
        self._sel.value = 0xF
        self._we.value = 0
        self._cti.value = 0
        self._bte.value = 0
        self._cyc.value = 1
        self._stb.value = 1
        while True:
            await ReadOnly()
            if int(self._ack.value) == 1:
                val = int(self._dat_r.value) & 0xFFFF_FFFF
                await RisingEdge(self._clk)
                break
            await RisingEdge(self._clk)
        self._cyc.value = 0
        self._stb.value = 0
        await RisingEdge(self._clk)
        return val

    async def write32(self, addr: int, value: int) -> None:
        from cocotb.triggers import ReadOnly, RisingEdge

        self._adr.value = (addr >> 2) & 0x3FFF_FFFF
        self._dat_w.value = int(value) & 0xFFFF_FFFF
        self._sel.value = 0xF
        self._we.value = 1
        self._cti.value = 0
        self._bte.value = 0
        self._cyc.value = 1
        self._stb.value = 1
        while True:
            await ReadOnly()
            if int(self._ack.value) == 1:
                await RisingEdge(self._clk)
                break
            await RisingEdge(self._clk)
        self._cyc.value = 0
        self._stb.value = 0
        self._we.value = 0
        await RisingEdge(self._clk)

    async def read_words(self, addr: int, count: int) -> list[int]:
        out: list[int] = []
        for idx in range(count):
            out.append(await self.read32(addr + 4 * idx))
        return out

    async def write_words(self, addr: int, words: Sequence[int]) -> None:
        for idx, word in enumerate(words):
            await self.write32(addr + 4 * idx, int(word))


class EtherboneBackend:
    """LiteX RemoteClient Etherbone backend (chunked & flow-controlled for serial2tcp safety)."""

    def __init__(self, client, poll_sleep_s: float = 0.0002) -> None:
        self._client = client
        self._poll_sleep_s = poll_sleep_s
        self._unacked_words = 0

    def _record_writes(self, n_words: int, fence_addr: int) -> None:
        self._unacked_words += n_words
        if self._unacked_words >= _ETHERBONE_MAX_BURST_WORDS:
            self._client.read(fence_addr)
            self._unacked_words = 0

    async def read32(self, addr: int) -> int:
        self._unacked_words = 0
        res = int(self._client.read(addr)) & 0xFFFF_FFFF
        if self._poll_sleep_s > 0.0:
            time.sleep(self._poll_sleep_s)
        return res

    async def write32(self, addr: int, value: int) -> None:
        self._client.write(addr, int(value) & 0xFFFF_FFFF)
        self._record_writes(1, addr)

    async def read_words(self, addr: int, count: int) -> list[int]:
        self._unacked_words = 0
        words: list[int] = []
        offset = 0
        while offset < count:
            chunk = min(_ETHERBONE_MAX_BURST_WORDS, count - offset)
            res = self._client.read(addr + 4 * offset, length=chunk)
            if isinstance(res, int):
                words.append(res & 0xFFFF_FFFF)
            else:
                words.extend(int(w) & 0xFFFF_FFFF for w in res)
            offset += chunk
        return words

    async def write_words(self, addr: int, words: Sequence[int]) -> None:
        total = len(words)
        offset = 0
        while offset < total:
            chunk = [
                int(w) & 0xFFFF_FFFF for w in words[offset : offset + _ETHERBONE_MAX_BURST_WORDS]
            ]
            chunk_addr = addr + 4 * offset
            self._client.write(chunk_addr, chunk)
            self._record_writes(len(chunk), chunk_addr)
            offset += len(chunk)


class RefBusBackend:
    """Bit-exact in-memory behavioral TernaryAccel + main_ram backend for fast unit tests."""

    def __init__(
        self,
        rows: int = 4,
        lanes: int = 64,
        group_size: int = 128,
        beat_bits: int = 256,
        max_k: int = 2048,
        max_rows: int = 2048,
        csr_layout: CSRLayout | None = None,
    ) -> None:
        self.layout = csr_layout or CSRLayout()
        self.rows = rows
        self.lanes = lanes
        self.group_size = group_size
        self.beat_bits = beat_bits
        self.max_k = max_k
        self.max_rows = max_rows
        self._ram = bytearray(self.layout.main_ram_size)
        self._csrs: dict[int, int] = {}
        self._init_csrs()

    def _reg_addr(self, name: str) -> int:
        return self.layout.resolve(name, "RefBusBackend").addr

    def _init_csrs(self) -> None:
        self._csrs[self._reg_addr("ctrl")] = 0x1
        self._csrs[self._reg_addr("cfg")] = (1 << 8) | 0
        self._csrs[self._reg_addr("info")] = (
            ((self.beat_bits & 0xFFF) << 20)
            | ((self.group_size & 0xFFF) << 8)
            | (self.lanes & 0xFF)
        )
        self._csrs[self._reg_addr("status")] = 0xC  # trit_ready | scale_ready
        self._csrs[self._reg_addr("tile_info")] = (
            ((self.group_size & 0xFFFF) << 16) | ((self.lanes & 0xFF) << 8) | (self.rows & 0xFF)
        )
        self._csrs[self._reg_addr("buf_caps")] = ((self.max_rows & 0xFFFF) << 16) | (
            self.max_k & 0xFFFF
        )
        self._csrs[self._reg_addr("dim_k")] = ((self.group_size & 0xFFFF) << 16) | (
            self.group_size & 0xFFFF
        )
        self._csrs[self._reg_addr("req_cfg")] = 1

    def _ram_slice(self, addr: int, n_bytes: int) -> slice:
        off = addr - self.layout.main_ram_base
        return slice(off, off + n_bytes)

    def _execute_launch(self, ctrl_val: int) -> None:
        cfg = self._csrs.get(self._reg_addr("cfg"), 0)
        shift = cfg & 0x3F
        num_rows = (cfg >> 8) & 0xFFFF
        dim_k = self._csrs.get(self._reg_addr("dim_k"), 0)
        k_cols = dim_k & 0xFFFF
        g_cfg = (dim_k >> 16) & 0xFFFF

        w_addr = self._csrs.get(self._reg_addr("weight_addr"), 0)
        s_addr = self._csrs.get(self._reg_addr("scale_addr"), 0)
        a_addr = self._csrs.get(self._reg_addr("act_addr"), 0)
        o_addr = self._csrs.get(self._reg_addr("out_addr"), 0)

        err = 0
        if k_cols == 0 or k_cols > self.max_k:
            err |= 0x1
        if num_rows == 0 or num_rows > self.max_rows:
            err |= 0x2
        if g_cfg != self.group_size:
            err |= 0x4
        if (
            (w_addr & 0x3)
            or (s_addr & 0x3)
            or (a_addr & 0x3)
            or (o_addr & 0x3)
            or ((w_addr == 0) != (s_addr == 0))
        ):
            err |= 0x8

        if err != 0:
            self._csrs[self._reg_addr("status")] = (err << 8) | (1 << 4)
            self._csrs[self._reg_addr("perf_cycles")] = 0
            self._csrs[self._reg_addr("perf_bus_beats")] = 0
            self._csrs[self._reg_addr("perf_sat_rows")] = 0
            return

        n_groups = (k_cols + self.group_size - 1) // self.group_size
        k_pad = n_groups * self.group_size
        trit_bytes = num_rows * (k_pad // 4)
        scale_bytes = num_rows * n_groups * 2

        packed_trits = np.frombuffer(
            bytes(self._ram[self._ram_slice(w_addr, trit_bytes)]), dtype=np.uint8
        ).copy()
        scales_u16 = (
            np.frombuffer(bytes(self._ram[self._ram_slice(s_addr, scale_bytes)]), dtype="<u2")
            .copy()
            .reshape(num_rows, n_groups)
        )
        acts_i8 = np.frombuffer(
            bytes(self._ram[self._ram_slice(a_addr, k_cols)]), dtype=np.int8
        ).copy()

        proj = PackedTernaryTensor(
            packed_trits=packed_trits,
            scales=scales_u16,
            n_rows=num_rows,
            n_cols=k_cols,
            group_size=self.group_size,
            shift=shift,
        )
        wide = ternary_matvec_integer(proj, acts_i8, lanes=self.lanes)

        req_enable = bool((ctrl_val >> 3) & 0x1)
        req_sym = bool((ctrl_val >> 4) & 0x1)
        sat_count = 0
        if req_enable:
            req_cfg = self._csrs.get(self._reg_addr("req_cfg"), 1)
            req_scale = req_cfg & 0xFFFF
            req_shift = (req_cfg >> 16) & 0x3F
            req_pairs = [
                requantize_int8(int(v), scale_u16=req_scale, shift=req_shift, symmetric=req_sym)
                for v in wide
            ]
            out_i8 = np.array([p[0] for p in req_pairs], dtype=np.int8)
            sat_count = sum(1 for p in req_pairs if p[1])
            if o_addr != 0:
                self._ram[self._ram_slice(o_addr, num_rows)] = out_i8.tobytes()
            out_beats = (num_rows + 3) // 4 if o_addr != 0 else 0
        else:
            out_i32 = wide.astype("<i4")
            if o_addr != 0:
                self._ram[self._ram_slice(o_addr, num_rows * 4)] = out_i32.tobytes()
            out_beats = num_rows if o_addr != 0 else 0

        read_beats = (k_pad // 4) + ((scale_bytes + 3) // 4) + (trit_bytes // 4)
        total_beats = read_beats + out_beats
        total_cycles = total_beats + 16

        self._csrs[self._reg_addr("perf_cycles")] = total_cycles
        self._csrs[self._reg_addr("perf_bus_beats")] = total_beats
        self._csrs[self._reg_addr("perf_sat_rows")] = sat_count
        status = (num_rows << 16) | ((1 if sat_count > 0 else 0) << 5) | 0xC | 0x2
        self._csrs[self._reg_addr("status")] = status

    async def read32(self, addr: int) -> int:
        if (
            self.layout.main_ram_base
            <= addr
            < self.layout.main_ram_base + self.layout.main_ram_size
        ):
            raw = bytes(self._ram[self._ram_slice(addr, 4)])
            return int(struct.unpack("<I", raw)[0])
        return int(self._csrs.get(addr, 0)) & 0xFFFF_FFFF

    async def write32(self, addr: int, value: int) -> None:
        val = int(value) & 0xFFFF_FFFF
        if (
            self.layout.main_ram_base
            <= addr
            < self.layout.main_ram_base + self.layout.main_ram_size
        ):
            self._ram[self._ram_slice(addr, 4)] = struct.pack("<I", val)
            return
        self._csrs[addr] = val
        if addr == self._reg_addr("ctrl"):
            if (val >> 1) & 0x1:  # clear
                self._csrs[self._reg_addr("status")] = 0xC
                self._csrs[self._reg_addr("perf_cycles")] = 0
                self._csrs[self._reg_addr("perf_bus_beats")] = 0
                self._csrs[self._reg_addr("perf_sat_rows")] = 0
            if (val >> 2) & 0x1:  # start
                self._execute_launch(val)

    async def read_words(self, addr: int, count: int) -> list[int]:
        return [await self.read32(addr + 4 * i) for i in range(count)]

    async def write_words(self, addr: int, words: Sequence[int]) -> None:
        for i, w in enumerate(words):
            await self.write32(addr + 4 * i, int(w))


class AsyncAccelDriver:
    """Async host driver implementing write_csr, dma_write, dma_read, launch, wait, and perf."""

    def __init__(
        self,
        backend: AsyncBusBackend,
        csr_layout: CSRLayout | None = None,
    ) -> None:
        self.backend = backend
        self.layout = csr_layout or CSRLayout()
        self._last_launch_bytes: int = 0

    @property
    def main_ram_base(self) -> int:
        return self.layout.main_ram_base

    @property
    def main_ram_size(self) -> int:
        return self.layout.main_ram_size

    def _check_dma_bounds(self, addr: int, n_bytes: int, caller: str) -> None:
        if addr & 0x3:
            raise ValueError(
                f"[setun.runtime.driver:{caller}] DMA address must be 4-byte aligned "
                f"(expected addr[1:0] == 0, got 0x{addr:08x})."
            )
        if (
            n_bytes < 0
            or addr < self.main_ram_base
            or (addr + n_bytes) > (self.main_ram_base + self.main_ram_size)
        ):
            raise ValueError(
                f"[setun.runtime.driver:{caller}] range [0x{addr:08x}, 0x{addr + n_bytes:08x}) "
                f"is out of main_ram bounds [0x{self.main_ram_base:08x}, "
                f"0x{self.main_ram_base + self.main_ram_size:08x})."
            )

    async def read_csr(self, name: str) -> int:
        spec = self.layout.resolve(name, "read_csr")
        return await self.backend.read32(spec.addr)

    async def write_csr(self, name: str, value: int) -> None:
        spec = self.layout.resolve(name, "write_csr")
        if spec.access == "ro":
            raise PermissionError(
                f"[setun.runtime.driver:write_csr] CSR {spec.name!r} at 0x{spec.addr:08x} "
                f"is read-only (expected access='rw', got 'ro')."
            )
        if not (0 <= int(value) <= 0xFFFF_FFFF):
            raise ValueError(
                f"[setun.runtime.driver:write_csr] CSR {spec.name!r} requires a 32-bit unsigned "
                f"integer in [0, 0xffffffff], got {value}."
            )
        await self.backend.write32(spec.addr, int(value))

    async def dma_write(self, addr: int, data: bytes | bytearray | memoryview | np.ndarray) -> None:
        raw = data.tobytes() if isinstance(data, np.ndarray) else bytes(data)
        self._check_dma_bounds(addr, len(raw), "dma_write")
        if not raw:
            return
        rem = len(raw) % 4
        padded = raw if rem == 0 else raw + (b"\x00" * (4 - rem))
        words = np.frombuffer(padded, dtype="<u4").tolist()
        await self.backend.write_words(addr, words)

    async def dma_read(self, addr: int, n_bytes: int) -> bytes:
        self._check_dma_bounds(addr, n_bytes, "dma_read")
        if n_bytes == 0:
            return b""
        n_words = (n_bytes + 3) // 4
        words = await self.backend.read_words(addr, n_words)
        raw = np.asarray(words, dtype="<u4").tobytes()
        return raw[:n_bytes]

    async def launch(
        self,
        *,
        num_rows: int,
        k_cols: int,
        group_size: int,
        shift: int,
        weight_addr: int = 0,
        scale_addr: int = 0,
        act_addr: int = 0,
        out_addr: int = 0,
        act_scale: float = 1.0,
        req_enable: bool = False,
        req_sym: bool = False,
        req_scale_u16: int = 1,
        req_shift: int = 0,
    ) -> None:
        cfg_word = ((int(num_rows) & 0xFFFF) << 8) | (int(shift) & 0x3F)
        dim_word = ((int(group_size) & 0xFFFF) << 16) | (int(k_cols) & 0xFFFF)
        act_scale_bits = struct.unpack("<I", struct.pack("<f", float(act_scale)))[0]

        await self.write_csr("cfg", cfg_word)
        await self.write_csr("dim_k", dim_word)
        await self.write_csr("weight_addr", weight_addr)
        await self.write_csr("scale_addr", scale_addr)
        await self.write_csr("act_addr", act_addr)
        await self.write_csr("out_addr", out_addr)
        await self.write_csr("act_scale", act_scale_bits)
        if req_enable:
            req_word = ((int(req_shift) & 0x3F) << 16) | (int(req_scale_u16) & 0xFFFF)
            await self.write_csr("req_cfg", req_word)

        n_groups = (max(k_cols, 1) + max(group_size, 1) - 1) // max(group_size, 1)
        k_pad = n_groups * max(group_size, 1)
        w_bytes = (num_rows * (k_pad // 4)) if weight_addr != 0 else 0
        s_bytes = (num_rows * n_groups * 2) if scale_addr != 0 else 0
        a_bytes = k_pad if act_addr != 0 else 0
        o_bytes = (num_rows if req_enable else num_rows * 4) if out_addr != 0 else 0
        self._last_launch_bytes = w_bytes + s_bytes + a_bytes + o_bytes

        ctrl_word = (
            0x1  # enable
            | (1 << 2)  # start
            | ((1 if req_enable else 0) << 3)
            | ((1 if req_sym else 0) << 4)
        )
        await self.write_csr("ctrl", ctrl_word)

    async def wait(self, max_polls: int = 100_000) -> int:
        last_status = 0
        for _ in range(max_polls):
            last_status = await self.read_csr("status")
            if (last_status >> 4) & 0x1:
                err_code = (last_status >> 8) & 0xF
                names = [label for bit, label in ERR_FLAGS if err_code & bit] or ["UNKNOWN_ERR"]
                raise AccelHardwareError(
                    f"[setun.runtime.driver:wait] TernaryAccel hardware legality error: "
                    f"status=0x{last_status:08x}, err_code=0x{err_code:x} ({'|'.join(names)})."
                )
            busy = last_status & 0x1
            done = (last_status >> 1) & 0x1
            if done == 1 and busy == 0:
                return (last_status >> 16) & 0xFFFF

        rows_so_far = (last_status >> 16) & 0xFFFF
        raise TimeoutError(
            f"[setun.runtime.driver:wait] timed out after {max_polls} status polls "
            f"(last status=0x{last_status:08x}, rows_completed={rows_so_far})."
        )

    async def perf(self, clk_freq_hz: float | None = None) -> PerfMetrics:
        cycles = await self.read_csr("perf_cycles")
        beats = await self.read_csr("perf_bus_beats")
        sat_rows = await self.read_csr("perf_sat_rows")
        status = await self.read_csr("status")
        req_sat = bool((status >> 5) & 0x1)

        freq = float(clk_freq_hz) if clk_freq_hz is not None else self.layout.clk_freq_hz
        util = (float(beats) / float(cycles)) if cycles > 0 else 0.0
        mb_s = (
            (float(self._last_launch_bytes) * freq) / (float(cycles) * 1e6) if cycles > 0 else 0.0
        )
        return PerfMetrics(
            cycles=cycles,
            bus_beats=beats,
            sat_rows=sat_rows,
            req_saturated=req_sat,
            bytes_moved=self._last_launch_bytes,
            bus_utilization=round(util, 6),
            mb_per_sec=round(mb_s, 4),
        )


def _run_immediate(coro):
    """Execute a non-yielding host coroutine synchronously without asyncio loop overhead."""
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    raise RuntimeError(
        "[setun.runtime.driver:_run_immediate] coroutine yielded unexpectedly on sync driver; "
        "use AsyncAccelDriver inside cocotb."
    )


class AccelDriver:
    """Synchronous host driver facade over AsyncAccelDriver for CLI & Etherbone/ref sessions."""

    def __init__(self, async_driver: AsyncAccelDriver) -> None:
        self.async_driver = async_driver

    @property
    def main_ram_base(self) -> int:
        return self.async_driver.main_ram_base

    @property
    def main_ram_size(self) -> int:
        return self.async_driver.main_ram_size

    def read_csr(self, name: str) -> int:
        return int(_run_immediate(self.async_driver.read_csr(name)))

    def write_csr(self, name: str, value: int) -> None:
        _run_immediate(self.async_driver.write_csr(name, value))

    def dma_write(self, addr: int, data: bytes | bytearray | memoryview | np.ndarray) -> None:
        _run_immediate(self.async_driver.dma_write(addr, data))

    def dma_read(self, addr: int, n_bytes: int) -> bytes:
        return bytes(_run_immediate(self.async_driver.dma_read(addr, n_bytes)))

    def launch(self, **kwargs) -> None:
        _run_immediate(self.async_driver.launch(**kwargs))

    def wait(self, max_polls: int = 100_000) -> int:
        return int(_run_immediate(self.async_driver.wait(max_polls=max_polls)))

    def perf(self, clk_freq_hz: float | None = None) -> PerfMetrics:
        return _run_immediate(self.async_driver.perf(clk_freq_hz=clk_freq_hz))


def _alloc_free_tcp_ports(count: int) -> list[int]:
    socks: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            ports.append(int(s.getsockname()[1]))
            socks.append(s)
        return ports
    finally:
        for s in socks:
            s.close()


def _wait_tcp_listening(port: int, proc: subprocess.Popen, timeout_s: float = 15.0) -> None:
    """Wait until `port` enters TCP_LISTEN (0A) in /proc/net/tcp without connecting a client."""
    hex_port = f":{port:04X}"
    deadline = time.monotonic() + timeout_s
    proc_tcp = Path("/proc/net/tcp")
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                f"[setun.runtime.driver:_wait_tcp_listening] subprocess exited early "
                f"(rc={proc.returncode}) while waiting for TCP port {port}:\n{out}"
            )
        if proc_tcp.exists():
            for line in proc_tcp.read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 4 and parts[1].endswith(hex_port) and parts[3] == "0A":
                    return
        time.sleep(0.02)
    raise TimeoutError(
        f"[setun.runtime.driver:_wait_tcp_listening] timed out after {timeout_s:.1f}s "
        f"waiting for port {port} to listen."
    )


def _terminate_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)


_RAND_RESET_CPP = """#include "verilated.h"
#include <cstdlib>
namespace {
struct SetunRandResetInit {
  SetunRandResetInit() {
    Verilated::randReset(2);
    if (const char* s = std::getenv("SETUN_SIM_SEED")) Verilated::randSeed(std::atoi(s));
  }
} g_setun_rand_reset_init;
}
"""


@contextmanager
def _open_litex_sim_driver(
    *,
    rows: int = 4,
    lanes: int = 64,
    group_size: int = 128,
    beat_bits: int = 256,
    max_k: int = 2048,
    max_rows: int = 2048,
    main_ram_size: int = 0x400000,
    ram_init: Path | str | None = None,
    rand_reset_seed: int | None = None,
    ram_wait_jitter_seed: int = 0,
) -> Iterator[AccelDriver]:
    """Build/cache SetunSimSoC(with_uartbone=True) and attach litex_server + RemoteClient."""
    from litex.build.sim.config import SimConfig
    from litex.soc.integration.builder import Builder
    from litex.tools.litex_client import RemoteClient

    from targets.sim import SetunSimSoC

    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    h = hashlib.sha256()
    for src_rel in ("targets/sim.py", "setun/soc/ternary_accel.py"):
        h.update((REPO_ROOT / src_rel).read_bytes())
    for acc_py in sorted((REPO_ROOT / "setun/accelerators").glob("*.py")):
        h.update(acc_py.read_bytes())
    xtag = (
        f"_J{ram_wait_jitter_seed}_RR{int(rand_reset_seed is not None)}"
        if (ram_wait_jitter_seed != 0 or rand_reset_seed is not None)
        else ""
    )
    cfg_tag = (
        f"R{rows}_L{lanes}_G{group_size}_B{beat_bits}_K{max_k}_N{max_rows}_M{main_ram_size}{xtag}"
    )
    h.update(cfg_tag.encode("utf-8"))
    if ram_init is not None:
        h.update(Path(ram_init).read_bytes())

    build_dir = (
        REPO_ROOT / "sim_build" / ".litex_sim_cache" / worker / f"{cfg_tag}_{h.hexdigest()[:12]}"
    )
    gateware_dir = build_dir / "gateware"
    vsim_bin = gateware_dir / "obj_dir" / "Vsim"
    csr_json_path = build_dir / "csr.json"

    uart_port, eb_port = _alloc_free_tcp_ports(2)

    if not vsim_bin.exists() or not csr_json_path.exists():
        build_dir.mkdir(parents=True, exist_ok=True)
        soc = SetunSimSoC(
            with_accel_stub=True,
            with_ternary_accel=True,
            with_uartbone=True,
            accel_rows=rows,
            accel_lanes=lanes,
            accel_group_size=group_size,
            accel_beat_bits=beat_bits,
            accel_max_k=max_k,
            accel_max_rows=max_rows,
            ram_init=ram_init,
            cpu_type=None,
            integrated_rom_size=0,
            integrated_main_ram_size=main_ram_size,
            ram_wait_jitter_seed=ram_wait_jitter_seed,
            bus_bursting=False,
        )
        sim_config = SimConfig()
        sim_config.add_clocker("sys_clk", freq_hz=soc.clk_freq)
        sim_config.add_module("serial2tcp", "serial", args={"port": str(uart_port)})
        builder = Builder(
            soc,
            output_dir=str(build_dir),
            csr_json=str(csr_json_path),
            compile_software=False,
            compile_gateware=True,
        )
        builder.build(sim_config=sim_config, run=False)
        if rand_reset_seed is not None:
            (gateware_dir / "rand_reset_init.cpp").write_text(_RAND_RESET_CPP, "utf-8")
            with (gateware_dir / "variables.mak").open("a", encoding="utf-8") as vf:
                vf.write(
                    "\nUSER_CPP_SRCS += rand_reset_init.cpp\n"
                    "INC_DIR += --x-initial unique --x-assign unique\n"
                )
        build_env = dict(os.environ)
        build_env["JOBS"] = "1"
        subprocess.run(
            ["bash", "build_sim.sh"],
            cwd=str(gateware_dir),
            env=build_env,
            check=True,
        )
    else:
        # Update only serial2tcp port in cached sim_config.js without recompiling Vsim C++
        sim_cfg_js = gateware_dir / "sim_config.js"
        cfg_list = json.loads(sim_cfg_js.read_text(encoding="utf-8"))
        for mod in cfg_list:
            if isinstance(mod, dict) and mod.get("module") == "serial2tcp":
                mod["args"]["port"] = str(uart_port)
        sim_cfg_js.write_text(json.dumps(cfg_list, indent=4), encoding="utf-8")

    vsim_proc: subprocess.Popen | None = None
    srv_proc: subprocess.Popen | None = None
    client: RemoteClient | None = None
    try:
        vsim_env = dict(os.environ)
        if rand_reset_seed is not None:
            vsim_env["SETUN_SIM_SEED"] = str(rand_reset_seed)
        vsim_proc = subprocess.Popen(
            [str(vsim_bin)],
            cwd=str(gateware_dir),
            env=vsim_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_tcp_listening(uart_port, vsim_proc, timeout_s=20.0)

        srv_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "litex.tools.litex_server",
                "--uart",
                "--uart-port",
                f"socket://127.0.0.1:{uart_port}",
                "--bind-ip",
                "127.0.0.1",
                "--bind-port",
                str(eb_port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_tcp_listening(eb_port, srv_proc, timeout_s=15.0)

        client = RemoteClient(host="127.0.0.1", port=eb_port, timeout=5.0)
        client.open()
        layout = CSRLayout(csr_json_path)
        yield AccelDriver(AsyncAccelDriver(EtherboneBackend(client), csr_layout=layout))
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        _terminate_proc(srv_proc)
        _terminate_proc(vsim_proc)


def add_backend_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = "ref",
) -> None:
    """Register canonical `--backend ref|sim|board` CLI flag (Task 3.3a)."""
    parser.add_argument(
        "--backend",
        choices=VALID_BACKENDS,
        default=default,
        help="Execution backend: 'ref' (golden Python), 'sim' (LiteX SimSoC), or 'board' (S3 FPGA)",
    )


@contextmanager
def open_backend(
    backend: str,
    *,
    rows: int = 4,
    lanes: int = 64,
    group_size: int = 128,
    beat_bits: int = 256,
    max_k: int = 2048,
    max_rows: int = 2048,
    main_ram_size: int = 0x400000,
    ram_init: Path | str | None = None,
    rand_reset_seed: int | None = None,
    ram_wait_jitter_seed: int = 0,
) -> Iterator[AccelDriver]:
    """Open an AccelDriver session for `backend` in ('ref', 'sim', 'board')."""
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"[setun.runtime.driver:open_backend] Unsupported backend {backend!r}: "
            f"expected one of {VALID_BACKENDS}."
        )
    if backend == "board":
        raise NotImplementedError(
            "[setun.runtime.driver:open_backend] '--backend board' hardware transport is "
            "scheduled for Season 3 Task 3.3b (FPGA board bring-up)."
        )
    if backend == "ref":
        ref_bus = RefBusBackend(
            rows=rows,
            lanes=lanes,
            group_size=group_size,
            beat_bits=beat_bits,
            max_k=max_k,
            max_rows=max_rows,
        )
        yield AccelDriver(AsyncAccelDriver(ref_bus, csr_layout=ref_bus.layout))
        return

    with _open_litex_sim_driver(
        rows=rows,
        lanes=lanes,
        group_size=group_size,
        beat_bits=beat_bits,
        max_k=max_k,
        max_rows=max_rows,
        main_ram_size=main_ram_size,
        ram_init=ram_init,
        rand_reset_seed=rand_reset_seed,
        ram_wait_jitter_seed=ram_wait_jitter_seed,
    ) as sim_drv:
        yield sim_drv
