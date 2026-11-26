"""Hybrid M0 decode engine executing linear GEMVs via AccelDriver and host non-linear ops."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from setun.ref.format import PackedTernaryTensor
from setun.ref.m0 import (
    KVCache,
    M0Model,
    TraceHook,
    apply_rope,
    quantize_activation_int8,
    rms_norm,
    silu,
    softmax,
)
from setun.runtime.driver import AccelDriver, open_backend

PROJ_NAMES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _align16(addr: int) -> int:
    return (addr + 15) & ~15


@dataclass(frozen=True)
class _StagedProj:
    w_addr: int
    s_addr: int
    proj: PackedTernaryTensor


@dataclass(frozen=True)
class HybridDecodeSummary:
    """Aggregated hardware CSR cycle and bandwidth metrics across a hybrid decode run."""

    backend_clk_hz: float
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    forward_steps: int
    gemv_calls: int
    total_cycles: int
    cycles_per_token: float
    total_bus_beats: int
    bus_utilization: float
    total_bytes_moved: int
    mb_per_sec: float


@contextmanager
def open_m0_backend(
    model: M0Model,
    backend: str,
    *,
    rows: int = 4,
    rand_reset_seed: int | None = None,
    ram_wait_jitter_seed: int = 0,
) -> Iterator[AccelDriver]:
    """Open an AccelDriver sized to `model.config` dimensions and group/lane geometry."""
    cfg = model.config
    projs = [getattr(ly, p) for ly in model.layers for p in PROJ_NAMES]
    max_k_raw = max(p.n_cols for p in projs)
    max_k_pad = ((max_k_raw + cfg.group_size - 1) // cfg.group_size) * cfg.group_size
    max_k = max(256, max_k_pad)
    max_rows = max(256, max(p.n_rows for p in projs))
    beat_bits = max(64, 2 * cfg.lanes)

    with open_backend(
        backend,
        rows=rows,
        lanes=cfg.lanes,
        group_size=cfg.group_size,
        beat_bits=beat_bits,
        max_k=max_k,
        max_rows=max_rows,
        rand_reset_seed=rand_reset_seed,
        ram_wait_jitter_seed=ram_wait_jitter_seed,
    ) as drv:
        yield drv


class HybridM0Runner:
    """Stages M0 weights in SimSoC/board RAM once and runs hybrid token decoding."""

    def __init__(
        self,
        model: M0Model,
        driver: AccelDriver,
        *,
        clk_freq_hz: float | None = None,
        gemv_fault_hook: Callable[[int, str, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.model = model
        self.driver = driver
        self.gemv_fault_hook = gemv_fault_hook
        self.clk_freq_hz = (
            float(clk_freq_hz)
            if clk_freq_hz is not None
            else driver.async_driver.layout.clk_freq_hz
        )
        self.forward_steps: int = 0
        self.gemv_calls: int = 0
        self.total_cycles: int = 0
        self.total_bus_beats: int = 0
        self.total_bytes_moved: int = 0
        self._staged: dict[str, _StagedProj] = {}

        base = driver.main_ram_base
        cursor = base
        pending_writes: list[tuple[int, bytes]] = []
        max_k_pad = 0
        max_n_rows = 0

        for l_idx, layer in enumerate(model.layers):
            for proj_name in PROJ_NAMES:
                proj: PackedTernaryTensor = getattr(layer, proj_name)
                w_bytes = np.ascontiguousarray(proj.packed_trits, dtype=np.uint8).tobytes()
                s_bytes = np.ascontiguousarray(proj.scales, dtype="<u2").tobytes()
                w_addr = cursor
                cursor = _align16(cursor + len(w_bytes))
                s_addr = cursor
                cursor = _align16(cursor + len(s_bytes))
                pending_writes.append((w_addr, w_bytes))
                pending_writes.append((s_addr, s_bytes))
                self._staged[f"layers.{l_idx}.{proj_name}"] = _StagedProj(w_addr, s_addr, proj)

                n_groups = (proj.n_cols + proj.group_size - 1) // proj.group_size
                max_k_pad = max(max_k_pad, n_groups * proj.group_size)
                max_n_rows = max(max_n_rows, proj.n_rows)

        self.act_addr = cursor
        cursor = _align16(cursor + max_k_pad)
        self.out_addr = cursor
        cursor = _align16(cursor + max_n_rows * 4)

        needed = cursor - base
        if needed > driver.main_ram_size:
            raise ValueError(
                f"[setun.runtime.m0_runner] M0 arena ({needed} B) exceeds "
                f"main_ram_size ({driver.main_ram_size} B)."
            )

        for addr, payload in pending_writes:
            driver.dma_write(addr, payload)

    def _linear_hw(self, key: str, x: np.ndarray, weight_unit_scale: float) -> np.ndarray:
        slot = self._staged[key]
        proj = slot.proj
        x_q, act_scale = quantize_activation_int8(x)
        if act_scale == 0.0:
            return np.zeros(proj.n_rows, dtype=np.float32)

        n_groups = (proj.n_cols + proj.group_size - 1) // proj.group_size
        k_pad = n_groups * proj.group_size
        x_pad = np.pad(x_q, (0, k_pad - proj.n_cols))

        self.driver.dma_write(self.act_addr, x_pad.tobytes())
        self.driver.launch(
            num_rows=proj.n_rows,
            k_cols=proj.n_cols,
            group_size=proj.group_size,
            shift=proj.shift,
            weight_addr=slot.w_addr,
            scale_addr=slot.s_addr,
            act_addr=self.act_addr,
            out_addr=self.out_addr,
            act_scale=act_scale,
            req_enable=False,
        )
        self.driver.wait()
        raw_out = self.driver.dma_read(self.out_addr, proj.n_rows * 4)
        y_i32 = np.frombuffer(raw_out, dtype="<i4").copy()
        if self.gemv_fault_hook is not None:
            y_i32 = self.gemv_fault_hook(self.forward_steps, key, y_i32)

        pm = self.driver.perf(clk_freq_hz=self.clk_freq_hz)
        self.gemv_calls += 1
        self.total_cycles += pm.cycles
        self.total_bus_beats += pm.bus_beats
        self.total_bytes_moved += pm.bytes_moved

        return (y_i32.astype(np.float32) * np.float32(act_scale * weight_unit_scale)).astype(
            np.float32
        )

    def forward_token(
        self,
        token_id: int,
        pos: int,
        kv_cache: KVCache,
        trace_hook: TraceHook | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute one hybrid decode step (GEMVs on AccelDriver, norms/RoPE/attn/SwiGLU on host)."""
        cfg = self.model.config
        if not (0 <= token_id < cfg.vocab_size):
            raise ValueError(f"token_id {token_id} out of range [0, {cfg.vocab_size})")

        def _trace(name: str, tensor: np.ndarray) -> None:
            if trace_hook is not None:
                trace_hook(name, tensor.copy())

        h = self.model.tok_embeddings[token_id].astype(np.float32).copy()
        _trace("embed", h)

        kv_repeat = cfg.n_heads // cfg.n_kv_heads
        attn_scale = 1.0 / np.sqrt(float(cfg.head_dim))

        for l_idx, layer in enumerate(self.model.layers):
            prefix = f"layers.{l_idx}"

            h_norm = rms_norm(h, layer.attn_norm, cfg.rms_eps)
            _trace(f"{prefix}.attn_norm", h_norm)

            q_flat = self._linear_hw(f"{prefix}.q_proj", h_norm, layer.weight_unit_scale)
            k_flat = self._linear_hw(f"{prefix}.k_proj", h_norm, layer.weight_unit_scale)
            v_flat = self._linear_hw(f"{prefix}.v_proj", h_norm, layer.weight_unit_scale)
            _trace(f"{prefix}.q_proj", q_flat)
            _trace(f"{prefix}.k_proj", k_flat)
            _trace(f"{prefix}.v_proj", v_flat)

            q = q_flat.reshape(cfg.n_heads, cfg.head_dim)
            k = k_flat.reshape(cfg.n_kv_heads, cfg.head_dim)
            v = v_flat.reshape(cfg.n_kv_heads, cfg.head_dim)

            q = rms_norm(q, layer.q_norm, cfg.rms_eps)
            k = rms_norm(k, layer.k_norm, cfg.rms_eps)
            q = apply_rope(q, pos=pos, theta=cfg.rope_theta)
            k = apply_rope(k, pos=pos, theta=cfg.rope_theta)
            _trace(f"{prefix}.q_rope", q)
            _trace(f"{prefix}.k_rope", k)

            kv_cache.k[l_idx] = np.concatenate([kv_cache.k[l_idx], k[None, :, :]], axis=0)
            kv_cache.v[l_idx] = np.concatenate([kv_cache.v[l_idx], v[None, :, :]], axis=0)

            k_seq = kv_cache.k[l_idx]
            v_seq = kv_cache.v[l_idx]

            attn_heads = np.zeros((cfg.n_heads, cfg.head_dim), dtype=np.float32)
            for h_idx in range(cfg.n_heads):
                kv_idx = h_idx // kv_repeat
                scores = (k_seq[:, kv_idx, :] @ q[h_idx]) * np.float32(attn_scale)
                probs = softmax(scores)
                attn_heads[h_idx] = probs @ v_seq[:, kv_idx, :]

            attn_concat = attn_heads.reshape(-1)
            _trace(f"{prefix}.attn_context", attn_concat)

            attn_out = self._linear_hw(f"{prefix}.o_proj", attn_concat, layer.weight_unit_scale)
            _trace(f"{prefix}.o_proj", attn_out)
            h = (h + attn_out).astype(np.float32)
            _trace(f"{prefix}.post_attn_residual", h)

            ffn_in = rms_norm(h, layer.ffn_norm, cfg.rms_eps)
            _trace(f"{prefix}.ffn_norm", ffn_in)

            gate = self._linear_hw(f"{prefix}.gate_proj", ffn_in, layer.weight_unit_scale)
            up = self._linear_hw(f"{prefix}.up_proj", ffn_in, layer.weight_unit_scale)
            _trace(f"{prefix}.gate_proj", gate)
            _trace(f"{prefix}.up_proj", up)

            hidden_ffn = (silu(gate) * up).astype(np.float32)
            _trace(f"{prefix}.swiglu", hidden_ffn)

            down = self._linear_hw(f"{prefix}.down_proj", hidden_ffn, layer.weight_unit_scale)
            _trace(f"{prefix}.down_proj", down)
            h = (h + down).astype(np.float32)
            _trace(f"{prefix}.post_ffn_residual", h)

        final_hidden = rms_norm(h, self.model.final_norm, cfg.rms_eps)
        _trace("final_norm", final_hidden)

        head_scale = np.float32(1.0 / np.sqrt(float(cfg.d_model)))
        raw_logits = self.model.tok_embeddings.astype(np.float32) @ final_hidden
        logits = (raw_logits * head_scale).astype(np.float32)
        _trace("logits", logits)
        self.forward_steps += 1
        return logits, final_hidden

    def summary(
        self,
        prompt_tokens: Sequence[int],
        generated_tokens: Sequence[int],
    ) -> HybridDecodeSummary:
        cpt = (
            float(self.total_cycles) / float(self.forward_steps) if self.forward_steps > 0 else 0.0
        )
        util = (
            float(self.total_bus_beats) / float(self.total_cycles) if self.total_cycles > 0 else 0.0
        )
        mb_s = (
            (float(self.total_bytes_moved) * self.clk_freq_hz) / (float(self.total_cycles) * 1e6)
            if self.total_cycles > 0
            else 0.0
        )
        return HybridDecodeSummary(
            backend_clk_hz=self.clk_freq_hz,
            prompt_tokens=tuple(int(t) for t in prompt_tokens),
            generated_tokens=tuple(int(t) for t in generated_tokens),
            forward_steps=self.forward_steps,
            gemv_calls=self.gemv_calls,
            total_cycles=self.total_cycles,
            cycles_per_token=round(cpt, 2),
            total_bus_beats=self.total_bus_beats,
            bus_utilization=round(util, 6),
            total_bytes_moved=self.total_bytes_moved,
            mb_per_sec=round(mb_s, 4),
        )

    def generate_greedy(
        self,
        prompt_tokens: Sequence[int],
        max_new_tokens: int,
        *,
        on_token: Callable[[int], None] | None = None,
        trace_hook: TraceHook | None = None,
    ) -> tuple[list[int], np.ndarray, np.ndarray, HybridDecodeSummary]:
        if not prompt_tokens:
            raise ValueError("prompt_tokens must be non-empty")
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

        kv_cache = KVCache.empty(self.model.config)
        pos = 0
        logits = np.zeros(self.model.config.vocab_size, dtype=np.float32)
        hidden = np.zeros(self.model.config.d_model, dtype=np.float32)

        for tok in prompt_tokens:
            logits, hidden = self.forward_token(
                int(tok), pos=pos, kv_cache=kv_cache, trace_hook=trace_hook
            )
            pos += 1

        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_tok = int(np.argmax(logits))
            generated.append(next_tok)
            if on_token is not None:
                on_token(next_tok)
            logits, hidden = self.forward_token(
                next_tok, pos=pos, kv_cache=kv_cache, trace_hook=trace_hook
            )
            pos += 1

        return generated, logits, hidden, self.summary(prompt_tokens, generated)


def run_m0_greedy(
    model: M0Model,
    *,
    backend: str = "ref",
    prompt_tokens: Sequence[int] = (1,),
    max_new_tokens: int = 16,
    clk_freq_hz: float | None = None,
    rand_reset_seed: int | None = None,
    ram_wait_jitter_seed: int = 0,
    on_token: Callable[[int], None] | None = None,
    trace_hook: TraceHook | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray, HybridDecodeSummary]:
    """Open `backend`, stage M0 weights once, and run greedy hybrid token generation."""
    with open_m0_backend(
        model, backend, rand_reset_seed=rand_reset_seed, ram_wait_jitter_seed=ram_wait_jitter_seed
    ) as drv:
        runner = HybridM0Runner(model, drv, clk_freq_hz=clk_freq_hz)
        return runner.generate_greedy(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            on_token=on_token,
            trace_hook=trace_hook,
        )
