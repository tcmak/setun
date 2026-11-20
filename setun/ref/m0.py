"""M0 Golden Integer & FP32 Reference Transformer Decoder (Tasks 1.5 & 1.6)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from setun.ref.format import (
    PackedTernaryTensor,
    deserialize_packed_weights,
    pack_ternary_tensor,
    serialize_packed_weights,
    unpack_ternary_tensor,
)
from setun.ref.ops import round_shift_right

TraceHook = Callable[[str, np.ndarray], None]


@dataclass(frozen=True)
class M0Config:
    """Configuration for the M0 tiny ternary transformer."""

    name: str = "m0-tiny"
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 16
    ffn_dim: int = 128
    vocab_size: int = 256
    max_seq_len: int = 64
    group_size: int = 64
    lanes: int = 16
    shift: int = 8
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0
    tied_embeddings: bool = True
    seed: int = 1958


PRESET_CONFIGS: dict[str, M0Config] = {
    "tiny": M0Config(
        name="m0-tiny",
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        ffn_dim=128,
        vocab_size=256,
        max_seq_len=64,
        group_size=64,
        lanes=16,
        shift=8,
    ),
    "small": M0Config(
        name="m0-small",
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        ffn_dim=256,
        vocab_size=256,
        max_seq_len=64,
        group_size=128,
        lanes=64,
        shift=8,
    ),
}


@dataclass
class M0LayerWeights:
    """Weights for a single M0 transformer decoder layer."""

    attn_norm: np.ndarray  # float32 (d_model,)
    q_norm: np.ndarray  # float32 (head_dim,)
    k_norm: np.ndarray  # float32 (head_dim,)
    ffn_norm: np.ndarray  # float32 (d_model,)
    q_proj: PackedTernaryTensor  # (n_heads * head_dim, d_model)
    k_proj: PackedTernaryTensor  # (n_kv_heads * head_dim, d_model)
    v_proj: PackedTernaryTensor  # (n_kv_heads * head_dim, d_model)
    o_proj: PackedTernaryTensor  # (d_model, n_heads * head_dim)
    gate_proj: PackedTernaryTensor  # (ffn_dim, d_model)
    up_proj: PackedTernaryTensor  # (ffn_dim, d_model)
    down_proj: PackedTernaryTensor  # (d_model, ffn_dim)
    weight_unit_scale: float = 1.0  # converts integer row_dot output to float unit


@dataclass
class M0Model:
    """Complete M0 model weights (packed ternary linear layers + FP32 embeddings/norms)."""

    config: M0Config
    tok_embeddings: np.ndarray  # float32 (vocab_size, d_model), also tied LM head
    final_norm: np.ndarray  # float32 (d_model,)
    layers: list[M0LayerWeights] = field(default_factory=list)


@dataclass
class KVCache:
    """Key-Value cache across layers for autoregressive decoding."""

    k: list[np.ndarray]  # per layer: float32 (seq_len, n_kv_heads, head_dim)
    v: list[np.ndarray]  # per layer: float32 (seq_len, n_kv_heads, head_dim)

    @classmethod
    def empty(cls, config: M0Config) -> KVCache:
        return cls(
            k=[
                np.zeros((0, config.n_kv_heads, config.head_dim), dtype=np.float32)
                for _ in range(config.n_layers)
            ],
            v=[
                np.zeros((0, config.n_kv_heads, config.head_dim), dtype=np.float32)
                for _ in range(config.n_layers)
            ],
        )


def quantize_activation_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-token symmetric absmax quantization to signed int8 [-127, 127] (ADR-002)."""
    x_f32 = np.asarray(x, dtype=np.float32)
    absmax = float(np.max(np.abs(x_f32)))
    if absmax == 0.0:
        return np.zeros_like(x_f32, dtype=np.int8), 0.0
    scale = absmax / 127.0
    q = np.clip(np.round(x_f32 / scale), -127, 127).astype(np.int8)
    return q, scale


def ternary_matvec_integer(
    proj: PackedTernaryTensor,
    x_int8: np.ndarray,
    lanes: int,
) -> np.ndarray:
    """Bit-exact vectorized implementation of row_dot across all N rows of a PackedTernaryTensor."""
    w_trits, scales_u16, shift = unpack_ternary_tensor(proj)
    n_rows, k = w_trits.shape
    g = proj.group_size
    if g % lanes != 0:
        raise ValueError(f"group_size {g} must be divisible by lanes {lanes}")

    n_groups = (k + g - 1) // g
    k_pad = n_groups * g

    w_pad = np.zeros((n_rows, k_pad), dtype=np.int64)
    x_pad = np.zeros(k_pad, dtype=np.int64)
    w_pad[:, :k] = w_trits.astype(np.int64)
    x_pad[:k] = x_int8.astype(np.int64)

    # Group partial sums: shape (n_rows, n_groups)
    w_grouped = w_pad.reshape(n_rows, n_groups, g)
    x_grouped = x_pad.reshape(1, n_groups, g)
    group_sums = np.sum(w_grouped * x_grouped, axis=2)

    # Multiply by uint16 group scale and accumulate across groups in wide fixed-point
    row_accs = np.sum(group_sums * scales_u16.astype(np.int64), axis=1)
    return np.asarray(round_shift_right(row_accs, shift), dtype=np.int64)


def dequantize_packed_tensor_fp32(
    proj: PackedTernaryTensor,
    weight_unit_scale: float,
) -> np.ndarray:
    """Reconstruct the exact effective FP32 weight matrix from a PackedTernaryTensor."""
    w_trits, scales_u16, shift = unpack_ternary_tensor(proj)
    n_rows, k = w_trits.shape
    g = proj.group_size
    n_groups = (k + g - 1) // g
    k_pad = n_groups * g

    w_pad = np.zeros((n_rows, k_pad), dtype=np.float32)
    w_pad[:, :k] = w_trits.astype(np.float32)
    w_grouped = w_pad.reshape(n_rows, n_groups, g)

    # Effective float scale per group matching (sum * scale_u16) >> shift * weight_unit_scale
    s_float = (scales_u16.astype(np.float32) / float(1 << shift)) * float(weight_unit_scale)
    w_scaled = (w_grouped * s_float[:, :, None]).reshape(n_rows, k_pad)[:, :k]
    return w_scaled.astype(np.float32)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Root-Mean-Square Layer Normalization in FP32."""
    x_f32 = np.asarray(x, dtype=np.float32)
    var = np.mean(np.square(x_f32), axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(var + np.float32(eps))
    return (x_f32 * inv_rms * weight.astype(np.float32)).astype(np.float32)


def apply_rope(x: np.ndarray, pos: int, theta: float = 10000.0) -> np.ndarray:
    """Apply Rotary Position Embeddings to shape (n_heads, head_dim)."""
    _, head_dim = x.shape
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / float(head_dim)))
    angles = float(pos) * freqs
    cos = np.cos(angles)[None, :]
    sin = np.sin(angles)[None, :]

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    out = np.empty_like(x, dtype=np.float32)
    out[:, 0::2] = x_even * cos - x_odd * sin
    out[:, 1::2] = x_even * sin + x_odd * cos
    return out


def silu(x: np.ndarray) -> np.ndarray:
    """Sigmoid Linear Unit: x * sigmoid(x)."""
    x_f32 = np.asarray(x, dtype=np.float32)
    return (x_f32 / (1.0 + np.exp(-x_f32))).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    x_f32 = np.asarray(x, dtype=np.float32)
    shifted = x_f32 - np.max(x_f32, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return (exps / np.sum(exps, axis=-1, keepdims=True)).astype(np.float32)


def _linear_step(
    proj: PackedTernaryTensor,
    x: np.ndarray,
    weight_unit_scale: float,
    lanes: int,
    mode: str,
) -> np.ndarray:
    if mode == "integer":
        x_q, act_scale = quantize_activation_int8(x)
        if act_scale == 0.0:
            return np.zeros(proj.n_rows, dtype=np.float32)
        y_int = ternary_matvec_integer(proj, x_q, lanes=lanes)
        return (y_int.astype(np.float32) * np.float32(act_scale * weight_unit_scale)).astype(
            np.float32
        )
    if mode == "fp32":
        w_fp32 = dequantize_packed_tensor_fp32(proj, weight_unit_scale)
        return (w_fp32 @ x.astype(np.float32)).astype(np.float32)
    raise ValueError(f"Unsupported execution mode: {mode!r}")


def forward_token(
    model: M0Model,
    token_id: int,
    pos: int,
    kv_cache: KVCache,
    mode: str = "integer",
    trace_hook: TraceHook | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one autoregressive decode step for `token_id` at position `pos`.

    Returns (logits[vocab_size], final_hidden[d_model]).
    """
    cfg = model.config
    if not (0 <= token_id < cfg.vocab_size):
        raise ValueError(f"token_id {token_id} out of range [0, {cfg.vocab_size})")

    def _trace(name: str, tensor: np.ndarray) -> None:
        if trace_hook is not None:
            trace_hook(name, tensor.copy())

    h = model.tok_embeddings[token_id].astype(np.float32).copy()
    _trace("embed", h)

    kv_repeat = cfg.n_heads // cfg.n_kv_heads
    attn_scale = 1.0 / np.sqrt(float(cfg.head_dim))

    for l_idx, layer in enumerate(model.layers):
        prefix = f"layers.{l_idx}"

        # 1. Pre-Attention RMSNorm
        h_norm = rms_norm(h, layer.attn_norm, cfg.rms_eps)
        _trace(f"{prefix}.attn_norm", h_norm)

        # 2. Q, K, V projections
        q_flat = _linear_step(layer.q_proj, h_norm, layer.weight_unit_scale, cfg.lanes, mode)
        k_flat = _linear_step(layer.k_proj, h_norm, layer.weight_unit_scale, cfg.lanes, mode)
        v_flat = _linear_step(layer.v_proj, h_norm, layer.weight_unit_scale, cfg.lanes, mode)
        _trace(f"{prefix}.q_proj", q_flat)
        _trace(f"{prefix}.k_proj", k_flat)
        _trace(f"{prefix}.v_proj", v_flat)

        q = q_flat.reshape(cfg.n_heads, cfg.head_dim)
        k = k_flat.reshape(cfg.n_kv_heads, cfg.head_dim)
        v = v_flat.reshape(cfg.n_kv_heads, cfg.head_dim)

        # 3. QK-norm + RoPE
        q = rms_norm(q, layer.q_norm, cfg.rms_eps)
        k = rms_norm(k, layer.k_norm, cfg.rms_eps)
        q = apply_rope(q, pos=pos, theta=cfg.rope_theta)
        k = apply_rope(k, pos=pos, theta=cfg.rope_theta)
        _trace(f"{prefix}.q_rope", q)
        _trace(f"{prefix}.k_rope", k)

        # 4. Update KV cache & compute GQA attention
        kv_cache.k[l_idx] = np.concatenate([kv_cache.k[l_idx], k[None, :, :]], axis=0)
        kv_cache.v[l_idx] = np.concatenate([kv_cache.v[l_idx], v[None, :, :]], axis=0)

        k_seq = kv_cache.k[l_idx]  # (seq_len, n_kv_heads, head_dim)
        v_seq = kv_cache.v[l_idx]

        attn_heads = np.zeros((cfg.n_heads, cfg.head_dim), dtype=np.float32)
        for h_idx in range(cfg.n_heads):
            kv_idx = h_idx // kv_repeat
            scores = (k_seq[:, kv_idx, :] @ q[h_idx]) * np.float32(attn_scale)
            probs = softmax(scores)
            attn_heads[h_idx] = probs @ v_seq[:, kv_idx, :]

        attn_concat = attn_heads.reshape(-1)
        _trace(f"{prefix}.attn_context", attn_concat)

        # 5. Output projection + residual
        attn_out = _linear_step(layer.o_proj, attn_concat, layer.weight_unit_scale, cfg.lanes, mode)
        _trace(f"{prefix}.o_proj", attn_out)
        h = (h + attn_out).astype(np.float32)
        _trace(f"{prefix}.post_attn_residual", h)

        # 6. Pre-FFN RMSNorm + SwiGLU FFN
        ffn_in = rms_norm(h, layer.ffn_norm, cfg.rms_eps)
        _trace(f"{prefix}.ffn_norm", ffn_in)

        gate = _linear_step(layer.gate_proj, ffn_in, layer.weight_unit_scale, cfg.lanes, mode)
        up = _linear_step(layer.up_proj, ffn_in, layer.weight_unit_scale, cfg.lanes, mode)
        _trace(f"{prefix}.gate_proj", gate)
        _trace(f"{prefix}.up_proj", up)

        hidden_ffn = (silu(gate) * up).astype(np.float32)
        _trace(f"{prefix}.swiglu", hidden_ffn)

        down = _linear_step(layer.down_proj, hidden_ffn, layer.weight_unit_scale, cfg.lanes, mode)
        _trace(f"{prefix}.down_proj", down)
        h = (h + down).astype(np.float32)
        _trace(f"{prefix}.post_ffn_residual", h)

    final_hidden = rms_norm(h, model.final_norm, cfg.rms_eps)
    _trace("final_norm", final_hidden)

    # Tied embeddings LM head
    head_scale = np.float32(1.0 / np.sqrt(float(cfg.d_model)))
    raw_logits = model.tok_embeddings.astype(np.float32) @ final_hidden
    logits = (raw_logits * head_scale).astype(np.float32)
    _trace("logits", logits)
    return logits, final_hidden


def generate_greedy(
    model: M0Model,
    prompt_tokens: list[int],
    max_new_tokens: int,
    mode: str = "integer",
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Run greedy decoding for `prompt_tokens` + `max_new_tokens`.

    Returns (generated_tokens, last_logits, last_final_hidden).
    """
    if not prompt_tokens:
        raise ValueError("prompt_tokens must be non-empty")
    kv_cache = KVCache.empty(model.config)
    pos = 0
    logits = np.zeros(model.config.vocab_size, dtype=np.float32)
    hidden = np.zeros(model.config.d_model, dtype=np.float32)

    for tok in prompt_tokens:
        logits, hidden = forward_token(model, tok, pos=pos, kv_cache=kv_cache, mode=mode)
        pos += 1

    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_tok = int(np.argmax(logits))
        generated.append(next_tok)
        logits, hidden = forward_token(model, next_tok, pos=pos, kv_cache=kv_cache, mode=mode)
        pos += 1

    return generated, logits, hidden


def build_toy_m0_model(config: M0Config | None = None, seed: int | None = None) -> M0Model:
    """Deterministically generate a seeded M0 model with well-scaled ternary projections."""
    cfg = config or PRESET_CONFIGS["tiny"]
    actual_seed = seed if seed is not None else cfg.seed
    rng = np.random.default_rng(actual_seed)

    # Normal embeddings scaled by 1/sqrt(d_model) so logits have healthy variance
    tok_embeddings = rng.standard_normal((cfg.vocab_size, cfg.d_model)).astype(
        np.float32
    ) * np.float32(1.0 / np.sqrt(float(cfg.d_model)))
    final_norm = rng.uniform(0.85, 1.15, size=(cfg.d_model,)).astype(np.float32)

    def _rand_packed(n_rows: int, n_cols: int) -> PackedTernaryTensor:
        trits = rng.choice(
            np.array([-1, 0, 1], dtype=np.int8),
            size=(n_rows, n_cols),
            p=[0.30, 0.40, 0.30],
        )
        n_groups = (n_cols + cfg.group_size - 1) // cfg.group_size
        # Keep scales around 128..384 so (group_sum * scale) >> shift stays O(1/sqrt(K))
        scales = rng.integers(128, 384, size=(n_rows, n_groups), dtype=np.uint16)
        return pack_ternary_tensor(trits, scales, group_size=cfg.group_size, shift=cfg.shift)

    # Unit scale chosen so projection variance is ~1/d_model
    weight_unit_scale = 2.5 / float(cfg.d_model)

    layers: list[M0LayerWeights] = []
    q_dim = cfg.n_heads * cfg.head_dim
    kv_dim = cfg.n_kv_heads * cfg.head_dim

    for _ in range(cfg.n_layers):
        layers.append(
            M0LayerWeights(
                attn_norm=rng.uniform(0.9, 1.1, size=(cfg.d_model,)).astype(np.float32),
                q_norm=rng.uniform(0.9, 1.1, size=(cfg.head_dim,)).astype(np.float32),
                k_norm=rng.uniform(0.9, 1.1, size=(cfg.head_dim,)).astype(np.float32),
                ffn_norm=rng.uniform(0.9, 1.1, size=(cfg.d_model,)).astype(np.float32),
                q_proj=_rand_packed(q_dim, cfg.d_model),
                k_proj=_rand_packed(kv_dim, cfg.d_model),
                v_proj=_rand_packed(kv_dim, cfg.d_model),
                o_proj=_rand_packed(cfg.d_model, q_dim),
                gate_proj=_rand_packed(cfg.ffn_dim, cfg.d_model),
                up_proj=_rand_packed(cfg.ffn_dim, cfg.d_model),
                down_proj=_rand_packed(cfg.d_model, cfg.ffn_dim),
                weight_unit_scale=weight_unit_scale,
            )
        )

    return M0Model(
        config=cfg,
        tok_embeddings=tok_embeddings,
        final_norm=final_norm,
        layers=layers,
    )


def serialize_m0_bundle(model: M0Model) -> bytes:
    """Serialize an entire M0Model into a deterministic byte-exact binary bundle."""
    cfg = model.config
    meta = {
        "name": cfg.name,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "ffn_dim": cfg.ffn_dim,
        "vocab_size": cfg.vocab_size,
        "max_seq_len": cfg.max_seq_len,
        "group_size": cfg.group_size,
        "lanes": cfg.lanes,
        "shift": cfg.shift,
        "rms_eps": cfg.rms_eps,
        "rope_theta": cfg.rope_theta,
        "tied_embeddings": cfg.tied_embeddings,
        "seed": cfg.seed,
    }

    chunks: list[bytes] = []
    tensor_index: list[dict[str, int | str]] = []
    offset = 0

    def _add_chunk(name: str, kind: str, data: bytes) -> None:
        nonlocal offset
        tensor_index.append({"name": name, "kind": kind, "offset": offset, "length": len(data)})
        chunks.append(data)
        offset += len(data)

    _add_chunk(
        "tok_embeddings",
        "f32",
        np.ascontiguousarray(model.tok_embeddings, dtype="<f4").tobytes(),
    )
    _add_chunk(
        "final_norm",
        "f32",
        np.ascontiguousarray(model.final_norm, dtype="<f4").tobytes(),
    )

    for l_idx, layer in enumerate(model.layers):
        p = f"layers.{l_idx}"
        _add_chunk(f"{p}.attn_norm", "f32", np.ascontiguousarray(layer.attn_norm, "<f4").tobytes())
        _add_chunk(f"{p}.q_norm", "f32", np.ascontiguousarray(layer.q_norm, "<f4").tobytes())
        _add_chunk(f"{p}.k_norm", "f32", np.ascontiguousarray(layer.k_norm, "<f4").tobytes())
        _add_chunk(f"{p}.ffn_norm", "f32", np.ascontiguousarray(layer.ffn_norm, "<f4").tobytes())
        for proj_name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            proj: PackedTernaryTensor = getattr(layer, proj_name)
            _add_chunk(f"{p}.{proj_name}", "ttape", serialize_packed_weights(proj))

    header_json = json.dumps(
        {"config": meta, "tensors": tensor_index},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    header_len = len(header_json).to_bytes(4, "little")
    return b"STM0" + header_len + header_json + b"".join(chunks)


def deserialize_m0_bundle(blob: bytes) -> M0Model:
    """Deserialize a deterministic M0 binary bundle back into an M0Model."""
    if len(blob) < 8 or blob[:4] != b"STM0":
        raise ValueError("Invalid M0 bundle magic; expected b'STM0'.")
    header_len = int.from_bytes(blob[4:8], "little")
    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    cfg = M0Config(**header["config"])
    payload = memoryview(blob)[8 + header_len :]

    by_name: dict[str, bytes] = {}
    for entry in header["tensors"]:
        start = int(entry["offset"])
        end = start + int(entry["length"])
        by_name[str(entry["name"])] = bytes(payload[start:end])

    tok_embeddings = (
        np.frombuffer(by_name["tok_embeddings"], dtype="<f4")
        .astype(np.float32)
        .reshape(cfg.vocab_size, cfg.d_model)
        .copy()
    )
    final_norm = (
        np.frombuffer(by_name["final_norm"], dtype="<f4")
        .astype(np.float32)
        .reshape(cfg.d_model)
        .copy()
    )

    weight_unit_scale = 2.5 / float(cfg.d_model)
    layers: list[M0LayerWeights] = []
    for l_idx in range(cfg.n_layers):
        p = f"layers.{l_idx}"
        layers.append(
            M0LayerWeights(
                attn_norm=np.frombuffer(by_name[f"{p}.attn_norm"], dtype="<f4")
                .astype(np.float32)
                .copy(),
                q_norm=np.frombuffer(by_name[f"{p}.q_norm"], dtype="<f4").astype(np.float32).copy(),
                k_norm=np.frombuffer(by_name[f"{p}.k_norm"], dtype="<f4").astype(np.float32).copy(),
                ffn_norm=np.frombuffer(by_name[f"{p}.ffn_norm"], dtype="<f4")
                .astype(np.float32)
                .copy(),
                q_proj=deserialize_packed_weights(by_name[f"{p}.q_proj"]),
                k_proj=deserialize_packed_weights(by_name[f"{p}.k_proj"]),
                v_proj=deserialize_packed_weights(by_name[f"{p}.v_proj"]),
                o_proj=deserialize_packed_weights(by_name[f"{p}.o_proj"]),
                gate_proj=deserialize_packed_weights(by_name[f"{p}.gate_proj"]),
                up_proj=deserialize_packed_weights(by_name[f"{p}.up_proj"]),
                down_proj=deserialize_packed_weights(by_name[f"{p}.down_proj"]),
                weight_unit_scale=weight_unit_scale,
            )
        )

    return M0Model(
        config=cfg,
        tok_embeddings=tok_embeddings,
        final_norm=final_norm,
        layers=layers,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 1.0 if np.allclose(va, vb) else 0.0
    return float(np.dot(va, vb) / denom)


def shannon_entropy(logits: np.ndarray) -> float:
    """Compute Shannon entropy H(p) in nats from raw logits."""
    p = softmax(logits).astype(np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def bundle_sha256(blob: bytes) -> str:
    """Return hex SHA256 digest of a serialized M0 bundle."""
    return hashlib.sha256(blob).hexdigest()
