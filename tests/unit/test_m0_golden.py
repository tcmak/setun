"""Unit tests for M0 Golden Integer Model (Task 1.5) and Toy Generator (Task 1.6)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.make_toy import generate_toy_artifacts
from setun.ref.format import load_ttape_layers, unpack_ternary_tensor
from setun.ref.m0 import (
    PRESET_CONFIGS,
    KVCache,
    build_toy_m0_model,
    bundle_sha256,
    cosine_similarity,
    deserialize_m0_bundle,
    forward_ternary_chain,
    forward_token,
    generate_greedy,
    serialize_m0_bundle,
    shannon_entropy,
)
from setun.ref.ops import row_dot

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "third_party" / "TernaryTape" / "fixtures"


@pytest.mark.parametrize("size", ["tiny", "small"])
def test_m0_integer_vs_fp32_cosine_similarity(size: str) -> None:
    """Task 1.5: M0 final hidden state matches FP32 reference with cosine >= 0.999."""
    cfg = PRESET_CONFIGS[size]
    model = build_toy_m0_model(cfg, seed=1958)

    prompt = [1, 42, 17, 99, 128]
    gen_int, logits_int, hidden_int = generate_greedy(
        model, prompt_tokens=prompt, max_new_tokens=4, mode="integer"
    )
    gen_fp, logits_fp, hidden_fp = generate_greedy(
        model, prompt_tokens=prompt, max_new_tokens=4, mode="fp32"
    )

    cos_hidden = cosine_similarity(hidden_int, hidden_fp)
    cos_logits = cosine_similarity(logits_int, logits_fp)
    assert cos_hidden >= 0.999, f"Expected hidden cosine >= 0.999, got {cos_hidden:.6f}"
    assert cos_logits >= 0.999, f"Expected logits cosine >= 0.999, got {cos_logits:.6f}"
    assert gen_int == gen_fp, f"Greedy token mismatch: {gen_int} vs {gen_fp}"


def test_m0_trace_hooks_capture_all_ops() -> None:
    """Task 1.5: Verify per-op trace hooks record embeddings, norms, projections, and logits."""
    cfg = PRESET_CONFIGS["tiny"]
    model = build_toy_m0_model(cfg, seed=1958)
    traces: dict[str, np.ndarray] = {}

    def hook(name: str, tensor: np.ndarray) -> None:
        traces[name] = tensor

    logits, hidden = forward_token(
        model, token_id=7, pos=0, kv_cache=KVCache.empty(cfg), mode="integer", trace_hook=hook
    )
    expected_keys = [
        "embed",
        "layers.0.attn_norm",
        "layers.0.q_proj",
        "layers.0.k_proj",
        "layers.0.v_proj",
        "layers.0.q_rope",
        "layers.0.k_rope",
        "layers.0.attn_context",
        "layers.0.o_proj",
        "layers.0.ffn_norm",
        "layers.0.gate_proj",
        "layers.0.up_proj",
        "layers.0.swiglu",
        "layers.0.down_proj",
        "layers.1.o_proj",
        "layers.1.down_proj",
        "final_norm",
        "logits",
    ]
    for key in expected_keys:
        assert key in traces, f"Missing trace key {key}"
    assert np.array_equal(traces["final_norm"], hidden)
    assert np.array_equal(traces["logits"], logits)


def test_toy_generator_deterministic_sha256_and_entropy() -> None:
    """Task 1.6: Same seed -> same SHA256; bundle round-trips; logits entropy in [3.5, 5.54]."""
    cfg = PRESET_CONFIGS["tiny"]
    m1 = build_toy_m0_model(cfg, seed=2026)
    m2 = build_toy_m0_model(cfg, seed=2026)
    m_diff = build_toy_m0_model(cfg, seed=2027)

    b1 = serialize_m0_bundle(m1)
    b2 = serialize_m0_bundle(m2)
    b_diff = serialize_m0_bundle(m_diff)

    assert bundle_sha256(b1) == bundle_sha256(b2), "Same seed must produce identical SHA256"
    assert bundle_sha256(b1) != bundle_sha256(b_diff), "Different seeds must differ"

    # Verify binary bundle round-trip
    restored = deserialize_m0_bundle(b1)
    logits1, _ = forward_token(m1, token_id=12, pos=0, kv_cache=KVCache.empty(cfg))
    logits2, _ = forward_token(restored, token_id=12, pos=0, kv_cache=KVCache.empty(cfg))
    assert np.array_equal(logits1, logits2)

    # Verify non-degenerate logit entropy
    ent = shannon_entropy(logits1)
    assert 3.5 <= ent <= 5.54, f"Logits entropy {ent:.3f} outside documented [3.5, 5.54] nats"
    assert not np.any(np.isnan(logits1)) and not np.any(np.isinf(logits1))

    # Verify make_toy.py artifact writer
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_a = generate_toy_artifacts(Path(tmpdir) / "run_a", size="tiny", seed=1958)
        manifest_b = generate_toy_artifacts(Path(tmpdir) / "run_b", size="tiny", seed=1958)
        assert manifest_a["sha256"] == manifest_b["sha256"]
        assert float(manifest_a["cosine_int_vs_fp32"]) >= 0.999


def test_lux_chain_grouped_mlp_ttape_forward_parity() -> None:
    """Task C1.3: Load Lux-exported grouped MLP .ttape and verify exact m0.py forward parity."""
    tape_path = FIXTURES_DIR / "accept" / "lux_chain_grouped_mlp.ttape"
    meta_path = FIXTURES_DIR / "accept" / "lux_chain_grouped_mlp.json"
    assert tape_path.exists(), f"Missing Lux golden tape at {tape_path}"
    assert meta_path.exists(), f"Missing Lux golden sidecar at {meta_path}"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    spots = meta["spot_checks"]

    layers = load_ttape_layers(tape_path.read_bytes(), reader_profile="grouped")
    assert len(layers) == int(spots["n_layers"]) == 2

    # Spot-check unpacked layer weights, group scales, and shift against sidecar metadata
    w0, s0, shift0 = unpack_ternary_tensor(layers[0])
    w1, s1, shift1 = unpack_ternary_tensor(layers[1])
    assert w0.shape == (int(spots["layer0_dim_out"]), int(spots["layer0_dim_in"]))
    assert w1.shape == (int(spots["layer1_dim_out"]), int(spots["layer1_dim_in"]))
    assert np.array_equal(w0, np.asarray(spots["layer0_weights"], dtype=np.int8))
    assert np.array_equal(w1, np.asarray(spots["layer1_weights"], dtype=np.int8))
    assert np.array_equal(s0, np.asarray(spots["layer0_scales_u16"], dtype=np.uint16))
    assert np.array_equal(s1, np.asarray(spots["layer1_scales_u16"], dtype=np.uint16))
    assert shift0 == int(spots["layer0_shift"])
    assert shift1 == int(spots["layer1_shift"])

    x_in = np.asarray(meta["input_vector"], dtype=np.int8)
    expected_hidden = np.asarray(spots["hidden_vector"], dtype=np.int64)
    expected_out = np.asarray(meta["expected_output_vector"], dtype=np.int64)

    traces: dict[str, np.ndarray] = {}

    def hook(name: str, tensor: np.ndarray) -> None:
        traces[name] = tensor

    y_int = forward_ternary_chain(layers, x_in, lanes=16, mode="integer", trace_hook=hook)
    assert np.array_equal(traces["layers.0"], expected_hidden), (
        f"m0.forward_ternary_chain layer 0 mismatch: expected {expected_hidden.tolist()}, "
        f"got {traces['layers.0'].tolist()}"
    )
    assert np.array_equal(y_int, expected_out), (
        f"m0.forward_ternary_chain output mismatch: expected {expected_out.tolist()}, "
        f"got {y_int.tolist()}"
    )

    # Cross-check against row-by-row setun.ref.ops.row_dot
    h_ops = np.array(
        [
            row_dot(w0[r], x_in, s0[r], group_size=layers[0].group_size, lanes=16, shift=shift0)
            for r in range(w0.shape[0])
        ],
        dtype=np.int64,
    )
    y_ops = np.array(
        [
            row_dot(
                w1[r],
                h_ops.astype(np.int8),
                s1[r],
                group_size=layers[1].group_size,
                lanes=16,
                shift=shift1,
            )
            for r in range(w1.shape[0])
        ],
        dtype=np.int64,
    )
    assert np.array_equal(y_ops, expected_out)

    # Also verify FP32 dequantized execution mode in m0.py matches exact sidecar output
    y_fp32 = forward_ternary_chain(layers, x_in.astype(np.float32), lanes=16, mode="fp32")
    assert np.array_equal(y_fp32, expected_out.astype(np.float32))

    # Goose-style diagnostic checks on invalid chain inputs
    with pytest.raises(ValueError, match="m0.forward_ternary_chain.*non-empty"):
        forward_ternary_chain([], x_in)
    with pytest.raises(ValueError, match="m0.forward_ternary_chain.*layer\\[0\\].*expected"):
        forward_ternary_chain(layers, x_in[:3])
    with pytest.raises(ValueError, match="m0.forward_ternary_chain.*signed int8"):
        forward_ternary_chain(layers, np.array([200, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64))
