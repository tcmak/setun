"""Unit tests for M0 Golden Integer Model (Task 1.5) and Toy Generator (Task 1.6)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.make_toy import generate_toy_artifacts
from setun.ref.m0 import (
    PRESET_CONFIGS,
    KVCache,
    build_toy_m0_model,
    bundle_sha256,
    cosine_similarity,
    deserialize_m0_bundle,
    forward_token,
    generate_greedy,
    serialize_m0_bundle,
    shannon_entropy,
)


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
