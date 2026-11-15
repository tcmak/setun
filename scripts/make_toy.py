"""Generate deterministic M0 toy ternary transformer bundles (Task 1.6)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from setun.ref.m0 import (
    PRESET_CONFIGS,
    KVCache,
    build_toy_m0_model,
    bundle_sha256,
    cosine_similarity,
    dequantize_packed_tensor_fp32,
    forward_token,
    serialize_m0_bundle,
    shannon_entropy,
)


def generate_toy_artifacts(
    out_dir: Path,
    size: str = "tiny",
    seed: int = 1958,
) -> dict[str, str | int | float]:
    """Generate the internal binary M0 bundle + float source weights NPZ and return metadata."""
    if size not in PRESET_CONFIGS:
        raise ValueError(f"size must be one of {tuple(PRESET_CONFIGS)}, got {size!r}")

    cfg = replace(PRESET_CONFIGS[size], seed=seed)
    model = build_toy_m0_model(cfg, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_bytes = serialize_m0_bundle(model)
    bundle_path = out_dir / f"m0_{size}.stm0"
    bundle_path.write_bytes(bundle_bytes)
    sha256_hex = bundle_sha256(bundle_bytes)

    # Also write dequantized FP32 source weights for reference inspection
    float_tensors: dict[str, np.ndarray] = {
        "tok_embeddings": model.tok_embeddings,
        "final_norm": model.final_norm,
    }
    for l_idx, layer in enumerate(model.layers):
        p = f"layers.{l_idx}"
        float_tensors[f"{p}.attn_norm"] = layer.attn_norm
        float_tensors[f"{p}.q_norm"] = layer.q_norm
        float_tensors[f"{p}.k_norm"] = layer.k_norm
        float_tensors[f"{p}.ffn_norm"] = layer.ffn_norm
        for proj_name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            proj = getattr(layer, proj_name)
            float_tensors[f"{p}.{proj_name}"] = dequantize_packed_tensor_fp32(
                proj, layer.weight_unit_scale
            )

    npz_path = out_dir / f"m0_{size}_fp32.npz"
    np.savez(npz_path, **float_tensors)

    # Verify forward pass sanity (cosine & entropy)
    logits_int, hidden_int = forward_token(
        model, token_id=1, pos=0, kv_cache=KVCache.empty(cfg), mode="integer"
    )
    logits_fp, hidden_fp = forward_token(
        model, token_id=1, pos=0, kv_cache=KVCache.empty(cfg), mode="fp32"
    )
    cos = cosine_similarity(hidden_int, hidden_fp)
    ent = shannon_entropy(logits_int)

    manifest = {
        "name": cfg.name,
        "size": size,
        "seed": seed,
        "bundle_file": bundle_path.name,
        "bundle_bytes": len(bundle_bytes),
        "sha256": sha256_hex,
        "cosine_int_vs_fp32": round(cos, 6),
        "logits_entropy_nats": round(ent, 4),
    }
    manifest_path = out_dir / f"m0_{size}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate M0 toy ternary transformer weights.")
    parser.add_argument(
        "--size",
        choices=list(PRESET_CONFIGS.keys()),
        default="tiny",
        help="Model size preset: tiny (CI) or small (nightly)",
    )
    parser.add_argument("--seed", type=int, default=1958, help="Deterministic RNG seed")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/toy_models"),
        help="Output directory for generated model bundle",
    )
    args = parser.parse_args(argv)

    manifest = generate_toy_artifacts(args.out_dir, size=args.size, seed=args.seed)
    print(
        f"[make_toy] Generated {manifest['bundle_file']} "
        f"({manifest['bundle_bytes']} B, sha256={manifest['sha256'][:16]}..., "
        f"cosine={manifest['cosine_int_vs_fp32']:.6f}, "
        f"entropy={manifest['logits_entropy_nats']:.3f} nats)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
