"""Tests for Task 3.4: M0 TTAPE-extended codec, HybridM0Runner, and `setun run` CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.make_toy import generate_toy_artifacts
from scripts.make_toy import main as make_toy_main
from setun.cli import main as cli_main
from setun.ref.format import (
    TtapeValidationError,
    pack_ternary_tensor,
    serialize_packed_weights,
    validate_ttape,
)
from setun.ref.m0 import (
    PRESET_CONFIGS,
    KVCache,
    build_toy_m0_model,
    deserialize_m0_ttape,
    forward_token,
    generate_greedy,
    load_m0_model,
    serialize_m0_bundle,
    serialize_m0_ttape,
)
from setun.runtime.driver import AccelDriver, AsyncAccelDriver, RefBusBackend
from setun.runtime.m0_runner import HybridM0Runner, open_m0_backend, run_m0_greedy

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_m0_ttape_extended_roundtrip_and_guards(tmp_path: Path) -> None:
    """Verify M0 serializes to canonical TTAPE-extended v1.0 and round-trips bit-exact."""
    cfg = PRESET_CONFIGS["tiny"]
    model = build_toy_m0_model(cfg, seed=1958)

    ttape_bytes = serialize_m0_ttape(model)
    assert ttape_bytes[:4] == b"TTAP"
    assert validate_ttape(ttape_bytes, reader_profile="extended") == "TTAPE_OK"
    assert validate_ttape(ttape_bytes, reader_profile="grouped") == "TTAPE_ERR_UNSUPPORTED_PROFILE"

    restored = deserialize_m0_ttape(ttape_bytes)
    assert restored.config == model.config
    assert np.array_equal(restored.tok_embeddings, model.tok_embeddings)
    assert np.array_equal(restored.final_norm, model.final_norm)

    logits_orig, hidden_orig = forward_token(model, 1, 0, KVCache.empty(cfg), mode="integer")
    logits_rest, hidden_rest = forward_token(restored, 1, 0, KVCache.empty(cfg), mode="integer")
    assert np.array_equal(logits_orig, logits_rest)
    assert np.array_equal(hidden_orig, hidden_rest)

    # load_m0_model handles both .ttape (bytes & Path) and .stm0 (bytes & Path)
    ttape_path = tmp_path / "m0.ttape"
    stm0_path = tmp_path / "m0.stm0"
    ttape_path.write_bytes(ttape_bytes)
    stm0_bytes = serialize_m0_bundle(model)
    stm0_path.write_bytes(stm0_bytes)

    assert load_m0_model(ttape_path).config == cfg
    assert load_m0_model(str(stm0_path)).config == cfg
    assert load_m0_model(stm0_bytes).config == cfg

    # Guard: corrupted TTAPE header, non-M0 single-layer TTAPE, and unknown magic
    bad_ttape = bytearray(ttape_bytes)
    bad_ttape[0x20] ^= 0xFF
    with pytest.raises(TtapeValidationError):
        deserialize_m0_ttape(bytes(bad_ttape))

    one_layer = serialize_packed_weights(
        pack_ternary_tensor(
            np.ones((4, 64), dtype=np.int8),
            np.ones((4, 1), dtype=np.uint16),
            group_size=64,
            shift=0,
        )
    )
    with pytest.raises(ValueError, match="expected TTAPE-extended M0"):
        deserialize_m0_ttape(one_layer)

    with pytest.raises(ValueError, match="Unsupported M0 model container"):
        load_m0_model(b"NOPE12345678")

    # Verify make_toy generates .ttape alongside .stm0 and supports --ttape-out
    out_dir = tmp_path / "toy_out"
    custom_ttape = tmp_path / "custom" / "m0_toy.ttape"
    manifest = generate_toy_artifacts(out_dir, size="tiny", seed=1958)
    assert (out_dir / "m0_tiny.ttape").exists()
    assert manifest["ttape_file"] == "m0_tiny.ttape"

    rc = make_toy_main(
        [
            "--size",
            "tiny",
            "--seed",
            "1958",
            "--out-dir",
            str(out_dir),
            "--ttape-out",
            str(custom_ttape),
        ]
    )
    assert rc == 0 and custom_ttape.exists()
    assert validate_ttape(custom_ttape.read_bytes(), reader_profile="extended") == "TTAPE_OK"


def test_hybrid_m0_runner_ref_parity_traces_and_guards() -> None:
    """Verify HybridM0Runner matches setun.ref.m0 bit-for-bit including trace hooks & guards."""
    cfg = PRESET_CONFIGS["tiny"]
    model = build_toy_m0_model(cfg, seed=1958)

    prompt = [1, 42]
    ref_tokens, ref_logits, ref_hidden = generate_greedy(
        model, prompt_tokens=prompt, max_new_tokens=6, mode="integer"
    )

    streamed: list[int] = []
    hyb_tokens, hyb_logits, hyb_hidden, summary = run_m0_greedy(
        model,
        backend="ref",
        prompt_tokens=prompt,
        max_new_tokens=6,
        clk_freq_hz=20_990_000.0,
        on_token=streamed.append,
    )

    assert hyb_tokens == ref_tokens == streamed
    assert np.array_equal(hyb_logits, ref_logits)
    assert np.array_equal(hyb_hidden, ref_hidden)
    assert summary.forward_steps == len(prompt) + 6
    assert summary.gemv_calls == summary.forward_steps * cfg.n_layers * 7
    assert summary.total_cycles > 0
    assert summary.cycles_per_token > 0.0
    assert summary.bus_utilization > 0.0
    assert summary.mb_per_sec > 0.0

    # Per-op TraceHook parity on a single forward step
    ref_traces: dict[str, np.ndarray] = {}
    hyb_traces: dict[str, np.ndarray] = {}
    forward_token(
        model,
        token_id=7,
        pos=0,
        kv_cache=KVCache.empty(cfg),
        mode="integer",
        trace_hook=lambda k, v: ref_traces.__setitem__(k, v),
    )
    with open_m0_backend(model, "ref") as drv:
        runner = HybridM0Runner(model, drv)
        runner.forward_token(
            token_id=7,
            pos=0,
            kv_cache=KVCache.empty(cfg),
            trace_hook=lambda k, v: hyb_traces.__setitem__(k, v),
        )
        # Guard: invalid token_id and empty prompt
        with pytest.raises(ValueError, match="token_id"):
            runner.forward_token(token_id=999, pos=0, kv_cache=KVCache.empty(cfg))
        with pytest.raises(ValueError, match="prompt_tokens must be non-empty"):
            runner.generate_greedy(prompt_tokens=[], max_new_tokens=1)
        with pytest.raises(ValueError, match="max_new_tokens must be >= 0"):
            runner.generate_greedy(prompt_tokens=[1], max_new_tokens=-1)

    assert list(hyb_traces.keys()) == list(ref_traces.keys())
    for key in ref_traces:
        assert np.array_equal(hyb_traces[key], ref_traces[key]), f"Mismatch on trace {key}"

    # Zero-activation branch (tok_embeddings = 0 -> rms_norm = 0 -> act_scale == 0.0)
    zero_model = build_toy_m0_model(cfg, seed=1958)
    zero_model.tok_embeddings.fill(0.0)
    with open_m0_backend(zero_model, "ref") as drv:
        z_runner = HybridM0Runner(zero_model, drv)
        z_logits, z_hidden = z_runner.forward_token(0, 0, KVCache.empty(cfg))
        assert np.all(z_logits == 0.0) and np.all(z_hidden == 0.0)
        z_sum = z_runner.summary(prompt_tokens=[0], generated_tokens=[])
        assert z_sum.cycles_per_token == 0.0 and z_sum.bus_utilization == 0.0

    # Guard: main_ram capacity overflow check
    tiny_csr_bus = RefBusBackend(rows=4, lanes=16, group_size=64, beat_bits=64)
    tiny_csr_bus.layout.main_ram_size = 256  # too small for M0 weights
    tiny_drv = AccelDriver(AsyncAccelDriver(tiny_csr_bus, csr_layout=tiny_csr_bus.layout))
    with pytest.raises(ValueError, match="exceeds main_ram_size"):
        HybridM0Runner(model, tiny_drv)


def test_cli_run_ref_and_argument_guards(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise `setun run` CLI over `--backend ref` and validate CLI error handling."""
    tape_path = tmp_path / "m0_tiny.ttape"
    model = build_toy_m0_model(PRESET_CONFIGS["tiny"], seed=1958)
    tape_path.write_bytes(serialize_m0_ttape(model))

    rc = cli_main(
        [
            "run",
            str(tape_path),
            "--backend",
            "ref",
            "--tokens",
            "8",
            "--prompt",
            "1,5",
            "--clk-mhz",
            "20.99",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    got_tokens = [int(tok) for tok in captured.out.strip().split()]
    exp_tokens, _, _ = generate_greedy(
        model, prompt_tokens=[1, 5], max_new_tokens=8, mode="integer"
    )
    assert got_tokens == exp_tokens
    assert "cycles/tok=" in captured.err
    assert "MB/s=" in captured.err
    assert "clk=20.99 MHz" in captured.err

    with pytest.raises(ValueError, match="--tokens must be >= 1"):
        cli_main(["run", str(tape_path), "--backend", "ref", "--tokens", "0"])

    with pytest.raises(ValueError, match="--prompt must contain"):
        cli_main(["run", str(tape_path), "--backend", "ref", "--prompt", " , "])

    with pytest.raises(ValueError, match="--clk-mhz must be > 0"):
        cli_main(["run", str(tape_path), "--backend", "ref", "--clk-mhz", "0"])


@pytest.mark.slow
def test_cli_run_sim_16_tokens_bit_exact_vs_ref(capsys: pytest.CaptureFixture[str]) -> None:
    """Task 3.4 Gate: `setun run models/m0_toy.ttape --backend sim --tokens 16` == ref bit-exact."""
    model_path = REPO_ROOT / "models" / "m0_toy.ttape"
    assert model_path.exists(), f"Expected canonical toy model at {model_path}"

    rc_ref = cli_main(["run", str(model_path), "--backend", "ref", "--tokens", "16"])
    assert rc_ref == 0
    ref_io = capsys.readouterr()
    ref_tokens = [int(t) for t in ref_io.out.strip().split()]
    assert len(ref_tokens) == 16

    rc_sim = cli_main(["run", str(model_path), "--backend", "sim", "--tokens", "16"])
    assert rc_sim == 0
    sim_io = capsys.readouterr()
    sim_tokens = [int(t) for t in sim_io.out.strip().split()]

    assert sim_tokens == ref_tokens
    assert "backend=sim" in sim_io.err
    assert "cycles/tok=" in sim_io.err
    assert "bus_util=" in sim_io.err
    assert "MB/s=" in sim_io.err
