# Reference & Toy Models (`docs/models.md`)

## 1. M0 — Deterministic Toy Ternary Transformer (Season 2)

`M0` is a deterministic, seeded random model sharing the exact architectural structure of `M1` (GQA attention, pre-RMSNorm, QK-norm, RoPE, SwiGLU FFN, and tied token embeddings / LM head) at miniature dimensions so full-chip Verilator simulations and initial FPGA bring-up run in seconds.

| Parameter | `tiny` (CI) | `small` (Nightly) |
|---|---|---|
| **Layers (`n_layers`)** | 2 | 2 |
| **Hidden dimension (`d_model`)** | 64 | 128 |
| **Query heads (`n_heads`)** | 4 | 4 |
| **KV heads (`n_kv_heads`, GQA)** | 2 | 2 |
| **Head dimension (`head_dim`)** | 16 | 32 |
| **FFN intermediate (`ffn_dim`, SwiGLU)** | 128 | 256 |
| **Vocabulary size (`vocab_size`)** | 256 | 256 |
| **Group size (`G`)** | 64 | 128 |
| **Hardware lanes (`L`)** | 16 | 64 |
| **Norm / Activation / Pos** | RMSNorm + QK-norm / SiLU (SwiGLU) / RoPE | RMSNorm + QK-norm / SiLU (SwiGLU) / RoPE |
| **Embeddings** | Tied (`tok_embeddings` == LM head) | Tied (`tok_embeddings` == LM head) |

### Logit Entropy Bounds
For `vocab_size = 256`, maximum uniform entropy is $\ln(256) \approx 5.545\text{ nats}$ and a collapsed one-hot distribution has entropy $0.0\text{ nats}$. A healthy non-degenerate `M0` forward pass must produce output logit Shannon entropy strictly within **$[3.5, 5.54]\text{ nats}$**.

### Important Note on Random Weights vs. Real Model Error Budgets
> **Why error budgets are judged on `M1`, not `M0`:** Seeded random weights (`M0`) follow sub-Gaussian distributions with **no emergent activation outliers**. In trained LLMs (`M1`), high-magnitude outlier channels emerge in hidden states, which stress dynamic `int8` per-token quantization much harder than random Gaussian activations. Therefore, `M0` validates **structural, plumbing, and bit-exact hardware-vs-golden equivalence** (requiring final hidden-state cosine similarity $\ge 0.999$ vs. FP32), whereas quantization error budgets (perplexity delta $\le +2\%$ and top-1 token agreement $\ge 95\%$) are evaluated strictly on real trained weights in `M1` (Track A, Tasks 1.5 & 1.8).
