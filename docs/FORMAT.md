Setun Internal Weight Format Specification (v0.1)
This document defines the internal binary and in-memory representation for grouped ternary weight tensors in Setun (ADR-003 and ADR-011).
1. Trit Encoding (ADR-003)
Each ternary weight $w \in {-1, 0, +1}$ is encoded as a 2-bit unsigned code:

Value ($w$)
2-bit Code (Binary)
Hex
Notes
0
0b00
0x0
Zero / skip lane (also used for ragged-$K$ padding)
+1
0b01
0x1
Add activation ($+x$)
-1
0b10
0x2
Subtract activation ($-x$)
Reserved
0b11
0x3
Rejected by software unpacker (ValueError); aliased to 0 in RTL
2. Bit-Packing Endianness (LSB-First)
Four consecutive trits along the reduction dimension $K$ are packed into one byte (uint8) in LSB-first order:

$$\text{byte} = (c_0 \ll 0) \mid (c_1 \ll 2) \mid (c_2 \ll 4) \mid (c_3 \ll 6)$$

where $c_i \in {0, 1, 2}$ is the 2-bit code for trit index $i \pmod 4$. In a wide hardware stream beat of $L$ lanes ($2L$ bits), lane $l \in [0, L-1]$ occupies bits [2*l + 1 : 2*l].
3. Grouping and Ragged-$K$ Padding
A 2D weight matrix has logical shape $(N, K)$, where $N$ is the number of output rows and $K$ is the inner reduction dimension.
Weights along each row are partitioned into groups of size $G \in {64, 128, 256}$.
The number of groups per row is: $$N_{\text{groups}} = \left\lceil \frac{K}{G} \right\rceil$$
The padded reduction length is $K_{\text{pad}} = N_{\text{groups}} \cdot G$.
If $K$ is not a multiple of $G$ (ragged $K$), the trailing $K_{\text{pad}} - K$ elements of the final group in each row are padded with 0 (0b00). Because $0 \cdot x = 0$, padded trits contribute zero to the group partial sum.
Each group of $G$ trits occupies $G / 4$ bytes (e.g., 16 bytes for $G=64$, 32 bytes for $G=128$, 64 bytes for $G=256$).
4. Group Scales (ADR-011)
Every group $(n, g)$ has an unsigned 16-bit fixed-point scale mantissa scale_u16 $\in [0, 65535]$ (uint16, little-endian), paired with a shared right-shift exponent shift (uint8).
Effective real scale: $\text{scale}_{\text{real}}(n, g) = \text{scale_u16}(n, g) \cdot 2^{-\text{shift}}$.
Lossless integer pack/unpack preserves both the exact ternary matrix $W \in {-1, 0, +1}^{N \times K}$ and the uint16 group scale table $S \in \mathbb{Z}{2^{16}}^{N \times N{\text{groups}}}$.
5. Binary Header & Layout
Binary blobs serialized by serialize_packed_weights start with a 24-byte little-endian header (<4sBBHIIII):

Offset
Type
Field
Description
0x00
char[4]
magic
ASCII "STNW" (0x574E5453 LE)
0x04
uint8
version
Format version (1)
0x05
uint8
shift
Shared fixed-point right-shift exponent
0x06
uint16
group_size
Group size $G \in {64, 128, 256}$
0x08
uint32
n_rows
Logical output rows $N$
0x0C
uint32
n_cols
Logical inner dimension $K$
0x10
uint32
n_groups
Groups per row $\lceil K / G \rceil$
0x14
uint32
reserved
0 (alignment padding to 24 bytes)

Followed immediately by:

packed_trits: uint8 array of shape $(N, N_{\text{groups}}, G / 4)$ in row-major order.
scales: little-endian uint16 array of shape $(N, N_{\text{groups}})$ in row-major order.
6. Byte-Exact Worked Example
Consider a 1-row ragged tensor with $N = 1$, $K = 6$, and group size $G = 64$ (1 group, padded to $K_{\text{pad}} = 64$):

Logical row weights: w = [+1, -1, 0, +1, -1, -1]
Group scale: scale_u16 = [[0x0180]] (384), shift = 8

Step 1: Pad to $G = 64$ with zeros

First 6 trits: +1, -1, 0, +1, -1, -1
Remaining 58 trits: 0, 0, ..., 0

Step 2: Encode 2-bit codes ({0: 0b00, +1: 0b01, -1: 0b10})

Byte 0 (trits 0..3: +1, -1, 0, +1): $$\text{byte}_0 = (1 \ll 0) \mid (2 \ll 2) \mid (0 \ll 4) \mid (1 \ll 6) = 1 + 8 + 0 + 64 = 73 = \text{0x49} \quad (\text{0b01_00_10_01})$$
Byte 1 (trits 4..7: -1, -1, 0, 0): $$\text{byte}_1 = (2 \ll 0) \mid (2 \ll 2) \mid (0 \ll 4) \mid (0 \ll 6) = 2 + 8 + 0 + 0 = 10 = \text{0x0A} \quad (\text{0b00_00_10_10})$$
Bytes 2..15 (trits 8..63: all 0): $$\text{byte}_{2..15} = \text{0x00}$$

Step 3: Serialized Binary Bytes (42 bytes total = 24B header + 16B trits + 2B scale)

Header (24 bytes): 53 54 4e 57 ("STNW") 01 (ver=1) 08 (shift=8) 40 00 (G=64) 01 00 00 00 (N=1) 06 00 00 00 (K=6) 01 00 00 00 (n_groups=1) 00 00 00 00 (reserved=0)
Packed trits (16 bytes): 49 0a 00 00 00 00 00 00 00 00 00 00 00 00 00 00
Scales (2 bytes, LE): 80 01

---

## 6. Planned Migration to VeriObjC `VTPE v1` (Preparation — Blocked on VeriObjC Task 5.6)

> **Status (`VTPE` Spec §12.5):** **PREPARATION ONLY — BLOCKED on `VeriObjC` Task 5.6 (`rt/` layout reconciliation).**
> Until `VeriObjC` reconciles its `rt/` C header layout and `setun/ref/format.py` is rewritten and verified against the dual-profile golden corpus, `setun`'s internal `STNW` / `STM0` framing and `VTPE` **coexist and must not be claimed unified**.

### 6.1 Concept Mapping (`STNW` / `STM0` → `VTPE v1`)

The 2-bit LSB-first ternary weight payload (`{00: 0, 01: +1, 10: -1, 11: reserved/reject}`, 4 trits/byte, zero-padded tail lanes) is already byte-identical between `STNW` and `VTPE v1` §3. Only the outer container framing and section tags will migrate once `VeriObjC` Task 5.6 lands:

| Current `setun` Artifact (`STNW` / `STM0`) | Target `VTPE v1` Construct | Profile (`flags` bit 2) | Notes |
|---|---|---|---|
| `STNW` 24-byte header (`magic="STNW"`) | `VTPE` 32-byte header (`magic="VTPE"`, FNV-1a-32 over `0x00..0x0B`, `artifact_id`) | Both (`0` or `1`) | Position-independent, little-endian, FNV-1a-32 framed |
| `packed_trits` (`uint8[N, n_groups, G//4]`) | Section Tag `0x0001` (`weights`) + `0x0003` (`arch descriptor`) | Both (`0` or `1`) | Identical 2-bit LSB-first packing; `0b11` and non-zero padding rejected |
| Power-of-two tensor shift (`scale_u16 = 1`, `shift`) | Section Tag `0x0002` (`scales`: `s16` exponent per tensor) | `VTPE-Core` (`bit 2 = 0`) | Strict `DSP=0` shift-only profile produced by `TernaryTape.jl` |
| `scales` (`uint16[N, n_groups]`, `G ∈ {64, 128, 256}`) + `shift` | Section Tag `0x0006` (`group_scales_u16`, explicit `G` in payload) | `VTPE-Grouped` (`bit 2 = 1`) | Requires consumer multiply (1 DSP48E1/row or `-nodsp` LUT multiplier) |
| `STM0` FP32 `tok_embeddings` & `rms_norm` vectors | Section Tag `0x0007` (`fp32_aux`) | `VTPE-Grouped` (`bit 2 = 1`) | **Host-consumed only** (hybrid decode); never hardware-visible |
| `m0_<size>_manifest.json` | Section Tag `0x0004` (`manifest`) | Optional (`bit 1 = 1`) | Length-prefixed ASCII provenance fields |

### 6.2 Dual-Consumer Design Sketch (`setun` Loader & Datapath)

When `setun/ref/format.py` migrates to `VTPE v1` after `VeriObjC` Task 5.6:
1. **Framing & Rejection Gate (§8):** `deserialize_vtpe()` will validate the 32-byte header, FNV-1a-32 header/section checksums, section count/length invariants, reserved bits, and profile tag rules (`0x0006`/`0x0007` legal if and only if `flags` bit 2 is set; `G ∈ {64, 128, 256}`), backed by an exhaustive per-byte mutation fuzz test.
2. **`VTPE-Core` Execution (`flags` bit 2 = 0):** Loaded into `GroupAccumulateScale` in shift mode (`scale_u16 = 1`, `shift = -s16_exp` across a single group or replicated power-of-two shift across groups), requiring zero multipliers (`DSP=0`).
3. **`VTPE-Grouped` Execution (`flags` bit 2 = 1):** Loaded into `GroupAccumulateScale` with per-group `u16` scales (`1 DSP48E1/row` by default, or `0 DSP` via Yosys `-nodsp`), while `0x0007` (`fp32_aux`) tensors stay on the host for hybrid non-linear ops.
