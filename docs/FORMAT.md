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
