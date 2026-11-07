Setun Numerics Specification (Season 1: Ternary Row Engine)
This document specifies the exact integer arithmetic, bit widths, sign extensions, and rounding rules for the Season 1 hardware datapath (TernaryDot, group accumulator, scale multiply, and TernaryRow). Every hardware module in Blocks D and E is verified bit-exact against setun/ref/ops.py.
1. Parameters and Bit-Width Summary
Stage
Symbol
Formula / Type
Width ($A=8, L=64, G=128$)
Signed?
Activation input
$x_i$
Two's complement intA
8 bits ([-128, 127])
Signed
Trit code input
$c_i$
{00: 0, 01: +1, 10: -1, 11: 0}
2 bits
Unsigned
Single-lane output
$y_i$
${-x_i, 0, +x_i}$
$A + 1 = 9$ bits ([-128, +128])
Signed
$L$-lane TernaryDot sum
$d_b$
$\sum_{l=0}^{L-1} y_l$
$\lceil \log_2(L \cdot 2^{A-1} + 1) \rceil + 1 = 15$ bits
Signed
Group partial sum
$p_g$
$\sum_{b=0}^{G/L - 1} d_b$
$\lceil \log_2(G \cdot 2^{A-1} + 1) \rceil + 1 = 16$ bits
Signed
Group scale mantissa
$s_g$
Fixed-point mantissa (ADR-011)
16 bits ([0, 65535])
Unsigned
Scaled group product
$m_g$
$p_g \times s_g$
$16 + 16 = 32$ bits
Signed
Cross-group row accumulator
$\text{acc}$
$\sum_{g=0}^{N_{\text{groups}}-1} m_g$
48 bits (supports up to $2^{16}$ groups without overflow)
Signed
Shifted & rounded row result
$\text{out}$
$\text{round_shift}(\text{acc}, \text{shift})$
32 bits (int32)
Signed
2. Stage-by-Stage Arithmetic Definitions
2.1 Single Lane (lane_select)
Given an $A$-bit signed activation $x \in [-2^{A-1}, 2^{A-1}-1]$ (for $A=8$, $x \in [-128, 127]$) and a 2-bit trit code $c \in {0, 1, 2, 3}$:

Extend $x$ to $A + 1$ signed bits before negation so that $x = -128$ negates cleanly to $+128$ without two's complement overflow: $$y = \begin{cases} +x & \text{if } c = \text{0b01} \ -x & \text{if } c = \text{0b10} \ 0 & \text{if } c \in {\text{0b00}, \text{0b11}} \end{cases}$$
Range for $A=8$: $y \in [-128, +128]$, fitting strictly inside a 9-bit signed integer ([-256, 255]).
2.2 $L$-Lane Dot Product (ternary_dot)
Given $L$ lane trits and $L$ signed $A$-bit activations: $$d = \sum_{l=0}^{L-1} \text{lane_select}(c_l, x_l)$$

Maximum magnitude occurs when all $L$ activations are $-2^{A-1}$ (e.g. $-128$) and all $L$ trits are $-1$ (0b10), giving $d_{\max} = +L \cdot 2^{A-1}$.
Minimum value occurs when all $L$ activations are $-2^{A-1}$ and all $L$ trits are $+1$ (0b01), giving $d_{\min} = -L \cdot 2^{A-1}$.
Required signed bit width: $$W_{\text{dot}}(L, A) = \lceil \log_2(L \cdot 2^{A-1} + 1) \rceil + 1 = \log_2(L) + A + 1 \quad (\text{for power-of-two } L)$$ For $A=8$: $L=4 \to 11$ bits; $L=16 \to 13$ bits; $L=64 \to 15$ bits; $L=128 \to 16$ bits.
2.3 Group Accumulator (group_accumulate)
A group of $G$ elements is processed over $B = G / L$ beats: $$p_g = \sum_{b=0}^{G/L - 1} d_b$$

Maximum value: $+G \cdot 2^{A-1}$ (e.g., for $G=128, A=8$, $+128 \times 128 = +16384$).
Minimum value: $-G \cdot 2^{A-1}$ ($-16384$).
Required signed bit width: $$W_{\text{group}}(G, A) = \lceil \log_2(G \cdot 2^{A-1} + 1) \rceil + 1$$ For $A=8$:
$G = 64 \implies \lceil \log_2(8193) \rceil + 1 = 14 + 1 = 15$ bits ([-16384, 16383], max $+8192$ fits).
$G = 128 \implies \lceil \log_2(16385) \rceil + 1 = 15 + 1 = 16$ bits ([-32768, 32767], max $+16384$ fits).
$G = 256 \implies \lceil \log_2(32769) \rceil + 1 = 16 + 1 = 17$ bits ([-65536, 65535], max $+32768$ fits).
2.4 Scale Multiply & Rounding (apply_scale)
Per ADR-011, each group partial sum $p_g$ is multiplied by its unsigned 16-bit group scale $s_g \in [0, 65535]$: $$m_g = p_g \cdot s_g$$

When accumulating across groups (row_dot), the unshifted products $m_g$ are summed in a 48-bit signed row accumulator: $$\text{acc} = \sum_{g=0}^{N_{\text{groups}}-1} (p_g \cdot s_g)$$
At the end of the row (or in standalone apply_scale with a right-shift shift $\ge 0$), the shared right-shift is applied using round-half-up (add $2^{\text{shift}-1}$ before arithmetic right shift): $$\text{round_shift}(v, \text{shift}) = \begin{cases} v & \text{if } \text{shift} = 0 \ (v + (1 \ll (\text{shift} - 1))) \gg \text{shift} & \text{if } \text{shift} > 0 \end{cases}$$
2.5 Ragged-$K$ Padding
When $K$ is not a multiple of $G$, both the weight trits and activations for indices $k \in [K, N_{\text{groups}} \cdot G - 1]$ are padded with 0 (0b00 for trits).
3. Hand-Computed Worked Example
Let $L = 4$, $G = 8$ ($B = G / L = 2$ beats per group), $A = 8$, $K = 6$ (ragged $K < G$, so $N_{\text{groups}} = 1$), scale_u16 = 300, and shift = 4:

Logical trits ($K=6$): w = [+1, -1, 0, +1, -1, -1] $\to$ padded to $G=8$: [+1, -1, 0, +1,  -1, -1, 0, 0]
Logical activations ($K=6$): x = [-128, 50, 99, 30,  -20, 10] $\to$ padded to $G=8$: [-128, 50, 99, 30,  -20, 10, 0, 0]
Beat 0 (lanes 0..3):
Lane 0: $w_0 = +1, x_0 = -128 \implies y_0 = -128$
Lane 1: $w_1 = -1, x_1 = 50 \implies y_1 = -50$
Lane 2: $w_2 = 0, x_2 = 99 \implies y_2 = 0$
Lane 3: $w_3 = +1, x_3 = 30 \implies y_3 = +30$
ternary_dot beat 0: $d_0 = -128 - 50 + 0 + 30 = -148$
Beat 1 (lanes 4..7, with lanes 6..7 padded to 0):
Lane 4: $w_4 = -1, x_4 = -20 \implies y_4 = +20$
Lane 5: $w_5 = -1, x_5 = 10 \implies y_5 = -10$
Lane 6: $w_6 = 0, x_6 = 0 \implies y_6 = 0$
Lane 7: $w_7 = 0, x_7 = 0 \implies y_7 = 0$
ternary_dot beat 1: $d_1 = 20 - 10 + 0 + 0 = +10$
Group Accumulate ($G=8$): $$p_0 = d_0 + d_1 = -148 + 10 = -138$$
Scale Multiply (scale_u16 = 300, shift = 4):
Unshifted product: $m_0 = -138 \times 300 = -41400$
Round-half-up shift (shift = 4, rounding bias $= 1 \ll 3 = 8$): $$(-41400 + 8) \gg 4 = -41392 \gg 4 = -2587$$
