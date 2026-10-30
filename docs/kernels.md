# Packed kernels

## Status

Status (v0.1): fake-quant reference path, verified bit-exact against the PACE evaluation harness
(`tests/test_vs_pace.py`: formats, DSQ, DAMP masks, head scores; `tests/e2e_vs_pace.py`: PACE-4 and PACE-4/8 logits on
Qwen3.5-0.8B, prompt and cached decoding).

## Gated DeltaNet

Packed decode path (`statequant.kernels`, Triton): int4 / int8 / bf16 state storage with fp16 scales (axis 'k' or 'v'),
a delta-value decode buffer of m tokens (int8 or bf16) and a device-side fold + re-round every m tokens, CUDA-graph
capturable. `unpack(pack(S, q)) == Quantizer(..., scale_dtype='fp16').fake(S)` bit for bit; the step is checked
against a torch reference of the same arithmetic (`tests/test_kernels.py`); latency: `benchmarks/decode_state.py`.

```python
from statequant.kernels import PackedGDNState
st = PackedGDNState(S0, sq.Quantizer(4, 'k', 64), m=8, hv=32, hk=16, buffer_bits=8)   # S0: [B*hv, d_k, d_v]
o = st.step(q, k, v, g, beta, scale)     # one token; q, k raw (l2-normalised inside), g = log decay, beta sigmoided
```

Head dims that are not powers of two (OLMo-Hybrid: d_k 96, d_v 192; any even d_k >= 16, d_v a multiple of 32): the
kernels pad the key tiles to the next power of two (128) and mask them, memory holds exactly d_k x d_v, value rows run
in blocks that divide d_v (the fused step: three blocks of 64). Key-axis groups may be ragged (64 on 96 = 64 + 32, as
the `Quantizer`); value-axis groups must divide d_v. A 96 x 192 run is bit-identical to the same problem zero-padded to
128 x 256 on the power-of-two path (stored codes, scales, buffers, decays; outputs too, up to 7e-8 relative in 0.6% of
entries for int8 at m = 1), and checked against the torch reference, fused vs five-kernel, slot vs lockstep, `read`,
graph replay (`tests/test_kernels_nonpow2.py`). Triton 3.8 fills the masked lanes of a vectorised uint8 load with a
sign-extended `other` (lanes read 0x88, 0xFF, 0xFF, 0xFF), so padded lanes are zeroed explicitly (`_pad0`) where they
could reach a ragged group's scale. The Mamba2 kernel loads padded key tiles the same way (`other=136`); there the
padded lanes only meet zero factors or groups that are not stored (its d_state 96 tests pass), but it was not changed.

### Fused decode step

Fused decode step (`kernels/_fused_triton.py`): the five launches of a packed step (`_prep`, `_state`, `_append`,
`_fold`, `_advance`) as ONE launch with one program per (sequence, value head), which owns the head's whole d_v row:
normalise q, k, the buffer coefficients, the state read, o and u, the append (int8 u with one absmax over the row),
and, when this sequence's buffer is full, the fold + re-rounding in place. GQA: only the group's first value head
writes the shared key slot; the others take this token's key from their own registers for the fold. The counter has
one copy per value head (`NPTR[seq, h]`; the five-kernel step reads copy 0 and writes all), so no atomics are needed
and both steps can be mixed on one state. CUDA-graph capturable. Outputs, codes, scales, buffers, decays and counters
are bit-identical to the five-kernel step (`tests/test_fused.py`: int4 k g64, int4 v g32, int8 k g32, bf16; m 1 / 8 /
16; int8 / bf16 buffer; GQA 2; slot subsets with padding; graph replay), with one exception: the slot path with a bf16
state at m = 1, where the outputs differ in the last fp32 bit in some entries (stored state still exact).
`PackedGDNState(..., fused=None|True|False)`, `SlotLayout(..., fused=...)` (and `step(..., fused=...)`); `None` = auto
(`kernels.fold.use_fused`: fused, except value-axis formats above 128 batch rows x value heads). vLLM plugin:
`STATEQUANT_VLLM_FUSED=auto|1|0` (default auto).

Per-layer decode step, us, Qwen3.5-4B shapes (HV 32, HK 16, d 128), m = 8, CUDA graph of 8 steps (one fold) / 8,
min over 20 repeats, A100 80GB PCIe (`benchmarks/decode_state.py --slots`; F = fused, 5L = five launches):

| batch | fla bf16 in place | fla fp32 in place | bf16 buf8 F (5L) | bf16 buf16 F (5L) | int4 k g64 F (5L) | int8 k g32 F (5L) | int4 v g32 F (5L) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.7 | 4.7 | 6.5 (13.5) | 5.7 (12.5) | 7.2 (14.9) | 7.0 (13.7) | 10.8 (13.9) |
| 2 | 8.3 | 6.9 | 6.6 (14.4) | 5.9 (14.0) | 7.3 (15.7) | 7.3 (14.5) | 11.0 (15.0) |
| 4 | 13.2 | 10.9 | 8.4 (16.8) | 7.8 (16.5) | 8.7 (17.8) | 8.6 (17.3) | 13.1 (17.3) |
| 8 | 22.1 | 18.7 | 10.5 (20.9) | 9.9 (20.4) | 14.0 (22.1) | 13.8 (21.2) | 22.4 (21.7) |
| 16 | 37.6 | 48.8 | 16.5 (29.4) | 15.0 (28.2) | 20.4 (31.2) | 20.6 (30.4) | 32.8 (31.2) |
| 32 | 68.8 | 93.2 | 41.4 (56.2) | 38.1 (55.8) | 35.9 (51.4) | 36.2 (49.5) | 57.1 (50.3) |
| 64 | 132.2 | 181.0 | 73.3 (101.3) | 68.9 (100.7) | 67.9 (92.2) | 80.5 (99.9) | 111.4 (91.2) |
| 128 | 258.1 | 355.5 | 133.0 (200.1) | 126.1 (194.7) | 151.2 (173.4) | 152.7 (192.8) | 227.7 (175.1) |
| 512 | 1094.7 | 1404.3 | 494.1 (765.5) | 469.9 (739.1) | 585.6 (665.0) | 591.5 (765.4) | 888.3 (692.4) |

### Serving layout

Serving layout (`SlotLayout`): one fixed-size byte blob per sequence slot (codes, fp16 scales, int8 key / delta-value
buffer with scales, log decays, a per-sequence fill counter); `step(views, slot_idx, ...)` decodes a batch of
sequences each at its own buffer position (slot <= 0 = padding, folds only the sequences whose buffer filled),
`read` / `write` dequantize (+ pending buffer) and store a state around prefill (`tests/test_slots.py`).

Decode step latency at OLMo-Hybrid-7B's shape (`benchmarks/decode_state.py --gdn-shape olmo-hybrid-7b --slots`; us per
layer per step, CUDA graph of 8 steps (one fold), min over 20 repeats; a shared L40 while other jobs ran, indicative
only; F = fused, 5L = five launches):

| batch | fla bf16 in place | fla fp32 in place | bf16 buf8 F (5L) | bf16 buf16 F (5L) | int4 k g64 F (5L) | int8 k g32 F (5L) | int4 v g32 F (5L) | slot int4 k g64 F (5L) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.4 | 2.9 | 6.4 (8.5) | 4.9 (8.0) | 5.6 (9.3) | 5.2 (8.9) | 9.8 (9.2) | 5.8 (9.9) |
| 8 | 17.5 | 13.7 | 8.2 (14.8) | 6.5 (14.4) | 7.0 (15.5) | 7.3 (14.3) | 12.0 (15.9) | 7.6 (16.8) |
| 64 | 133.8 | 476.6 | 42.5 (61.2) | 45.2 (69.7) | 42.6 (72.4) | 40.5 (58.5) | 77.0 (68.2) | 43.4 (70.2) |
| 256 | 911.2 | 1914.6 | 494.7 (607.3) | 520.6 (632.4) | 192.2 (280.5) | 296.5 (459.8) | 352.8 (301.5) | 191.1 (294.2) |

## Mamba2 / SSD

Mamba2 is the same packed decode without the delta term: per head, `S <- exp(A dt) S + (dt x) B^T`,
`y = S C + D x`, so B (one per group, shared by H/G heads like GQA keys) is the buffered key, C the query,
`u = dt x` the buffered value, and the decay is a scalar per head. The value does not depend on S, so a step reads the
stored state once, against C only. q and k are not normalised.
`kernels/_mamba2_triton.py` does the whole step in one launch per (sequence, head): dt bias, softplus, the
dt_limit clamp and D are applied in the kernel, followed by the state read, the buffer append and, every m tokens,
the fold with re-rounding in place. Head dims that are not powers of two (d_v 80, d_k 96) are padded inside the kernel
and masked; memory holds exactly d_k x d_v. Formats are the same as GDN: int4 / int8 with fp16 scales on the key or
value axis (the group must be a power of two that divides its axis), or bf16. The buffer is int8 or bf16,
m <= 16 (m = 32 works but spills registers in the fold). The kernel is CUDA-graph capturable.

```python
from statequant.kernels import PackedMamba2State, Mamba2SlotLayout, ReferenceMamba2State
st = PackedMamba2State(S0, sq.Quantizer(4, 'k', 64), m=8, hv=H, hk=G)   # S0: [B, H, head_dim, d_state] (vLLM's layout)
y = st.step(x, dt, B, C, A, D, dt_bias, dt_softplus=True)               # x [B,H,P], dt [B,H], B/C [B,G,N], A = -exp(A_log)
```

Tests (`tests/test_mamba2.py`):
- **Kernel vs the torch reference of the same arithmetic.** Covered: int4 / int8 on the k and v axes, bf16; m 1 / 4 / 8 / 16; int8 / bf16 buffer; shapes 8 heads / 2 groups / 128 x 64 and 6 / 2 / 96 x 80.
  - Decays are bit-exact, and so are the buffer entries (0 flips).
  - Codes and scales after each fold are exact except at rounding ties of fp32 sums: <= 8e-6 of int codes, <= 8e-5 of bf16 entries.
  - Outputs agree to <= 4e-7 relative.
- **Slot path vs lockstep:** bit-identical, over random slot subsets with padding rows (slot 0 / -1) and diverging counters.
- **fp32 controls vs the plain fp32 recurrence:** the fp32 / Identity reference matches to 5e-7; the bf16 packed state to 1e-2 (int8 buffer) or 3e-3 (bf16 buffer) relative L2 over 64 tokens.
- **Also covered:** in-kernel gating, one group shared by 64 heads, prefill-side `read` / `write`, and graph replay with vLLM-style strided inputs.

**Decode step latency** (`benchmarks/decode_state.py --mamba2`, vLLM venv). Per-layer decode step in us: CUDA graph
of 8 steps (so one fold per packed arm), min over 20 repeats, A100 80GB PCIe, slot-indexed as in vLLM. The vLLM
column is `selective_state_update` in place.

| shape | batch | vLLM fp32 | vLLM bf16 | bf16 buf8 | bf16 buf16 | int4 k g64 | int8 k g32 | int4 v g32 (g16 for P 80) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Nemotron-H-4B (112 h, 8 g, P 64, N 128) | 1 | 5.8 | 5.8 | 5.5 | 4.9 | 5.8 | 6.1 | 8.4 |
| | 16 | 81.5 | 63.9 | 26.4 | 24.2 | 29.7 | 28.2 | 44.7 |
| | 128 | 642.7 | 504.3 | 202.0 | 200.2 | 232.5 | 223.9 | 322.4 |
| | 512 | 2581.2 | 2015.6 | 781.9 | 782.5 | 910.4 | 869.8 | 1260.9 |
| Nemotron-3-Nano-4B (96 h, 8 g, P 80, N 128) | 1 | 6.0 | 6.1 | 6.1 | 5.2 | 6.7 | 6.9 | 8.8 |
| | 16 | 86.5 | 69.1 | 34.5 | 28.2 | 38.6 | 33.1 | 55.0 |
| | 128 | 792.8 | 525.6 | 225.7 | 210.1 | 285.3 | 258.5 | 407.5 |
| | 512 | 3243.9 | 2178.5 | 870.5 | 817.1 | 1119.9 | 1002.5 | 1602.2 |
| Granite-4.0-H-1B (48 h, 1 g, P 64, N 128) | 1 | 3.6 | 3.6 | 7.3 | 4.0 | 5.1 | 5.5 | 6.9 |
| | 16 | 26.6 | 27.1 | 18.8 | 11.5 | 14.5 | 14.5 | 20.1 |
| | 128 | 280.0 | 218.0 | 130.9 | 90.8 | 101.2 | 99.1 | 140.1 |
| | 512 | 1117.7 | 875.8 | 502.3 | 338.2 | 389.6 | 374.9 | 541.7 |

## KDA

**Packed decode (`kernels/kda.py`, `kernels/_kda_triton.py`).** This is GDN's delta-value buffer with the scalar decays
replaced by per-key-channel vectors. With a_t = exp(g_t) in R^{d_k} and the buffer of (G_i, k_i, u_i) since the last
fold, where G_i in R^{d_k} is the cumulative log decay:
- read: S'^T x = S_0^T (exp(G_n) * x) + sum_i [exp(G_n - G_i) * k_i] . x  u_i, for x = q, k;
- step: u_n = beta (v - S'^T k), o_n = S'^T q + (k.q) u_n;
- fold, every m tokens: S = diag(exp(G_n)) S_0 + sum_i u_i (exp(G_n - G_i) * k_i)^T, then re-rounded.

So the stored state is read once, against the decayed q and k. The buffer keeps G per channel ([m, d_k] fp32 per
head, 64 KiB per layer and slot at m = 8 for Ling's 16 x 128 heads, plus 8 KiB for the running G).
- One launch per (sequence, head) does the whole step: normalise q and k, compute the buffer coefficients (one
  [m, d_k] tile), the state pass (up to four value blocks kept in registers), the append (int8 u with one absmax scale
  over the row, or bf16), and the fold with re-rounding when this sequence's buffer is full.
- No GQA (KDA has none), so every buffer entry and counter copy is owned by one program.
- Formats as for GDN: int4 / int8 with fp16 scales on the key or value axis, or bf16. The buffer is int8 or bf16 and
  m <= 32. d_k and d_v must be powers of two, d_v <= 256. CUDA-graph capturable.
- `PackedKDAState` (lockstep), `KDASlotLayout` (slots, with `read` / `write` around prefill), `ReferenceKDAState`
  (torch, same arithmetic) and `kda_recurrence` (the plain fp32 step).

Tests (`tests/test_kda_kernels.py`, 105 cases):
- **Kernel vs the torch reference.** Covered: int4 / int8 on the k and v axes, bf16; m 1 / 2 / 4 / 8 / 16 / 32; int8 /
  bf16 buffer; shapes 128 x 128, 64 x 32 and 128 x 256.
  - Decays are bit-exact.
  - Buffer entries flip at <= 4.3e-5 of entries.
  - Codes and scales after each fold are exact except at ties of fp32 sums: <= 3.3e-5 of stored int code bytes,
    <= 6.5e-5 of bf16 entries, <= 1.0e-3 of scales (worst: int4 v g16 at m = 1).
  - Outputs agree to <= 3.6e-7 relative.
- **Slot path vs lockstep:** bit-identical, over random slot subsets with padding rows and diverging counters.
- **Controls:**
  - The Identity / fp32-buffer reference equals the plain recurrence to 4e-7.
  - The plain recurrence equals fla's `fused_recurrent_kda` to 3e-7.
  - The bf16 packed state tracks the fp32 recurrence to 8.9e-3 (int8 buffer) or 2.6e-3 (bf16 buffer) relative L2
    over 64 tokens.
- **Also covered:** prefill-side `read` / `write`, and graph replay with vLLM-style strided inputs.
