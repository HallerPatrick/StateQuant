# Methods

Every method is a `StateQuantizer`: a storage format, an optional allocation (which heads / channels get which
format) and a rounding schedule (when the recurrent state is re-rounded). `sq.apply(model, q)` installs it on every
recurrent layer of a supported model ([models](models.md)); `sq.report(model, q)` gives bits per state value and MiB
per sequence.

## Presets

| preset | method | source |
|---|---|---|
| `pace4()`, `pace48()` | PACE-4 (int4 key-axis groups, decode buffer, periodic re-rounding); PACE-4/8 (int8 Hadamard heads by a calibrated head score) | PACE, [arXiv 2610.29545](https://arxiv.org/abs/2610.29545) |
| `damp_int8()`, `damp_int4()` | DAMP-style protected key channels (`ChannelProtect`), as configured in the PACE evaluation harness | [PACE](https://arxiv.org/abs/2610.29545) evaluation harness |
| `damp_paper_int8()`, `damp_paper_int4()` | DAMP's own recipe: fp16 key channels, INT8 sym / INT4 asym g32, calibrated value scaling + reordering, SR | DAMP, [arXiv 2608.27513](https://arxiv.org/abs/2608.27513) v2 |
| `qmamba_dsq()` | decoupled scale quantization (`DSQ`) | Q-Mamba, [Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.551) |
| `nemotron()` | fp16 state with stochastic rounding, every token | Nemotron 3 Super, [arXiv 2604.12374](https://arxiv.org/abs/2604.12374) |
| `stepquant(bits)` | per-head bit allocation from the state's lifetime (`StepQuant`) | STEPQuant, [arXiv 2609.38169](https://arxiv.org/abs/2609.38169), [code](https://github.com/Dreamer-Toby/STEPQuant) |
| `rtn(bits, axis, group)` | round to nearest, every token (baseline) | |
| `int4_g128()`, `bdr_int4()` | INT4 asym g128, every token; + Bounded Directional Rounding | [arXiv 2610.15709](https://arxiv.org/abs/2610.15709) |
| `lrb_bdr(...)` | Low-Rank Bypass + BDR | [arXiv 2610.15709](https://arxiv.org/abs/2610.15709) |
| `leapquant(bits=8\|6\|4)` | per-window quantization + compensator tokens + smoothing | LeapQuant, [arXiv 2609.38166](https://arxiv.org/abs/2609.38166) |
| `gramian(bits, C, allocation)` | calibration-free decay-weighted per-head / per-row widths | [arXiv 2609.30950](https://arxiv.org/abs/2609.30950) |

Per-model quality and bits per value: [results](results.md).

## Usage

```python
import statequant as sq

q = sq.pace48()                          # PACE-4/8; also sq.pace4(), sq.damp_int8(), sq.damp_int4(),
                                         # sq.qmamba_dsq(), sq.nemotron(), sq.rtn(bits, axis, group)
sq.calibrate(model, q, calib_ids)        # only methods with a fitted allocation (HeadMix, ChannelProtect)
with sq.apply(model, q):                 # the method's published schedule, or sq.apply(model, q, sq.Schedule(...))
    out = model.generate(ids, max_new_tokens=256)
sq.report(model, q)                      # bits/value (payload + decode buffer), MiB per sequence
```

Compose your own:

```python
fmt = sq.Quantizer(bits=4, axis='k', group=64)                 # int4, one scale per 64 key rows of a value column
q = sq.HeadMix(lo=fmt, hi=sq.Quantizer(8, 'k', 32, affine=True, rotation='hadamard'), frac=0.25)
sched = sq.Schedule(prefill=256, decode=8, buffer_bits=8)      # round every 256 prompt / 8 generated tokens
```

| piece | options |
|---|---|
| formats | `Quantizer` (bits, axis k/v/head, group, symmetric/affine, int/fp zero point, Hadamard, RTN/SR, scale bits, fp32/fp16 scale), `FloatCast` (bf16/fp16/fp8, SR), `Identity` |
| methods | `Uniform`, `HeadMix` (PACE-4/8), `ChannelProtect` (DAMP), `DSQ` (Q-Mamba) |
| schedules | `Schedule(prefill, decode, buffer_bits, buffer)`, `per_token()`, `periodic()` |

## Stateful methods (`statequant.methods_ext`)

| preset | method | stored bits/value on Qwen3.5 (d_k = d_v = 128) |
|---|---|---|
| `int4_g128()`, `bdr_int4()` | INT4 asym g128 (fp32 step + zero point), every token; + BDR (arXiv 2610.15709) | 4.5 |
| `lrb_bdr(arch='gdn', records='fp16'\|'fp32', bdr=True)` | Low-Rank Bypass + Bounded Directional Rounding (2610.15709, Table 6 configs in `LRB_CONFIGS`) | avg 6.50, peak 8.50 (paper's 1.3B GDN, d_k 256 / d_v 512: 5.25 / 6.00) |
| `leapquant(bits=8\|6\|4, records=...)` | LeapQuant per-window + compensator tokens + smoothing (2609.38166) | 9.5 / 7.5 / 6.5 (+ 4.0 window buffer) |
| `gramian(bits, C=1\|64, allocation='gramian'\|'uniform')` | calibration-free decay-weighted per-head widths (2609.30950) | b + 0.5 + widths |
| `damp_paper_int8()`, `damp_paper_int4()` | DAMP's own recipe: fp16 key channels, INT8 sym / INT4 asym g32 fp32 scales, calibrated value scaling + reordering (2608.27513 v2) | 9.875 / 7.25 |

These methods are *stateful*: they set `stateful = True` and `statequant.apply` calls `method.piece(S, layer, piece)`
after every scan piece and every in-place decode step (a `families.Piece` with `.state_before`, `.position`, `.hit`,
`.start`, `.prompt_end`, `.inputs()`) instead of only at rounding points. They run on every family with a diagonal
decay and a rank-one write (GDN, KDA, Mamba2, lightning, RWKV-6; RWKV-7 raises `NotImplementedError`):
`lrb_bdr()` picks the paper's configuration per family (`FAMILY_ARCH`: Mamba2 k1/m5, lightning = RetNet's), Gramian
allocation uses heads (GDN), rows (KDA) or rows normalised within the gated-norm group (Mamba2). `methods_ext/hooks.py`
lists what each method needs from a family adapter.
