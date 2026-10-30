# Supported models

Every recurrent-state family with open-weight LLMs runs on the fake-quant reference path (transformers / fla). Packed kernels and the vLLM plugin cover Gated DeltaNet, Mamba2 / SSD and KDA. Results per model: [results](results.md); vLLM details: [vllm](vllm.md).

## Support matrix

| family | update rule (canonical [B, H, d_k, d_v]) | transformers (reference) | fla layers (reference) | packed kernels | vLLM plugin |
|---|---|---|---|---|---|
| G1 Gated DeltaNet | delta, scalar decay / head | Qwen3-Next, Qwen3.5(-MoE), Qwen4-Exp, OLMo-Hybrid (d_k 96, d_v 192, neg. eigenvalues) | `GatedDeltaNet` | yes (`backend='packed'`; any even d_k >= 16, d_v a multiple of 32) | Qwen3.5 / Qwen3-Next, OLMo-Hybrid (Qwen4-Exp allowed, not run); other GDN layers refused |
| G2 KDA | delta, decay / key channel | Kimi Linear, GLM-5-Next (text backbone, gate lower bound), Ling-3.0 / Bailing hybrid (transformers 5.20; load with `statequant.compat`) | `KimiDeltaAttention` | yes (`kernels.kda`) | Ling-3.0 (`BailingMoeV3KimiDeltaAttention`); other KDA layers stock |
| G3 Mamba2 / SSD | additive, scalar decay / head; d_k = d_state, d_v = head_dim | Mamba2, Nemotron-H, Falcon-H1, GraniteMoeHybrid, Bamba, Zamba2 | no (fla's `Mamba2` needs mamba_ssm) | yes (`kernels.mamba2`) | MambaMixer2: Nemotron-H, Falcon-H1, Granite-4.0-H, Zamba2, Mamba2 |
| G5 lightning | additive, constant decay exp(-slope) / head | MiniMax (`MiniMaxLightningAttention`, forward replaced inside `apply`) | - | no | no |
| G6 RWKV-7 | diagonal + rank one, decay / key channel | - | `RWKV7Attention` | no | no |
| G6 RWKV-6 | additive, decay / key channel | - | `RWKV6Attention` | no | no |
| G4 Mamba1 (Jamba, Falcon-Mamba) | | not supported (transformers' scan has no initial state) | | | |

Tested (`tests/test_families.py`, tiny random models from the real config classes; fla rows on GPU only): Identity
through `apply` reproduces the unpatched logits bit for bit (prompt + cached decoding); boundaries fall exactly on the
schedule's positions and each prompt piece starts from the previous rounded state; cached decoding equals a manual loop
that runs the unpatched model token by token and rounds the cached state in place (bit for bit); one prompt call, two
prompt calls and prompt + decoding round at identical positions; `calibrate` + HeadMix / ChannelProtect / StepQuant
end to end; `report`. Real checkpoints, if cached (`tests/test_e2e_families.py`): Mamba2-130m (CPU/GPU), RWKV-7 0.1B
(GPU), Qwen3.5-0.8B (`tests/test_packed_backend.py`); every preset on OLMo-Hybrid-7B in [results](results.md).

Methods x families: `Uniform`, `DSQ`, `HeadMix`, `ChannelProtect` run on every family (decay-based scores use the
head's mean decay over key channels for KDA / RWKV; ChannelProtect's persistence P_u is per channel there). `StepQuant`
runs on every family except RWKV-7 (its row impact assumes a delta or additive transition; RWKV-7 raises
`NotImplementedError`); per-channel decays get a per-key-row lifetime. Not in the papers for families other than GDN.

Per-family notes
- Adapters live in `statequant/families.py` (`Family`: ops to wrap, native <-> canonical state, per-token log decay,
  canonical per-token inputs q / k / v / beta, decode-buffer arguments). At every rounding point the method gets the
  canonical state and a `Piece` (op arguments, `.state_before`, `.position`, `.log_decay()`, `.inputs()`).
- transformers dispatches the GDN / KDA ops to fla's Triton kernels when fla is installed (GPU only); without fla its
  torch fallbacks run on CPU.
- Mamba2: the decode op updates the state in place; a boundary re-rounds it in place. The decode buffer rounds B and x
  (dt exact); the D skip reads the unrounded x.
- Lightning: a multi-token call with a state runs the block algorithm (transformers loops token by token; equal up to
  float rounding). transformers' `MiniMaxCache` reads the sequence length from layer 0, so cached decoding is wrong
  whenever layer 0 is a lightning layer (as in MiniMax-Text-01), with or without statequant.
- Hadamard rotation on an axis that is not a power of two (OLMo-Hybrid d_k 96 / d_v 192) is block diagonal (blocks of
  32 / 64).
- A `Quantizer` group that does not divide its axis is ragged: ceil(n / group) groups, the last one shorter (PACE-4's
  key groups of 64 on OLMo-Hybrid's d_k 96 are 64 + 32, 4.33 instead of 4.25 b/value); bits per value count the extra
  scale (`tests/test_ragged_groups.py`). `AffineGrid` (BDR / LRB's INT4 g128) takes one group per column when d_k is
  below its group (96 on OLMo-Hybrid: 4.67 b/value). DAMP's 16 (INT8) / 32 (INT4) protected key channels are kept as
  counts, i.e. 1/6 and 1/3 of d_k 96 (1/8, 1/4 of 128 on Qwen).
- vLLM plugin: with `STATEQUANT_VLLM` set, GDN layers other than Qwen's and OLMo-Hybrid's, and other callers of the
  GDN state calculators, raise `NotImplementedError` when the model is built (`tests/test_vllm_plugin.py`, CPU).

## Model-specific notes

### OLMo-Hybrid

allenai/Olmo-Hybrid-7B: 24 GDN layers (30 key = 30 value heads, d_k 96, d_v 192, beta = 2 sigmoid(b), i.e. negative
eigenvalues) and 8 attention layers. Everything in this README runs on it: the reference path with every preset, the
packed kernels (padded key tiles, ragged key groups; see the packed decode path above) and the vLLM plugin
(`integrations/vllm_olmo.py`).

Head-dim effects on the formats: PACE-4's key groups of 64 are ragged on 96 (4.33 instead of 4.25 b/value); DAMP keeps
16 / 32 protected key channels, i.e. 1/6 / 1/3 of d_k (`damp_int8` / `damp_int4` 9.96 / 6.00 b/value vs 9.66 /
5.66 on Qwen's 128; `damp_paper_int8` 10.17 vs 9.875); the INT4 g128 grid of `int4_g128` / `bdr_int4` / LRB uses one
group per value column (96 < 128: 4.67 b/value); Hadamard rotations (PACE-4/8's int8 heads, DAMP) are block diagonal (32 on d_k, 64 on d_v); PACE-4/8
puts 8 of 30 heads per layer at int8 (frac 0.25 rounds up); STEPQuant scales its 32 fp16 pivots of 2,304 heads to 10
of 720. LeapQuant and DAMP's paper recipe keep the prompt in full precision, so their prompt delta is zero by
construction.

### Ling-3.0 (KDA)

**Loading Ling-3.0 in transformers 5.20.2.** `inclusionAI/Ling-3.0-*` checkpoints declare `model_type: bailing_hybrid`
(transformers' native `BailingHybridForCausalLM`, which renames the checkpoint's weights itself) and
`architectures: BailingMoeV3ForCausalLM` (the remote code, and vLLM's model class). Native loading fails in two places:
the strict `BailingHybridConfig` rejects the integer `kda_lower_bound: -5`, and the native MLA layer has a full-rank
`q_proj` only, while Ling-3.0-tiny's MLA uses a low-rank query (`q_lora_rank: 256`). `statequant.compat.load_ling3(repo)`
fixes both: the config fields are coerced to float, and `BailingHybridAttention` gets the q-LoRA path. With the fix
the checkpoint loads with no missing or unexpected keys. `compat.from_pretrained` / `compat.load_tokenizer` apply the
fix where a checkpoint needs it (`examples/presets_loss.py` uses them).
- The remote code (`trust_remote_code`) does not run on 5.20.2 as shipped. With import shims and its rotary
  `inv_freq` recomputed (transformers' meta-device init leaves it uninitialised), it agrees with the fixed native model
  to MoE-routing noise. On 1024 WikiText tokens: loss 2.1920 vs 2.1887, argmax agreement 0.976.

```python
from statequant.compat import load_ling3, load_tokenizer
model = load_ling3('inclusionAI/Ling-3.0-tiny').cuda().eval()     # then sq.calibrate / sq.apply as for any model
```
