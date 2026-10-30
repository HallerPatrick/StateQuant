# vLLM plugin

vLLM plugin (`statequant.integrations.vllm_plugin`, entry point `vllm.general_plugins`): with
`STATEQUANT_VLLM=pace4|int4v|int8k|bf16` (optional `STATEQUANT_VLLM_M=8`, `STATEQUANT_VLLM_BUFFER_BITS=8`) Qwen3.5 /
Qwen3-Next and OLMo-Hybrid GDN layers keep their recurrent state in the packed slot layout (any other GDN layer is
refused), vLLM's memory planner sizes the cache from it, and decode runs the slot kernels (CUDA-graph capturable;
vLLM wraps OLMo-Hybrid's whole GDN layer in one custom op, `olmo_hybrid_gdn_full_forward`); prefill starts
vLLM's chunk kernel from the dequantized slot and stores the final state packed. Unset = stock vLLM. Prefix caching and
speculative decoding must be off. End-to-end checks: `tests/vllm_packed_check.py` (`--model allenai/Olmo-Hybrid-7B
--out runs_vllm_olmo` for OLMo-Hybrid). OLMo-Hybrid specifics are in `integrations/vllm_olmo.py` and its section
below.

| layer type | models (vLLM) | formats | dims | decode | prefill |
|---|---|---|---|---|---|
| gated delta (GDN) | Qwen3.5 / Qwen3-Next, OLMo-Hybrid | int4 / int8 k / v, bf16; buffer int8 / bf16 | d_k even >= 16 (padded, masked; key groups may be ragged), d_v a multiple of 32 | fused or 5-launch slot step | chunk kernel from dequantized slot |
| Mamba2 / SSD | MambaMixer2: NemotronH, Falcon-H1, Granite-4.0-H, Zamba2, Mamba2 | same | any (padded, masked) | one-launch slot step | `mamba_chunk_scan_combined_varlen` from dequantized slot |
| KDA | Ling-3.0 (`BailingMoeV3KimiDeltaAttention`) | same | d_k = d_v powers of two, d_v <= 256 | one-launch slot step | `chunk_kda` from dequantized slot |

## Qwen3.5-4B

**vLLM 0.32 end to end** (`tests/vllm_packed_check.py`, A100, m = 8, int8 buffer, 32 WikiText-103 windows; concurrency at a fixed 16 GiB KV budget).

| engine | prefill loss | teacher-forced decode loss | max concurrency 4232 tok (16 GiB KV) |
|---|---:|---:|---:|
| stock fp32 | 2.3554 | 2.1155 | 82.67x |
| stock bf16 | 2.3554 | 2.1154 | 101.42x |
| plugin bf16 | 2.3554 | 2.1102 | 101.11x |
| int8k | 2.3554 | 2.1102 | 106.36x |
| pace4 | 2.3554 | 2.1199 | 113.77x |
| int4v | 2.3549 | 2.1090 | 114.17x |

## Mamba2 / SSD

vLLM: with `STATEQUANT_VLLM=<format>`, every `MambaMixer2` layer (NemotronH / Nemotron-3, Falcon-H1, Granite-4.0-H,
Zamba2, Mamba2) keeps its ssm state in `Mamba2SlotLayout`. To pack only some layer types, set
`STATEQUANT_VLLM_LAYERS=gdn,mamba2` (the default is all of them). Decode rows run the slot kernel. Prefill rows
dequantize the slot plus its pending buffer to fp32, run vLLM's `mamba_chunk_scan_combined_varlen`, and store the
final state packed. The state shape and dtype calculators are patched only for calls from `MambaMixer2` and the
models built on it, so other callers get vLLM's stock answer. A value-axis group that does not divide head_dim is
halved until it does (int4v g32 becomes g16 on head_dim 80). Restrictions (asserted): TP = 1, no speculative
decoding, no prefix caching, no ReplaySSM. E2E check: `tests/vllm_mamba2_check.py`.

**End-to-end in vLLM 0.32.** Model: NVIDIA-Nemotron-3-Nano-4B-BF16, a hybrid with 96 heads x head_dim 80 x d_state 128 and 8 groups. The run used a shared L40 with m = 8, an int8 buffer and 32 WikiText-103 windows; concurrency is for 4232-token requests at a fixed 6 GiB KV budget (`tests/vllm_mamba2_check.py`, summary in docs/figures/data/vllm_e2e.csv).
- `stock` keeps vLLM's default fp32 ssm state for NemotronH.
- Greedy agreement is a noisy measure on this model: stock fp32 and stock bf16 agree on only 0.82 of tokens.

| engine | ssm state B/slot/layer | prefill loss | teacher-forced decode loss | greedy agr. vs stock bf16 | mixed-batch agr. vs stock bf16 | max concurrency |
|---|---:|---:|---:|---:|---:|---:|
| stock (fp32) | 3932160 | 2.6175 | 2.5245 | 0.820 | 0.793 | 36.6x |
| stock bf16 | 1966080 | 2.6177 | 2.5256 | - | - | 52.8x |
| plugin bf16 (control) | 2041216 | 2.6180 | 2.5256 | 0.615 | 0.731 | 49.6x |
| int8 k g32 | 1119616 | 2.6181 | 2.5248 | 0.701 | 0.768 | 65.0x |
| int4 k g64 (pace4) | 597376 | 2.6179 | 2.5366 | 0.455 | 0.479 | 72.1x |
| int4 v g16 (int4v) | 689536 | 2.6183 | 2.5512 | 0.363 | 0.372 | 70.6x |

The concurrency gain is capped by the 4 attention layers' KV cache at this context length and by vLLM's hybrid page
alignment.

Limitations (Mamba2): TP > 1 is not supported, even though group replication is in place. There is no HF / fla
reference path here; it is planned in `patch.py`. The bf16 state with an int8 buffer is slower than with a bf16
buffer when one group serves many heads (Granite G = 1: 7.3 vs 4.0 us at batch 1). Value-axis formats need a group
that divides head_dim.

## OLMo-Hybrid

**vLLM 0.32 end to end** (`tests/vllm_packed_check.py --model allenai/Olmo-Hybrid-7B --out runs_vllm_olmo`, L40,
m = 8, int8 buffer). Losses: 32 WikiText-103 windows, prompt logprobs over 640 tokens (prefill path) and the last 128
of them teacher-forced after a 512-token prompt (decode path). Greedy agreement: 8 prompts x 64 tokens; mixed batch:
48 concurrent requests with chunked prefill (256 tokens / step), both against stock vLLM (bf16 state, OLMo-Hybrid's
default). Concurrency: vLLM's reported maximum for requests of 4232 / 648 tokens at a fixed 6 GiB KV budget
(`mem`). HF column: the same decode windows through statequant's HF packed backend (`vllm_packed_check.py hf`, the
same kernels and formats, one rounding at the end of the prompt).

| engine | GDN state B/slot/layer | prefill loss | teacher-forced decode loss | HF packed backend | greedy agr. | mixed-batch agr. | max concurrency 4232 / 648 tok |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock (bf16) | 1105920 | 2.0664 | 1.8002 | 1.8006 | - | - | 11.70x / 54.58x |
| stock fp32 (`mamba_ssm_cache_dtype='float32'`) | 2211840 | 2.0664 | 1.8003 | | 0.762 | 0.775 | 10.90x / 40.88x |
| plugin bf16 (control) | 1177216 | 2.0664 | 1.8004 | 1.8003 | 0.877 | 0.733 | 11.38x / 54.60x |
| int8 k g32 (`int8k`) | 658816 | 2.0662 | 1.8003 | 1.8002 | 0.877 | 0.758 | 11.87x / 64.24x |
| int4 k g64 (`pace4`, ragged 64 + 32) | 370816 | 2.0666 | 1.8049 | 1.8051 | 0.852 | 0.509 | 12.04x / 68.25x |
| int4 v g32 (`int4v`) | 382336 | 2.0665 | 1.7999 | | 0.711 | 0.445 | 12.04x / 68.25x |

Greedy agreement is a noisy measure here too: stock fp32 and stock bf16 agree on 0.762 of the tokens. At 4232 tokens the
8 attention layers' KV cache (122,880 B / token) dominates a sequence's memory, so the smaller recurrent state buys
little; at 648 tokens int4 gives 1.25x stock bf16's concurrency (1.67x stock fp32's). pace4 and int4v reach the same
concurrency because vLLM's hybrid page alignment rounds both to the same block size.

vLLM 0.32 cannot load allenai/Olmo-Hybrid-7B as shipped: the AutoWeightsLoader passes the q / k / v conv shard ids on
the tensor (`weight.shard_id`) while OLMo-Hybrid's fused conv1d weight loader expects an argument (`TypeError: list
indices must be integers or slices, not NoneType`). `integrations/vllm_extras.py` fixes that loader by default
(`STATEQUANT_VLLM_OLMO_CONV=0` turns the fix off); the stock rows above ran with it.

Limitations (OLMo-Hybrid): TP > 1 is not tested (the layout is per rank; heads split as vLLM splits them). Value-axis
groups must divide d_v (32 or 64 on 192; 128 is refused). Speculative decoding and prefix caching must be off, as for
Qwen. The fused int4 value-axis step is slower than the five-kernel step from batch 8 here as well (the auto policy
switches above 128 rows x heads).

## KDA (Ling-3.0)

**vLLM.** With `STATEQUANT_VLLM=<format>`, every `BailingMoeV3KimiDeltaAttention` layer keeps its recurrent state in
`KDASlotLayout`. Select it with `STATEQUANT_VLLM_LAYERS=kda` (it is included in the default of all types).
- Decode rows run the slot kernel. The gate comes from vLLM's `fused_kda_gate`, which applies the safe gate and the
  lower bound.
- Prefill rows dequantize the slot plus its pending buffer, run vLLM's `chunk_kda`, and store the final state packed.
  In mixed batches the decode rows take the packed step.
- The KDA state calculators are patched only for calls from vLLM's `bailing_moe_v3` module. vLLM's other KDA layers
  (Kimi Linear's `KimiGatedDeltaNetAttention`, GLM-5-Next, Kimi-K3) get the stock state and are not packed.
- Restrictions (asserted): TP = 1, no speculative decoding / MTP, no prefix caching.
- E2E check: `tests/vllm_kda_check.py`; unit tests: `tests/test_vllm_kda_plugin.py`.

**End-to-end in vLLM 0.32.** Model: Ling-3.0-tiny on a shared A100, m = 8, int8 buffer, 16 WikiText-103 windows
(prefill loss over 640 tokens; teacher-forced decode of the last 128 tokens after a 512-token prompt). Greedy runs
8 prompts x 64 tokens; the mixed batch is 48 requests with chunked prefill. Concurrency is for 4232-token requests at
a fixed 6 GiB KV budget. Results are in docs/figures/data/vllm_e2e.csv.
- vLLM 0.32 always keeps Ling's KDA state in fp32: `mamba_ssm_cache_dtype='bfloat16'` is not passed to
  `kda_state_dtype` by `bailing_moe_v3`. The "stock (2nd run)" row is therefore a second fp32 run and shows the
  run-to-run noise.

| engine | KDA state B/slot/layer | prefill loss | teacher-forced decode loss | greedy agr. vs stock | mixed-batch agr. vs stock | max concurrency |
|---|---:|---:|---:|---:|---:|---:|
| stock (fp32) | 1048576 | 2.3916 | 2.1357 | - | - | 113.75x |
| stock (2nd run; bf16 request ignored) | 1048576 | 2.3928 | 2.1370 | 0.982 | 1.000 | 113.75x |
| plugin bf16 (control) | 631360 | 2.3921 | 2.1305 | 0.828 | 0.582 | 145.60x |
| int8 k g32 | 385600 | 2.3922 | 2.1327 | 0.775 | 0.619 | 161.80x |
| int4 k g64 (pace4) | 246336 | 2.3931 | 2.1620 | 0.451 | 0.301 | 182.00x |
| int4 v g32 (int4v) | 254528 | 2.3911 | 2.1415 | 0.602 | 0.333 | 182.00x |

The prefill loss rounds the state only once, at the end of each window, so it barely moves. The concurrency gain is
capped by the 6 MLA layers' KV cache and by vLLM's hybrid page alignment: int4 k and int4 v land on the same page
size.

Limitations (KDA):
- TP > 1, MTP / speculative decoding and prefix caching are not supported.
- Only Ling-3.0's vLLM layer is packed.
- The HF `backend='packed'` path is GDN only; on the HF side KDA runs the fake-quant reference.
- The per-channel decay buffer costs 72 KiB per layer and slot at m = 8 (16 heads x 128): 29% of the int4 k g64
  slot (246336 B).
- No latency table: the GPU was shared during this work.
