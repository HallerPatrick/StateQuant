# statequant

[![PyPI](https://img.shields.io/pypi/v/statequant)](https://pypi.org/project/statequant/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/HallerPatrick/StateQuant/blob/main/LICENSE)

Recurrent-state quantization for linear-attention and SSM hybrid language models. statequant implements published
state-quantization methods (PACE, DAMP, STEPQuant, Q-Mamba DSQ, LRB + BDR, LeapQuant, decay-weighted allocation, ...)
behind one interface and runs them on every recurrent-state family with open-weight LLMs: Gated DeltaNet, KDA,
Mamba2 / SSD, MiniMax lightning and RWKV-6 / RWKV-7.

- **Reference path** (transformers / fla): every method on every supported model, for evaluation.
- **Packed Triton kernels**: int4 / int8 / bf16 state storage with a decode buffer, for Gated DeltaNet, Mamba2 and KDA.
- **vLLM plugin**: serve Qwen3.5 / Qwen3-Next, OLMo-Hybrid, Nemotron-H-style Mamba2 hybrids and Ling-3.0 with a packed
  recurrent state.

## Installation

```bash
pip install statequant              # core (torch)
pip install "statequant[hf]"        # + transformers, for the reference path
pip install "statequant[hf,fla]"    # + flash-linear-attention kernels (GPU)
pip install "statequant[vllm]"      # + vLLM, for the serving plugin
```

## Quickstart

```python
import statequant as sq

q = sq.pace4()                           # or sq.pace48(), sq.damp_int8(), sq.lrb_bdr(), sq.leapquant(8), ...
sq.calibrate(model, q, calib_ids)        # only for methods with a fitted allocation
with sq.apply(model, q):                 # the method's published rounding schedule
    out = model.generate(ids, max_new_tokens=256)
print(sq.report(model, q))               # bits per state value, MiB per sequence
```

Serving with vLLM:

```bash
STATEQUANT_VLLM=pace4 vllm serve Qwen/Qwen3.5-4B --no-enable-prefix-caching
```

## Implemented methods

| method | preset | reference |
|---|---|---|
| PACE-4, PACE-4/8 | `pace4()`, `pace48()` | Haller et al., [arXiv 2610.29545](https://arxiv.org/abs/2610.29545) |
| DAMP (paper recipe) | `damp_paper_int8()`, `damp_paper_int4()` | Zhang et al., [arXiv 2608.27513](https://arxiv.org/abs/2608.27513) |
| DAMP (PACE evaluation variant) | `damp_int8()`, `damp_int4()` | as configured in [PACE](https://arxiv.org/abs/2610.29545) |
| STEPQuant | `stepquant(4)`, `stepquant(6)` | Yao et al., [arXiv 2609.38169](https://arxiv.org/abs/2609.38169); [code](https://github.com/Dreamer-Toby/STEPQuant) |
| Q-Mamba DSQ | `qmamba_dsq()` | Chen et al., [Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.551) |
| LRB + BDR (Low-Rank Bypass, Bounded Directional Rounding) | `lrb_bdr()`, `bdr_int4()` | Li et al., [arXiv 2610.15709](https://arxiv.org/abs/2610.15709) |
| LeapQuant | `leapquant(8\|6\|4)` | Pan et al., [arXiv 2609.38166](https://arxiv.org/abs/2609.38166) |
| Decay-weighted (Gramian) bit allocation | `gramian(bits)` | [arXiv 2609.30950](https://arxiv.org/abs/2609.30950) |
| fp16 + stochastic rounding (Nemotron 3) | `nemotron()` | NVIDIA, [arXiv 2604.12374](https://arxiv.org/abs/2604.12374) |
| Round to nearest (baseline) | `rtn(bits, axis, group)`, `int4_g128()` | |

Formats, schedules and how to compose your own: [docs/methods.md](https://github.com/HallerPatrick/StateQuant/blob/main/docs/methods.md). Per-model quality of every
method: [docs/results.md](https://github.com/HallerPatrick/StateQuant/blob/main/docs/results.md).

## Supported models

| family | models | reference path | packed kernels | vLLM plugin |
|---|---|:---:|:---:|:---:|
| Gated DeltaNet | Qwen3.5, Qwen3-Next, Qwen4-Exp, OLMo-Hybrid | ✓ | ✓ | Qwen3.5 / Qwen3-Next, OLMo-Hybrid |
| KDA | Ling-3.0, Kimi Linear, GLM-5-Next | ✓ | ✓ | Ling-3.0 |
| Mamba2 / SSD | Nemotron-H / Nemotron-3, Falcon-H1, Granite-4.0-H, Bamba, Zamba2, Mamba2 | ✓ | ✓ | MambaMixer2 models |
| lightning | MiniMax | ✓ | | |
| RWKV-7 / RWKV-6 | fla models | ✓ | | |

Details per family, method coverage and model-specific notes:
[docs/models.md](https://github.com/HallerPatrick/StateQuant/blob/main/docs/models.md).

## Results

vLLM 0.32 end to end, packed state vs stock:

![vLLM concurrency and decode loss](https://raw.githubusercontent.com/HallerPatrick/StateQuant/main/docs/figures/vllm_e2e.png)

Decode-step latency of the packed kernels:

![decode step latency](https://raw.githubusercontent.com/HallerPatrick/StateQuant/main/docs/figures/decode_latency.png)

Full tables and setups: [docs/results.md](https://github.com/HallerPatrick/StateQuant/blob/main/docs/results.md).

## Documentation

- [Methods](https://github.com/HallerPatrick/StateQuant/blob/main/docs/methods.md): presets, formats, schedules,
  stateful methods
- [Supported models](https://github.com/HallerPatrick/StateQuant/blob/main/docs/models.md): support matrix,
  per-family notes
- [Packed kernels](https://github.com/HallerPatrick/StateQuant/blob/main/docs/kernels.md): storage layout, fused decode
  step, tests, latency
- [vLLM plugin](https://github.com/HallerPatrick/StateQuant/blob/main/docs/vllm.md): configuration, restrictions,
  end-to-end checks
- [Results](https://github.com/HallerPatrick/StateQuant/blob/main/docs/results.md): every preset per model

## Citation

If you use statequant, please cite:

```bibtex
@article{haller2026pace,
  title   = {{PACE}: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States},
  author  = {Haller, Patrick and Laitenberger, Filipe and Akbik, Alan},
  journal = {arXiv preprint arXiv:2610.29545},
  year    = {2026}
}

@software{haller2026statequant,
  title   = {statequant: Recurrent-State Quantization for Hybrid Language Models},
  author  = {Haller, Patrick},
  url     = {https://github.com/HallerPatrick/StateQuant},
  version = {0.1.1},
  year    = {2026}
}
```

Please also cite the papers of the methods you use (listed in
[docs/methods.md](https://github.com/HallerPatrick/StateQuant/blob/main/docs/methods.md)).

## License

Apache-2.0. See [LICENSE](https://github.com/HallerPatrick/StateQuant/blob/main/LICENSE).
