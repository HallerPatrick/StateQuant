"""Loading fixes for checkpoints whose native transformers class does not match them yet.

Ling-3.0 (inclusionAI ``Ling-3.0-tiny`` / ``-flash``, ``architectures: BailingMoeV3ForCausalLM``, ``model_type:
bailing_hybrid``) maps to transformers' native ``BailingHybridForCausalLM`` (5.20), which converts the checkpoint's
weight names itself, but in 5.20.2:

  * ``BailingHybridConfig`` is a strict dataclass and rejects the checkpoint's integer ``kda_lower_bound: -5``
    (and would reject other int-for-float fields), so ``from_pretrained`` fails before building the model;
  * its MLA layer (``BailingHybridAttention``) has a full-rank ``q_proj`` only, while Ling-3.0-tiny's MLA uses a
    low-rank query (``q_lora_rank: 256``: ``q_a_proj`` -> RMSNorm ``q_a_layernorm`` -> ``q_b_proj``, as the
    checkpoint's remote code and vLLM's ``BailingMoeV3MLAAttention``). Loaded as is, those weights would be dropped
    and ``q_proj`` left at its random init.

``load_ling3(repo)`` fixes both: the config is read from ``config.json``, its float fields coerced, and
``patch_bailing_hybrid()`` (idempotent) gives ``BailingHybridAttention`` the q-LoRA path whenever
``config.q_lora_rank`` is set (the full-rank path is unchanged otherwise). Checked against the checkpoint's own
remote code (``trust_remote_code``) on Ling-3.0-tiny.
"""
from __future__ import annotations

import json
import types

import torch

_FLOAT_FIELDS = ('kda_lower_bound', 'rope_theta', 'routed_scaling_factor', 'rms_norm_eps', 'initializer_range',
                 'attention_dropout', 'mtp_loss_scaling_factor')


def ling3_config(repo, **kw):
    """``BailingHybridConfig`` of a Ling-3.0 checkpoint (repo id or local dir), float fields coerced."""
    from transformers import BailingHybridConfig
    from transformers.utils import cached_file
    d = json.load(open(cached_file(repo, 'config.json')))
    for f in _FLOAT_FIELDS:
        if isinstance(d.get(f), int) and not isinstance(d.get(f), bool):
            d[f] = float(d[f])
    for f in ('auto_map', 'architectures'):                   # the remote-code classes; we build the native one
        d.pop(f, None)
    d['partial_rotary_factor'] = 1.0                          # RoPE over all qk_rope_head_dim dims, as the remote
                                                              # code (the native default RoPE ignores it anyway)
    d.update(kw)
    return BailingHybridConfig.from_dict(d)


def _q_lora(self, hidden_states):
    return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))


def patch_bailing_hybrid():
    """Give transformers' ``BailingHybridAttention`` (MLA) the low-rank query path of ``config.q_lora_rank``."""
    from transformers.models.bailing_hybrid import modeling_bailing_hybrid as M
    cls = M.BailingHybridAttention
    if getattr(cls, '_statequant_q_lora', False):
        return
    init0 = cls.__init__

    def __init__(self, config, layer_idx):
        init0(self, config, layer_idx)
        r = getattr(config, 'q_lora_rank', None)
        if r:
            del self.q_proj                                   # the forward calls self.q_proj(hidden_states)
            bias = bool(getattr(config, 'use_qkv_bias', False))
            self.q_a_proj = torch.nn.Linear(config.hidden_size, r, bias=bias)
            self.q_a_layernorm = M.BailingHybridRMSNorm(r)  # eps 1e-6, as the checkpoint's code
            self.q_b_proj = torch.nn.Linear(r, self.num_heads * self.qk_head_dim, bias=False)
            self.q_proj = types.MethodType(_q_lora, self)
    cls.__init__ = __init__
    cls._statequant_q_lora = True


def load_ling3(repo, dtype=torch.bfloat16, **kw):
    """Ling-3.0 as transformers' native ``BailingHybridForCausalLM`` (fixed as above)."""
    from transformers import BailingHybridForCausalLM
    patch_bailing_hybrid()
    return BailingHybridForCausalLM.from_pretrained(repo, config=ling3_config(repo), dtype=dtype, **kw)


def load_tokenizer(repo, **kw):
    """``AutoTokenizer`` for any checkpoint (AutoTokenizer reads the config too, which fails for Ling-3.0)."""
    from transformers import AutoTokenizer
    if needs_ling3_fix(repo):
        kw.setdefault('config', ling3_config(repo))
    return AutoTokenizer.from_pretrained(repo, **kw)


def needs_ling3_fix(repo) -> bool:
    """True for a Ling-3.0 (``BailingMoeV3ForCausalLM`` / ``bailing_hybrid``) checkpoint."""
    try:
        from transformers.utils import cached_file
        d = json.load(open(cached_file(repo, 'config.json')))
    except Exception:
        return False
    return d.get('model_type') == 'bailing_hybrid' or 'BailingMoeV3ForCausalLM' in (d.get('architectures') or [])


def from_pretrained(repo, dtype=torch.bfloat16, **kw):
    """``AutoModelForCausalLM.from_pretrained`` with the fixes of this module where a checkpoint needs them."""
    if needs_ling3_fix(repo):
        return load_ling3(repo, dtype=dtype, **kw)
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, **kw)


__all__ = ['ling3_config', 'patch_bailing_hybrid', 'load_ling3', 'load_tokenizer', 'needs_ling3_fix', 'from_pretrained']
