"""Tiny randomly initialised models from the real transformers config classes (no weight downloads), one or more per
recurrent-state family. ``build(name)`` returns an eval-mode fp32 model on CPU; ``FAMILY[name]`` its family."""
import torch

V = 256                                                  # vocabulary of every tiny model


def _qwen3_next():
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    cfg = Qwen3NextConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_num_key_heads=2,
                          linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                          num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
                          shared_expert_intermediate_size=32, layer_types=['linear_attention', 'full_attention'])
    return Qwen3NextForCausalLM(cfg)


def _qwen4_exp():
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
    cfg = Qwen4ExpTextConfig(vocab_size=V, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
                             linear_key_head_dim=32, linear_value_head_dim=32, num_experts=4, num_experts_per_tok=2,
                             moe_intermediate_size=32, shared_expert_intermediate_size=32, hc_count=2, hc_lowrank=8,
                             layer_types=['linear_attention', 'full_attention'], ngram_vocab_size_base=1024,
                             split_ngram_parts=8, heads_per_ngram=2, indexer_n_heads=2, indexer_kv_heads=1,
                             indexer_head_dim=32, indexer_budget=64, indexer_compress_ratio=4)
    return Qwen4ExpForCausalLM(cfg)


def _olmo_hybrid():
    from transformers import OlmoHybridConfig, OlmoHybridForCausalLM
    cfg = OlmoHybridConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                           num_attention_heads=2, linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=96, linear_value_head_dim=192, linear_allow_neg_eigval=True,
                           layer_types=['linear_attention', 'full_attention'], pad_token_id=0, eos_token_id=1)
    return OlmoHybridForCausalLM(cfg)


def _kimi_linear():
    from transformers import KimiLinearConfig, KimiLinearForCausalLM
    cfg = KimiLinearConfig(vocab_size=V, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, kv_lora_rank=16,
                           qk_rope_head_dim=8, v_head_dim=16, qk_nope_head_dim=16, num_local_experts=4,
                           num_experts_per_tok=2, linear_head_dim=32, linear_num_heads=2,
                           layer_types=['linear_attention', 'full_attention'], pad_token_id=0, bos_token_id=1,
                           eos_token_id=2)
    return KimiLinearForCausalLM(cfg)


def _bailing_hybrid():
    from transformers import BailingHybridConfig, BailingHybridForCausalLM
    cfg = BailingHybridConfig(vocab_size=V, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                              num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, kv_lora_rank=16,
                              qk_rope_head_dim=8, v_head_dim=16, qk_nope_head_dim=16, num_local_experts=4,
                              num_experts_per_tok=2, n_group=1, topk_group=1, first_k_dense_replace=1, head_dim=32,
                              layer_group_size=2, pad_token_id=0, eos_token_id=1)
    return BailingHybridForCausalLM(cfg)


class _TextLM(torch.nn.Module):
    """A transformers text backbone + a random lm head, for architectures shipped only inside a multimodal model."""

    def __init__(self, backbone, hidden, vocab):
        super().__init__()
        self.backbone, self.head = backbone, torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, use_cache=True, past_key_values=None, **kw):
        from types import SimpleNamespace
        out = self.backbone(input_ids=input_ids, use_cache=use_cache, past_key_values=past_key_values, **kw)
        return SimpleNamespace(logits=self.head(out.last_hidden_state), past_key_values=out.past_key_values)


def _glm5_next():
    from transformers import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
    cfg = Glm5NextTextConfig(vocab_size=V, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, n_routed_experts=4,
                             num_experts_per_tok=2, kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=0,
                             v_head_dim=16, qk_nope_head_dim=16, index_topk=16, index_head_dim=16, index_n_heads=2,
                             linear_head_dim=32, linear_num_heads=2, hc_mult=2,
                             layer_types=['full_attention', 'linear_attention'], mlp_layer_types=['dense'] * 2,
                             pad_token_id=0)
    return _TextLM(Glm5NextTextModel(cfg), 64, V)


def _mamba2():
    from transformers import Mamba2Config, Mamba2ForCausalLM
    cfg = Mamba2Config(vocab_size=V, hidden_size=64, num_heads=4, head_dim=32, state_size=16, n_groups=2,
                       num_hidden_layers=2, chunk_size=32, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return Mamba2ForCausalLM(cfg)


def _nemotron_h():
    from transformers import NemotronHConfig, NemotronHForCausalLM
    cfg = NemotronHConfig(vocab_size=V, hidden_size=64, layers_block_type=['mamba', 'attention', 'mlp'],
                          num_attention_heads=2, num_key_value_heads=1, head_dim=32, intermediate_size=128,
                          ssm_state_size=16, mamba_num_heads=4, mamba_head_dim=32, n_groups=2, chunk_size=32,
                          initializer_range=0.1)
    return NemotronHForCausalLM(cfg)


_M2 = dict(mamba_n_heads=4, mamba_d_head=32, mamba_n_groups=2, mamba_d_state=16, mamba_chunk_size=32)


def _falcon_h1():
    from transformers import FalconH1Config, FalconH1ForCausalLM
    cfg = FalconH1Config(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=32, mamba_d_ssm=128,
                         initializer_range=0.1, **_M2)
    return FalconH1ForCausalLM(cfg)


def _granitemoehybrid():
    from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM
    cfg = GraniteMoeHybridConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=1, num_local_experts=2,
                                 num_experts_per_tok=1, shared_intermediate_size=64,
                                 layer_types=['attention', 'mamba'], initializer_range=0.1, **_M2)
    return GraniteMoeHybridForCausalLM(cfg)


def _bamba():
    from transformers import BambaConfig, BambaForCausalLM
    cfg = BambaConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, attn_layer_indices=[1], initializer_range=0.1,
                      **_M2)
    return BambaForCausalLM(cfg)


def _zamba2():
    from transformers import Zamba2Config, Zamba2ForCausalLM
    cfg = Zamba2Config(vocab_size=V, hidden_size=64, num_hidden_layers=2, layers_block_type=['mamba', 'hybrid'],
                       mamba_d_state=16, n_mamba_heads=4, mamba_ngroups=2, chunk_size=32, num_attention_heads=2,
                       intermediate_size=128, initializer_range=0.1, use_mem_eff_path=False)
    return Zamba2ForCausalLM(cfg)


def _minimax():
    from transformers import MiniMaxConfig, MiniMaxForCausalLM
    cfg = MiniMaxConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=32, num_local_experts=2,
                        num_experts_per_tok=1, block_size=32, layer_types=['full_attention', 'linear_attention'])
    return MiniMaxForCausalLM(cfg)


def _fla_gdn():
    from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    return GatedDeltaNetForCausalLM(GatedDeltaNetConfig(hidden_size=128, num_hidden_layers=2, num_heads=2, head_dim=64,
                                                        expand_v=1, vocab_size=V, fuse_cross_entropy=False))


def _fla_kda():
    from fla.models import KDAConfig, KDAForCausalLM
    return KDAForCausalLM(KDAConfig(hidden_size=128, num_hidden_layers=2, num_heads=2, head_dim=64, vocab_size=V,
                                    fuse_cross_entropy=False))


def _fla_rwkv7():
    from fla.models import RWKV7Config, RWKV7ForCausalLM
    return RWKV7ForCausalLM(RWKV7Config(hidden_size=128, num_hidden_layers=2, head_dim=64, vocab_size=V,
                                        decay_low_rank_dim=16, gate_low_rank_dim=16, a_low_rank_dim=16,
                                        v_low_rank_dim=16, fuse_cross_entropy=False))


def _fla_rwkv6():
    from fla.models import RWKV6Config, RWKV6ForCausalLM
    return RWKV6ForCausalLM(RWKV6Config(hidden_size=128, num_hidden_layers=2, num_heads=2, expand_k=1, expand_v=1,
                                        vocab_size=V, fuse_cross_entropy=False))


FLA = {'fla_gdn': _fla_gdn, 'fla_kda': _fla_kda, 'fla_rwkv7': _fla_rwkv7, 'fla_rwkv6': _fla_rwkv6}

BUILDERS = {'qwen3_next': _qwen3_next, 'qwen4_exp': _qwen4_exp, 'olmo_hybrid': _olmo_hybrid,
            'kimi_linear': _kimi_linear, 'glm5_next': _glm5_next, 'bailing_hybrid': _bailing_hybrid, 'mamba2': _mamba2,
            'nemotron_h': _nemotron_h, 'falcon_h1': _falcon_h1, 'granitemoehybrid': _granitemoehybrid,
            'bamba': _bamba, 'zamba2': _zamba2, 'minimax': _minimax}
FAMILY = {'qwen3_next': 'gdn', 'qwen4_exp': 'gdn', 'olmo_hybrid': 'gdn', 'kimi_linear': 'kda', 'glm5_next': 'kda',
          'bailing_hybrid': 'kda', 'mamba2': 'mamba2', 'nemotron_h': 'mamba2', 'falcon_h1': 'mamba2',
          'granitemoehybrid': 'mamba2', 'bamba': 'mamba2', 'zamba2': 'mamba2', 'minimax': 'lightning',
          'fla_gdn': 'gdn', 'fla_kda': 'kda', 'fla_rwkv7': 'rwkv7', 'fla_rwkv6': 'rwkv6'}
BUILDERS.update(FLA)


def _slow_decay(model, lo=0.01, hi=0.5):
    """Qwen3-Next / Qwen4-Exp initialise A ~ U(0, 16): a state that forgets in a token or two, so state rounding
    hardly reaches the logits. Draw A ~ U(lo, hi) instead."""
    for m in model.modules():
        if type(m).__name__.endswith('GatedDeltaNet'):
            with torch.no_grad():
                m.A_log.copy_(torch.empty_like(m.A_log).uniform_(lo, hi).log())
    return model


def build(name, seed=0):
    torch.manual_seed(seed)
    m = BUILDERS[name]()
    if name in ('qwen3_next', 'qwen4_exp'):
        _slow_decay(m)
    if name == 'fla_rwkv7':                              # fla zero-initialises RWKV-7's output projection
        for mod in m.modules():
            if type(mod).__name__ == 'RWKV7Attention':
                torch.nn.init.normal_(mod.o_proj.weight, std=0.1)
    return m.float().eval()


def _cache(out):
    pkv = getattr(out, 'past_key_values', None)
    return ('past_key_values', pkv) if pkv is not None else ('cache_params', out.cache_params)


@torch.no_grad()
def run(model, ids, n=0, forced=None, split=None):
    """Logits [B, T + n, V] of the prompt (in one call, or two calls split at ``split``) followed by ``n`` cached
    single-token decode steps (greedy, or feeding ``forced``)."""
    if split is None:
        out = model(input_ids=ids, use_cache=True)
        logits = [out.logits]
    else:
        out = model(input_ids=ids[:, :split], use_cache=True)
        name, c = _cache(out)
        logits = [out.logits]
        out = model(input_ids=ids[:, split:], use_cache=True, **{name: c})
        logits.append(out.logits)
    nxt = out.logits[:, -1:].argmax(-1)
    for i in range(n):
        if forced is not None:
            nxt = forced[:, i:i + 1]
        name, c = _cache(out)
        out = model(input_ids=nxt, use_cache=True, **{name: c})
        logits.append(out.logits)
        nxt = out.logits[:, -1:].argmax(-1)
    return torch.cat(logits, 1).float()
