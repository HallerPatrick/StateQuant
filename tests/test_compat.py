"""statequant.compat: Ling-3.0 loading fixes for transformers' native BailingHybrid classes (config coercion, MLA
q-LoRA), on a tiny random model whose saved config.json is given Ling-3.0's int ``kda_lower_bound`` and remote-code
``architectures``. (The checkpoint's remote-code weight names are renamed by transformers' own conversion mapping;
that path is exercised on the real Ling-3.0-tiny checkpoint, README.)"""
import importlib.util
import json

import pytest
import torch

transformers = pytest.importorskip('transformers')
pytest.importorskip('transformers.models.bailing_hybrid')

from statequant import compat  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEV == 'cpu' and importlib.util.find_spec('fla') is not None:
    pytest.skip('transformers dispatches the KDA op to fla (Triton): needs a GPU', allow_module_level=True)


def _tiny_cfg(**kw):
    from transformers import BailingHybridConfig
    return BailingHybridConfig(vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                               num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2, head_dim=32,
                               kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16,
                               num_local_experts=4, num_experts_per_tok=2, n_group=1, topk_group=1,
                               first_k_dense_replace=1, layer_group_size=2, num_mtp_layers=0, pad_token_id=0,
                               eos_token_id=1, **kw)


def test_q_lora_attention():
    compat.patch_bailing_hybrid()
    from transformers import BailingHybridForCausalLM
    torch.manual_seed(0)
    m = BailingHybridForCausalLM(_tiny_cfg(q_lora_rank=8)).eval().to(DEV)
    att = m.model.layers[1].self_attn
    assert type(att).__name__ == 'BailingHybridAttention'
    assert 'q_proj' not in dict(att.named_children()) and att.q_a_proj.out_features == 8
    names = {n for n, _ in m.named_parameters()}
    assert 'model.layers.1.self_attn.q_a_layernorm.weight' in names and 'model.layers.1.self_attn.q_b_proj.weight' in names
    x = torch.randn(1, 3, 64, device=DEV)
    want = att.q_b_proj(att.q_a_layernorm(att.q_a_proj(x)))
    assert torch.equal(att.q_proj(x), want)
    ids = torch.randint(0, 128, (1, 6), device=DEV)
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True)
    assert torch.isfinite(out.logits).all()
    # full-rank MLA unchanged when q_lora_rank is unset
    m2 = BailingHybridForCausalLM(_tiny_cfg())
    assert isinstance(m2.model.layers[1].self_attn.q_proj, torch.nn.Linear)


def test_ling3_config_coerces_and_loads(tmp_path):
    compat.patch_bailing_hybrid()
    from transformers import BailingHybridForCausalLM
    torch.manual_seed(0)
    m = BailingHybridForCausalLM(_tiny_cfg(q_lora_rank=8)).eval()
    m.save_pretrained(tmp_path)
    d = json.load(open(tmp_path / 'config.json'))
    d.update(kda_lower_bound=-5, architectures=['BailingMoeV3ForCausalLM'], model_type='bailing_hybrid')
    json.dump(d, open(tmp_path / 'config.json', 'w'))
    with pytest.raises(Exception):                          # transformers' strict config rejects the int
        transformers.AutoConfig.from_pretrained(tmp_path)
    cfg = compat.ling3_config(str(tmp_path))
    assert cfg.kda_lower_bound == -5.0 and cfg.q_lora_rank == 8
    assert compat.needs_ling3_fix(str(tmp_path))
    m2, info = compat.load_ling3(str(tmp_path), dtype=torch.float32, output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys']
    ids = torch.randint(0, 128, (1, 7), device=DEV)
    with torch.no_grad():
        assert torch.allclose(m2.eval().to(DEV)(input_ids=ids).logits, m.to(DEV)(input_ids=ids).logits, atol=1e-5)
