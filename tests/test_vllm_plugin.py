"""The vLLM plugin's state-calculator patch is shared by every GDN model in vLLM: with STATEQUANT_VLLM set, the
Qwen3-Next / Qwen3.5 / OLMo-Hybrid hooks and their GDN layers (QwenGatedDeltaNetAttention,
OlmoHybridGatedDeltaNetAttention) get the packed uint8 slot; any other caller or GDN layer is refused loudly instead of
being handed a blob it cannot use. CPU only: calls the real vLLM model hooks and layer methods with stand-in configs;
no engine, no GPU."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('vllm')


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setenv('STATEQUANT_VLLM', 'pace4')
    from vllm.model_executor.layers.mamba import mamba_utils as mu
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q

    from vllm.model_executor.layers.mamba.gdn import base as gdn_base

    from statequant.integrations import vllm_olmo
    from statequant.integrations import vllm_plugin as P
    base_init = gdn_base.GatedDeltaNetAttention.__init__
    vllm_olmo.unpatch()
    saved = (mu.MambaStateShapeCalculator.__dict__['gated_delta_net_state_shape'],
             mu.MambaStateDtypeCalculator.__dict__['gated_delta_net_state_dtype'],
             {n: Q.QwenGatedDeltaNetAttention.__dict__.get(n) for n in ('__init__', '_warmup_prefill_kernels',
                                                                         '_forward_core', '_statequant_patched')},
             dict(P._STATE))
    P._STATE.clear()
    P.register()
    yield P
    gdn_base.GatedDeltaNetAttention.__init__ = base_init
    mu.MambaStateShapeCalculator.gated_delta_net_state_shape = saved[0]
    mu.MambaStateDtypeCalculator.gated_delta_net_state_dtype = saved[1]
    for n, f in saved[2].items():
        if f is None:
            if n in Q.QwenGatedDeltaNetAttention.__dict__:
                delattr(Q.QwenGatedDeltaNetAttention, n)
        else:
            setattr(Q.QwenGatedDeltaNetAttention, n, f)
    vllm_olmo.unpatch()
    P._STATE.clear()
    P._STATE.update(saved[3])


def _vllm_config(dk=128, dv=128, hk=16, hv=32):
    hf = SimpleNamespace(linear_num_key_heads=hk, linear_num_value_heads=hv, linear_key_head_dim=dk,
                         linear_value_head_dim=dv, linear_conv_kernel_dim=4)
    return SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1), speculative_config=None,
                           model_config=SimpleNamespace(hf_config=hf, hf_text_config=hf, dtype=torch.bfloat16),
                           cache_config=SimpleNamespace(mamba_cache_dtype='auto', mamba_ssm_cache_dtype='auto'))


def _layer(cls, dk=128, dv=128, hk=16, hv=32):
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    obj.tp_size, obj.num_k_heads, obj.num_v_heads, obj.head_k_dim, obj.head_v_dim = 1, hk, hv, dk, dv
    obj.conv_kernel_size, obj.num_spec = 4, 0
    cfg = _vllm_config()
    obj.model_config, obj.cache_config = cfg.model_config, cfg.cache_config
    return obj


def test_qwen_gets_packed_slot(plugin):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM
    nbytes = plugin.layout(32, 16, 128, 128).nbytes
    for model in (Qwen3NextForCausalLM, Qwen3_5ForConditionalGeneration):
        conv, ssm = model.get_mamba_state_shape_from_config(_vllm_config())
        assert ssm == (nbytes,)
        assert model.get_mamba_state_dtype_from_config(_vllm_config())[1] == torch.uint8
    layer = _layer(QwenGatedDeltaNetAttention)
    assert layer.get_state_shape()[1] == (nbytes,) and layer.get_state_dtype()[1] == torch.uint8


def test_olmo_hybrid_gets_packed_slot(plugin):
    """OLMo-Hybrid (d_k 96, d_v 192, 30 / 30 heads): the model hooks and the layer get the packed slot, sized for the
    real head dims (ragged key groups of 64: two fp16 scales per value row), and the layer runs the plugin's
    forward."""
    from vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn import OlmoHybridGatedDeltaNetAttention
    from vllm.model_executor.models.olmo_hybrid import OlmoHybridForCausalLM

    from statequant.integrations import vllm_olmo
    cfg = _vllm_config(dk=96, dv=192, hk=30, hv=30)
    lay = plugin.layout(30, 30, 96, 192)
    assert lay.breakdown()['S'] == 30 * 192 * 48 and lay.breakdown()['SC'] == 30 * 192 * 2 * 2
    conv, ssm = OlmoHybridForCausalLM.get_mamba_state_shape_from_config(cfg)
    stock_conv, stock_ssm = plugin._STATE['calculators'][0](1, 30, 30, 96, 192, 4)
    assert ssm == (lay.nbytes,) and conv == stock_conv and stock_ssm == (30, 192, 96)
    assert OlmoHybridForCausalLM.get_mamba_state_dtype_from_config(cfg)[1] == torch.uint8
    layer = _layer(OlmoHybridGatedDeltaNetAttention, dk=96, dv=192, hk=30, hv=30)
    assert layer.get_state_shape()[1] == (lay.nbytes,) and layer.get_state_dtype()[1] == torch.uint8
    assert OlmoHybridGatedDeltaNetAttention._forward_core is vllm_olmo._forward_core
    assert lay.nbytes < 0.34 * 30 * 96 * 192 * 2                 # int4 + scales + buffer: a third of bf16


def test_other_gdn_layer_construction_refused(plugin):
    """Any other GDN layer refuses in __init__, i.e. while the model is built and before any weights are loaded."""
    from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention

    class OtherGatedDeltaNet(GatedDeltaNetAttention):
        def get_state_shape(self):
            return None
    with pytest.raises(NotImplementedError, match='OtherGatedDeltaNet'):
        OtherGatedDeltaNet(SimpleNamespace(), _vllm_config(), prefix='model.layers.0')


def test_unknown_caller_refused(plugin):
    from vllm.model_executor.layers.mamba import mamba_utils as mu
    ns = {'__name__': 'some.other.model', 'mu': mu}
    exec('def hook():\n    return mu.MambaStateShapeCalculator.gated_delta_net_state_shape(1, 16, 32, 128, 128, 4)', ns)
    with pytest.raises(NotImplementedError, match='some.other.model'):
        ns['hook']()


def test_off_is_stock(monkeypatch):
    """Without STATEQUANT_VLLM the plugin patches nothing (OLMo-Hybrid gets vLLM's own shapes)."""
    monkeypatch.delenv('STATEQUANT_VLLM', raising=False)
    from vllm.model_executor.layers.mamba import mamba_utils as mu

    from statequant.integrations import vllm_plugin as P
    if P._STATE.get('registered'):
        pytest.skip('plugin already registered in this process')
    before = mu.MambaStateShapeCalculator.__dict__['gated_delta_net_state_shape']
    P.register()
    assert mu.MambaStateShapeCalculator.__dict__['gated_delta_net_state_shape'] is before
    from vllm.model_executor.models.olmo_hybrid import OlmoHybridForCausalLM
    conv, ssm = OlmoHybridForCausalLM.get_mamba_state_shape_from_config(_vllm_config(dk=96, dv=192))
    assert ssm == (32, 192, 96)
