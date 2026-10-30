"""The KDA part of the vLLM plugin without an engine: the state calculators are patched only for vLLM's
``bailing_moe_v3`` (Ling-3.0) module (its model hooks and ``BailingMoeV3KimiDeltaAttention``); vLLM's other KDA users
(Kimi Linear's GDN-style layer, GLM-5-Next, Kimi-K3) get the stock shapes and dtypes; TP > 1 and speculative decoding
are refused. Needs vLLM (skipped otherwise); the end-to-end check is ``tests/vllm_kda_check.py``."""
import pytest
import torch

pytest.importorskip('vllm')

import statequant as sq  # noqa: E402
from statequant.integrations import vllm_kda  # noqa: E402

ARGS = dict(tp_world_size=1, num_heads=16, head_dim=128, conv_kernel_size=4)      # Ling-3.0-tiny KDA layer
OTHER = ('vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn', 'vllm.models.glm5next.common.kda',
         'vllm.models.kimi_k3.nvidia.kda', 'some.other.model')


def _call_from(module_name, src):
    import vllm.model_executor.layers.mamba.mamba_utils as mu
    g = {'__name__': module_name, 'mu': mu, 'ARGS': ARGS, 'torch': torch}
    exec(src, g)
    return g['out']


@pytest.fixture
def patched():
    from vllm.model_executor.layers.mamba import mamba_utils as mu
    S, D = mu.MambaStateShapeCalculator, mu.MambaStateDtypeCalculator
    saved = (S.__dict__['kda_state_shape'], D.__dict__['kda_state_dtype'], dict(vllm_kda._STATE))
    cfg = (sq.Quantizer(4, 'k', 64), 8, 8, None)
    vllm_kda._STATE.clear()
    vllm_kda._patch_calculators(cfg)
    yield cfg
    S.kda_state_shape, D.kda_state_dtype = saved[0], saved[1]
    vllm_kda._STATE.clear()
    vllm_kda._STATE.update(saved[2])


SHAPE_SRC = 'out = mu.MambaStateShapeCalculator.kda_state_shape(**ARGS)'
DTYPE_SRC = "out = mu.MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, 'auto', 'auto')"


def test_calculators_scoped_to_bailing_moe_v3(patched):
    stock_conv, stock_rec = _call_from('some.other.model', SHAPE_SRC)
    assert stock_rec == (16, 128, 128)
    for caller in OTHER:
        assert _call_from(caller, SHAPE_SRC) == (stock_conv, stock_rec)
        assert _call_from(caller, DTYPE_SRC) == (torch.bfloat16, torch.float32)
    conv, rec = _call_from('vllm.model_executor.models.bailing_moe_v3', SHAPE_SRC)
    lay = vllm_kda.layout(patched, 16, 128)
    assert conv == stock_conv and rec == (lay.nbytes,)
    assert _call_from('vllm.model_executor.models.bailing_moe_v3', DTYPE_SRC) == (torch.bfloat16, torch.uint8)
    # int4 k g64 + scales + int8 buffer + per-channel decays: under a quarter of the stock fp32 state
    assert lay.nbytes < 0.25 * 16 * 128 * 128 * 4
    assert lay.bits == 4 and lay.fmt.axis == 'k' and lay.group == 64


def test_tp_and_spec_asserted(patched):
    with pytest.raises(AssertionError):
        _call_from('vllm.model_executor.models.bailing_moe_v3',
                   'out = mu.MambaStateShapeCalculator.kda_state_shape(**{**ARGS, "tp_world_size": 2})')
    with pytest.raises(AssertionError):
        _call_from('vllm.model_executor.models.bailing_moe_v3',
                   'out = mu.MambaStateShapeCalculator.kda_state_shape(**ARGS, num_spec=1)')
    _call_from('some.other.model', 'out = mu.MambaStateShapeCalculator.kda_state_shape(**{**ARGS, '
               '"tp_world_size": 2})')


def test_model_hook_goes_through_the_calculator(patched):
    """Ling-3.0's model-level hook (used for the hybrid page alignment) sees the packed blob too."""
    from types import SimpleNamespace

    from vllm.model_executor.models import bailing_moe_v3 as M
    hf = SimpleNamespace(num_attention_heads=16, head_dim=128, short_conv_kernel_size=4)
    vc = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf), speculative_config=None,
                         parallel_config=SimpleNamespace(tensor_parallel_size=1))
    conv, rec = M.BailingMoeV3ForCausalLM.get_mamba_state_shape_from_config(vc)
    assert rec == (vllm_kda.layout(patched, 16, 128).nbytes,)
