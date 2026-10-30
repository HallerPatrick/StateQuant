"""The Mamba2 part of the vLLM plugin without an engine: the state calculators are patched only for MambaMixer2 and
the model classes built on it (other callers get vLLM's stock shapes), the layer-type selection, the group fallback
for head dims a group does not divide. Needs vLLM (skipped otherwise); the end-to-end check is
``tests/vllm_mamba2_check.py``."""
import pytest
import torch

pytest.importorskip('vllm')

import statequant as sq  # noqa: E402
from statequant.integrations import vllm_mamba2, vllm_plugin  # noqa: E402

ARGS = dict(tp_world_size=1, intermediate_size=96 * 80, n_groups=8, num_heads=96, head_dim=80, state_size=128,
            conv_kernel=4)


def _call_from(module_name, src):
    import vllm.model_executor.layers.mamba.mamba_utils as mu
    g = {'__name__': module_name, 'mu': mu, 'ARGS': ARGS, 'torch': torch}
    exec(src, g)
    return g['out']


@pytest.fixture
def patched():
    from vllm.model_executor.layers.mamba import mamba_utils as mu
    S, D = mu.MambaStateShapeCalculator, mu.MambaStateDtypeCalculator
    saved = (S.__dict__['mamba2_state_shape'], D.__dict__['mamba2_state_dtype'], dict(vllm_mamba2._STATE))
    cfg = (sq.Quantizer(4, 'v', 32), 8, 8, None)
    vllm_mamba2._STATE.clear()
    vllm_mamba2._patch_calculators(cfg)
    yield cfg
    S.mamba2_state_shape, D.mamba2_state_dtype = saved[0], saved[1]
    vllm_mamba2._STATE.clear()
    vllm_mamba2._STATE.update(saved[2])


SHAPE_SRC = 'out = mu.MambaStateShapeCalculator.mamba2_state_shape(**ARGS)'
DTYPE_SRC = "out = mu.MambaStateDtypeCalculator.mamba2_state_dtype(torch.bfloat16, 'auto', 'float32')"


def test_calculators_scoped_to_mamba_mixer2_callers(patched):
    stock_conv, stock_ssm = _call_from('some.other.model', SHAPE_SRC)
    assert stock_ssm == (96, 80, 128)
    assert _call_from('some.other.model', DTYPE_SRC) == (torch.bfloat16, torch.float32)
    for caller in ('vllm.model_executor.layers.mamba.mamba_mixer2', 'vllm.model_executor.models.nemotron_h',
                   'vllm.model_executor.models.falcon_h1', 'vllm.model_executor.models.granitemoehybrid'):
        conv, ssm = _call_from(caller, SHAPE_SRC)
        lay = vllm_mamba2.layout(patched, 96, 8, 128, 80)
        assert conv == stock_conv and ssm == (lay.nbytes,)
        assert _call_from(caller, DTYPE_SRC) == (torch.bfloat16, torch.uint8)
    # value-axis group 32 does not divide head_dim 80: halved to 16
    assert lay.group == 16 and lay.fmt.axis == 'v' and lay.bits == 4
    assert lay.nbytes < 0.36 * 96 * 80 * 128 * 2                 # int4 + g16 scales + buffer: 35% of bf16


def test_tp_and_spec_asserted(patched):
    with pytest.raises(AssertionError):
        _call_from('vllm.model_executor.models.nemotron_h',
                   'out = mu.MambaStateShapeCalculator.mamba2_state_shape(**{**ARGS, "tp_world_size": 2})')
    with pytest.raises(AssertionError):
        _call_from('vllm.model_executor.models.nemotron_h',
                   'out = mu.MambaStateShapeCalculator.mamba2_state_shape(**ARGS, num_spec=1)')
    # a foreign caller is not asserted on
    _call_from('some.other.model', 'out = mu.MambaStateShapeCalculator.mamba2_state_shape(**{**ARGS, '
               '"tp_world_size": 2})')


def test_layer_kinds(monkeypatch):
    monkeypatch.delenv('STATEQUANT_VLLM_LAYERS', raising=False)
    assert vllm_plugin.layer_kinds() == ('gdn', 'mamba2', 'kda')
    monkeypatch.setenv('STATEQUANT_VLLM_LAYERS', 'mamba2')
    assert vllm_plugin.layer_kinds() == ('mamba2',)
    monkeypatch.setenv('STATEQUANT_VLLM_LAYERS', 'gdn, mamba2')
    assert vllm_plugin.layer_kinds() == ('gdn', 'mamba2')
    monkeypatch.setenv('STATEQUANT_VLLM_LAYERS', 'kda,mamba2')
    assert vllm_plugin.layer_kinds() == ('kda', 'mamba2')
    monkeypatch.setenv('STATEQUANT_VLLM_LAYERS', 'lightning')
    with pytest.raises(AssertionError):
        vllm_plugin.layer_kinds()
