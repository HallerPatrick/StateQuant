"""``statequant.apply(..., backend='packed')`` on a small random fla GatedDeltaNet and, if cached, on Qwen3.5-0.8B
(transformers path): the bf16 packed control tracks the unpatched model through cached decoding; int4 decodes run the
kernels and stay close to the reference path. Needs a GPU, fla and Triton."""
import os

import pytest
import torch

pytest.importorskip('fla')
if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)

import statequant as sq  # noqa: E402


def greedy(model, ids, n, forced=None):
    """Logits of the prompt's last position and of n decode steps; ``forced`` feeds a fixed continuation."""
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        logits, pkv = [out.logits[:, -1]], out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        for i in range(n):
            if forced is not None:
                nxt = forced[:, i:i + 1]
            out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            logits.append(out.logits[:, -1])
            nxt = out.logits[:, -1:].argmax(-1)
    return torch.stack(logits, 1).float()


@pytest.fixture(scope='module')
def fla_model():
    from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    torch.manual_seed(0)
    cfg = GatedDeltaNetConfig(hidden_size=256, num_hidden_layers=2, num_heads=2, head_dim=128, expand_v=1,
                              vocab_size=512, fuse_cross_entropy=False)
    return GatedDeltaNetForCausalLM(cfg).cuda().to(torch.bfloat16).eval()


def _ids(V, T=300, B=2):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(1)).cuda()


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize('m', [1, 8])
def test_fla_bf16_control(fla_model, m):
    x = _ids(512)
    forced = _ids(512, T=24)
    ref = greedy(fla_model, x, 24, forced)
    sch = sq.Schedule(prefill=256, decode=m, buffer_bits=0)
    with sq.apply(fla_model, sq.FloatCast('bf16'), sch, backend='packed'):
        out = greedy(fla_model, x, 24, forced)
    assert _rel(out, ref) < 2e-2, _rel(out, ref)


@pytest.mark.parametrize('axis,group', [('k', 64), ('v', 32)])
def test_fla_int4_matches_reference(fla_model, axis, group):
    """Like for like: prompt ends on a rounding boundary, fp16 scales on both paths, bf16 buffer. The reference rounds
    the raw state every m tokens; the packed path folds a buffer of bf16 (key, delta value) pairs - equal up to the
    bf16 rounding of the buffer. That rounding flips a few int4 codes at each fold, which moves the output by a small
    fraction of the quantisation error itself (measured: 1.4% vs 8.0% for k64); the kernel arithmetic is pinned exactly
    in test_kernels.py."""
    x, forced = _ids(512, T=256), _ids(512, T=24)
    fmt = sq.Quantizer(4, axis, group, scale_dtype='fp16')
    sch = sq.Schedule(prefill=256, decode=8, buffer_bits=16)
    with sq.apply(fla_model, fmt, sch, backend='packed'):
        packed = greedy(fla_model, x, 24, forced)
    with sq.apply(fla_model, fmt, sq.Schedule(prefill=256, decode=8, buffer_bits=0)):
        ref = greedy(fla_model, x, 24, forced)
    base = greedy(fla_model, x, 24, forced)
    print(f'{axis}{group}: packed vs reference {_rel(packed, ref):.3e}, reference vs fp {_rel(ref, base):.3e}')
    assert _rel(packed, ref) < 0.25 * _rel(ref, base), (_rel(packed, ref), _rel(ref, base))


REPO = os.environ.get('SQ_TEST_HF', 'Qwen/Qwen3.5-0.8B')


@pytest.fixture(scope='module')
def qwen():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(REPO)
        model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16).cuda().eval()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f'{REPO} not available: {e}')
    return model, tok


def test_qwen_packed(qwen):
    model, tok = qwen
    text = ('The history of the region begins with the first settlements along the river, where farmers grew '
            'wheat and barley. Over the following centuries the towns expanded, trade routes connected them ') * 12
    x = tok(text, return_tensors='pt').input_ids[:, :400].cuda()
    forced = tok(' and the cities grew into centres of learning and commerce for the whole valley',
                 return_tensors='pt').input_ids.cuda()
    n = forced.shape[1]
    base = greedy(model, x, n, forced)
    with sq.apply(model, sq.FloatCast('bf16'), sq.Schedule(256, 8, 0), backend='packed'):
        ctl = greedy(model, x, n, forced)
    with sq.apply(model, sq.pace4(paper=False), backend='packed'):
        p4 = greedy(model, x, n, forced)
    with sq.apply(model, sq.pace4(paper=False)):
        r4 = greedy(model, x, n, forced)
    agree = lambda a, b: (a.argmax(-1) == b.argmax(-1)).float().mean().item()  # noqa: E731
    print(f'bf16 control rel {_rel(ctl, base):.3e}; packed PACE-4 rel {_rel(p4, base):.3e} agree {agree(p4, base):.3f};'
          f' reference PACE-4 rel {_rel(r4, base):.3e} agree {agree(r4, base):.3f}')
    assert _rel(ctl, base) < 2e-2
    assert agree(p4, base) > 0.8


@pytest.fixture(scope='module')
def olmo_tiny():
    """A tiny OLMo-Hybrid (transformers' real layer: d_k 96, d_v 192, beta in (0, 2)) with slow decays."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from _tiny import build
    try:
        model = build('olmo_hybrid')
    except ImportError as e:
        pytest.skip(f'OLMo-Hybrid not in this transformers: {e}')
    for mod in model.modules():
        if type(mod).__name__.endswith('GatedDeltaNet'):
            with torch.no_grad():
                mod.A_log.copy_(torch.empty_like(mod.A_log).uniform_(0.01, 0.5).log())
    return model.cuda().eval()                              # fp32: the bf16 rounding of activations would
                                                             # hide the state format


@pytest.mark.parametrize('fmt', [sq.Quantizer(4, 'k', 64, scale_dtype='fp16'), sq.Quantizer(4, 'v', 32, scale_dtype='fp16')],
                         ids=['k64-ragged', 'v32'])
def test_olmo_hybrid_packed(olmo_tiny, fmt):
    """Non-power-of-two head dims through the packed backend: the bf16 control tracks the unpatched model, int4
    (ragged key groups of 64 on d_k 96) stays close to the reference path (as test_fla_int4_matches_reference)."""
    from _tiny import V
    x, forced = _ids(V, T=256), _ids(V, T=24)
    base = greedy(olmo_tiny, x, 24, forced)
    with sq.apply(olmo_tiny, sq.FloatCast('bf16'), sq.Schedule(prefill=256, decode=8, buffer_bits=0),
                  backend='packed'):
        ctl = greedy(olmo_tiny, x, 24, forced)
    sch = sq.Schedule(prefill=256, decode=8, buffer_bits=16)
    with sq.apply(olmo_tiny, fmt, sch, backend='packed'):
        packed = greedy(olmo_tiny, x, 24, forced)
    with sq.apply(olmo_tiny, fmt, sq.Schedule(prefill=256, decode=8, buffer_bits=0)):
        ref = greedy(olmo_tiny, x, 24, forced)
    print(f'\nOLMo tiny: bf16 control rel {_rel(ctl, base):.3e}; {fmt.axis}{fmt.group}: packed vs reference '
          f'{_rel(packed, ref):.3e}, reference vs fp {_rel(ref, base):.3e}')
    assert _rel(ctl, base) < 2e-2
    assert _rel(packed, ref) < 0.25 * _rel(ref, base), (_rel(packed, ref), _rel(ref, base))
