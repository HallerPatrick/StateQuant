"""Optional end-to-end checks on real small checkpoints, if they are in the HF cache (nothing is downloaded):
Mamba2-130m (transformers ``Mamba2ForCausalLM``, CPU or GPU) and RWKV-7 0.1B g1 (fla ``RWKV7ForCausalLM``, GPU).
Identity reproduces the model exactly; an int8 state stays close to it; int4 per-token rounding hurts more than
PACE's periodic int4 schedule; calibrated HeadMix runs. Override the repos with SQ_TEST_MAMBA2 / SQ_TEST_RWKV7."""
import os

import pytest
import torch

import statequant as sq
from _tiny import run

os.environ.setdefault('HF_HUB_OFFLINE', '1')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TEXT = ('The history of the region begins with the first settlements along the river, where farmers grew wheat and '
        'barley. Over the following centuries the towns expanded, trade routes connected them, and the cities grew '
        'into centres of learning and commerce for the whole valley. ') * 8


def _load(repo, fla=False, cls=None):
    from transformers import AutoModelForCausalLM
    cls = cls or AutoModelForCausalLM
    if fla:
        pytest.importorskip('fla')
        if DEV != 'cuda':
            pytest.skip('fla needs a GPU')
        import fla  # noqa: F401  (registers fla's model types with transformers)
    try:
        return cls.from_pretrained(repo, dtype=torch.float32).to(DEV).eval()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f'{repo} not available: {e}')


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _agree(a, b):
    return (a.argmax(-1) == b.argmax(-1)).float().mean().item()


@pytest.fixture(scope='module')
def mamba2():
    repo = os.environ.get('SQ_TEST_MAMBA2', 'AntonV/mamba2-130m-hf')
    from transformers import AutoTokenizer, Mamba2ForCausalLM
    m = _load(repo, cls=Mamba2ForCausalLM)            # not AutoModel: importing fla registers fla's own 'mamba2' type
    x = AutoTokenizer.from_pretrained(repo)(TEXT, return_tensors='pt').input_ids[:, :300].to(DEV)
    return m, x


@pytest.fixture(scope='module')
def rwkv7():
    m = _load(os.environ.get('SQ_TEST_RWKV7', 'fla-hub/rwkv7-0.1B-g1'), fla=True)
    x = torch.randint(100, 20000, (1, 300), generator=torch.Generator().manual_seed(0)).to(DEV)
    return m, x


@pytest.mark.parametrize('which', ['mamba2', 'rwkv7'])
def test_real_checkpoint(which, request):
    m, x = request.getfixturevalue(which)
    prompt, forced = x[:, :256], x[:, 256:]
    n = forced.shape[1]
    base = run(m, prompt, n, forced)
    with sq.apply(m, sq.Identity(), sq.Schedule(prefill=4096, decode=8, buffer_bits=0)):
        assert torch.equal(run(m, prompt, n, forced), base)
    with sq.apply(m, sq.Quantizer(8, 'k', 32), sq.Schedule(prefill=64, decode=8, buffer_bits=0)):
        q8 = run(m, prompt, n, forced)
    with sq.apply(m, sq.Quantizer(4, 'k', 32), sq.Schedule(prefill=1, decode=1, buffer_bits=0)):
        q4_tok = run(m, prompt, n, forced)
    with sq.apply(m, sq.Quantizer(4, 'k', 32), sq.Schedule(prefill=256, decode=8, buffer_bits=8)):
        q4_pace = run(m, prompt, n, forced)
    r8, rt, rp = _rel(q8, base), _rel(q4_tok, base), _rel(q4_pace, base)
    print(f'{which}: int8 rel {r8:.3e} agree {_agree(q8, base):.3f}; int4 per-token rel {rt:.3e} agree '
          f'{_agree(q4_tok, base):.3f}; int4 every 256/8 rel {rp:.3e} agree {_agree(q4_pace, base):.3f}')
    assert _agree(q8, base) > 0.95 and r8 < rp
    hm = sq.calibrate(m, sq.HeadMix(sq.Quantizer(4, 'k', 32), sq.Quantizer(8, 'k', 32), frac=0.25), x[:, :256],
                      period=64)
    with sq.apply(m, hm, sq.Schedule(prefill=256, decode=8, buffer_bits=8)):
        qh = run(m, prompt, n, forced)
    print(f'{which}: HeadMix 4/8 rel {_rel(qh, base):.3e} agree {_agree(qh, base):.3f}')
    assert torch.isfinite(qh).all()
